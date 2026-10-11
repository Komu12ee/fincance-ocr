'use strict';

/**
 * GAD (fg2) auth + account-management routes. Mounted at /gad-auth.
 *
 * Sign-in:
 *   POST /gad-auth/login            { identifier, password }         password login
 *   POST /gad-auth/otp/request      { identifier, channel, purpose } send a code
 *   POST /gad-auth/otp/login        { identifier, code }             passwordless login
 *   POST /gad-auth/reset-password   { identifier, code, new_password } reset via code
 *   POST /gad-auth/logout
 *   GET  /gad-auth/session                                            whoami
 *   GET  /gad-auth/channels                                           email/sms availability
 *
 * `identifier` may be a username, email, or mobile number. Password login also
 * still accepts `username` for backward compatibility.
 *
 * Admin-only account management (admin session required):
 *   GET/POST /gad-auth/users · POST /users/:u/password · /role · /contact · DELETE /users/:u
 */

const express = require('express');
const store = require('./store');
const otp = require('./otp');
const notify = require('./notify');
const limiter = require('./ratelimit');
const verifytoken = require('./verifytoken');
const { COOKIE_NAME, cookieOptions, issueToken, verifyToken, getToken, requireGadAuth, requireGadAdmin } = require('./auth');

// Signup is unauthenticated, so it is limited per source IP: a cooldown between
// attempts and a daily ceiling. Without these one script can enumerate usernames
// or flood the mail relay.
const SIGNUP_COOLDOWN_SECONDS = () => parseInt(process.env.SIGNUP_COOLDOWN_SECONDS || '', 10) || 30;
const SIGNUP_DAILY_CAP = () => parseInt(process.env.SIGNUP_DAILY_CAP || '', 10) || 10;

function clientIp(req) {
  return String((req.headers && req.headers['x-forwarded-for']) || req.ip || 'unknown')
    .split(',')[0].trim();
}

function escapeHtml(s) {
  return String(s == null ? '' : s)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}

/** Result page for the emailed confirmation link (opened in a mail client). */
function verifyPage(ok, message) {
  const app = (BRANDING && BRANDING.label) || 'Document Assistant';
  return `<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>${escapeHtml(app)} — email confirmation</title>
<style>
 body{font-family:system-ui,-apple-system,Segoe UI,Arial,sans-serif;background:#f4f6f9;color:#1b2430;
      display:flex;min-height:100vh;align-items:center;justify-content:center;margin:0;padding:20px}
 .card{background:#fff;border:1px solid #e2e8f0;border-radius:10px;padding:32px;max-width:440px;text-align:center;
       box-shadow:0 1px 3px rgba(0,0,0,.06)}
 .icon{font-size:42px;line-height:1}
 h1{font-size:19px;margin:14px 0 8px;color:#1c3d5c}
 p{margin:0 0 20px;color:#475569;font-size:14px;line-height:1.5}
 a.btn{display:inline-block;background:#1c3d5c;color:#fff;text-decoration:none;padding:11px 22px;
       border-radius:6px;font-weight:600;font-size:14px}
</style></head><body><div class="card">
 <div class="icon">${ok ? '✅' : '⚠️'}</div>
 <h1>${ok ? 'Email confirmed' : 'Could not confirm'}</h1>
 <p>${escapeHtml(message)}</p>
 <a class="btn" href="../login">Go to sign in</a>
</div></body></html>`;
}

let BRANDING = null;
try { BRANDING = require('../config').BRANDING; } catch (_) { /* branding optional */ }

const router = express.Router();
router.use(express.json());

// ── helpers ──────────────────────────────────────────────────────────────────
function setSession(res, user) {
  res.cookie(COOKIE_NAME, issueToken({ username: user.username, role: user.role }), cookieOptions());
}
function maskEmail(e) {
  if (!e) return '';
  const [u, d] = e.split('@');
  const head = u.length <= 2 ? u[0] || '' : u.slice(0, 2);
  return `${head}${'*'.repeat(Math.max(1, u.length - 2))}@${d}`;
}
function maskMobile(m) {
  return m ? `${'*'.repeat(6)}${m.slice(-4)}` : '';
}
function destFor(user, channel) {
  return channel === 'sms' ? user.mobile : user.email;
}

// An account that has not opened its confirmation link cannot sign in by ANY route.
// Enforced in one helper used by both login paths, so adding a third sign-in method
// later cannot accidentally skip the check.
const UNVERIFIED_MSG = 'Confirm your email address first — check your inbox for the link we sent.';

function blockUnverified(res, user) {
  if (store.isEmailVerified(user)) return false;
  res.status(403).json({ success: false, error: UNVERIFIED_MSG, emailVerified: false });
  return true;
}

// ── password login ───────────────────────────────────────────────────────────
router.post('/login', async (req, res) => {
  const { identifier, username, password } = req.body || {};
  const id = identifier || username;
  if (!id || typeof id !== 'string' || !password || typeof password !== 'string') {
    return res.status(400).json({ success: false, error: 'Username/email/mobile and password are required.' });
  }
  const user = await store.verifyCredentials(id, password);
  if (!user) return res.status(401).json({ success: false, error: 'Invalid credentials.' });
  // Checked AFTER the password so an unverified-account message can never be used
  // to discover which addresses are registered.
  if (blockUnverified(res, store.getUserByIdentifier(id))) return;
  setSession(res, user);
  console.log(`[gad-auth] Password login: ${user.username} (${user.role})`);
  return res.json({ success: true, user: { username: user.username, role: user.role } });
});

// ── self-service signup ──────────────────────────────────────────────────────
// Open to holders of a government email address (see store.isGovEmail). The account
// is created as a plain 'user' — signup can never mint an admin — and stays inactive
// until the emailed link is opened.
function baseUrl(req) {
  // Honour the reverse proxy so the confirmation link points at the public URL
  // rather than the container's own host/port.
  const proto = (req.headers['x-forwarded-proto'] || req.protocol || 'http').split(',')[0].trim();
  const host = (req.headers['x-forwarded-host'] || req.headers.host || '').split(',')[0].trim();
  const prefix = (req.headers['x-forwarded-prefix'] || '').replace(/\/$/, '');
  return `${process.env.PIPELINE_PUBLIC_URL || `${proto}://${host}${prefix}`}`.replace(/\/$/, '');
}

async function issueVerification(req, user) {
  const link = `${baseUrl(req)}/gad-auth/verify?token=${encodeURIComponent(verifytoken.issue(user.username))}`;
  const sent = await notify.sendVerification({
    to: user.email, link, ttlSeconds: verifytoken.ttlSeconds(), brand: BRANDING,
  });
  if (sent.mock) console.log(`[gad-auth] (mock) confirmation link for ${user.username}: ${link}`);
  return sent;
}

router.post('/signup', async (req, res) => {
  const { username, password, email, mobile } = req.body || {};

  // Rate-limit by source IP: signup is unauthenticated, so without this one script
  // can enumerate usernames or flood the mail relay.
  const ip = clientIp(req);
  const gate = limiter.cooldown(`signup:cool:${ip}`, SIGNUP_COOLDOWN_SECONDS());
  if (!gate.ok) {
    return res.status(429).json({ success: false, error: `Please wait ${gate.retryAfter}s before signing up again.`, retryAfter: gate.retryAfter });
  }
  const cap = limiter.counter(`signup:day:${ip}`, SIGNUP_DAILY_CAP(), 24 * 3600);
  if (!cap.ok) {
    return res.status(429).json({ success: false, error: 'Too many signups from this network today. Contact an administrator.', retryAfter: cap.retryAfter });
  }

  let created;
  try {
    created = store.signupUser({ username, password, email, mobile });
  } catch (err) {
    // Validation failure — a typo'd address or a taken username. Deliberately does
    // NOT start the cooldown: making someone wait 30s to correct a typo is hostile,
    // and nothing expensive happened (no account written, no mail sent). The daily
    // counter above still caps how many attempts one network gets.
    return res.status(400).json({ success: false, error: err.message });
  }
  // From here an account exists and mail is about to go out — the part worth
  // rate-limiting. Start the cooldown now.
  limiter.mark(`signup:cool:${ip}`);

  const sent = await issueVerification(req, { username: created.username, email: created.email });
  if (!sent.ok) {
    // The account exists but is unreachable. Say so plainly — silently succeeding
    // would leave someone waiting for mail that is never coming.
    console.error(`[gad-auth] Signup ${created.username}: confirmation email failed — ${sent.error}`);
    return res.status(502).json({
      success: false,
      error: 'Account created, but the confirmation email could not be sent. Contact an administrator.',
    });
  }
  console.log(`[gad-auth] Signup pending confirmation: ${created.username} <${created.email}>`);
  return res.status(201).json({
    success: true,
    message: `Almost done — open the confirmation link we sent to ${maskEmail(created.email)}.`,
    destination: maskEmail(created.email),
  });
});

// Opened from the email client, so this answers with a PAGE, not JSON.
router.get('/verify', (req, res) => {
  const v = verifytoken.verify(req.query && req.query.token);
  if (!v.ok) return res.status(400).send(verifyPage(false, v.error));
  try {
    store.markEmailVerified(v.username);
  } catch (err) {
    return res.status(400).send(verifyPage(false, err.message));
  }
  console.log(`[gad-auth] Email confirmed: ${v.username}`);
  return res.send(verifyPage(true, 'Your email is confirmed. You can sign in now.'));
});

// Re-send the link. Always answers the same way, whether or not the account exists
// or is already confirmed — otherwise this becomes an account-enumeration oracle.
router.post('/resend-verification', async (req, res) => {
  const { identifier } = req.body || {};
  const generic = { success: true, message: 'If that account exists and is unconfirmed, a new link is on its way.' };
  if (!identifier) return res.json(generic);

  const ip = clientIp(req);
  const gate = limiter.cooldown(`resend:cool:${ip}`, SIGNUP_COOLDOWN_SECONDS());
  if (!gate.ok) return res.status(429).json({ success: false, error: `Please wait ${gate.retryAfter}s.`, retryAfter: gate.retryAfter });
  limiter.mark(`resend:cool:${ip}`);

  const user = store.getUserByIdentifier(identifier);
  if (user && user.email && !store.isEmailVerified(user)) {
    const sent = await issueVerification(req, user);
    if (!sent.ok) console.error(`[gad-auth] Resend failed for ${user.username}: ${sent.error}`);
  }
  return res.json(generic);
});

// ── OTP request (login or reset) ─────────────────────────────────────────────
router.post('/otp/request', async (req, res) => {
  const { identifier, channel, purpose } = req.body || {};
  const ch = channel === 'sms' ? 'sms' : 'email';
  const pp = purpose === 'reset' ? 'reset' : 'login';
  if (!identifier || typeof identifier !== 'string') {
    return res.status(400).json({ success: false, error: 'Enter your username, email, or mobile number.' });
  }
  const user = store.getUserByIdentifier(identifier);
  if (!user) return res.status(404).json({ success: false, error: 'No account found for that identifier.' });

  const dest = destFor(user, ch);
  if (!dest) {
    return res.status(400).json({ success: false, error: `No ${ch === 'sms' ? 'mobile number' : 'email'} on file for this account. Ask an admin to add one.` });
  }

  const issued = otp.issue(user.username, pp, ch);
  if (!issued.ok) return res.status(429).json({ success: false, error: issued.error, retryAfter: issued.retryAfter });

  const sent = await notify.sendOtp({
    channel: ch, to: dest, code: issued.code, purpose: pp,
    ttlSeconds: issued.expiresInSeconds, brand: BRANDING,
  });
  if (!sent.ok) return res.status(502).json({ success: false, error: sent.error || 'Could not send code.' });

  return res.json({
    success: true,
    channel: ch,
    destination: ch === 'sms' ? maskMobile(dest) : maskEmail(dest),
    expiresInSeconds: issued.expiresInSeconds,
    mock: !!sent.mock,           // dev hint: code was logged, not really delivered
  });
});

// ── passwordless OTP login ───────────────────────────────────────────────────
router.post('/otp/login', (req, res) => {
  const { identifier, code } = req.body || {};
  if (!identifier || !code) return res.status(400).json({ success: false, error: 'Identifier and code are required.' });
  const user = store.getUserByIdentifier(identifier);
  if (!user) return res.status(404).json({ success: false, error: 'No account found for that identifier.' });

  const v = otp.verify(user.username, 'login', code);
  if (!v.ok) return res.status(401).json({ success: false, error: v.error });
  if (blockUnverified(res, user)) return;
  setSession(res, user);
  console.log(`[gad-auth] OTP login: ${user.username} (${user.role})`);
  return res.json({ success: true, user: { username: user.username, role: user.role } });
});

// ── password reset via OTP ───────────────────────────────────────────────────
router.post('/reset-password', (req, res) => {
  const { identifier, code, new_password } = req.body || {};
  if (!identifier || !code || !new_password) {
    return res.status(400).json({ success: false, error: 'Identifier, code, and new password are required.' });
  }
  const user = store.getUserByIdentifier(identifier);
  if (!user) return res.status(404).json({ success: false, error: 'No account found for that identifier.' });

  const v = otp.verify(user.username, 'reset', code);
  if (!v.ok) return res.status(401).json({ success: false, error: v.error });
  try {
    store.setPassword(user.username, new_password);
  } catch (err) {
    return res.status(400).json({ success: false, error: err.message });
  }
  console.log(`[gad-auth] Password reset via OTP: ${user.username}`);
  return res.json({ success: true, message: 'Password updated. You can sign in now.' });
});

// ── logout / session / channels ──────────────────────────────────────────────
router.post('/logout', (_req, res) => {
  res.clearCookie(COOKIE_NAME, { path: '/' });
  return res.json({ success: true });
});

router.get('/session', (req, res) => {
  const payload = verifyToken(getToken(req));
  if (!payload || !payload.role) return res.json({ authenticated: false });
  return res.json({ authenticated: true, user: { username: payload.username, role: payload.role } });
});

router.get('/channels', (_req, res) => res.json({ success: true, channels: notify.channelStatus() }));

// Public: lets the signup page show the operator's ACTUAL allowed domains rather
// than a hardcoded guess that drifts when SIGNUP_EMAIL_DOMAINS is changed.
router.get('/signup-info', (_req, res) =>
  res.json({ success: true, domains: store.allowedEmailDomains() }));

// ── account management (admin only) ──────────────────────────────────────────
const adminOnly = [requireGadAuth({ loginUrl: '/login' }), requireGadAdmin()];

router.get('/users', adminOnly, (_req, res) => res.json({ success: true, users: store.listUsers() }));

router.post('/users', adminOnly, (req, res) => {
  const { username, password, role, email, mobile } = req.body || {};
  try {
    const created = store.createUser({ username, password, role, email, mobile });
    console.log(`[gad-auth] Account created: ${created.username} (${created.role})`);
    return res.status(201).json({ success: true, user: created });
  } catch (err) {
    return res.status(400).json({ success: false, error: err.message });
  }
});

router.post('/users/:username/password', adminOnly, (req, res) => {
  try {
    return res.json({ success: true, user: store.setPassword(req.params.username, (req.body || {}).password) });
  } catch (err) {
    return res.status(400).json({ success: false, error: err.message });
  }
});

router.post('/users/:username/role', adminOnly, (req, res) => {
  try {
    return res.json({ success: true, user: store.setRole(req.params.username, (req.body || {}).role) });
  } catch (err) {
    return res.status(400).json({ success: false, error: err.message });
  }
});

router.post('/users/:username/contact', adminOnly, (req, res) => {
  const { email, mobile } = req.body || {};
  try {
    return res.json({ success: true, user: store.setContact(req.params.username, { email, mobile }) });
  } catch (err) {
    return res.status(400).json({ success: false, error: err.message });
  }
});

router.delete('/users/:username', adminOnly, (req, res) => {
  if (req.gadUser && req.gadUser.username === String(req.params.username).trim().toLowerCase()) {
    return res.status(400).json({ success: false, error: 'You cannot delete your own account while signed in.' });
  }
  try {
    return res.json({ success: true, deleted: store.deleteUser(req.params.username).username });
  } catch (err) {
    return res.status(400).json({ success: false, error: err.message });
  }
});

module.exports = router;
