'use strict';

/**
 * GAD (fg2) user store — standalone, file-backed accounts with roles + contacts.
 *
 * Two roles:  admin (full access incl. document workspace) · user (chatbot only).
 * Accounts are ADMIN-PROVISIONED (no public signup); an admin creates them from
 * the account-management panel. Each account may carry an email and/or mobile
 * number, which enable the optional email/SMS OTP flows (see gad/otp.js + notify/).
 *
 * Persisted as JSON under the gitignored data root, so it survives restarts with
 * no database. Passwords are bcrypt-hashed; contacts are stored in the clear
 * (needed to send OTPs). The store is intentionally the ONLY module that touches
 * the file — otp/notify/routes depend on this small surface, so the backing store
 * can later be swapped (SQLite, Postgres…) without changing callers.
 */

const fs = require('fs');
const path = require('path');
const bcrypt = require('bcryptjs');

const ROLES = ['admin', 'user'];

// Which bot this process serves. Every pipeline (chips/fg/fg2) now runs this same
// standalone login, so the account store and seed env are per-pipeline.
const PIPELINE = (process.env.PIPELINE || 'chips').toLowerCase();

function storeFile() {
  // GAD_USERS_FILE forces a single explicit location (rarely used; note that in a
  // shared-env deployment it would make every bot share ONE account list).
  if (process.env.GAD_USERS_FILE) return process.env.GAD_USERS_FILE;
  const repoRoot = path.join(__dirname, '..', '..', '..', '..');
  const dataRoot = process.env.RAG_DATA_ROOT || path.join(repoRoot, 'data');
  return path.join(dataRoot, PIPELINE, 'auth', 'users.json');
}

function readAll() {
  try {
    const parsed = JSON.parse(fs.readFileSync(storeFile(), 'utf8'));
    return Array.isArray(parsed.users) ? parsed.users : [];
  } catch (_) {
    return [];
  }
}

function writeAll(users) {
  const file = storeFile();
  fs.mkdirSync(path.dirname(file), { recursive: true });
  const tmp = `${file}.tmp`;
  fs.writeFileSync(tmp, JSON.stringify({ users }, null, 2), { mode: 0o600 });
  fs.renameSync(tmp, file); // atomic replace
}

// ── Normalisers ──────────────────────────────────────────────────────────────
function normUser(name) {
  return String(name || '').trim().toLowerCase();
}
function normEmail(email) {
  const e = String(email || '').trim().toLowerCase();
  return e || null;
}
/** Reduce a phone number to the 10-digit CDAC form (drop +91 / leading 0). */
function normMobile(mobile) {
  let d = String(mobile || '').replace(/\D/g, '');
  if (d.length > 10 && d.startsWith('91')) d = d.slice(-10);
  else if (d.length === 11 && d.startsWith('0')) d = d.slice(1);
  return d.length === 10 ? d : (d || null);
}

function isEmail(s) { return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(String(s || '')); }

// ── Government email allowlist ───────────────────────────────────────────────
// Self-signup is open to government staff only, so the address must sit under an
// approved domain. nic.in is included alongside gov.in because NIC runs the mail
// infrastructure for much of Indian government and a great many officials hold
// @nic.in rather than @<dept>.gov.in addresses.
//
// Matching is on DOMAIN BOUNDARIES, not "endsWith": "gov.in" must accept
// user@cg.gov.in and user@gov.in, and reject user@notgov.in and user@gov.in.evil.com.
// Override with SIGNUP_EMAIL_DOMAINS (comma-separated) — no code edit needed to
// widen this to another state, or narrow it to cg.gov.in only.
const DEFAULT_EMAIL_DOMAINS = ['gov.in', 'nic.in'];

function allowedEmailDomains() {
  const raw = (process.env.SIGNUP_EMAIL_DOMAINS || '').trim();
  if (!raw) return DEFAULT_EMAIL_DOMAINS;
  const list = raw.split(',').map((d) => d.trim().toLowerCase().replace(/^@/, '')).filter(Boolean);
  return list.length ? list : DEFAULT_EMAIL_DOMAINS;
}

/** Is this address under an approved government domain? */
function isGovEmail(email) {
  const e = normEmail(email);
  if (!e || !isEmail(e)) return false;
  const domain = e.slice(e.lastIndexOf('@') + 1);
  return allowedEmailDomains().some((d) => domain === d || domain.endsWith(`.${d}`));
}

function govEmailError() {
  return `Use your official government email address (${allowedEmailDomains()
    .map((d) => '@' + d).join(' or ')}).`;
}

/**
 * Has this account proved it owns its email address?
 *
 * A MISSING flag counts as verified, deliberately. Accounts created before this
 * feature (and every admin-provisioned account, which an admin vouched for by
 * creating it) have no such field, and must not be locked out by adding one.
 * Only the self-signup path writes emailVerified:false.
 */
function isEmailVerified(u) {
  return !u || u.emailVerified !== false;
}

// ── Reads ────────────────────────────────────────────────────────────────────
function publicView(u) {
  return {
    username: u.username, role: u.role, email: u.email || '', mobile: u.mobile || '',
    createdAt: u.createdAt,
    // Surfaced so the admin panel can show who is still pending confirmation.
    emailVerified: isEmailVerified(u),
  };
}

/** All users without password hashes (safe for the admin panel). */
function listUsers() {
  return readAll().map(publicView);
}

/** Full record (incl. hash) by username, or null. */
function getUser(name) {
  const key = normUser(name);
  return readAll().find((u) => u.username === key) || null;
}

/**
 * Look up by ANY identifier — username, email, or mobile. This is what lets the
 * login box and OTP request accept whichever the user knows. Returns the full
 * record or null.
 */
function getUserByIdentifier(identifier) {
  const raw = String(identifier || '').trim();
  if (!raw) return null;
  const asUser = normUser(raw);
  const asEmail = normEmail(raw);
  const asMobile = normMobile(raw);
  return readAll().find((u) =>
    u.username === asUser ||
    (u.email && u.email === asEmail) ||
    (u.mobile && asMobile && u.mobile === asMobile)
  ) || null;
}

/** Verify a plaintext password. Returns the public user or null (timing-safe). */
async function verifyCredentials(identifier, password) {
  const user = getUserByIdentifier(identifier);
  const hash = user ? user.passwordHash : '$2b$10$invalidsaltinvalidsaltinvalidsa.invalidhashxxxxxxxx';
  const ok = await bcrypt.compare(String(password || ''), hash);
  return user && ok ? publicView(user) : null;
}

// ── Writes ───────────────────────────────────────────────────────────────────
function assertUniqueContact(users, { email, mobile }, exceptUsername = null) {
  for (const u of users) {
    if (exceptUsername && u.username === exceptUsername) continue;
    if (email && u.email && u.email === email) throw new Error('That email is already used by another account.');
    if (mobile && u.mobile && u.mobile === mobile) throw new Error('That mobile number is already used by another account.');
  }
}

/**
 * Self-service signup. Distinct from createUser (the admin path) in three ways:
 * the email is MANDATORY and must be a government address, the role is forced to
 * 'user' so signup can never mint an admin, and the account starts UNVERIFIED so
 * it cannot sign in until the emailed link is opened.
 */
function signupUser({ username, password, email, mobile }) {
  const em = normEmail(email);
  if (!em) throw new Error('Email address is required.');
  if (!isGovEmail(em)) throw new Error(govEmailError());
  const created = createUser({ username, password, role: 'user', email: em, mobile });

  const users = readAll();
  const u = users.find((x) => x.username === created.username);
  if (u) {
    u.emailVerified = false;
    writeAll(users);
  }
  return { ...created, emailVerified: false };
}

/** Mark an account's email as proved. Idempotent. */
function markEmailVerified(username) {
  const key = normUser(username);
  const users = readAll();
  const u = users.find((x) => x.username === key);
  if (!u) throw new Error('User not found.');
  u.emailVerified = true;
  writeAll(users);
  return publicView(u);
}

function createUser({ username, password, role, email, mobile }) {
  const key = normUser(username);
  const r = ROLES.includes(role) ? role : 'user';
  if (!/^[a-z0-9._-]{3,32}$/.test(key)) {
    throw new Error('Username must be 3–32 chars: letters, digits, dot, underscore or hyphen.');
  }
  if (!password || String(password).length < 8) throw new Error('Password must be at least 8 characters.');

  const em = normEmail(email);
  if (em && !isEmail(em)) throw new Error('Email address is not valid.');
  const mob = mobile ? normMobile(mobile) : null;
  if (mobile && (!mob || mob.length !== 10)) throw new Error('Mobile number must be a 10-digit Indian number.');

  const users = readAll();
  if (users.some((u) => u.username === key)) throw new Error('That username already exists.');
  assertUniqueContact(users, { email: em, mobile: mob });

  const record = {
    username: key,
    role: r,
    email: em || '',
    mobile: mob || '',
    passwordHash: bcrypt.hashSync(String(password), 10),
    createdAt: new Date().toISOString(),
  };
  users.push(record);
  writeAll(users);
  return publicView(record);
}

function setPassword(name, password) {
  const key = normUser(name);
  if (!password || String(password).length < 8) throw new Error('Password must be at least 8 characters.');
  const users = readAll();
  const u = users.find((x) => x.username === key);
  if (!u) throw new Error('User not found.');
  u.passwordHash = bcrypt.hashSync(String(password), 10);
  writeAll(users);
  return publicView(u);
}

function setRole(name, role) {
  const key = normUser(name);
  if (!ROLES.includes(role)) throw new Error('Invalid role.');
  const users = readAll();
  const u = users.find((x) => x.username === key);
  if (!u) throw new Error('User not found.');
  u.role = role;
  writeAll(users);
  return publicView(u);
}

/** Update an account's contact details (admin, or self-service later). */
function setContact(name, { email, mobile }) {
  const key = normUser(name);
  const users = readAll();
  const u = users.find((x) => x.username === key);
  if (!u) throw new Error('User not found.');

  let em = u.email, mob = u.mobile;
  if (email !== undefined) {
    em = normEmail(email) || '';
    if (em && !isEmail(em)) throw new Error('Email address is not valid.');
  }
  if (mobile !== undefined) {
    mob = mobile ? normMobile(mobile) : '';
    if (mobile && (!mob || mob.length !== 10)) throw new Error('Mobile number must be a 10-digit Indian number.');
  }
  assertUniqueContact(users, { email: em, mobile: mob }, key);
  u.email = em || '';
  u.mobile = mob || '';
  writeAll(users);
  return publicView(u);
}

function deleteUser(name) {
  const key = normUser(name);
  const users = readAll();
  const next = users.filter((u) => u.username !== key);
  if (next.length === users.length) throw new Error('User not found.');
  if (!next.some((u) => u.role === 'admin')) throw new Error('Cannot delete the last remaining admin account.');
  writeAll(next);
  return { username: key };
}

// Per-bot seed env: <PIPELINE>_ADMIN_USER/PASSWORD/EMAIL/MOBILE (e.g. CHIPS_ADMIN_USER,
// FG_ADMIN_PASSWORD). fg2 also honours the legacy GAD_ADMIN_* names. This lets each
// bot seed a DISTINCT first admin instead of a shared default.
function seedEnv(key) {
  const prefixes = PIPELINE === 'fg2' ? ['FG2', 'GAD'] : [PIPELINE.toUpperCase()];
  for (const p of prefixes) {
    const v = process.env[`${p}_ADMIN_${key}`];
    if (v != null && v !== '') return v;
  }
  return undefined;
}

function ensureSeedAdmin() {
  const users = readAll();
  if (users.some((u) => u.role === 'admin')) return;
  const username = normUser(seedEnv('USER') || 'admin');
  const seededPassword = seedEnv('PASSWORD');
  const password = seededPassword || 'changeme123';
  // The seed path writes straight to the store, so it never went through
  // createUser's validation. Warn loudly rather than throw: refusing to seed would
  // leave the bot with NO admin and lock everyone out over a typo'd address. The
  // address is still recorded — it just can't be trusted for OTP/notification.
  let email = normEmail(seedEnv('EMAIL')) || '';
  if (email && !isEmail(email)) {
    console.warn(`[${PIPELINE}-auth] Seed admin email ${email} is not a valid address — ignoring it.`);
    email = '';
  } else if (email && !isGovEmail(email)) {
    console.warn(
      `[${PIPELINE}-auth] Seed admin email ${email} is outside the allowed government ` +
      `domains (${allowedEmailDomains().join(', ')}). Keeping it because an admin set it ` +
      `deliberately, but self-signup would reject this address.`
    );
  }
  const mobileEnv = seedEnv('MOBILE');
  const mobile = mobileEnv ? (normMobile(mobileEnv) || '') : '';
  try {
    const existing = users.find((u) => u.username === username);
    if (existing) {
      existing.role = 'admin';
    } else {
      users.push({
        username, role: 'admin', email, mobile,
        passwordHash: bcrypt.hashSync(String(password), 10),
        createdAt: new Date().toISOString(),
      });
    }
    writeAll(users);
    const P = PIPELINE.toUpperCase();
    if (!seededPassword) {
      console.warn(
        `[${PIPELINE}-auth] Seeded default admin "${username}" with a well-known password. ` +
        `Set ${P}_ADMIN_USER/${P}_ADMIN_PASSWORD, or change it in the Accounts panel immediately.`
      );
    } else {
      console.log(`[${PIPELINE}-auth] Seeded admin account "${username}" from env.`);
    }
  } catch (err) {
    console.error(`[gad-auth] Failed to seed admin: ${err.message}`);
  }
}

module.exports = {
  ROLES,
  storeFile,
  normUser, normEmail, normMobile, isEmail,
  isGovEmail, govEmailError, allowedEmailDomains, isEmailVerified,
  listUsers,
  getUser,
  getUserByIdentifier,
  verifyCredentials,
  createUser,
  signupUser,
  markEmailVerified,
  setPassword,
  setRole,
  setContact,
  deleteUser,
  ensureSeedAdmin,
};
