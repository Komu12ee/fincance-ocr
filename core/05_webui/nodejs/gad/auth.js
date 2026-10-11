'use strict';

/**
 * GAD (fg2) session auth — standalone and independent of the selection gateway.
 *
 * This is a deliberate parallel to shared/webAuth.js, kept separate so the GAD
 * pipeline has its OWN login that does not depend on (and cannot be satisfied by)
 * a gateway session. It signs its own cookie (`gad_session`) carrying the user's
 * role, so every request can be authorised (admin vs user) without a DB lookup.
 *
 * Uses only Node's built-in `crypto` — no extra dependency.
 */

const crypto = require('crypto');
// The shared verifier is used ONLY to recognise internal service tokens (e.g. the
// gateway's server-to-server /api/init call), never to authenticate a browser.
const shared = require('../../../../shared/webAuth');

// Which bot this process serves. Every pipeline runs this same standalone login,
// so sessions must be isolated per bot: a session minted for one bot must never be
// accepted by another. We do that two ways — a per-pipeline cookie NAME (so cookies
// don't even collide across localhost ports / a shared parent domain) and a
// `pipeline` claim baked into the token that verifyToken() checks.
const PIPELINE = (process.env.PIPELINE || 'chips').toLowerCase();

const COOKIE_NAME = process.env.GAD_SESSION_COOKIE_NAME || `${PIPELINE}_session`;

function secret() {
  const s = process.env.GAD_SESSION_SECRET || process.env.SESSION_SECRET || process.env.JWT_SECRET;
  if (!s) throw new Error('[gad-auth] No signing secret set (GAD_SESSION_SECRET/SESSION_SECRET/JWT_SECRET).');
  return s;
}

function ttlSeconds() {
  return parseInt(process.env.GAD_SESSION_TTL_SECONDS || '', 10) || 8 * 60 * 60; // 8h
}

function hmac(data) {
  return crypto.createHmac('sha256', secret()).update(data).digest('base64url');
}

/** Sign { username, role } into a compact `<body>.<sig>` token (JWT-like, HS256). */
function issueToken(payload, ttl = ttlSeconds()) {
  const now = Math.floor(Date.now() / 1000);
  // Stamp the bot identity so a token can only authenticate on the bot that issued it.
  const body = { pipeline: PIPELINE, ...payload, iat: now, exp: now + ttl };
  const data = Buffer.from(JSON.stringify(body)).toString('base64url');
  return `${data}.${hmac(data)}`;
}

/** Verify a GAD token's signature + expiry. Returns the payload, or null. */
function verifyToken(token) {
  if (!token || typeof token !== 'string') return null;
  const dot = token.indexOf('.');
  if (dot <= 0) return null;
  const data = token.slice(0, dot);
  const sig = token.slice(dot + 1);
  let a, b;
  try {
    a = Buffer.from(sig);
    b = Buffer.from(hmac(data));
  } catch (_) {
    return null;
  }
  if (a.length !== b.length || !crypto.timingSafeEqual(a, b)) return null;
  let body;
  try {
    body = JSON.parse(Buffer.from(data, 'base64url').toString('utf8'));
  } catch (_) {
    return null;
  }
  if (typeof body.exp !== 'number' || body.exp < Math.floor(Date.now() / 1000)) return null;
  // Reject a session minted for a different bot (defence-in-depth beyond the
  // per-pipeline cookie name — matters if cookies are shared across a parent domain).
  if (body.pipeline && body.pipeline !== PIPELINE) return null;
  return body;
}

function cookieOptions() {
  const opts = {
    httpOnly: true,
    sameSite: 'lax',
    path: '/',
    secure: process.env.NODE_ENV === 'production' && process.env.COOKIE_INSECURE !== '1',
    maxAge: ttlSeconds() * 1000,
  };
  if (process.env.COOKIE_DOMAIN) opts.domain = process.env.COOKIE_DOMAIN;
  return opts;
}

function getToken(req) {
  const raw = req.headers && req.headers.cookie;
  if (raw) {
    for (const part of raw.split(';')) {
      const i = part.indexOf('=');
      if (i > -1 && part.slice(0, i).trim() === COOKIE_NAME) {
        return decodeURIComponent(part.slice(i + 1).trim());
      }
    }
  }
  const h = (req.headers && req.headers['authorization']) || '';
  return h.startsWith('Bearer ') ? h.slice(7) : null;
}

// Recognise the gateway's internal service token (used by auto-init). It is
// signed with the SHARED secret and marked internal — accept it for machine
// calls only, never as a browser identity (it has no role).
function internalServiceUser(req) {
  const h = (req.headers && req.headers['authorization']) || '';
  const token = h.startsWith('Bearer ') ? h.slice(7) : null;
  if (!token) return null;
  const payload = shared.verifyToken(token);
  if (payload && (payload.sub === 'internal' || payload.svc)) {
    return { username: '__internal__', role: 'service' };
  }
  return null;
}

/**
 * Require a valid GAD session. Sets req.gadUser = { username, role }.
 *  - Unauthenticated top-level HTML navigation → redirect to the GAD login page.
 *  - Everything else (XHR/API) → 401 JSON.
 * Also lets internal service tokens through so the gateway can auto-init the pipeline.
 */
// Reverse-proxy sub-path: nginx sets X-Forwarded-Prefix (e.g. /pipeline/fg2) when
// the pipeline is served under a path. Redirect Location headers must carry it so
// the browser is sent to the proxied URL, not the origin root. Empty otherwise.
function fwdPrefix(req) {
  return (req.headers && req.headers['x-forwarded-prefix']) || '';
}

function requireGadAuth(opts = {}) {
  const loginUrl = opts.loginUrl || '/login';
  return (req, res, next) => {
    const user = verifyToken(getToken(req)) || internalServiceUser(req);
    if (user) {
      req.gadUser = user;
      return next();
    }
    const wantsHtml = req.method === 'GET' && req.accepts && req.accepts('html') && !req.xhr;
    if (wantsHtml) return res.redirect(fwdPrefix(req) + loginUrl);
    return res.status(401).json({ success: false, error: 'Unauthorized — please sign in.' });
  };
}

/**
 * Require an admin (or an internal service call). Assumes requireGadAuth ran first.
 * HTML navigation by a non-admin is redirected to the chat home; APIs get 403 JSON.
 */
function requireGadAdmin(opts = {}) {
  const redirectUrl = opts.redirectUrl || '/';
  return (req, res, next) => {
    const role = req.gadUser && req.gadUser.role;
    if (role === 'admin' || role === 'service') return next();
    const wantsHtml = req.method === 'GET' && req.accepts && req.accepts('html') && !req.xhr;
    if (wantsHtml) return res.redirect(fwdPrefix(req) + redirectUrl);
    return res.status(403).json({ success: false, error: 'Forbidden — admin access required.' });
  };
}

module.exports = {
  COOKIE_NAME,
  issueToken,
  verifyToken,
  getToken,
  cookieOptions,
  requireGadAuth,
  requireGadAdmin,
};
