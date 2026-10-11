'use strict';

/**
 * Signed email-verification links for self-signup.
 *
 * Deliberately NOT an OTP. A 6-digit code (gad/otp.js) is right when the user is
 * sitting in front of the form and can type it back; proving you own an inbox is
 * better served by a link you click from that inbox — no transcription, and it works
 * from a phone mail app without the browser session that issued it.
 *
 * The token is STATELESS: an HMAC over {username, purpose, exp}, verified by
 * recomputing the signature. That means no server-side table to persist, expire or
 * clean up, and it survives a restart — which matters because the gap between signing
 * up and opening your mail can be hours.
 *
 * Replay is harmless by construction: the only thing a valid token does is set
 * emailVerified = true, which is idempotent. Tokens still expire so an address that
 * changes hands cannot be claimed with an ancient link.
 */

const crypto = require('crypto');

const PIPELINE = (process.env.PIPELINE || 'chips').toLowerCase();

function ttlSeconds() {
  return parseInt(process.env.SIGNUP_VERIFY_TTL_SECONDS || '', 10) || 24 * 60 * 60; // 24h
}

function secret() {
  const s = process.env.GAD_SESSION_SECRET || process.env.SESSION_SECRET || process.env.JWT_SECRET;
  if (!s) throw new Error('[verifytoken] No signing secret set (GAD_SESSION_SECRET/SESSION_SECRET/JWT_SECRET).');
  return s;
}

function hmac(data) {
  return crypto.createHmac('sha256', secret()).update(data).digest('base64url');
}

/** Signed token proving the holder received mail sent to `username`'s address. */
function issue(username, ttl = ttlSeconds()) {
  const now = Math.floor(Date.now() / 1000);
  // The pipeline is baked in so a link minted by one bot cannot verify an account
  // on another — the same isolation rule the session cookie follows.
  const body = { pipeline: PIPELINE, sub: String(username), purpose: 'verify-email', iat: now, exp: now + ttl };
  const data = Buffer.from(JSON.stringify(body)).toString('base64url');
  return `${data}.${hmac(data)}`;
}

/** Verify and decode. Returns { ok:true, username } or { ok:false, error }. */
function verify(token) {
  if (!token || typeof token !== 'string') return { ok: false, error: 'Missing verification token.' };
  const dot = token.indexOf('.');
  if (dot <= 0) return { ok: false, error: 'Malformed verification link.' };

  const data = token.slice(0, dot);
  let a, b;
  try {
    a = Buffer.from(token.slice(dot + 1));
    b = Buffer.from(hmac(data));
  } catch (_) {
    return { ok: false, error: 'Malformed verification link.' };
  }
  if (a.length !== b.length || !crypto.timingSafeEqual(a, b)) {
    return { ok: false, error: 'This verification link is not valid.' };
  }

  let body;
  try {
    body = JSON.parse(Buffer.from(data, 'base64url').toString('utf8'));
  } catch (_) {
    return { ok: false, error: 'Malformed verification link.' };
  }
  if (body.purpose !== 'verify-email') return { ok: false, error: 'Wrong link type.' };
  if (body.pipeline && body.pipeline !== PIPELINE) return { ok: false, error: 'This link belongs to a different workspace.' };
  if (typeof body.exp !== 'number' || body.exp < Math.floor(Date.now() / 1000)) {
    return { ok: false, error: 'This verification link has expired. Request a new one from the sign-in page.' };
  }
  return { ok: true, username: body.sub };
}

module.exports = { issue, verify, ttlSeconds };
