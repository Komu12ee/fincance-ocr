'use strict';

/**
 * One-time-passcode lifecycle for GAD: issue → (send, done by notify/) → verify.
 *
 * Codes are 6 digits, stored only as a salted hash with an expiry, a small attempt
 * budget, and per-recipient resend cooldown + daily cap (via gad/ratelimit.js — no
 * Redis). State is in-memory (a Map); this module owns it so a future refactor can
 * persist it elsewhere without changing routes.
 *
 * `purpose` scopes a code so a "login" OTP can't be replayed as a "reset" OTP.
 * `channel` ("email" | "sms") is remembered only for messaging/rate-key clarity.
 */

const crypto = require('crypto');
const limiter = require('./ratelimit');

const TTL_SECONDS = () => parseInt(process.env.OTP_TTL_SECONDS || '', 10) || 600;          // 10 min
const RESEND_COOLDOWN = () => parseInt(process.env.OTP_RESEND_COOLDOWN_SECONDS || '', 10) || 60;
const DAILY_CAP = () => parseInt(process.env.OTP_DAILY_CAP || '', 10) || 5;
const MAX_ATTEMPTS = () => parseInt(process.env.OTP_MAX_ATTEMPTS || '', 10) || 5;

// key -> { hash, salt, purpose, channel, expiresAt, attempts }
const _codes = new Map();

function _key(userKey, purpose) {
  return `${purpose}:${userKey}`;
}
function _hash(code, salt) {
  return crypto.createHmac('sha256', salt).update(String(code)).digest('hex');
}

/**
 * Issue a code for (userKey, purpose). Enforces resend cooldown + daily cap.
 * Returns { ok:true, code, expiresInSeconds } — the caller (routes) hands `code`
 * to notify/ to actually deliver; it is never returned to the browser.
 * On limit: { ok:false, error, retryAfter }.
 */
function issue(userKey, purpose, channel = 'email') {
  const rlKey = _key(userKey, purpose);

  const cool = limiter.cooldown(`otp:cool:${rlKey}`, RESEND_COOLDOWN());
  if (!cool.ok) {
    return { ok: false, error: `Please wait ${cool.retryAfter}s before requesting another code.`, retryAfter: cool.retryAfter };
  }
  const cap = limiter.counter(`otp:day:${rlKey}`, DAILY_CAP(), 24 * 3600);
  if (!cap.ok) {
    return { ok: false, error: `Too many code requests. Try again in ${Math.ceil(cap.retryAfter / 3600)}h or contact an admin.`, retryAfter: cap.retryAfter };
  }

  const code = String(crypto.randomInt(0, 1_000_000)).padStart(6, '0');
  const salt = crypto.randomBytes(16).toString('hex');
  _codes.set(rlKey, {
    hash: _hash(code, salt), salt, purpose, channel,
    expiresAt: Date.now() + TTL_SECONDS() * 1000, attempts: 0,
  });
  limiter.mark(`otp:cool:${rlKey}`);
  return { ok: true, code, expiresInSeconds: TTL_SECONDS() };
}

/**
 * Verify a submitted code for (userKey, purpose). Consumes the code on success.
 * Returns { ok:true } or { ok:false, error }.
 */
function verify(userKey, purpose, code) {
  const rlKey = _key(userKey, purpose);
  const rec = _codes.get(rlKey);
  if (!rec) return { ok: false, error: 'No active code — request a new one.' };
  if (Date.now() > rec.expiresAt) {
    _codes.delete(rlKey);
    return { ok: false, error: 'Code expired — request a new one.' };
  }
  if (rec.attempts >= MAX_ATTEMPTS()) {
    _codes.delete(rlKey);
    return { ok: false, error: 'Too many incorrect attempts — request a new code.' };
  }
  rec.attempts += 1;
  const submitted = _hash(String(code || '').trim(), rec.salt);
  const a = Buffer.from(submitted), b = Buffer.from(rec.hash);
  const good = a.length === b.length && crypto.timingSafeEqual(a, b);
  if (!good) return { ok: false, error: 'Incorrect code.' };
  _codes.delete(rlKey);                 // single-use
  limiter.reset(`otp:cool:${rlKey}`);
  return { ok: true };
}

module.exports = { issue, verify, TTL_SECONDS };
