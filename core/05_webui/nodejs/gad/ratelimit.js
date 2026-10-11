'use strict';

/**
 * Tiny in-memory rate limiter — deliberately NOT Redis-backed.
 *
 * Two primitives the OTP flow needs:
 *   · cooldown(key, seconds)   — reject if the same key acted < seconds ago
 *   · counter(key, max, windowSeconds) — reject once max actions happen in a window
 *
 * State lives in a Map in this process. For the single-process GAD server that is
 * sufficient; the interface is small and synchronous so a future refactor can drop
 * in a shared/persistent backend (Redis, a table) without touching callers.
 *
 * Trade-off (documented on purpose): counters reset on restart and are per-process.
 * For a brute-force/abuse guard on a low-traffic internal government tool that is an
 * acceptable simplification — the user explicitly opted out of Redis.
 */

class RateLimiter {
  constructor() {
    this._last = new Map();     // key -> epoch ms of last action (for cooldowns)
    this._hits = new Map();     // key -> { count, resetAt }
    this._sweepAt = 0;
  }

  _sweep(now) {
    if (now < this._sweepAt) return;
    this._sweepAt = now + 60_000; // at most once a minute
    for (const [k, v] of this._hits) if (v.resetAt <= now) this._hits.delete(k);
    for (const [k, t] of this._last) if (now - t > 24 * 3600_000) this._last.delete(k);
  }

  /**
   * Enforce a minimum gap between actions on `key`.
   * Returns { ok:true } or { ok:false, retryAfter:<seconds> }.
   */
  cooldown(key, seconds) {
    const now = Date.now();
    this._sweep(now);
    const last = this._last.get(key) || 0;
    const waited = (now - last) / 1000;
    if (waited < seconds) return { ok: false, retryAfter: Math.ceil(seconds - waited) };
    return { ok: true };
  }

  /** Record that an action on `key` happened now (starts its cooldown). */
  mark(key) {
    this._last.set(key, Date.now());
  }

  /**
   * Enforce at most `max` actions per rolling `windowSeconds` for `key`.
   * Increments on success. Returns { ok:true, remaining } or
   * { ok:false, retryAfter:<seconds> } when the cap is hit.
   */
  counter(key, max, windowSeconds) {
    const now = Date.now();
    this._sweep(now);
    let rec = this._hits.get(key);
    if (!rec || rec.resetAt <= now) {
      rec = { count: 0, resetAt: now + windowSeconds * 1000 };
      this._hits.set(key, rec);
    }
    if (rec.count >= max) return { ok: false, retryAfter: Math.ceil((rec.resetAt - now) / 1000) };
    rec.count += 1;
    return { ok: true, remaining: max - rec.count };
  }

  /** Clear all limits for a key (e.g. after a successful login). */
  reset(key) {
    this._last.delete(key);
    this._hits.delete(key);
  }
}

// One shared limiter for the process. Callers import this singleton; tests can
// construct their own `new RateLimiter()`.
module.exports = new RateLimiter();
module.exports.RateLimiter = RateLimiter;
