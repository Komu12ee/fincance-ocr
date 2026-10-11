'use strict';

const express  = require('express');
const bcrypt   = require('bcryptjs');
const jwt      = require('jsonwebtoken');
const { JWT_SECRET, JWT_EXPIRY, loadUsers } = require('../config');

const router = express.Router();

/**
 * POST /auth/login
 *
 * Body: { username: string, password: string }
 *
 * Success → 200  { success: true,  token: <jwt>, user: { username } }
 * Failure → 401  { success: false, error: <message> }
 *
 * Rate-limiting note: for production add express-rate-limit on this route.
 */
router.post('/login', async (req, res) => {
  const { username, password } = req.body || {};

  // ── Basic input validation ───────────────────────────────────────
  if (!username || typeof username !== 'string' ||
      !password || typeof password !== 'string') {
    return res.status(400).json({ success: false, error: 'Username and password are required.' });
  }

  const users = loadUsers();
  const key   = username.trim().toLowerCase();

  // ── Look up user ─────────────────────────────────────────────────
  const hash = users[key];
  if (!hash) {
    // Timing-safe: run a dummy compare so response time doesn't reveal
    // whether the user exists or not.
    await bcrypt.compare(password, '$2b$10$invalidsaltinvalidsaltinvalidsa.invalidhashxxxxxxxx');
    return res.status(401).json({ success: false, error: 'Invalid credentials.' });
  }

  // ── Verify password ──────────────────────────────────────────────
  const match = await bcrypt.compare(password, hash);
  if (!match) {
    return res.status(401).json({ success: false, error: 'Invalid credentials.' });
  }

  // ── Issue JWT ────────────────────────────────────────────────────
  const token = jwt.sign(
    {
      username: key,
      app:      'chips-rag',
      iat:      Math.floor(Date.now() / 1000),
    },
    JWT_SECRET,
    { expiresIn: JWT_EXPIRY }
  );

  console.log(`[auth] Login successful: ${key}`);

  return res.status(200).json({
    success: true,
    token,
    user: { username: key },
  });
});

/**
 * POST /auth/logout
 * Client-side logout — just tells the frontend to clear its token.
 * (Stateless JWT: no server-side invalidation unless you add a denylist.)
 */
router.post('/logout', (_req, res) => {
  res.json({ success: true });
});

module.exports = router;
