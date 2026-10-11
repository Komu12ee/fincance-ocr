'use strict';

const jwt = require('jsonwebtoken');
const { JWT_SECRET } = require('../config');

/**
 * Express middleware that enforces JWT authentication.
 *
 * Expects:  Authorization: Bearer <token>
 * On valid token  → attaches decoded payload to req.user, calls next()
 * On missing/bad  → 401 JSON so the frontend can redirect to login
 */
module.exports = function authMiddleware(req, res, next) {
  const header = req.headers['authorization'] || '';
  const token  = header.startsWith('Bearer ') ? header.slice(7) : null;

  if (!token) {
    return res.status(401).json({ success: false, error: 'No token — please log in.' });
  }

  try {
    req.user = jwt.verify(token, JWT_SECRET);
    next();
  } catch (err) {
    const expired = err.name === 'TokenExpiredError';
    return res.status(401).json({
      success: false,
      error:   expired ? 'Session expired — please log in again.' : 'Invalid token.',
      expired,
    });
  }
};
