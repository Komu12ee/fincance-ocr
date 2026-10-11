'use strict';

/**
 * Email provider (SMTP via nodemailer). Reuses the SMTP_* env the repo already
 * defines. When SMTP is not configured it becomes a MOCK that logs the message —
 * so every OTP flow is fully testable in dev without real mail credentials.
 *
 * Swap this file to change the email backend (SES, an HTTP API, …); the contract
 * the dispatcher relies on is just: { id, isConfigured(), send({to,subject,html,text}) }.
 */

let _nodemailer = null;
try { _nodemailer = require('nodemailer'); } catch (_) { /* optional */ }

function cfg() {
  return {
    host: process.env.SMTP_HOST || '',
    port: parseInt(process.env.SMTP_PORT || '', 10) || 587,
    // SMTP_SECURE=true (or port 465) → implicit TLS; else STARTTLS.
    secure: String(process.env.SMTP_SECURE || '').toLowerCase() === 'true' ||
            parseInt(process.env.SMTP_PORT || '', 10) === 465,
    user: process.env.SMTP_USER || '',
    pass: process.env.SMTP_PASSWORD || '',
    from: process.env.SMTP_FROM_EMAIL || process.env.SMTP_USER || '',
    senderName: process.env.SMTP_SENDER_NAME || 'GAD Governance Assistant',
  };
}

function isConfigured() {
  return !!(_nodemailer && cfg().host);
}

let _transport = null;
function transport() {
  if (_transport) return _transport;
  const c = cfg();
  _transport = _nodemailer.createTransport({
    host: c.host, port: c.port, secure: c.secure,
    auth: c.user ? { user: c.user, pass: c.pass } : undefined,
    connectionTimeout: 15000, greetingTimeout: 15000,
  });
  return _transport;
}

async function send({ to, subject, html, text }) {
  const c = cfg();
  if (!isConfigured()) {
    console.log(`[gad-notify][MOCK EMAIL] to=${to} subject="${subject}"\n${text || html}`);
    return { ok: true, mock: true };
  }
  const from = c.senderName ? `"${c.senderName}" <${c.from}>` : c.from;
  await transport().sendMail({ from, to, subject, text, html });
  return { ok: true };
}

module.exports = { id: 'email', isConfigured, send };
