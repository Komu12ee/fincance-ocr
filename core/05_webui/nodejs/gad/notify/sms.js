'use strict';

/**
 * SMS provider — CDAC MSDG Government SMS API Gateway (same gateway SDC uses).
 * Uses Node 18+ built-in fetch, so no npm dependency.
 *
 * DLT rules: the government gateway only delivers messages that match a registered
 * template tied to your sender ID. So GAD needs its OWN CDAC credentials and a
 * DLT-approved OTP template; the message text must match that template. All of it
 * is env-configured (SMS_*). When unconfigured this is a MOCK that logs the OTP,
 * so the flow is testable in dev.
 *
 * Contract for the dispatcher: { id, isConfigured(), send({to, text}) }.
 */

const crypto = require('crypto');

const CDAC_URL = process.env.SMS_GATEWAY_URL || 'https://msdgweb.mgov.gov.in/esms/sendsmsrequestDLT';

function cfg() {
  return {
    username: (process.env.SMS_USERNAME || '').trim(),
    password: process.env.SMS_PASSWORD || '',
    senderId: (process.env.SMS_SENDER_ID || '').trim(),
    secureKey: (process.env.SMS_SECURE_KEY || '').trim(),
    templateId: (process.env.SMS_OTP_TEMPLATE_ID || '').trim(),
    serviceType: process.env.SMS_SERVICE_TYPE || 'singlemsg',
  };
}

function isConfigured() {
  const c = cfg();
  return !!(c.username && c.senderId && c.secureKey);
}

function tenDigits(phone) {
  let d = String(phone || '').replace(/\D/g, '');
  if (d.length > 10 && d.startsWith('91')) d = d.slice(-10);
  else if (d.length === 11 && d.startsWith('0')) d = d.slice(1);
  return d.length === 10 ? d : '';
}

async function send({ to, text }) {
  const mobile = tenDigits(to);
  if (!mobile) return { ok: false, error: 'Invalid mobile number.' };

  if (!isConfigured()) {
    console.log(`[gad-notify][MOCK SMS] to=${mobile}\n${text}`);
    return { ok: true, mock: true };
  }

  const c = cfg();
  // CDAC request key: SHA-512 of username + senderid + message + secureKey.
  const key = crypto.createHash('sha512').update(`${c.username}${c.senderId}${text}${c.secureKey}`).digest('hex');
  const payload = {
    username: c.username, password: c.password, smsservicetype: c.serviceType,
    senderid: c.senderId, mobileno: mobile, content: text, key, templateid: c.templateId,
  };
  // urlencode, then restore spaces (CDAC expects raw spaces, matching PHP/urllib quirk).
  const body = new URLSearchParams(payload).toString().replace(/\+/g, ' ');

  const resp = await fetch(CDAC_URL, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/x-www-form-urlencoded',
      'User-Agent': 'Mozilla/4.0 (compatible; MSIE 5.0; Windows 98; DigExt)',
    },
    body,
    signal: AbortSignal.timeout(15000),
  });
  const respText = await resp.text().catch(() => '');
  if (!resp.ok) return { ok: false, error: `SMS gateway HTTP ${resp.status}: ${respText.slice(0, 120)}` };
  return { ok: true, response: respText.slice(0, 200) };
}

module.exports = { id: 'sms', isConfigured, send };
