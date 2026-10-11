'use strict';

/**
 * Notification dispatcher — the single entry point routes/otp use to deliver a
 * code. It composes the message and hands off to a channel provider (email/sms).
 *
 * Providers are looked up in a registry, so adding WhatsApp or a different backend
 * later means registering one module that satisfies { id, isConfigured(), send() } —
 * no caller changes. Keeping composition here (not in the providers) means the OTP
 * wording lives in one place.
 */

const email = require('./email');
const sms = require('./sms');

const PROVIDERS = { email, sms };

/** Which channels can actually deliver right now (configured, else mock in dev). */
function channelStatus() {
  return {
    email: { available: true, live: email.isConfigured() },
    sms: { available: true, live: sms.isConfigured() },
  };
}

function otpMinutes(ttlSeconds) {
  return Math.max(1, Math.round((ttlSeconds || 600) / 60));
}

function buildEmail(code, mins, purpose, brand) {
  const app = (brand && brand.label) || 'GAD Governance Assistant';
  const what = purpose === 'reset' ? 'password reset' : 'sign-in';
  const subject = `${app} — your ${what} code`;
  const text =
    `Your ${what} verification code is ${code}. It expires in ${mins} minutes. ` +
    `If you did not request this, ignore this message.`;
  const html = `
    <div style="font-family:Arial,sans-serif;max-width:520px;margin:0 auto;padding:20px;border:1px solid #e2e8f0;border-radius:8px;color:#1b2430">
      <h2 style="color:#1c3d5c;border-bottom:2px solid #e2e8f0;padding-bottom:10px;margin-top:0">${app}</h2>
      <p>Use this code to complete your ${what}. It expires in <strong>${mins} minutes</strong>:</p>
      <div style="background:#f7f8fa;border:1px solid #cbd5e1;border-radius:6px;padding:14px;font-size:26px;font-weight:bold;text-align:center;letter-spacing:6px;margin:18px 0">${code}</div>
      <p style="font-size:12px;color:#64748b">If you did not request this, you can safely ignore this email.</p>
    </div>`;
  return { subject, text, html };
}

/**
 * SMS body. Government DLT delivery requires the text to match a registered
 * template, so the wording is env-overridable (SMS_OTP_TEMPLATE_TEXT) with {code}
 * and {mins} placeholders. The default is a sensible OTP line for dev/mock.
 */
function buildSms(code, mins) {
  const tmpl = process.env.SMS_OTP_TEMPLATE_TEXT ||
    'Your GAD Governance Assistant OTP is {code}. Valid for {mins} minutes. Do not share it. -GAD CG';
  return tmpl.replace(/\{code\}/g, code).replace(/\{mins\}/g, String(mins));
}

/**
 * Deliver an OTP over the chosen channel.
 * @returns { ok:true, mock? } or { ok:false, error }
 */
async function sendOtp({ channel, to, code, purpose = 'login', ttlSeconds = 600, brand = null }) {
  const provider = PROVIDERS[channel];
  if (!provider) return { ok: false, error: `Unknown channel: ${channel}` };
  if (!to) return { ok: false, error: `No ${channel} on file for this account.` };
  const mins = otpMinutes(ttlSeconds);
  try {
    if (channel === 'email') {
      const { subject, text, html } = buildEmail(code, mins, purpose, brand);
      return await provider.send({ to, subject, text, html });
    }
    return await provider.send({ to, text: buildSms(code, mins) });
  } catch (err) {
    return { ok: false, error: `${channel} delivery failed: ${err.message}` };
  }
}

/**
 * Signup confirmation. Sent once, to prove the address belongs to the person who
 * registered — the account cannot sign in until this link is opened.
 *
 * Email only, on purpose: an SMS equivalent would need its own DLT-registered
 * template (the government gateway rejects anything that does not match one), and
 * that is days of paperwork for no extra assurance here.
 */
function buildVerifyEmail(link, hours, brand) {
  const app = (brand && brand.label) || 'GAD Governance Assistant';
  const subject = `${app} — confirm your email address`;
  const text =
    `Confirm your email address to activate your ${app} account: ${link}\n\n` +
    `This link expires in ${hours} hour(s). If you did not create this account, ignore this message — ` +
    `no account is active until the link is opened.`;
  const html = `
    <div style="font-family:Arial,sans-serif;max-width:520px;margin:0 auto;padding:20px;border:1px solid #e2e8f0;border-radius:8px;color:#1b2430">
      <h2 style="color:#1c3d5c;border-bottom:2px solid #e2e8f0;padding-bottom:10px;margin-top:0">${app}</h2>
      <p>Confirm your email address to activate your account.</p>
      <p style="text-align:center;margin:24px 0">
        <a href="${link}" style="background:#1c3d5c;color:#fff;text-decoration:none;padding:12px 22px;border-radius:6px;display:inline-block;font-weight:bold">Confirm my email</a>
      </p>
      <p style="font-size:12px;color:#64748b">Or paste this link into your browser:<br><span style="word-break:break-all">${link}</span></p>
      <p style="font-size:12px;color:#64748b">This link expires in <strong>${hours} hour(s)</strong>. If you did not create this account you can ignore this email — it stays inactive until the link is opened.</p>
    </div>`;
  return { subject, text, html };
}

async function sendVerification({ to, link, ttlSeconds = 86400, brand = null }) {
  if (!to) return { ok: false, error: 'No email address to confirm.' };
  const hours = Math.max(1, Math.round((ttlSeconds || 86400) / 3600));
  try {
    const { subject, text, html } = buildVerifyEmail(link, hours, brand);
    return await email.send({ to, subject, text, html });
  } catch (err) {
    return { ok: false, error: `email delivery failed: ${err.message}` };
  }
}

module.exports = { sendOtp, sendVerification, channelStatus, PROVIDERS };
