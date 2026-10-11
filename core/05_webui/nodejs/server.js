'use strict';

const express = require('express');
const { createProxyMiddleware } = require('http-proxy-middleware');
const morgan = require('morgan');
const path = require('path');

const PORT = process.env.PORT || 3002;
const FLASK_PORT = process.env.FLASK_PORT || 5002;
const FLASK_URL = `http://127.0.0.1:${FLASK_PORT}`;
const IS_PROD = process.env.NODE_ENV === 'production';

const app = express();

// CORS for the pipeline UIs. In the supported topologies the browser only ever
// talks to its OWN origin (same-origin behind the reverse proxy, or the pipeline's
// own subdomain), so NO cross-origin request is needed — hence CORS is DISABLED by
// default. Reflecting the request origin with credentials:true (the old default)
// let any website make credentialed requests to a reachable pipeline port and read
// the response. Only enable CORS by explicitly listing trusted origins in
// CORS_ORIGIN (e.g. https://rag.example.gov).
const CORS_ORIGIN = process.env.CORS_ORIGIN
    ? process.env.CORS_ORIGIN.split(',').map((s) => s.trim()).filter(Boolean)
    : false;
if (CORS_ORIGIN) {
    app.use(require('cors')({ origin: CORS_ORIGIN, credentials: true }));
}

// ─── Logging ────────────────────────────────────────────────────────────────
app.use(morgan(IS_PROD ? 'combined' : 'dev'));

// ─── Security headers ───────────────────────────────────────────────────────
app.use((_req, res, next) => {
  res.setHeader('X-Content-Type-Options', 'nosniff');
  // SAMEORIGIN (not DENY) so the Documents page can render the source PDF in a
  // same-origin <iframe>; still blocks cross-origin framing (clickjacking).
  res.setHeader('X-Frame-Options', 'SAMEORIGIN');
  res.setHeader('Referrer-Policy', 'strict-origin-when-cross-origin');
  next();
});

// ─── Health check (UNAUTHENTICATED) ──────────────────────────────────────────
// The launcher polls this to know the server is up, so it must stay public. It
// exposes nothing sensitive. Registered before the auth gate below.
app.get('/health', (_req, res) => res.status(200).json({ success: true, status: 'ok' }));

// ─── Branding (PUBLIC) ───────────────────────────────────────────────────────
// Which department this instance serves. Display strings only — no secrets — so
// it is exposed before the auth gate, letting the login page brand itself.
// Registered before the /api proxy so it is answered here, not forwarded to Flask.
const { BRANDING } = require('./config');
app.get('/api/branding', (_req, res) => res.json(BRANDING));

// ─── Auth gate ────────────────────────────────────────────────────────────────
// EVERY bot (chips/fg/fg2) now runs its OWN standalone login with admin/user roles
// (see gad/*.js) — independent of the selection gateway, which is just a public
// launcher. Admins get the document-verification workspace, users get the chatbot
// only. No bot UI, asset, or proxied API/PDF call is served without a valid session
// for THIS bot. CORS preflight (OPTIONS) is answered above, so it never reaches here.
{
  const botStore = require('./gad/store');
  const botRoutes = require('./gad/routes');
  const { requireGadAuth, requireGadAdmin } = require('./gad/auth');

  botStore.ensureSeedAdmin(); // guarantee an admin exists on first run

  // Standalone login page (PUBLIC) and the auth/account API (login/session are
  // public; the account-management routes gate themselves to admins).
  app.get('/login', (_req, res) => res.sendFile(path.join(__dirname, 'public', 'gad-login.html')));
  // Self-service signup is public by design: government staff register with their
  // official address and confirm it by email before the account can sign in.
  app.get('/signup', (_req, res) => res.sendFile(path.join(__dirname, 'public', 'gad-signup.html')));
  app.use('/gad-auth', botRoutes);

  // The login page is public, so the handful of NON-SENSITIVE assets it references
  // must be public too. Without this, the auth gate below treats each CSS/JS/img
  // request as HTML navigation (Accept: */*) and redirects it to the login page —
  // so the login renders unstyled with a broken emblem. Only branding assets are
  // exposed; the app shell (index.html), documents.html and gad-admin.html stay
  // gated (served by express.static AFTER the gate).
  const publicDir = path.join(__dirname, 'public');
  const assetOpts = { maxAge: IS_PROD ? '7d' : 0, etag: true };
  app.get('/design-system.css', (_req, res) => res.sendFile(path.join(publicDir, 'design-system.css')));
  app.get('/branding.js', (_req, res) => res.sendFile(path.join(publicDir, 'branding.js')));
  app.use('/img', express.static(path.join(publicDir, 'img'), assetOpts));

  // Gate everything below on a valid session for this bot (or an internal service token).
  app.use(requireGadAuth({ loginUrl: '/login' }));

  // Admin-only surfaces, BEFORE static so they can't be served to a plain user:
  //  · the document-management pages (HTML)
  //  · every /api/documents route (upload / verify / edit — the write surface)
  // Chat citations use /01_preprocessing/used_files, which stays open to any
  // signed-in user so answers can still show their source PDFs.
  app.get(['/documents.html', '/gad-admin.html'], requireGadAdmin({ redirectUrl: '/' }));
  app.use('/api/documents', requireGadAdmin());
}

// ─── Static assets ──────────────────────────────────────────────────────────
app.use(
  express.static(path.join(__dirname, 'public'), {
    maxAge: IS_PROD ? '7d' : 0,
    etag: true,
  })
);

// ─── Proxies ─────────────────────────────────────────────────────────────────
// hpm v3 strips the mount path before forwarding (e.g. /api/query → /query).
// pathRewrite as a FUNCTION (not object) re-adds the prefix.
// Object form { '^/api': '/api' } is silently ignored in hpm v3.

const apiProxy = createProxyMiddleware({
  target: FLASK_URL,
  changeOrigin: true,
  selfHandleResponse: false,
  pathRewrite: (path) => '/api' + path,   // /query → /api/query
  on: {
    error: (err, _req, res) => {
      console.error('[proxy] API error:', err.message);
      if (!res.headersSent) {
        res.status(502).json({
          success: false,
          error: 'RAG backend is unreachable. Is Flask running?',
        });
      }
    },
    proxyReq: (_proxyReq, req) => {
      if (!IS_PROD) {
        console.log(`[proxy] → ${req.method} ${FLASK_URL}${req.url}`);
      }
    },
  },
});

const pdfProxy = createProxyMiddleware({
  target: FLASK_URL,
  changeOrigin: true,
  pathRewrite: (path) => '/01_preprocessing' + path,
  on: {
    error: (err, _req, res) => {
      console.error('[proxy] PDF error:', err.message);
      if (!res.headersSent) {
        res.status(502).json({ error: 'PDF service unavailable' });
      }
    },
  },
});

// Guarded by the gateway-session auth gate mounted above.
app.use('/api', apiProxy);
app.use('/01_preprocessing', pdfProxy);

// ─── SPA fallback ───────────────────────────────────────────────────────────
app.get('*', (_req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

// ─── Start ───────────────────────────────────────────────────────────────────
app.listen(PORT, '0.0.0.0', () => {
  console.log('');
  console.log('═══════════════════════════════════════════════════════════');
  console.log('  CHiPS-RAG  –  Express UI Server');
  console.log('═══════════════════════════════════════════════════════════');
  console.log(`  UI      →  http://0.0.0.0:${PORT}`);
  console.log(`  Flask   →  ${FLASK_URL}  (proxied, gateway-session required)`);
  console.log(`  Mode    →  ${IS_PROD ? 'production' : 'development'}`);
  console.log('═══════════════════════════════════════════════════════════');
  console.log('');
});