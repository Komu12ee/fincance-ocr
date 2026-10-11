/**
 * CHiPS-RAG  –  Frontend application
 *
 * Single sign-on lives at the selection gateway. This pipeline UI trusts the
 * gateway session (an httpOnly cookie sent automatically on same-origin /api/*
 * requests); there is no separate pipeline login. On any 401 the app reloads so
 * the server's auth gate redirects to the gateway login.
 * PDFs are fetched as blobs (same-origin cookie) and shown in a half-screen panel.
 */

'use strict';
// Same-origin: the pipeline's Node server serves this page and proxies /api/* to
// its Flask backend. Empty base works for every pipeline (chips/fg/fg2) when hit
// directly (:3001/:3002/:3003). Set to "/pipeline/<name>" only behind a path-based reverse proxy.
// Empty for direct access; '/pipeline/<name>' when served under the reverse proxy
// (set by the inline snippet in the page <head>). Every same-origin call below is
// prefixed with it, so the app works in both topologies unchanged.
const BASE = (typeof window !== 'undefined' && window.__APP_BASE__) || '';

const THEME_KEY = 'chips_rag_theme';

function applyTheme(theme) {
  document.body.dataset.theme = theme;
  const toggle = document.getElementById('theme-toggle');
  if (toggle) {
    toggle.textContent = theme === 'light' ? 'Dark mode' : 'Light mode';
    toggle.setAttribute('aria-label', theme === 'light' ? 'Switch to dark mode' : 'Switch to light mode');
  }
}

function initTheme() {
  applyTheme('light');

  const toggle = document.getElementById('theme-toggle');
  if (toggle) {
    toggle.addEventListener('click', () => {
      const nextTheme = document.body.dataset.theme === 'light' ? 'dark' : 'light';
      localStorage.setItem(THEME_KEY, nextTheme);
      applyTheme(nextTheme);
    });
  }
}

// ── Token storage (sessionStorage: cleared on tab/browser close) ──────────
const TOKEN_KEY = 'chips_rag_token';
const USER_KEY = 'chips_rag_user';

const session = {
  save(token, user) {
    sessionStorage.setItem(TOKEN_KEY, token);
    sessionStorage.setItem(USER_KEY, JSON.stringify(user));
  },
  token() { return sessionStorage.getItem(TOKEN_KEY); },
  user() { try { return JSON.parse(sessionStorage.getItem(USER_KEY)); } catch { return null; } },
  clear() { sessionStorage.removeItem(TOKEN_KEY); sessionStorage.removeItem(USER_KEY); },
  exists() { return !!sessionStorage.getItem(TOKEN_KEY); },
};

// ── State ─────────────────────────────────────────────────────────────────
const state = {
  initialized: false,
  loading: false,
  lastResults: [],
  pdfBlobUrl: null,   // current blob URL so we can revoke it on close
  // Thinking mode is sticky for the session but deliberately NOT persisted — it is
  // slower and experimental, so each visit starts on the fast path.
  thinkingMode: true,   // default ON: multi-hop retrieval with query rewrite
};

// ── DOM refs ──────────────────────────────────────────────────────────────
const $ = id => document.getElementById(id);

const ui = {
  // login
  loginScreen: $('login-screen'),
  loginForm: $('login-form'),
  loginUsername: $('login-username'),
  loginPassword: $('login-password'),
  loginError: $('login-error'),
  loginSubmit: $('login-submit'),
  loginBtnText: $('login-btn-text'),
  loginBtnSpinner: $('login-btn-spinner'),

  // app shell
  appShell: $('app-shell'),
  userDisplay: $('user-display'),
  btnLogout: $('btn-logout'),

  // rag ui
  btnSend: $('btn-send'),
  btnThinking: $('btn-thinking'),
  queryInput: $('query-input'),
  chatEmpty: $('chat-empty'),
  chatMessages: $('chat-messages'),
  exampleList: $('example-list'),

  // source chunk drawer (right)
  drawerOverlay: $('drawer-overlay'),
  sourceDrawer: $('source-drawer'),
  drawerBody: $('drawer-body'),
  drawerClose: $('drawer-close'),

  // pdf panel (left half)
  pdfPanel: $('pdf-panel'),
  pdfOverlay: $('pdf-overlay'),
  pdfIframe: $('pdf-iframe'),
  pdfTitle: $('pdf-title'),
  pdfClose: $('pdf-close'),
  pdfLoading: $('pdf-loading'),

  toastContainer: $('toast-container'),
};

// ── API client ────────────────────────────────────────────────────────────
const api = {
  async _request(method, path, body) {
    // Auth is the same-origin gateway session cookie (sent automatically); no
    // Authorization header needed. The server's auth gate enforces it.
    const headers = { 'Content-Type': 'application/json' };
    const opts = { method, headers };
    if (body !== undefined) opts.body = JSON.stringify(body);

    const res = await fetch(path, opts);
    const json = await res.json().catch(() => ({}));

    if (res.status === 401) {
      // Gateway session missing/expired — bounce to the single sign-on gateway.
      goToGatewayLogin();
      throw new Error('UNAUTHENTICATED');
    }

    return { ok: res.ok, status: res.status, data: json };
  },

  // Fetch a PDF as a blob (relies on the same-origin session cookie).
  async fetchPdf(path) {
    const res = await fetch(path);
    if (res.status === 401) {
      goToGatewayLogin();
      throw new Error('UNAUTHENTICATED');
    }
    if (!res.ok) throw new Error(`PDF fetch failed: ${res.status}`);
    return res.blob();
  },

  login: (username, password) =>
    api._request('POST', `${BASE}/auth/login`, { username, password }, false),
  logout: () =>
    api._request('POST', `${BASE}/auth/logout`, undefined, false),
  health: () => api._request('GET', `${BASE}/api/health`),
  init: () => api._request('POST', `${BASE}/api/init`),
  dbStatus: () => api._request('GET', `${BASE}/api/db-status`),
  examples: () => api._request('GET', `${BASE}/api/examples`),
  settings: (body) => api._request(body ? 'POST' : 'GET', `${BASE}/api/settings`, body),
  query: (q, n, thinking) => api._request('POST', `${BASE}/api/query`,
    { query: q, num_results: n, thinking: !!thinking }),
};

// ── Session ───────────────────────────────────────────────────────────────
// Single sign-on lives at the gateway. When the pipeline session is missing or
// expired, a full-page reload hits the server's auth gate, which 302-redirects
// to the gateway login — so we never show a (now removed) local login screen.
let _redirecting = false;
function goToGatewayLogin() {
  if (_redirecting) return;
  _redirecting = true;
  window.location.reload();
}

// ── Screen switching ──────────────────────────────────────────────────────
function showLogin(errorMsg) {
  ui.appShell.classList.add('hidden');
  ui.loginScreen.classList.remove('hidden');
  ui.loginUsername.value = '';
  ui.loginPassword.value = '';
  if (errorMsg) showLoginError(errorMsg);
  else hideLoginError();
  ui.loginUsername.focus();
}

// Shown in the sidebar when no real account is signed in (login is skipped in
// this deployment). Placeholder name of a Chhattisgarh government user.
const PLACEHOLDER_USER = 'Lalit Kumar';

function showApp() {
  ui.loginScreen.classList.add('hidden');
  ui.appShell.classList.remove('hidden');
  const user = session.user();
  ui.userDisplay.textContent = (user && user.username) ? user.username : PLACEHOLDER_USER;
}

function showLoginError(msg) {
  ui.loginError.textContent = msg;
  ui.loginError.classList.remove('hidden');
}
function hideLoginError() {
  ui.loginError.classList.add('hidden');
}

// ── Login form ────────────────────────────────────────────────────────────
async function handleLogin(e) {
  e.preventDefault();
  hideLoginError();

  const username = ui.loginUsername.value.trim();
  const password = ui.loginPassword.value;

  if (!username || !password) {
    showLoginError('Please enter both username and password.');
    return;
  }

  ui.loginSubmit.disabled = true;
  ui.loginBtnText.classList.add('hidden');
  ui.loginBtnSpinner.classList.remove('hidden');

  try {
    const { ok, data } = await api.login(username, password);
    if (ok && data.success) {
      session.save(data.token, data.user);
      showApp();
      bootRagUI();
    } else {
      showLoginError(data.error || 'Invalid credentials.');
    }
  } catch (err) {
    if (err.message !== 'UNAUTHENTICATED') {
      showLoginError('Cannot reach server. Please try again.');
    }
  }

  ui.loginSubmit.disabled = false;
  ui.loginBtnText.classList.remove('hidden');
  ui.loginBtnSpinner.classList.add('hidden');
}

// ── Logout ────────────────────────────────────────────────────────────────
async function handleLogout() {
  closePdfPanel();
  closeDrawer();
  if (gadMode) {
    // GAD has its own session; clear it and return to the standalone GAD login.
    await fetch(`${BASE}/gad-auth/logout`, { method: 'POST', credentials: 'same-origin' }).catch(() => { });
    window.location.href = `${BASE}/login`;
    return;
  }
  // Other pipelines: clear the shared gateway session, then a reload lands on the
  // gateway login via the auth gate.
  await api.logout().catch(() => { });
  goToGatewayLogin();
}

// ── Toast ─────────────────────────────────────────────────────────────────
function toast(message, type = 'info', duration = 3500) {
  const el = document.createElement('div');
  el.className = `toast ${type}`;
  el.textContent = message;
  ui.toastContainer.appendChild(el);
  setTimeout(() => el.remove(), duration);
}

// ── Readiness ─────────────────────────────────────────────────────────────
// The pipeline is auto-initialised by the launcher, so the end user never sees
// or triggers any of this. We just confirm it's ready and, if not, warm it up
// silently before enabling the question box.
async function ensureReady() {
  try {
    const { ok, data } = await api.health();
    if (ok && data.pipeline_initialized) {
      state.initialized = true;
      enableQueryBar();
      return;
    }
  } catch (_) { /* fall through to warm-up */ }

  // Not ready yet — warm it up in the background.
  disableQueryBar('Preparing…');
  try {
    const { ok, data } = await api.init();
    if (ok && data.success) {
      state.initialized = true;
      showChatEmptyDefault();
      enableQueryBar();
      return;
    }
    // Distinguish an EMPTY knowledge base (no documents indexed yet) from a
    // transient warm-up failure. An empty KB is a real, explainable state — not a
    // reason to leave the user staring at a disabled "Preparing…" box forever.
    const details = (data && data.details) || {};
    const err = ((data && data.error) || '').toLowerCase();
    if (details.collection_exists === false || err.includes('does not exist')) {
      showNoDocumentsState();
    }
  } catch (_) { /* handled below */ }

  // Keep polling: the moment a document is verified/indexed the collection
  // appears and the next pass enables the bar automatically.
  setTimeout(ensureReady, 6000);
}

// Shown when the pipeline is up but its vector store has no indexed documents.
// Replaces the misleading permanent "Preparing…" with an honest, actionable note.
function showNoDocumentsState() {
  disableQueryBar('No documents indexed yet');
  const empty = ui.chatEmpty;
  if (empty && !empty.dataset.noDocs) {
    empty.dataset.noDocs = '1';
    const p = empty.querySelector('p');
    if (p) {
      p.innerHTML = 'No documents have been indexed yet, so there is nothing to search. ' +
        'An administrator needs to verify documents in the <strong>Documents</strong> workspace — ' +
        'each verified document joins the knowledge base, and this assistant activates automatically.';
    }
  }
}

function showChatEmptyDefault() {
  const empty = ui.chatEmpty;
  if (empty && empty.dataset.noDocs) {
    delete empty.dataset.noDocs;
    const p = empty.querySelector('p');
    if (p) {
      p.innerHTML = 'Ask a question about your department&rsquo;s documents and letters. ' +
        'Type your question below and press <strong>Send</strong>.';
    }
  }
}

// ── Query bar ─────────────────────────────────────────────────────────────
function enableQueryBar() {
  ui.btnSend.disabled = false;
  ui.queryInput.placeholder = 'Type your question here…';
}
function disableQueryBar(msg = 'One moment…') {
  ui.btnSend.disabled = true;
  ui.queryInput.placeholder = msg;
}
function autoResize() {
  ui.queryInput.style.height = 'auto';
  ui.queryInput.style.height = `${ui.queryInput.scrollHeight}px`;
}

// ── Chat ──────────────────────────────────────────────────────────────────
function hideChatEmpty() { ui.chatEmpty.style.display = 'none'; }

function escapeHtml(str) {
  return str
    .replace(/&/g, '&amp;').replace(/</g, '&lt;')
    .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

// Inline formatting: **bold** (the model emits this around headings, filenames
// and figures). Escaping runs first, so the ** markers are the only thing left
// to interpret — no raw HTML can slip through.
function renderInline(escaped) {
  // Input is already HTML-escaped. [label](url) becomes a clickable link — the
  // answer's Sources list uses that form, and without this the raw markdown text
  // would show on screen. Only http(s) URLs match, so the label cannot inject markup.
  return escaped
    .replace(/\[([^\]\n]+?)\]\((https?:\/\/[^)\s]+?)\)/g,
             '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>')
    .replace(/\*\*([^*\n]+?)\*\*/g, '<strong>$1</strong>');
}

// Block renderer for assistant answers. Turns the model's plain text into real
// paragraphs, bold, and bullet lists so "**Total = 4.44**" shows bold and
// "- item" / "* item" / "1. item" lines become a proper list.
function renderAnswer(raw) {
  const lines = escapeHtml(raw).split('\n');
  let html = '';
  let para = [];
  let listOpen = false;

  const flushPara = () => {
    if (para.length) { html += `<p>${renderInline(para.join('<br>'))}</p>`; para = []; }
  };
  const closeList = () => { if (listOpen) { html += '</ul>'; listOpen = false; } };

  for (const line of lines) {
    const t = line.trim();
    const bullet = t.match(/^(?:[-*•]|\d+\.)\s+(.*)$/);
    if (bullet && !t.startsWith('**')) {
      flushPara();
      if (!listOpen) { html += '<ul class="answer-list">'; listOpen = true; }
      html += `<li>${renderInline(bullet[1])}</li>`;
    } else if (t === '') {
      flushPara(); closeList();
    } else {
      closeList(); para.push(line);
    }
  }
  flushPara(); closeList();
  return html;
}

// Avatar images beside each message. The assistant is represented by the
// Government of Chhattisgarh emblem; the user by a neutral person mark.
const AVATARS = {
  user: 'img/avatar-user.svg',
  assistant: 'img/cg-emblem.png',
  thinking: 'img/cg-emblem.png',
};

// ── Reasoning trace (thinking mode) ───────────────────────────────────────
// Renders what the assistant actually did: which sub-questions it searched, what
// came back, what it judged missing, and how it ordered the documents by date. The
// point is that a precedence answer ("notice B supersedes notice A") can be checked
// rather than taken on faith, so each step shows its evidence — including the exact
// date string and the exact phrase that decided precedence.
const TRACE_LABEL = {
  plan: 'Plan', retrieve: 'Search', assess: 'Check',
  timeline: 'Dates', synthesize: 'Answer',
};

function buildTrace(steps, hops) {
  const wrap = document.createElement('div');
  wrap.className = 'trace';

  const btn = document.createElement('button');
  btn.className = 'trace-toggle';
  btn.type = 'button';
  btn.setAttribute('aria-expanded', 'false');
  const hopTxt = hops ? `${hops} search round${hops === 1 ? '' : 's'}` : `${steps.length} steps`;
  btn.innerHTML = `<span class="chev">▸</span> How this answer was reached (${escapeHtml(hopTxt)})`;

  const list = document.createElement('ul');
  list.className = 'trace-steps';
  list.hidden = true;
  steps.forEach(s => list.appendChild(buildTraceStep(s)));

  btn.addEventListener('click', () => {
    const open = btn.getAttribute('aria-expanded') === 'true';
    btn.setAttribute('aria-expanded', String(!open));
    list.hidden = open;
  });

  wrap.appendChild(btn);
  wrap.appendChild(list);
  return wrap;
}

function buildTraceStep(step) {
  const li = document.createElement('li');
  li.className = 'trace-step';
  li.dataset.kind = step.kind;

  const head = document.createElement('div');
  head.className = 'trace-head';
  head.innerHTML =
    `<span class="trace-kind">${escapeHtml(TRACE_LABEL[step.kind] || step.kind)}</span>` +
    `<span class="trace-title">${escapeHtml(step.title || '')}</span>` +
    `<span class="trace-time">${Number(step.elapsed_s || 0).toFixed(1)}s</span>`;
  li.appendChild(head);

  const detail = renderTraceDetail(step);
  if (detail) li.appendChild(detail);
  return li;
}

function renderTraceDetail(step) {
  const d = step.detail || {};
  const box = document.createElement('div');
  box.className = 'trace-detail';
  let html = '';

  if (step.kind === 'plan' && d.sub_questions?.length) {
    if (d.reasoning) html += `<div>${escapeHtml(d.reasoning)}</div>`;
    html += '<ul>' + d.sub_questions.map(q => `<li class="q">${escapeHtml(q)}</li>`).join('') + '</ul>';
  }

  if (step.kind === 'retrieve') {
    if (d.queries?.length) {
      html += '<ul>' + d.queries.map(q =>
        `<li class="q">${escapeHtml(q.question)} <span class="trace-time">${q.new ?? 0} new</span></li>`
      ).join('') + '</ul>';
    }
    (d.new_chunks || []).slice(0, 6).forEach(c => {
      html += `<div class="trace-chunk"><span class="src">${escapeHtml(c.source)}</span>` +
        (c.supporting ? ' <em>(adjacent)</em>' : '') +
        ` <span class="trace-time">${c.score}</span>` +
        `<span class="snip">${escapeHtml(c.snippet || '')}</span></div>`;
    });
  }

  if (step.kind === 'assess') {
    if (d.answered?.length) {
      html += '<ul>' + d.answered.map(a => `<li>✓ ${escapeHtml(a)}</li>`).join('') + '</ul>';
    }
    if (d.gaps?.length) {
      html += '<ul>' + d.gaps.map(g => `<li class="gap">Missing: ${escapeHtml(g)}</li>`).join('') + '</ul>';
    }
  }

  // The precedence evidence. Showing the matched date string and the matched
  // supersession phrase is the whole point — it is what makes the ruling checkable.
  if (step.kind === 'timeline' && d.documents?.length) {
    html += '<ul>' + d.documents.map(doc => {
      const when = doc.latest_date
        ? `${escapeHtml(doc.latest_date)}${doc.date_evidence ? ` <span class="trace-time">“${escapeHtml(doc.date_evidence)}”</span>` : ''}`
        : '<span class="trace-time">undated</span>';
      const rels = (doc.relations || [])
        .map(r => ` <span class="trace-rel">${escapeHtml(r.kind)}: “${escapeHtml(r.phrase)}”</span>`).join('');
      return `<li><span class="q">${escapeHtml(doc.source)}</span> — ${when}${rels}</li>`;
    }).join('') + '</ul>';
  }

  if (step.kind === 'synthesize' && d.sources?.length) {
    html += '<ul>' + d.sources.map(s => `<li>${escapeHtml(s)}</li>`).join('') + '</ul>';
  }

  if (d.error) html += `<div class="gap">${escapeHtml(d.error)}</div>`;
  if (!html) return null;
  box.innerHTML = html;
  return box;
}

function appendMessage(role, text, meta = {}) {
  hideChatEmpty();
  const msg = document.createElement('div');
  msg.className = `msg msg-${role}`;

  // Avatar
  const avatar = document.createElement('img');
  avatar.className = 'msg-avatar';
  avatar.src = AVATARS[role] || AVATARS.assistant;
  avatar.alt = role === 'user' ? 'You' : 'Assistant';

  // Content column (bubble + any attachments)
  const content = document.createElement('div');
  content.className = 'msg-content';

  const body = document.createElement('div');
  body.className = `msg-body ${role}-body`;

  if (role === 'thinking') {
    body.innerHTML =
      '<div class="thinking-dot"></div>' +
      '<div class="thinking-dot"></div>' +
      '<div class="thinking-dot"></div>';
  } else if (role === 'assistant') {
    body.innerHTML = renderAnswer(text);
  } else {
    // User's own message — plain text, no markdown interpretation.
    body.innerHTML = escapeHtml(text)
      .split('\n\n').filter(Boolean)
      .map(p => `<p>${p.replace(/\n/g, '<br>')}</p>`)
      .join('');
  }
  // Reasoning trace (thinking mode only), above the sources control. Collapsed
  // by default — the answer is the product, this is the audit trail.
  if (role === 'assistant' && meta.trace?.length) {
    body.appendChild(buildTrace(meta.trace, meta.hops));
  }
  // Sources control lives inside the bubble, under a hairline, so it reads as
  // part of the assistant's message rather than a detached button beneath it.
  if (role === 'assistant' && meta.results?.length) {
    const foot = document.createElement('div');
    foot.className = 'msg-sources';
    const btn = document.createElement('button');
    btn.className = 'sources-btn';
    btn.innerHTML = `<span class="source-count">${meta.results.length}</span> View sources`;
    btn.addEventListener('click', () => openDrawer(meta.results));
    foot.appendChild(btn);
    body.appendChild(foot);
  }
  content.appendChild(body);

  msg.appendChild(avatar);
  msg.appendChild(content);
  ui.chatMessages.appendChild(msg);
  msg.scrollIntoView({ behavior: 'smooth', block: 'end' });
  return msg;
}

// ── Send query ────────────────────────────────────────────────────────────
async function sendQuery() {
  const text = ui.queryInput.value.trim();
  if (!text || state.loading) return;

  state.loading = true;
  disableQueryBar('Finding the answer…');
  appendMessage('user', text);
  addToLastPrompts(text);
  ui.queryInput.value = '';
  autoResize();

  const thinkingEl = appendMessage('thinking', '');

  try {
    const deep = !!state.thinkingMode;
    const { ok, data } = await api.query(text, 8, deep);
    thinkingEl.remove();

    if (ok && data.success) {
      state.lastResults = data.results || [];
      appendMessage('assistant', data.answer, {
        results: state.lastResults,
        trace: data.trace,
        hops: data.hops,
      });
    } else {
      appendMessage('assistant', `Sorry, something went wrong: ${data.error || 'Unknown error'}`);
      toast(data.error || 'Query failed', 'error');
    }
  } catch (err) {
    if (err.message !== 'UNAUTHENTICATED') {
      thinkingEl.remove();
      appendMessage('assistant', 'Sorry, the assistant could not be reached. Please try again in a moment.');
      toast('Network error', 'error');
    }
  }

  state.loading = false;
  enableQueryBar();
}

// ── PDF panel ─────────────────────────────────────────────────────────────
async function openPdfPanel(fname) {
  // Show panel immediately with loading state
  ui.pdfTitle.textContent = fname;
  ui.pdfIframe.src = '';
  ui.pdfLoading.classList.remove('hidden');
  ui.pdfIframe.classList.add('hidden');
  ui.pdfPanel.classList.remove('hidden');
  ui.pdfOverlay.classList.remove('hidden');

  try {
    const pdfPath = `${BASE}/01_preprocessing/used_files/${encodeURIComponent(fname)}`;
    const blob = await api.fetchPdf(pdfPath);

    // Revoke previous blob URL to free memory
    if (state.pdfBlobUrl) {
      URL.revokeObjectURL(state.pdfBlobUrl);
    }

    state.pdfBlobUrl = URL.createObjectURL(blob);
    ui.pdfIframe.src = state.pdfBlobUrl;

    // Show iframe once loaded
    ui.pdfIframe.onload = () => {
      ui.pdfLoading.classList.add('hidden');
      ui.pdfIframe.classList.remove('hidden');
    };
  } catch (err) {
    if (err.message !== 'UNAUTHENTICATED') {
      ui.pdfLoading.classList.add('hidden');
      ui.pdfPanel.innerHTML +=
        `<div class="pdf-error">Could not load PDF: ${escapeHtml(err.message)}</div>`;
      toast('Could not load PDF', 'error');
    }
  }
}

function closePdfPanel() {
  ui.pdfPanel.classList.add('hidden');
  ui.pdfOverlay.classList.add('hidden');
  ui.pdfIframe.src = '';
  if (state.pdfBlobUrl) {
    URL.revokeObjectURL(state.pdfBlobUrl);
    state.pdfBlobUrl = null;
  }
}

// Chunks carry an internal header ("# Chunk 191\nDocument: …\nPages: …\n---\n").
// Strip it so the reader sees only the passage from the document.
function cleanExcerpt(text) {
  const marker = text.indexOf('---');
  let body = marker !== -1 ? text.slice(marker + 3) : text;
  body = body.replace(/^#\s*Chunk.*$/im, '').trim();
  return body.slice(0, 320) + (body.length > 320 ? '…' : '');
}

// ── Source drawer (right panel) ───────────────────────────────────────────
function openDrawer(results) {
  ui.drawerBody.innerHTML = '';

  // De-duplicate by document — the end user cares which documents the answer
  // came from, not how many internal chunks matched.
  const seen = new Set();
  results.forEach(r => {
    // Dedupe by the document itself (doc_dir), not its display name — several
    // distinct budget pages legitimately share the label "Main Budget".
    const key = r.source || r.actual_pdf || r.label || 'Document';
    if (seen.has(key)) return;
    seen.add(key);

    const label = r.label || r.actual_pdf || r.source || 'Document';
    const card = document.createElement('div');
    card.className = 'source-card';
    card.innerHTML = `
      <div class="source-card-header">
        <span class="source-filename">${escapeHtml(label)}</span>
      </div>
      <div class="source-excerpt">${escapeHtml(cleanExcerpt(r.excerpt || r.text || ''))}</div>
    `;

    // Most sources are scraped web pages with no local PDF — open the live source in
    // a new tab. Only fall back to the in-panel PDF viewer for uploaded documents
    // that actually have a used_files/<hash>.pdf on disk (no source_url).
    const pdfBtn = document.createElement('button');
    pdfBtn.className = 'pdf-open-btn';
    pdfBtn.textContent = r.source_url ? 'Open source' : 'Open document';
    pdfBtn.addEventListener('click', () => {
      if (r.source_url) window.open(r.source_url, '_blank', 'noopener');
      else openPdfPanel(r.actual_pdf || r.source);
    });
    card.appendChild(pdfBtn);

    ui.drawerBody.appendChild(card);
  });

  ui.drawerOverlay.classList.remove('hidden');
  ui.sourceDrawer.classList.remove('hidden');
}

function closeDrawer() {
  ui.drawerOverlay.classList.add('hidden');
  ui.sourceDrawer.classList.add('hidden');
}

// ── Last Prompts (session history) ────────────────────────────────────────
const sessionPrompts = [];

function addToLastPrompts(text) {
  // Avoid duplicate consecutive entries
  if (sessionPrompts[sessionPrompts.length - 1] === text) return;
  sessionPrompts.push(text);

  ui.exampleList.innerHTML = '';
  // Show newest first, up to 20
  [...sessionPrompts].reverse().slice(0, 20).forEach(prompt => {
    const li = document.createElement('li');
    li.textContent = prompt;
    li.title = prompt;
    li.addEventListener('click', () => {
      ui.queryInput.value = prompt;
      autoResize();
      ui.queryInput.focus();
    });
    ui.exampleList.appendChild(li);
  });
}


// ── RAG UI boot ───────────────────────────────────────────────────────────
async function bootRagUI() {
  disableQueryBar('Preparing…');
  await ensureReady();

  ui.btnSend.addEventListener('click', sendQuery);

  // Thinking mode toggle. The placeholder changes too, so it is obvious which
  // path the next question will take without reading the button.
  if (ui.btnThinking) {
    ui.btnThinking.addEventListener('click', () => {
      state.thinkingMode = !state.thinkingMode;
      ui.btnThinking.setAttribute('aria-pressed', String(state.thinkingMode));
      ui.queryInput.placeholder = state.thinkingMode
        ? 'Ask something that needs digging — it will search several times…'
        : 'Type your question here…';
      toast(state.thinkingMode
        ? 'Thinking mode on — slower, shows its reasoning'
        : 'Thinking mode off', 'info');
    });
    // Reflect the default (thinking mode ON) in the button + placeholder on load.
    ui.btnThinking.setAttribute('aria-pressed', String(state.thinkingMode));
    ui.queryInput.placeholder = state.thinkingMode
      ? 'Ask something that needs digging — it will search several times…'
      : 'Type your question here…';
  }

  ui.queryInput.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (!ui.btnSend.disabled) sendQuery();
    }
  });
  ui.queryInput.addEventListener('input', autoResize);

  // Drawer close
  ui.drawerClose.addEventListener('click', closeDrawer);
  ui.drawerOverlay.addEventListener('click', closeDrawer);

  // PDF panel close
  ui.pdfClose.addEventListener('click', closePdfPanel);
  ui.pdfOverlay.addEventListener('click', closePdfPanel);

  document.addEventListener('keydown', e => {
    if (e.key === 'Escape') { closeDrawer(); closePdfPanel(); }
  });
}

// ── GAD role gating ─────────────────────────────────────────────────────────
// GAD (fg2) has its own standalone login with admin/user roles. On other
// pipelines the /gad-auth/session endpoint does not exist, so this is a no-op and
// the UI behaves as before. When it IS the GAD pipeline:
//   · show the real signed-in username
//   · role "user"  → hide the Documents workspace (chatbot only)
//   · role "admin" → keep Documents and reveal the Accounts panel link
let gadMode = false;
async function applyGadRoleGating() {
  let data;
  try {
    const res = await fetch(`${BASE}/gad-auth/session`, { credentials: 'same-origin' });
    if (!res.ok) return;              // chips/fg: endpoint absent → leave UI as-is
    data = await res.json();
  } catch (_) { return; }
  if (!data || !data.authenticated || !data.user) return;

  gadMode = true;
  const { username, role } = data.user;
  if (ui.userDisplay) ui.userDisplay.textContent = username;

  const docs = document.getElementById('nav-documents');
  const accts = document.getElementById('nav-accounts');
  if (role === 'admin') {
    if (docs) docs.classList.remove('hidden');
    if (accts) accts.classList.remove('hidden');
  } else {
    if (docs) docs.classList.add('hidden');       // users: chatbot only
    if (accts) accts.classList.add('hidden');
  }
}

// ── Entry point ───────────────────────────────────────────────────────────
function boot() {
  ui.btnLogout.addEventListener('click', handleLogout);
  applyGadRoleGating();
  // Skip login UI for FG deployment (no auth). Boot the app immediately.
  // Keep logout handler available if someone wants to clear state.
  try { ui.loginForm.removeEventListener && ui.loginForm.removeEventListener('submit', handleLogin); } catch (_) { }
  try { ui.loginPassword.removeEventListener && ui.loginPassword.removeEventListener('keydown', handleLogin); } catch (_) { }

  showApp();
  bootRagUI();
}

initTheme();
boot();
