'use strict';

// Same-origin: this page's Node server proxies /api/* to the pipeline's Flask backend.
const BASE = (typeof window !== 'undefined' && window.__APP_BASE__) || '';

// ── Reviewer identity ────────────────────────────────────────────────────────
// On GAD (fg2) this page is admin-only, so the reviewer IS the signed-in admin —
// lock the field to their username. On other pipelines the /gad-auth/session
// endpoint is absent, so fall back to the previous free-text behaviour.
const reviewerInput = document.getElementById('reviewer');
reviewerInput.value = localStorage.getItem('doc_reviewer') || 'admin';
reviewerInput.addEventListener('change', () => {
  localStorage.setItem('doc_reviewer', reviewerInput.value.trim() || 'admin');
});
const reviewer = () => reviewerInput.value.trim() || 'admin';

(async function bindReviewerIdentity() {
  try {
    const res = await fetch(`${BASE}/gad-auth/session`, { credentials: 'same-origin' });
    if (!res.ok) return;
    const data = await res.json();
    if (data && data.authenticated && data.user) {
      reviewerInput.value = data.user.username;
      reviewerInput.readOnly = true;
      reviewerInput.title = 'Signed in as ' + data.user.username;
      localStorage.setItem('doc_reviewer', data.user.username);
    }
  } catch (_) { /* leave free-text default */ }
})();

// ── Collapsible document list ────────────────────────────────────────────────
const toggleListBtn = document.getElementById('toggle-list');
if (localStorage.getItem('doc_list_collapsed') === '1') document.body.classList.add('list-collapsed');
toggleListBtn.addEventListener('click', () => {
  const collapsed = document.body.classList.toggle('list-collapsed');
  localStorage.setItem('doc_list_collapsed', collapsed ? '1' : '0');
});

// ── State ────────────────────────────────────────────────────────────────────
let docs = [];
let selectedId = null;
let mdDirty = false;
let lastDetailSig = '';   // state of the currently-rendered detail bar (avoids needless re-render/PDF reload)
let filter = 'review';    // 'review' | 'all' | 'verified' | 'error'
let searchTerm = '';
let visibleFlat = [];     // docs in rendered order — for keyboard nav + auto-advance
let procById = {};        // id -> processing summary (how the md was produced)

// Human labels for the re-process methods offered in the panel.
const REPROCESS_OPTIONS = [
  ['auto',       'OCR: Docling + Sarvam (auto)'],
  ['ocr',        'OCR: Docling / EasyOCR only'],
  ['ocr_sarvam', 'OCR: Sarvam (all pages)'],
  ['extract',    'Text extraction (embedded PDF)'],
];

const $ = (id) => document.getElementById(id);
const toast = (msg) => {
  const t = $('toast'); t.textContent = msg; t.classList.add('show');
  setTimeout(() => t.classList.remove('show'), 2500);
};

// ── API ──────────────────────────────────────────────────────────────────────
const api = {
  list:   () => fetch(`${BASE}/api/documents`).then(r => r.json()),
  md:     (id) => fetch(`${BASE}/api/documents/${id}/md`).then(r => r.json()),
  processing: (id) => fetch(`${BASE}/api/documents/${id}/processing`).then(r => r.json()),
  reprocess:  (id, method) => fetch(`${BASE}/api/documents/${id}/reprocess`, {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ method }) }).then(r => r.json()),
  spellcheck: (id) => fetch(`${BASE}/api/documents/${id}/spellcheck`).then(r => r.json()),
  saveMd: (id, markdown) => fetch(`${BASE}/api/documents/${id}/md`, {
            method: 'PUT', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ markdown, edited_by: reviewer() }) }).then(r => r.json()),
  verify: (id) => fetch(`${BASE}/api/documents/${id}/verify`, {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ verified_by: reviewer() }) }).then(r => r.json()),
  unverify: (id) => fetch(`${BASE}/api/documents/${id}/unverify`, { method: 'POST' }).then(r => r.json()),
  remove: (id) => fetch(`${BASE}/api/documents/${id}`, { method: 'DELETE' }).then(r => r.json()),
  verifyBulk: (ids) => fetch(`${BASE}/api/documents/verify-bulk`, {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ ids, verified_by: reviewer() }) }).then(r => r.json()),
  upload: (file) => {
    const fd = new FormData();
    fd.append('file', file);
    fd.append('uploaded_by', reviewer());
    return fetch(`${BASE}/api/documents/upload`, { method: 'POST', body: fd }).then(r => r.json());
  },
};

// ── Status helpers ───────────────────────────────────────────────────────────
function statusOf(d) {
  if (d.status === 'error') return { cls: 'error', label: 'Error' };
  if (d.status === 'processing') return { cls: 'processing', label: 'Processing…' };
  if (d.verified && d.indexed) return { cls: 'verified', label: 'Verified ✓' };
  if (d.verified && !d.indexed) return { cls: 'indexing', label: 'Indexing…' };
  return { cls: 'pending', label: 'Needs verification' };
}

// ── List: filter + search + progress ─────────────────────────────────────────
function matchesFilter(d) {
  switch (filter) {
    case 'review':   return d.status === 'processed' && !d.verified;
    case 'verified': return !!d.verified;
    case 'error':    return d.status === 'error';
    // Method filters. 'auto' means the page split used BOTH engines, so it counts
    // as Sarvam — those are exactly the pages a reviewer wants to re-read.
    case 'sarvam':   return d.ocr_method === 'ocr_sarvam' || d.ocr_method === 'auto';
    case 'ocr':      return d.ocr_method === 'ocr';
    case 'extract':  return d.ocr_method === 'extract';
    default:         return true; // 'all'
  }
}

// Short badge for the list row: enough to triage at a glance, full detail on hover.
const METHOD_BADGE = {
  auto:       { label: 'Sarvam + Docling', cls: 'm-sarvam' },
  ocr_sarvam: { label: 'Sarvam OCR',       cls: 'm-sarvam' },
  ocr:        { label: 'Local OCR',        cls: 'm-ocr' },
  extract:    { label: 'Extracted',        cls: 'm-extract' },
};
function matchesSearch(d) {
  if (!searchTerm) return true;
  return `${d.original_name || ''} ${d.source_url || ''} ${d.uploaded_by || ''}`
    .toLowerCase().includes(searchTerm);
}

// While searching, look across ALL statuses (find-anything); otherwise honour the
// active chip. Returns docs grouped for display + a flat list in rendered order.
function computeVisible() {
  const base = searchTerm ? docs.filter(matchesSearch)
                          : docs.filter(d => matchesFilter(d) && matchesSearch(d));
  const groups = {
    'Processing':         base.filter(d => d.status === 'processing'),
    'Needs verification': base.filter(d => d.status === 'processed' && !d.verified),
    'Verified':           base.filter(d => d.verified),
    'Errors':             base.filter(d => d.status === 'error'),
  };
  const flat = [];
  for (const items of Object.values(groups)) for (const d of items) flat.push(d);
  return { groups, flat };
}

function renderProgress() {
  const total = docs.length;
  const box = $('progress');
  if (!total) { box.hidden = true; return; }
  const verified = docs.filter(d => d.verified).length;
  const pending = docs.filter(d => d.status === 'processed' && !d.verified).length;
  box.hidden = false;
  $('progress-fill').style.width = `${Math.round((verified / total) * 100)}%`;
  $('progress-text').textContent = `${verified} of ${total} verified · ${pending} to review`;
}

// Show a bulk-verify button for the pending docs currently in view (respects the
// active filter + search — you verify exactly what you see).
function updateBulkButton() {
  const btn = $('bulk-verify');
  if (!btn) return;
  const pending = visibleFlat.filter(d => d.status === 'processed' && !d.verified);
  btn.hidden = pending.length === 0;
  if (!btn.hidden) { btn.disabled = false; btn.textContent = `Verify all ${pending.length} shown ✓`; }
}

function renderList() {
  renderProgress();
  const { groups, flat } = computeVisible();
  visibleFlat = flat;
  updateBulkButton();
  const list = $('list');
  list.innerHTML = '';

  if (!docs.length) {
    list.innerHTML = '<div class="list-empty">No documents yet — upload a PDF or import a scraped batch.</div>';
    return;
  }
  if (!flat.length) {
    list.innerHTML = `<div class="list-empty">${searchTerm
      ? 'No documents match your search.' : 'Nothing here — try a different filter.'}</div>`;
    return;
  }

  for (const [label, items] of Object.entries(groups)) {
    if (!items.length) continue;
    const gl = document.createElement('div');
    gl.className = 'group-label';
    gl.textContent = `${label} (${items.length})`;
    list.appendChild(gl);
    for (const d of items) {
      const s = statusOf(d);
      const el = document.createElement('div');
      el.className = 'doc' + (d.id === selectedId ? ' active' : '');
      const srcLine = d.source === 'scrape' && d.source_url
        ? `<div class="meta url" title="${escapeHtml(d.source_url)}">🌐 ${escapeHtml(d.source_url)}</div>` : '';
      // The method sits next to the status so "Needs verification · Sarvam OCR"
      // reads as one line — you can see what produced the text before opening it.
      const m = METHOD_BADGE[d.ocr_method];
      const methodBadge = m
        ? `<span class="badge method ${m.cls}" title="${escapeHtml(d.ocr_label || m.label)}">${m.label}</span>`
        : '';
      el.innerHTML =
        `<div class="name">${escapeHtml(d.original_name)}</div>` +
        srcLine +
        `<div class="meta">${escapeHtml(d.uploaded_by || '')} · ${fmtDate(d.uploaded_at)}</div>` +
        `<span class="badge ${s.cls}">${s.label}</span>${methodBadge}`;
      el.addEventListener('click', () => selectDoc(d.id));
      list.appendChild(el);
    }
  }
}

// Select the next reviewable doc after acting on the current one (keeps flow going).
function advanceAfterAction(actedId) {
  const pending = docs.filter(d => d.status === 'processed' && !d.verified && d.id !== actedId);
  if (pending.length) { selectDoc(pending[0].id); return true; }
  return false;
}

// ── Detail view ──────────────────────────────────────────────────────────────
async function selectDoc(id) {
  if (mdDirty && !confirm('Discard unsaved edits to the current document?')) return;
  selectedId = id;
  mdDirty = false;
  renderList();
  renderDetail();

  const d = docs.find(x => x.id === id);
  if (!d) return;
  if (d.status === 'error') return;

  const res = await api.md(id);
  if (selectedId !== id) return; // switched away
  const ta = $('md-text');
  if (ta) {
    if (res.success) ta.value = res.markdown;
    else ta.value = `(${res.error || 'not ready'})`;
  }
}

// A short signature of the state that affects the detail *bar* (not the PDF/md).
function detailSig(d) { return d ? `${d.status}|${d.verified}|${d.indexed}|${!!d.error}` : ''; }

// HTML for just the top bar (title + status + action buttons).
function detailBarHTML(d) {
  const verifiedInfo = d.verified
    ? `<span class="status-note">Verified by ${escapeHtml(d.verified_by || '?')} · ${fmtDate(d.verified_at)}${d.indexed ? '' : ' · indexing…'}</span>`
    : `<span class="status-note" style="color:var(--amber,#a35a10)">Not verified</span>`;
  const verifyBtn = d.verified
    ? `<button class="act ghost" id="btn-unverify">Unverify</button>`
    : `<button class="act verify" id="btn-verify" ${d.status !== 'processed' ? 'disabled' : ''}>Mark Verified ✓</button>`;
  const srcLink = d.source === 'scrape' && d.source_url
    ? `<a class="title-src" href="${escapeHtml(d.source_url)}" target="_blank" rel="noopener" title="${escapeHtml(d.source_url)}">🌐 ${escapeHtml(d.source_url)}</a>` : '';
  return `
    <div class="title-wrap">
      <div class="title">${escapeHtml(d.original_name)}</div>
      ${srcLink}
      ${methodRowHTML(d)}
    </div>
    <div class="actions">
      ${verifiedInfo}
      <button class="act discard" id="btn-discard" title="Remove this document from the workspace (does not enter the RAG)">Discard</button>
      <button class="act save" id="btn-save" ${mdDirty ? '' : 'disabled'}>Save edits</button>
      ${verifyBtn}
    </div>`;
}

// "How this text was produced" badge + a re-process control. The method comes from
// the /processing endpoint (cached in procById); until it loads we show a spinner.
function methodRowHTML(d) {
  const p = procById[d.id];
  const label = p && p.label ? p.label : 'checking method…';
  const busy = d.status === 'processing';
  let control = '';
  if (p && p.can_reprocess) {
    const opts = REPROCESS_OPTIONS
      .map(([v, t]) => `<option value="${v}"${p.method === v ? ' selected' : ''}>${t}</option>`).join('');
    control = `
      <select id="reproc-method" class="reproc-sel" ${busy ? 'disabled' : ''} title="Choose how to re-produce the text">${opts}</select>
      <button class="act ghost" id="btn-reprocess" ${busy ? 'disabled' : ''}>${busy ? 'Processing…' : 'Re-process'}</button>`;
  }
  return `<div class="method-row">
      <span class="method-badge" title="How this document's markdown was produced">⚙ ${escapeHtml(label)}</span>
      ${control}
    </div>`;
}

// Fetch (once) how a doc was processed, cache it, and refresh the bar in place.
async function loadProcessing(id, force) {
  if (!force && procById[id]) return;
  try {
    const res = await api.processing(id);
    if (res && res.success) {
      procById[id] = res.processing;
      if (selectedId === id) updateDetailBar();
    }
  } catch (_) { /* leave the badge as 'checking…' */ }
}

// Left review pane: the SOURCE that produced the markdown. For uploads that is
// the PDF; for scraped docs it is the raw fetched content (html/pdf/json) — or,
// when the scraper kept no per-doc snapshot, the source URL to open directly.
function sourcePaneHTML(d) {
  if (d.source === 'scrape') {
    const url = d.source_url || '';
    const label = `<div class="pane-label">Scraped source${url
      ? ` · <a class="src-url" href="${escapeHtml(url)}" target="_blank" rel="noopener">${escapeHtml(url)}</a>` : ''}</div>`;
    if (d.raw_ext) {
      const sandbox = (d.raw_ext === 'html' || d.raw_ext === 'htm') ? 'sandbox=""' : '';
      return label + `<iframe src="${BASE}/api/documents/${d.id}/raw" ${sandbox}></iframe>`;
    }
    return label + `<div class="no-raw">No stored snapshot of the raw source for this item —
      it was extracted from a listing/API response.${url
        ? `<br><br><a href="${escapeHtml(url)}" target="_blank" rel="noopener">Open source URL ↗</a>` : ''}</div>`;
  }
  // Uploads normally have a source PDF; OCR-imported docs may not. Show a clear
  // note instead of a broken frame (the /pdf endpoint would 404 for these).
  if (!d.pdf_path) {
    return `<div class="pane-label">Source PDF</div>` +
      `<div class="no-raw">No source PDF is stored for this document — it was ingested as
       OCR text. Review the extracted markdown on the right and verify it into the knowledge base.</div>`;
  }
  return `<div class="pane-label">Source PDF</div>` +
         `<iframe src="${BASE}/api/documents/${d.id}/pdf#toolbar=1"></iframe>`;
}

// (Re)wire the buttons in the bar. Safe to call repeatedly (fresh nodes each time).
function wireBar(d) {
  const ta = () => $('md-text');
  const saveBtn = $('btn-save');
  if (saveBtn) saveBtn.addEventListener('click', async () => {
    saveBtn.disabled = true; saveBtn.textContent = 'Saving…';
    const res = await api.saveMd(d.id, ta().value);
    saveBtn.textContent = 'Save edits';
    if (res.success) { mdDirty = false; toast(res.reindexing ? 'Saved — re-indexing into RAG…' : 'Saved.'); await refresh(); }
    else { saveBtn.disabled = false; toast('Save failed: ' + (res.error || 'error')); }
  });
  const vb = $('btn-verify');
  if (vb) vb.addEventListener('click', async () => {
    vb.disabled = true; vb.textContent = 'Verifying…';
    const res = await api.verify(d.id);
    if (res.success) {
      toast('Verified — indexing into RAG…');
      mdDirty = false;
      await refresh();
      advanceAfterAction(d.id);   // keep the reviewer moving to the next pending doc
    } else { vb.disabled = false; vb.textContent = 'Mark Verified ✓'; toast('Verify failed: ' + (res.error || 'error')); }
  });
  const rp = $('btn-reprocess');
  if (rp) rp.addEventListener('click', async () => {
    const method = ($('reproc-method') || {}).value || 'auto';
    const label = (REPROCESS_OPTIONS.find(o => o[0] === method) || [, method])[1];
    if (!confirm(`Re-process "${d.original_name}" with:\n\n${label}\n\nThis rebuilds its text and, if the doc is verified, re-indexes it into the RAG.`)) return;
    rp.disabled = true; rp.textContent = 'Starting…';
    const res = await api.reprocess(d.id, method);
    if (res.success) {
      toast('Re-processing… the panel will update when it finishes.');
      // Optimistic: show it as processing immediately; polling + /processing refresh do the rest.
      const doc = docs.find(x => x.id === d.id); if (doc) doc.status = 'processing';
      delete procById[d.id];
      updateDetailBar();
    } else {
      rp.disabled = false; rp.textContent = 'Re-process';
      toast('Re-process failed: ' + (res.error || 'error'));
    }
  });
  const db = $('btn-discard');
  if (db) db.addEventListener('click', async () => {
    if (!confirm(`Discard "${d.original_name}"?\n\nIt is removed from the workspace and the RAG. The source/OCR files stay on disk (a re-import can restore it).`)) return;
    db.disabled = true; db.textContent = 'Discarding…';
    const res = await api.remove(d.id);
    if (res.success) {
      toast('Discarded.');
      mdDirty = false;
      await refresh();
      if (!advanceAfterAction(d.id)) { selectedId = null; renderDetail(); }
    } else { db.disabled = false; db.textContent = 'Discard'; toast('Discard failed: ' + (res.error || 'error')); }
  });
  const ub = $('btn-unverify');
  if (ub) ub.addEventListener('click', async () => {
    if (!confirm('Remove this document from the RAG (unverify)?')) return;
    ub.disabled = true;
    const res = await api.unverify(d.id);
    if (res.success) { toast('Removed from RAG.'); await refresh(); }
    else { ub.disabled = false; toast('Failed: ' + (res.error || 'error')); }
  });
}

// Update ONLY the top bar in place — never touches the PDF iframe or the textarea.
function updateDetailBar() {
  const d = docs.find(x => x.id === selectedId);
  const bar = $('detail-bar');
  if (!d || !bar) return;
  bar.innerHTML = detailBarHTML(d);
  wireBar(d);
  lastDetailSig = detailSig(d);
}

// Full render — builds the whole detail incl. the iframe. Called only when the
// selected document CHANGES (or transitions to/from the error view).
function renderDetail() {
  const d = docs.find(x => x.id === selectedId);
  const detail = $('detail');
  if (!d) { detail.innerHTML = '<div class="placeholder">Select a document.</div>'; lastDetailSig = ''; return; }
  lastDetailSig = detailSig(d);

  if (d.status === 'error') {
    // Keep the full bar so a failed doc can be re-processed a different way.
    detail.innerHTML =
      `<div class="detail-bar" id="detail-bar">${detailBarHTML(d)}</div>` +
      `<div class="err-box">Processing failed:\n\n${escapeHtml(d.error || 'unknown error')}</div>`;
    wireBar(d);
    loadProcessing(d.id);
    return;
  }

  detail.innerHTML = `
    <div class="detail-bar" id="detail-bar">${detailBarHTML(d)}</div>
    <div class="panes">
      <div class="pane pdf">${sourcePaneHTML(d)}</div>
      <div class="pane md">
        <div class="pane-label">
          <span>Extracted Markdown (editable)</span>
          <button class="check-words" id="btn-check" type="button"
                  title="Highlight words OCR may have misread, with suggested corrections">
            Check words
          </button>
        </div>
        <textarea id="md-text" spellcheck="false" placeholder="${d.status === 'processing' ? 'Processing…' : ''}"></textarea>
        <div id="md-review" class="md-review hidden" tabindex="0"></div>
      </div>
    </div>`;

  $('md-text').addEventListener('input', () => {
    mdDirty = true; const sb = $('btn-save'); if (sb) sb.disabled = false;
    // Highlights are computed from the text; once it is edited by hand they are
    // stale, so drop back to the plain editor rather than showing wrong offsets.
    if (reviewMode) exitReview();
  });
  const checkBtn = $('btn-check');
  if (checkBtn) checkBtn.addEventListener('click', () => (reviewMode ? exitReview() : enterReview(d.id)));
  wireBar(d);
  loadProcessing(d.id);
}

// ── Word review: highlight suspect OCR words, suggest, replace ───────────────
// The textarea stays the source of truth. Review mode renders a read-only overlay of
// the SAME text with flagged words marked, so a reviewer can fix words by clicking
// instead of hunting through prose — then falls back to the textarea for free editing.
//
// Nothing is auto-corrected: every replacement is an explicit click, and the result
// still has to be saved and verified. That is deliberate for official records, where a
// silently "improved" date or post count is worse than a visible error.
let reviewMode = false;
let reviewIssues = [];        // [{word,start,end,reason,suggestions}], offsets into md
let popEl = null;

const REASON_TEXT = {
  broken_encoding: 'Broken character encoding — the PDF font mapped this to non-text codes',
  orphan_matra: 'Starts with a vowel mark — the letter it attaches to was lost',
  unknown_word: 'Not found in the Hindi dictionary',
};

async function enterReview(docId) {
  const btn = $('btn-check');
  if (btn) { btn.disabled = true; btn.textContent = 'Checking…'; }
  let res;
  try {
    res = await api.spellcheck(docId);
  } catch (_) {
    res = { success: false, error: 'Could not reach the spellchecker.' };
  }
  if (btn) btn.disabled = false;

  if (!res || !res.success) {
    if (btn) btn.textContent = 'Check words';
    toast((res && res.error) || 'Spellcheck failed', 'error');
    return;
  }
  reviewIssues = res.issues || [];
  reviewMode = true;
  if (btn) btn.textContent = 'Back to editing';
  renderReview();

  // Word-by-word fixing only pays off when the errors are few. Past roughly one bad
  // word in eight, the page was misread rather than mis-spelled and re-running OCR
  // (Sarvam, all pages) is faster and better than clicking through it — so say so
  // instead of leaving a wall of highlights to grind through.
  const n = reviewIssues.length;
  const words = countDevanagariWords($('md-text').value);
  const ratio = words ? n / words : 0;
  if (!n) {
    toast('No suspect words found', 'success');
  } else if (ratio > 0.125) {
    toast(`${n} of ~${words} words look wrong (${Math.round(ratio * 100)}%) — ` +
          `this page may be better re-read: use Reprocess → Sarvam (all pages).`, 'error');
  } else {
    toast(`${n} word(s) to check${res.truncated ? ' (first 400 shown)' : ''}`, 'info');
  }
}

/** Devanagari words in the text — the denominator for "how bad is this page?". */
function countDevanagariWords(text) {
  return ((text || '').match(/[ऀ-ॿ]{3,}/g) || []).length;
}

function exitReview() {
  reviewMode = false;
  closePopover();
  const btn = $('btn-check');
  if (btn) btn.textContent = 'Check words';
  const rv = $('md-review'), ta = $('md-text');
  if (rv) rv.classList.add('hidden');
  if (ta) ta.classList.remove('hidden');
}

/** Build the highlighted read-only view from the textarea's current text. */
function renderReview() {
  const ta = $('md-text'), rv = $('md-review');
  if (!ta || !rv) return;
  const text = ta.value;

  // Walk the issue offsets in order, escaping the untouched spans between them.
  let html = '', cursor = 0;
  reviewIssues.forEach((iss, idx) => {
    if (iss.start < cursor) return;                 // overlapping/stale — skip
    html += escapeHtml(text.slice(cursor, iss.start));
    const has = (iss.suggestions || []).length;
    html += `<mark class="sus ${iss.reason}${has ? '' : ' nofix'}" data-idx="${idx}"` +
            ` title="${escapeHtml(REASON_TEXT[iss.reason] || '')}">` +
            `${escapeHtml(text.slice(iss.start, iss.end))}</mark>`;
    cursor = iss.end;
  });
  html += escapeHtml(text.slice(cursor));

  rv.innerHTML = html || '<em class="empty">No text.</em>';
  rv.classList.remove('hidden');
  ta.classList.add('hidden');

  // Hover opens the suggestion list; it stays while the pointer is on the word OR
  // the popover, so it can actually be clicked into.
  rv.querySelectorAll('mark.sus').forEach((el) => {
    el.addEventListener('mouseenter', () => openPopover(el));
    el.addEventListener('mouseleave', () => scheduleClose());
    el.addEventListener('click', () => openPopover(el));   // touch/keyboard friendly
  });
}

let closeTimer = null;
function scheduleClose() { clearTimeout(closeTimer); closeTimer = setTimeout(closePopover, 260); }
function cancelClose() { clearTimeout(closeTimer); }

function closePopover() {
  if (popEl) { popEl.remove(); popEl = null; }
}

function openPopover(markEl) {
  cancelClose();
  closePopover();
  const idx = Number(markEl.dataset.idx);
  const iss = reviewIssues[idx];
  if (!iss) return;

  const pop = document.createElement('div');
  pop.className = 'sus-pop';
  const sugs = iss.suggestions || [];
  pop.innerHTML =
    `<div class="sus-why">${escapeHtml(REASON_TEXT[iss.reason] || 'Check this word')}</div>` +
    (sugs.length
      ? `<div class="sus-list">` + sugs.map((s, i) =>
          `<button class="sus-opt" data-term="${escapeHtml(s.term)}" data-idx="${idx}">` +
          `<span class="t">${escapeHtml(s.term)}</span>` +
          `<span class="d">edit ${s.distance} · seen ${s.frequency}×</span></button>`).join('')
        + `</div>`
      : `<div class="sus-none">No close match in the dictionary — edit by hand if it is wrong.</div>`) +
    `<button class="sus-keep" data-idx="${idx}">Keep as is</button>`;

  document.body.appendChild(pop);
  const r = markEl.getBoundingClientRect();
  // Prefer below; flip above when it would fall off the viewport.
  const top = r.bottom + 6 + window.scrollY;
  pop.style.left = `${Math.max(8, Math.min(r.left + window.scrollX, window.innerWidth - pop.offsetWidth - 8))}px`;
  pop.style.top = (top + pop.offsetHeight > window.scrollY + window.innerHeight)
    ? `${r.top + window.scrollY - pop.offsetHeight - 6}px` : `${top}px`;

  pop.addEventListener('mouseenter', cancelClose);
  pop.addEventListener('mouseleave', scheduleClose);
  pop.querySelectorAll('.sus-opt').forEach((b) =>
    b.addEventListener('click', () => applySuggestion(Number(b.dataset.idx), b.dataset.term)));
  pop.querySelector('.sus-keep').addEventListener('click', () => dismissIssue(idx));
  popEl = pop;
}

/** Replace one flagged word, shifting the offsets of everything after it. */
function applySuggestion(idx, term) {
  const ta = $('md-text');
  const iss = reviewIssues[idx];
  if (!ta || !iss) return;

  const text = ta.value;
  ta.value = text.slice(0, iss.start) + term + text.slice(iss.end);
  mdDirty = true;
  const sb = $('btn-save'); if (sb) sb.disabled = false;

  // Offsets after this word move by the length difference. Adjusting locally keeps
  // the interaction instant — re-running the whole check per click would not.
  const delta = term.length - (iss.end - iss.start);
  reviewIssues = reviewIssues
    .filter((_, i) => i !== idx)
    .map((o) => (o.start > iss.start ? { ...o, start: o.start + delta, end: o.end + delta } : o));

  closePopover();
  renderReview();
  toast(`Replaced with “${term}”`, 'success');
}

/** Accept the word as correct — drops the highlight without changing the text. */
function dismissIssue(idx) {
  reviewIssues = reviewIssues.filter((_, i) => i !== idx);
  closePopover();
  renderReview();
}

// ── Refresh / polling ────────────────────────────────────────────────────────
async function refresh() {
  const res = await api.list();
  if (!res.success) return;
  docs = res.documents || [];
  renderList();

  // If the selected doc just finished processing and the textarea is empty, load its md.
  const d = docs.find(x => x.id === selectedId);
  const ta = $('md-text');
  if (d && ta && !mdDirty && d.md_path !== null && ta.value.startsWith('(')) {
    const md = await api.md(d.id);
    if (md.success && !mdDirty) ta.value = md.markdown;
  }
  // Update ONLY the status bar when the doc's state actually changed — never
  // rebuild the panes, so the PDF iframe (and any scroll position) is preserved.
  if (d && !mdDirty && detailSig(d) !== lastDetailSig) {
    if (d.status === 'error' || !$('detail-bar')) renderDetail();  // structure differs → full render
    else updateDetailBar();
    loadProcessing(d.id, true);   // status changed (e.g. re-process finished) → refresh the method badge
  }
}

// ── Upload ───────────────────────────────────────────────────────────────────
$('upload-btn').addEventListener('click', () => $('file-input').click());
$('file-input').addEventListener('change', async (e) => {
  const file = e.target.files[0];
  if (!file) return;
  e.target.value = '';
  toast('Uploading…');
  const res = await api.upload(file);
  if (res.success) { toast('Uploaded — OCR running…'); await refresh(); selectDoc(res.id); }
  else toast('Upload failed: ' + (res.error || 'error'));
});

// ── Search + status filter ────────────────────────────────────────────────────
const searchEl = $('doc-search');
if (searchEl) searchEl.addEventListener('input', () => {
  searchTerm = searchEl.value.trim().toLowerCase();
  renderList();
});
$('filter-chips').addEventListener('click', (e) => {
  const chip = e.target.closest('.chip');
  if (!chip) return;
  filter = chip.dataset.filter;
  for (const c of $('filter-chips').querySelectorAll('.chip')) c.classList.toggle('active', c === chip);
  renderList();
});

// ── Bulk verify (everything currently in view) ─────────────────────────────────
$('bulk-verify').addEventListener('click', async () => {
  const pending = visibleFlat.filter(d => d.status === 'processed' && !d.verified);
  if (!pending.length) return;
  if (!confirm(`Verify all ${pending.length} document(s) shown?\n\nThey will be embedded into the RAG (indexing runs in the background). You can unverify any of them later.`)) return;
  const btn = $('bulk-verify');
  btn.disabled = true; btn.textContent = 'Verifying…';
  const res = await api.verifyBulk(pending.map(d => d.id));
  if (res.success) { toast(`Verifying ${res.verified} document(s) — indexing in the background…`); await refresh(); }
  else { btn.disabled = false; toast('Bulk verify failed: ' + (res.error || 'error')); }
});

// ── Keyboard navigation (throughput over hundreds of docs) ─────────────────────
// ↑/↓ (or j/k) move through the visible list; ⌘/Ctrl+Enter verifies the current doc.
document.addEventListener('keydown', (e) => {
  const el = document.activeElement;
  const inField = el && /^(INPUT|TEXTAREA|SELECT)$/.test(el.tagName);
  if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {   // verify from anywhere, even mid-edit
    const vb = $('btn-verify');
    if (vb && !vb.disabled) { e.preventDefault(); vb.click(); }
    return;
  }
  if (inField) return;                                    // don't hijack normal typing
  const dir = (e.key === 'ArrowDown' || e.key === 'j') ? 1
            : (e.key === 'ArrowUp' || e.key === 'k') ? -1 : 0;
  if (!dir || !visibleFlat.length) return;
  e.preventDefault();
  const idx = visibleFlat.findIndex(d => d.id === selectedId);
  const next = idx < 0 ? (dir > 0 ? 0 : visibleFlat.length - 1)
                       : Math.min(visibleFlat.length - 1, Math.max(0, idx + dir));
  const target = visibleFlat[next];
  if (target && target.id !== selectedId) {
    selectDoc(target.id);
    const active = $('list').querySelector('.doc.active');
    if (active) active.scrollIntoView({ block: 'nearest' });
  }
});

// ── Utils ────────────────────────────────────────────────────────────────────
function escapeHtml(s) {
  return String(s == null ? '' : s).replace(/[&<>"']/g, c =>
    ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}
function fmtDate(iso) {
  if (!iso) return '';
  try { return new Date(iso).toLocaleString([], { dateStyle: 'short', timeStyle: 'short' }); }
  catch { return iso; }
}

// ── Boot ─────────────────────────────────────────────────────────────────────
refresh();
setInterval(refresh, 4000);
