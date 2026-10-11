"""Produce the text for every scraped PDF — by OCR, or by keeping a clean text layer.

The scraper stores PDFs raw and never opens them (see Scraper/extract.py), so this
is the ONLY component allowed to decide what is inside a PDF. It writes
stage2/<urlhash>/structured.md for each one, which is exactly where the OCR
pipeline would have put it — so scrape_import and the chunker see one uniform shape
regardless of how the text was obtained.

Two ways a PDF gets its text:

  OCR (stage1 -> stage2)   the default for scans and for broken-font PDFs.
  embedded text layer      kept ONLY when it is confidently clean.

Why the gate exists: many CG govt Hindi PDFs embed non-Unicode fonts, so their
"text layer" decodes to mojibake — measured on the live GAD corpus, 45 of 207
extractions were junk like "Effisrc $rFT gFIrq gvnff{". Embedding that is worse
than useless. But OCR'ing a genuinely clean digital PDF is also a loss (layout
scrambling, hyphenation, ~45s/page), so we keep text only when it clearly reads as
language, and OCR everything else. The asymmetry is deliberate: OCR'ing a clean PDF
costs time, keeping mojibake corrupts the index permanently.

    # preview the decision for every PDF WITHOUT spending the OCR hours
    PIPELINE=fg2 uv run python core/ocr_scanned.py --corpus ~/CHiPS-RAG/data/scrapped_data --gate-report

    # do the work
    PIPELINE=fg2 uv run python core/ocr_scanned.py --corpus ~/CHiPS-RAG/data/scrapped_data
    PIPELINE=fg2 uv run python core/ocr_scanned.py --corpus … --force-ocr-all  # ignore the gate
    PIPELINE=fg2 uv run python core/ocr_scanned.py --corpus … --limit 25       # produce ≤25 new this run
    PIPELINE=fg2 uv run python core/ocr_scanned.py --corpus … --head 50        # only the 50 most-recent (crawl order)
    PIPELINE=fg2 uv run python core/ocr_scanned.py --corpus … --head 50 --limit 1  # one round-robin step
    PIPELINE=fg2 uv run python core/ocr_scanned.py --corpus … --force          # redo existing
    PIPELINE=fg2 uv run python core/ocr_scanned.py --corpus … --max-pages 40   # raise the page budget
    PIPELINE=fg2 uv run python core/ocr_scanned.py --corpus … --max-pages 0    # drain the deferred large PDFs

PDFs beyond --max-pages (default 25) are skipped and recorded to
<data_root>/skipped_large_pdfs.jsonl (pipeline + url + page count) so a later
dedicated pass can OCR exactly those with --max-pages 0. Re-running is safe: docs
that already produced stage2/<hash>/structured.md are skipped, and the skip list
is deduped by urlhash.

Run with the stack STOPPED — OCR contends for CPU and Qdrant is single-writer.
Must run BEFORE scrape_import (which refuses to register a PDF that has no text yet).
"""

import argparse
import json
import os
import re
import subprocess
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

CORE = Path(__file__).resolve().parent
sys.path.insert(0, str(CORE))
import pipeline_config as pc

STAGE1 = CORE / "01_preprocessing" / "run_stage1.py"
STAGE2 = CORE / "01_preprocessing" / "run_stage2.py"
PY = sys.executable
_LIVE = {"fetched_new", "fetched_changed", "not_modified"}

# ── text-quality gate ────────────────────────────────────────────────────────
# Deciding "keep the PDF's embedded text vs OCR it" hinges on ONE hard question at
# scale: is the text real language, or broken-font mojibake (glyphs that decode to
# Latin/Devanagari garbage like "gts g*s* fffd GFis")? Dictionaries and language-ID
# both fail on this govt corpus — real docs are full of names/codes/abbreviations,
# and mojibake is script-valid. What works, cheaply, is a character-TRIGRAM language
# model: real words/names follow a language's letter phonotactics; mojibake does not.
# We build a tiny LM once from wordfreq's top words (µs per doc, ~0 RAM, no model
# file, no daemon — the right-sized tool vs a neural LLM on a busy 16GB box), and
# gate the Latin branch on an ENSEMBLE (trigram AND vowel-less AND junk) so a family
# that fools one signal is still caught by another. Errors are asymmetric: keeping
# mojibake poisons the RAG (bad), OCR'ing a clean doc just costs time — so we default
# to OCR whenever confidence is short.
MIN_CHARS_PER_PAGE = 40      # below this the pages are images -> OCR
DEVANAGARI_RATIO = 0.15      # confident Hindi (script share)
MAX_JUNK_RATIO = 0.04        # symbol soup typical of broken font encodings
MAX_VOWELLESS_RATIO = 0.15   # real English ~3% vowel-less words; mojibake ~20-40%
TRIGRAM_EN_MIN = -2.75       # Latin mean-trigram-logprob below this is not real English
                             #   (measured: real English >= -2.63, mojibake <= -2.85)
TRIGRAM_HI_FLOOR = -5.0      # egregious-garbage floor for Devanagari; real Hindi >= -4.3,
                             #   so this only catches Devanagari garbled far below any real Hindi
NGRAM_TOP_WORDS = 40000      # vocabulary size used to train each model
FUNCTION_WORD_RATIO = 0.06   # (fallback heuristic only, when wordfreq is unavailable)

_JUNK_CHARS = set("~{}<>|\\^`_=[]/*")

# Devanagari dependent vowel signs + virama/nukta/anusvara etc. These are COMBINING
# marks: they attach to a preceding consonant and cannot legally begin a word. A text
# layer that dropped its conjunct glyphs leaves them stranded at token starts.
_DEV_MATRAS = set("ािीुूृॄॅॆेैॉॊोौ्ंःँ़ॢॣ")
MAX_ORPHAN_MATRA_RATIO = 0.02   # real Devanagari is ~0; 2% tolerates stray OCR/format noise


def _orphan_matra_ratio(text: str) -> float:
    """Share of whitespace-separated tokens that BEGIN with a combining mark.

    Structurally impossible in real Devanagari, and the signature of an extractor
    that silently dropped unmapped conjunct glyphs (see text_quality). Returns 0.0
    when there is no Devanagari to judge, so non-Hindi text is unaffected.
    """
    tokens = [t for t in text.split() if t]
    dev_tokens = [t for t in tokens if any("ऀ" <= c <= "ॿ" for c in t)]
    if len(dev_tokens) < 10:          # too little Devanagari to draw a conclusion
        return 0.0
    orphans = sum(1 for t in dev_tokens if t[0] in _DEV_MATRAS)
    return orphans / len(dev_tokens)
_VOWELS = set("aeiou")
_FUNCTION_WORDS = {"the", "of", "and", "for", "to", "in", "is", "by", "on",
                   "with", "as", "shall", "be", "are", "that", "from"}

# Lazily-built, cached char-trigram log-prob models: {"en": {...}, "hi": {...}} or
# {"en": None, "hi": None} if wordfreq isn't installed (then we fall back to heuristics).
_ngram_cache: dict = {}


def _build_ngram(lang: str, lo: str, hi: str) -> dict:
    """Char-trigram log-prob model over the language's top words (add-1 smoothed)."""
    import math
    from collections import defaultdict
    from wordfreq import top_n_list
    tri: dict = defaultdict(lambda: defaultdict(int))
    charset: set = set()
    for w in top_n_list(lang, NGRAM_TOP_WORDS):
        w = w.lower()
        if len(w) < 2 or not all(lo <= c <= hi for c in w):
            continue
        charset |= set(w)
        s = "  " + w + " "
        for i in range(len(s) - 2):
            tri[s[i:i + 2]][s[i + 2]] += 1
    v = len(charset) + 2
    lp = {}
    for k, d in tri.items():
        tot = sum(d.values())
        lp[k] = {c: math.log((n + 1) / (tot + v)) for c, n in d.items()}
        lp[k]["_d"] = math.log(1 / (tot + v))
    return {"lp": lp, "floor": math.log(1.0 / NGRAM_TOP_WORDS)}


def _ngram_models():
    if not _ngram_cache:
        try:
            _ngram_cache["en"] = _build_ngram("en", "a", "z")
            _ngram_cache["hi"] = _build_ngram("hi", "ऀ", "ॿ")
        except Exception:   # wordfreq missing/unloadable -> heuristic fallback
            _ngram_cache["en"] = _ngram_cache["hi"] = None
    return _ngram_cache["en"], _ngram_cache["hi"]


def _ngram_score(model, words) -> float | None:
    """Mean char-trigram log-prob of `words` under `model` (higher = more real)."""
    if model is None or len(words) < 5:
        return None
    lp, floor = model["lp"], model["floor"]
    s = n = 0
    for w in words:
        t = "  " + w + " "
        for i in range(len(t) - 2):
            k = t[i:i + 2]
            s += lp[k].get(t[i + 2], lp[k]["_d"]) if k in lp else floor
            n += 1
    return s / n if n else None


def text_quality(text: str, page_count: int) -> tuple[bool, str]:
    """(keep_embedded_text, reason). Conservative: anything unclear goes to OCR."""
    text = unicodedata.normalize("NFC", text or "")
    letters = sum(1 for c in text if c.isalpha())
    if page_count > 0 and len(text.strip()) < MIN_CHARS_PER_PAGE * page_count:
        return False, "image-only (too little text for its page count)"
    if letters < 50:
        return False, "no meaningful text"

    # C0 control characters mean the PDF's font maps glyphs to codepoints that are
    # not text at all. Measured on the live finance corpus: circulars extracted as
    # "छ\x02ीसगढ़ शासन" / "िव\x02 िवभाग" — every conjunct (त्त, वि) replaced by 0x02,
    # 0x0F, 0x19. The script-ratio test scored that 67% Devanagari and KEPT it, so
    # the mangled text went into the index, where it is actively harmful: a user
    # asking the correctly-spelled "अनंतिम पदक्रम सूची" does not match the stored
    # "अनंितम पदम सूची", so the document is never retrieved and the answer becomes
    # "not mentioned in the documents" for a fact that IS in the corpus.
    # This runs before the script checks because it is decisive: no legitimate text
    # layer contains these bytes, and OCR recovers the page correctly.
    ctrl = sum(1 for c in text if (ord(c) < 0x20 and c not in "\t\n\r") or ord(c) == 0x7F)
    if ctrl:
        return False, (f"broken font encoding ({ctrl} control char(s) — conjuncts map to "
                       f"non-text codepoints)")

    # The same broken-cmap PDF read by a different extractor loses the conjuncts
    # SILENTLY instead of noisily: poppler's pdftotext drops the unmapped glyphs
    # altogether, turning "छत्तीसगढ़" into "छ ीसगढ़" — zero control characters, so the
    # check above would pass it straight into the index. What it leaves behind is
    # structurally impossible Devanagari: a dependent vowel sign (matra) with no
    # consonant to attach to. Matras are combining marks; one that opens a token is
    # always damage, never valid text.
    orphan = _orphan_matra_ratio(text)
    if orphan > MAX_ORPHAN_MATRA_RATIO:
        return False, (f"dropped conjuncts ({orphan:.0%} of tokens start with a matra — "
                       f"text layer lost characters)")

    # Count Devanagari LETTERS only. The block also holds matras/combining marks,
    # which isalpha() excludes from `letters` — counting them would push the ratio
    # above 100% and make the gate far more permissive than intended.
    dev = sum(1 for c in text if c.isalpha() and "ऀ" <= c <= "ॿ") / max(letters, 1)
    en_model, hi_model = _ngram_models()

    # ── Devanagari-dominant -> Hindi ─────────────────────────────────────────
    # Devanagari is the corpus's real content and the script share keeps it; the
    # trigram acts only as a low FLOOR that rejects Devanagari garbled far below any
    # real Hindi, so real Hindi is never sent to OCR by mistake.
    if dev > DEVANAGARI_RATIO:
        hs = _ngram_score(hi_model, re.findall(r"[ऀ-ॿ]{2,}", text))
        if hs is not None and hs < TRIGRAM_HI_FLOOR:
            return False, f"garbled Devanagari (trigram {hs:.2f} < {TRIGRAM_HI_FLOOR})"
        return True, f"clean Devanagari ({dev:.0%})"

    # ── Latin-dominant -> where broken-font mojibake lives ───────────────────
    low = text.lower()
    words = [w for w in re.findall(r"[a-z']+", low) if len(w) > 1]
    junk = sum(1 for c in text if c in _JUNK_CHARS) / max(len(text), 1)
    vowelless = sum(1 for w in words if not (set(w) & _VOWELS)) / max(len(words), 1)
    es = _ngram_score(en_model, re.findall(r"[a-z]{2,}", low))

    if es is None:
        # wordfreq unavailable -> fall back to the function-word + vowel-less heuristic.
        fnr = sum(1 for w in words if w in _FUNCTION_WORDS) / max(len(words), 1)
        if fnr > FUNCTION_WORD_RATIO and junk < MAX_JUNK_RATIO and vowelless < MAX_VOWELLESS_RATIO:
            return True, f"clean English (heuristic: function words {fnr:.0%}, vowel-less {vowelless:.0%})"
        return False, f"mojibake/unreadable (heuristic: function words {fnr:.0%}, vowel-less {vowelless:.0%})"

    # Ensemble: keep ONLY if the trigram LM reads it as English AND it isn't full of
    # vowel-less clusters AND not symbol soup. Any one failing routes it to OCR.
    if es >= TRIGRAM_EN_MIN and vowelless < MAX_VOWELLESS_RATIO and junk < MAX_JUNK_RATIO:
        return True, f"clean English (trigram {es:.2f}, vowel-less {vowelless:.0%}, junk {junk:.0%})"
    return False, f"mojibake/unreadable (trigram {es:.2f}, vowel-less {vowelless:.0%}, junk {junk:.0%})"


def embedded_text(pdf: Path) -> tuple[str, int]:
    """(text, page_count) from the PDF's own text layer. Never trusted blindly."""
    import fitz  # PyMuPDF — a core dependency (stage1 uses it to rasterise)

    try:
        fitz.TOOLS.mupdf_display_errors(False)   # damaged govt PDFs spew zlib warnings
    except Exception:
        pass
    doc = fitz.open(str(pdf))
    try:
        return "\n".join(p.get_text("text") for p in doc), doc.page_count
    finally:
        doc.close()


def pdf_page_count(pdf: Path) -> int:
    """Page count only — cheap, so the page-budget gate can decide before OCR."""
    import fitz

    try:
        fitz.TOOLS.mupdf_display_errors(False)
    except Exception:
        pass
    doc = fitz.open(str(pdf))
    try:
        return doc.page_count
    finally:
        doc.close()


# ── deferred-work registry: PDFs skipped for exceeding the page budget ────────
# Written per pipeline (under its data root) so a later, dedicated pass can pick
# exactly these up — see --max-pages 0 in the module docstring. Append-only and
# deduplicated by urlhash so re-runs (including --resume) never double-record.
def skip_registry_path() -> Path:
    return pc.DATA_ROOT / "skipped_large_pdfs.jsonl"


def already_skipped(path: Path) -> set:
    seen = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                seen.add(json.loads(line)["urlhash"])
            except Exception:
                continue
    return seen


def record_skip(uhash: str, raw: Path, url: str, pages, reason: str) -> Path:
    path = skip_registry_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "pipeline": pc.PIPELINE,
        "urlhash": uhash,
        "url": url,
        "raw_path": str(raw),
        "pages": pages,
        "size_bytes": raw.stat().st_size if raw.exists() else None,
        "reason": reason,
        "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return path


def _write_extraction_marker(uhash: str, method: str, reason: str) -> None:
    """Note how stage2/<uhash>/structured.md was produced.

    OCR leaves a *_confidence.json sidecar; keeping the embedded text layer left
    nothing behind, so downstream could not tell the two apart and assumed OCR.
    This is the missing half of that signal.
    """
    d = pc.paths.stage2 / uhash
    d.mkdir(parents=True, exist_ok=True)
    (d / "extraction.json").write_text(json.dumps({
        "method": method,        # "text_layer" — the OCR path writes *_confidence.json
        "reason": reason,
        "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }, ensure_ascii=False, indent=2), encoding="utf-8")


def _run(args):
    env = dict(os.environ)
    env["PIPELINE"] = pc.PIPELINE
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    # Stage 2's Sarvam fallback reads the unprefixed SARVAM_API_KEY; keys are set
    # per pipeline (SARVAM_API_KEY_<PIPELINE>). Bridge the active pipeline's key
    # through so the fallback actually runs for scanned/low-confidence pages.
    if pc.SARVAM_API_KEY:
        env["SARVAM_API_KEY"] = pc.SARVAM_API_KEY
    return subprocess.run([PY] + [str(a) for a in args], env=env,
                          capture_output=True, text=True, encoding="utf-8", errors="replace")


def scraped_pdfs(corpus: Path):
    """(urlhash, raw_pdf_path) for every PDF in the corpus, newest record per URL.

    Selects on the raw file's extension rather than a doc_kind label: `classify`
    detects PDFs by magic bytes too, so this still works when a server lies about
    its content-type. (The old `doc_kind == "pdf_scanned"` label no longer exists —
    the scraper doesn't read PDFs, so it cannot classify their interiors.)
    """
    manifest = corpus / "manifest.jsonl"
    latest = {}
    if manifest.exists():
        for line in manifest.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if rec.get("outcome") in _LIVE and str(rec.get("raw_path") or "").lower().endswith(".pdf"):
                latest[rec.get("normalized_url") or rec.get("url")] = rec
    out = []
    for rec in latest.values():
        raw = corpus / str(rec["raw_path"]).replace("\\", "/")
        if raw.exists() and raw.suffix.lower() == ".pdf":
            url = rec.get("normalized_url") or rec.get("url") or ""
            out.append((raw.stem, raw, url))   # stem == urlhash
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--corpus", required=True, help="scraper corpus dir (holds manifest.jsonl + raw/)")
    ap.add_argument("--force", action="store_true", help="redo docs that already have stage2 output")
    ap.add_argument("--force-ocr-all", action="store_true",
                    help="ignore the quality gate and OCR every PDF, even clean digital ones")
    ap.add_argument("--gate-report", action="store_true",
                    help="DRY RUN: print the keep-vs-OCR decision for every PDF and exit. "
                         "Use this to see the OCR bill before committing to it.")
    ap.add_argument("--limit", type=int, default=None,
                    help="stop after PRODUCING N docs this run (newly OCR'd or kept). Unlike a "
                         "slice, this skips already-done/failed docs first, so --limit 1 called "
                         "repeatedly walks forward one NEW doc at a time — the primitive the "
                         "round-robin driver loops on.")
    ap.add_argument("--only", default=None,
                    help="process ONLY these urlhashes (comma-separated, or @file with one "
                         "per line). For repairing specific documents — e.g. re-OCR'ing the "
                         "ones a bad text layer poisoned — without walking the whole corpus. "
                         "Combine with --force to redo docs that already have output.")
    ap.add_argument("--head", type=int, default=None,
                    help="only consider the first N PDFs in crawl order. The crawler is BFS from "
                         "each site's landing/notice pages (which list newest first), so the first "
                         "N are a proxy for 'the N most recent documents' — use it to OCR just a "
                         "recent slice per site for a demo instead of the whole corpus.")
    ap.add_argument("--max-pages", type=int, default=25,
                    help="skip (and record) any PDF with more than N pages — OCR is ~45s/page, "
                         "so huge scans are deferred to a dedicated pass. 0 = no limit "
                         "(use this later to drain the skipped list; already-OCR'd docs are "
                         "left untouched). Default 25.")
    args = ap.parse_args()

    corpus = Path(os.path.expanduser(args.corpus)).resolve()
    if not pc.SARVAM_API_KEY:
        print(f"[{pc.PIPELINE}] WARNING: no Sarvam key (set SARVAM_API_KEY_{pc.PIPELINE.upper()} "
              "in .env). EasyOCR still runs, but low-confidence pages get NO Sarvam fallback "
              "— weaker text on scanned pages. Ctrl-C now if that's not intended.")
    items = scraped_pdfs(corpus)
    print(f"[{pc.PIPELINE}] scraped PDFs: {len(items)}   (data root {pc.DATA_ROOT})")

    if args.only:
        raw = args.only
        if raw.startswith("@"):
            raw = Path(raw[1:]).read_text(encoding="utf-8").replace("\n", ",")
        wanted = {h.strip() for h in raw.split(",") if h.strip()}
        items = [it for it in items if it[0] in wanted]
        missing = wanted - {it[0] for it in items}
        print(f"[{pc.PIPELINE}] --only: {len(items)} of {len(wanted)} requested doc(s) found"
              + (f"; not in this corpus: {', '.join(sorted(missing))}" if missing else ""))

    if not items:
        print("nothing to do.")
        return

    # ── dry run: decide for every PDF, spend nothing ─────────────────────────
    if args.gate_report:
        keep = ocr = unreadable = toobig = 0
        for uhash, raw, url in items:
            try:
                text, pages = embedded_text(raw)
            except Exception as e:
                unreadable += 1
                print(f"  OCR   {uhash}  (cannot open: {type(e).__name__})")
                continue
            if args.max_pages and args.max_pages > 0 and pages > args.max_pages:
                toobig += 1
                print(f"  SKIP  {uhash}  {pages:>4}p  > {args.max_pages}-page budget (deferred)")
                continue
            good, why = text_quality(text, pages)
            if good:
                keep += 1
                print(f"  keep  {uhash}  {pages:>4}p  {why}")
            else:
                ocr += 1
                print(f"  OCR   {uhash}  {pages:>4}p  {why}")
        print(f"\ngate: keep embedded text {keep}, OCR {ocr + unreadable} "
              f"({unreadable} unopenable), skip-too-large {toobig} (> {args.max_pages}p). "
              f"Re-run without --gate-report to do the work.")
        return

    pc.paths.input_pdfs.mkdir(parents=True, exist_ok=True)
    # --head N: the "N most recent" window (crawl order, newest-first — see the flag
    # help). --limit N: stop once N docs have actually been PRODUCED this run. The two
    # compose: `--head 50 --limit 1` walks the 50 most-recent PDFs and produces the
    # next single undone one — exactly one round-robin step.
    candidates = items[: args.head] if args.head else items
    n = len(candidates)
    kept = ocrd = skip = fail = toobig = 0
    produced = 0                       # docs that gained a structured.md this run
    skipped_hashes = already_skipped(skip_registry_path())

    def _mark_failed(uh: str, why: str):
        """Persist an OCR_FAILED marker so a re-run (round-robin re-attempts every
        undone candidate) doesn't burn OCR time on the same broken PDF each pass."""
        m = pc.paths.stage2 / uh / "OCR_FAILED"
        m.parent.mkdir(parents=True, exist_ok=True)
        m.write_text(f"{datetime.now(timezone.utc).isoformat(timespec='seconds')}  {why}\n",
                     encoding="utf-8")

    def _clear_failed(uh: str):
        m = pc.paths.stage2 / uh / "OCR_FAILED"
        if m.exists():
            m.unlink()

    for i, (uhash, raw, url) in enumerate(candidates, 1):
        if args.limit is not None and produced >= args.limit:
            break
        dest_md = pc.paths.stage2 / uhash / "structured.md"
        if dest_md.exists() and not args.force:
            skip += 1
            continue
        # Known-bad from a previous pass — skip fast so the driver moves to the next
        # candidate instead of re-failing this one every cycle. --force retries it.
        if (pc.paths.stage2 / uhash / "OCR_FAILED").exists() and not args.force:
            skip += 1
            continue

        # 0. Over the page budget? Defer it — OCR at ~45s/page is what blows up a run.
        #    Recorded once (deduped) so a later `--max-pages 0` pass can drain these.
        if args.max_pages and args.max_pages > 0:
            try:
                npages = pdf_page_count(raw)
            except Exception:
                npages = None
            if npages is not None and npages > args.max_pages:
                if uhash not in skipped_hashes:
                    reg = record_skip(uhash, raw, url, npages, f">{args.max_pages} pages")
                    skipped_hashes.add(uhash)
                    print(f"  [{i}/{n}] {uhash}: SKIP — {npages}p > {args.max_pages} "
                          f"(recorded in {reg})")
                else:
                    print(f"  [{i}/{n}] {uhash}: SKIP — {npages}p > {args.max_pages} "
                          f"(already recorded)")
                toobig += 1
                continue

        # 1. Can we keep the PDF's own text layer?
        if not args.force_ocr_all:
            try:
                text, pages = embedded_text(raw)
                good, why = text_quality(text, pages)
            except Exception as e:
                good, why, text = False, f"cannot open ({type(e).__name__})", ""
            if good:
                dest_md.parent.mkdir(parents=True, exist_ok=True)
                dest_md.write_text(unicodedata.normalize("NFC", text).strip() + "\n",
                                   encoding="utf-8")
                # Record that this text came from the PDF, NOT from OCR. Without a
                # marker the two paths are indistinguishable downstream — stage2
                # writes structured.md either way — so scrape_import labelled every
                # scraped PDF "ocr", and the Documents panel showed an OCR badge
                # (and matched the Sarvam filter) on text no OCR engine ever touched.
                _write_extraction_marker(uhash, "text_layer", why)
                _clear_failed(uhash)
                kept += 1
                produced += 1
                print(f"  [{i}/{n}] {uhash}: kept text layer — {why}")
                continue
            print(f"  [{i}/{n}] {uhash}: -> OCR — {why}")

        # 2. Otherwise run the real OCR pipeline. Staging the raw PDF as
        #    <urlhash>.pdf makes stage1/stage2 write to stage2/<urlhash>/, which is
        #    the key scrape_import joins on.
        staged = pc.paths.input_pdfs / f"{uhash}.pdf"
        staged.write_bytes(raw.read_bytes())
        r1 = _run([STAGE1, staged])
        if r1.returncode != 0:
            fail += 1
            _mark_failed(uhash, f"stage1: {r1.stderr[-160:].strip()}")
            print(f"  [{i}/{n}] {uhash}: stage1 FAILED — {r1.stderr[-200:].strip()}")
            continue
        r2 = _run([STAGE2, pc.paths.stage1 / uhash, "--archive-source", "--force"])
        if r2.returncode != 0 or not dest_md.exists():
            fail += 1
            _mark_failed(uhash, f"stage2: {r2.stderr[-160:].strip()}")
            print(f"  [{i}/{n}] {uhash}: stage2 FAILED — {r2.stderr[-200:].strip()}")
            continue
        _clear_failed(uhash)
        ocrd += 1
        produced += 1
        print(f"  [{i}/{n}] {uhash}: OCR ok")

    print(f"\ndone: {ocrd} OCR'd, {kept} kept clean text layer, {skip} already present/failed, "
          f"{toobig} skipped (> {args.max_pages}p), {fail} failed.")
    if toobig:
        print(f"{toobig} large PDF(s) deferred -> {skip_registry_path()}\n"
              f"  drain them later with:  PIPELINE={pc.PIPELINE} uv run python core/ocr_scanned.py "
              f"--corpus {corpus} --max-pages 0")
    # Machine-readable control line for the round-robin driver (deploy/ingest-all.sh):
    # produced=0 means this pipeline had nothing left to do within its --head window,
    # so the driver stops cycling it. `window` is the candidate count considered.
    print(f"RR produced={produced} window={n} pipeline={pc.PIPELINE}")
    print("Next: scrape_import to register everything with provenance.")


if __name__ == "__main__":
    main()
