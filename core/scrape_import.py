"""Batch-import scraped website data into the verification workspace.

The scraper (Scraper/) fetches and stores; it does not process. Its corpus:

    corpus/
      manifest.jsonl                 one JSON record per fetch — THE provenance
                                     authority (url, doc_kind, raw_path, txt_path…)
      raw/<domain>/<urlhash>.<ext>   original fetched bytes (html/pdf/json)
      txt/<domain>/<urlhash>.txt     HTML pages only, tag-stripped to plain text

So a document arrives here by one of exactly two routes:

    PDF   raw bytes only. Registered IMMEDIATELY — even before OCR — so scraped PDFs
          appear in the Documents panel as `status:'processing'` (source PDF openable)
          the moment they're imported, then flip to `processed` when OCR produces
          stage2/<urlhash>/structured.md. ocr_scanned flips them live; a re-import
          reconciles any it missed. Only VERIFIED docs are ever embedded, so a
          'processing' doc never reaches the RAG — registering early is safe.

    TXT   the scraper's tag-stripped page text, copied to
          stage2/<urlhash>/structured.md so it enters at the post-OCR boundary. Its
          text is ready at import time, so it registers straight to `processed`.

Either way the record carries its `source_url` and a viewable source, satisfying
the INGESTION.md invariant. Idempotent + reconciling: new docs are added, and an
already-registered scraped PDF that has since been OCR'd is upgraded to `processed`;
verified docs and reviewer edits are never clobbered.
"""

import hashlib
import json
import shutil
import sys
import uuid
from datetime import datetime
from pathlib import Path

CORE = Path(__file__).resolve().parent
sys.path.insert(0, str(CORE))
import pipeline_config as pc
import doc_registry as registry

# Manifest outcomes that represent a live, successful fetch worth importing.
_LIVE_OUTCOMES = {"fetched_new", "fetched_changed", "not_modified"}


def resolve_corpus(path_str: str) -> Path | None:
    """Accept the corpus dir itself, or a scraper dir that contains `corpus/`.

    A manifest is mandatory: it is the only place provenance lives now that the
    stored text carries no front-matter."""
    if not path_str:
        return None
    p = Path(path_str.strip().strip('"')).expanduser()
    if not p.exists():
        return None
    for cand in (p, p / "corpus"):
        if (cand / "manifest.jsonl").exists():
            return cand.resolve()
    return None


def _iter_from_manifest(corpus: Path):
    """Latest live fetch record per normalized URL.

    Keys off raw_path first, txt_path second — never on extracted text existing,
    because a scraped PDF legitimately has no text until OCR has run."""
    mpath = corpus / "manifest.jsonl"
    latest: dict[str, dict] = {}
    with open(mpath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("outcome") in _LIVE_OUTCOMES and (rec.get("raw_path") or rec.get("txt_path")):
                latest[rec.get("normalized_url") or rec.get("url")] = rec

    for rec in latest.values():
        raw_rel = rec.get("raw_path") or ""
        txt_rel = rec.get("txt_path") or ""
        # urlhash/domain come from whichever file exists — both buckets use the
        # same <domain>/<urlhash> naming, so either is authoritative.
        ref = corpus / Path(str(raw_rel or txt_rel).replace("\\", "/"))
        raw_path = corpus / Path(str(raw_rel).replace("\\", "/")) if raw_rel else None
        txt_path = corpus / Path(str(txt_rel).replace("\\", "/")) if txt_rel else None
        yield {
            "url": rec.get("url") or rec.get("normalized_url"),
            "title": rec.get("title") or "",
            "doc_kind": rec.get("doc_kind") or "",
            "department": rec.get("department_id") or "",
            "lang": rec.get("lang") or "",
            "sha256": rec.get("sha256") or "",
            "domain": ref.parent.name,
            "urlhash": ref.stem,
            "raw_path": raw_path,
            "raw_ext": (raw_path.suffix.lstrip(".").lower() if raw_path else None),
            "txt_path": txt_path,
        }


_CONTROL_CHARS = {c for c in range(0x20) if c not in (0x09, 0x0a, 0x0d)} | {0x7f}


def _strip_control_chars(text: str) -> tuple[str, int]:
    """Remove C0 control characters, returning (clean_text, how_many_removed).

    These are never text. In scraped HTML they come from the same broken font
    encodings that corrupt PDF text layers, and left in place they poison the vector
    store exactly as they did there — a query's correctly-spelled Hindi cannot match
    a stored word with a control byte wedged inside it.
    """
    removed = sum(1 for c in text if ord(c) in _CONTROL_CHARS)
    if not removed:
        return text, 0
    return "".join(c for c in text if ord(c) not in _CONTROL_CHARS), removed


def _mark_web_cleaned(doc_dir: str, removed: int) -> None:
    """Record that a scraped page needed cleaning, so the panel can flag it.

    Stripping makes the text safe to index but does NOT recover the lost letters —
    the page is still missing characters. Recording it keeps that visible instead of
    quietly shipping degraded text as if it were fine.
    """
    d = pc.paths.stage2 / doc_dir
    d.mkdir(parents=True, exist_ok=True)
    (d / "extraction.json").write_text(json.dumps({
        "method": "web_text_cleaned",
        "reason": f"stripped {removed} control character(s) from the scraped page text",
        "at": datetime.now().isoformat(timespec="seconds"),
    }, ensure_ascii=False, indent=2), encoding="utf-8")


def _text_source_for(doc_dir: str) -> str:
    """How stage2 actually produced this doc's text: "ocr" or "extract".

    ocr_scanned takes one of two paths and both end at stage2/<dir>/structured.md,
    so the only way to tell them apart is the sidecar each leaves:
        *_confidence.json  -> the OCR pipeline ran (per-page scores + engine)
        extraction.json    -> the PDF's own text layer was kept, no OCR
    This used to be hardcoded to "ocr" for every scraped PDF, which made the
    Documents panel show an OCR badge — and match the Sarvam filter — on text no
    OCR engine had touched. Defaults to "ocr" when neither marker exists, matching
    the historical label for documents processed before markers were written.
    """
    d = pc.paths.stage2 / (doc_dir or "")
    if not d.is_dir():
        return "ocr"
    if any(d.glob("*_confidence.json")):
        return "ocr"
    marker = d / "extraction.json"
    if marker.is_file():
        try:
            if json.loads(marker.read_text(encoding="utf-8")).get("method") == "text_layer":
                return "extract"
        except Exception:
            pass
    return "ocr"


def import_corpus(corpus_path: str, imported_by: str = "scrape-import") -> dict:
    """Import every scraped document from a corpus into this pipeline's registry.

    Returns {ok, corpus, imported, skipped, pending_ocr, unsupported, errors, details}.
    `pending_ocr` > 0 means: run core/ocr_scanned.py, then re-run this."""
    corpus = resolve_corpus(corpus_path)
    if corpus is None:
        return {"ok": False, "error": f"No scraper corpus found at: {corpus_path!r} "
                "(expected a manifest.jsonl — a corpus without one has no provenance)."}

    items = list(_iter_from_manifest(corpus))
    if not items:
        return {"ok": True, "corpus": str(corpus), "imported": 0, "skipped": 0,
                "pending_ocr": 0, "unsupported": 0, "errors": 0, "details": [],
                "message": "No scraped documents found in corpus."}

    known = registry.all()
    by_dir = {d.get("doc_dir"): d for d in known if d.get("doc_dir")}
    existing_urls = {d.get("source_url") for d in known if d.get("source_url")}
    pc.paths.stage2.mkdir(parents=True, exist_ok=True)
    pc.paths.scrape_raw.mkdir(parents=True, exist_ok=True)

    imported = processing = upgraded = skipped = errors = unsupported = 0
    details = []
    now = datetime.now().isoformat()

    def _copy_raw(dest_name: str) -> str | None:
        """Copy the scraped raw file into scrape_raw/ so the reviewer can open it,
        and return its path. Prefers an already-archived used_files/ copy for PDFs."""
        if it["raw_ext"] == "pdf":
            used = pc.paths.used_files / f"{doc_dir}.pdf"
            if used.exists():
                return str(used)
        if it["raw_path"] and it["raw_path"].exists():
            dest_raw = pc.paths.scrape_raw / dest_name
            shutil.copyfile(it["raw_path"], dest_raw)
            return str(dest_raw)
        return None

    for it in items:
        url = it["url"]
        # doc_dir = the scraper's urlhash: stable, traceable to the manifest, and the
        # SAME dir OCR writes to — exactly one stage2 dir per source.
        doc_dir = it["urlhash"]
        dest_md = pc.paths.stage2 / doc_dir / "structured.md"
        existing = by_dir.get(doc_dir)
        try:
            # ── Already registered ─────────────────────────────────────────────
            # The only useful reconcile is upgrading a still-'processing' scraped PDF
            # to 'processed' once ocr_scanned has produced its text. (ocr_scanned also
            # does this live; this catches any it missed.) Everything else is left as
            # is — never clobber a verified doc or a reviewer's edits.
            if existing is not None:
                if (it["raw_ext"] == "pdf" and dest_md.exists()
                        and existing.get("status") == "processing"):
                    md_text = dest_md.read_text(encoding="utf-8")
                    used = pc.paths.used_files / f"{doc_dir}.pdf"
                    pdfp = str(used) if used.exists() else existing.get("pdf_path")
                    registry.update(existing["id"],
                        status="processed", md_path=str(dest_md),
                        text_source=_text_source_for(doc_dir),
                        md_hash=hashlib.sha256(md_text.encode("utf-8")).hexdigest(),
                        pdf_path=pdfp, raw_ext="pdf",
                        raw_path=pdfp or existing.get("raw_path"), error=None)
                    upgraded += 1
                    details.append({"url": url, "doc_dir": doc_dir, "upgraded": True})
                else:
                    skipped += 1
                continue
            if url in existing_urls:
                skipped += 1
                continue

            base = {
                "id": uuid.uuid4().hex,
                "original_name": it["title"] or url,
                "doc_dir": doc_dir,
                "verified": False, "verified_by": None, "verified_at": None,
                "uploaded_by": imported_by, "uploaded_at": now,
                "edited_by": None, "edited_at": None,
                "indexed": False, "chunk_count": 0, "error": None,
                "source": "scrape", "source_url": url,
                "department": it["department"], "lang": it["lang"],
            }

            # ── New scraped PDF: register NOW, regardless of OCR state ──────────
            # Text may not exist yet — that is exactly the point. It shows in the
            # panel as 'processing' with its source PDF openable, and flips to
            # 'processed' when OCR lands. Registering early is safe: doc_dir is the
            # join key, so ocr_scanned/re-import find and upgrade this same record.
            if it["raw_ext"] == "pdf":
                pdf_path = _copy_raw(f"{doc_dir}.pdf")
                if dest_md.exists():
                    md_text = dest_md.read_text(encoding="utf-8")
                    status, md_path = "processed", str(dest_md)
                    md_hash = hashlib.sha256(md_text.encode("utf-8")).hexdigest()
                    imported += 1
                else:
                    status, md_path, md_hash = "processing", None, None
                    processing += 1
                registry.add({**base,
                    "pdf_path": pdf_path, "md_path": md_path, "status": status,
                    "md_hash": md_hash, "text_source": _text_source_for(doc_dir),
                    "source_type": "pdf", "doc_kind": "pdf",
                    "raw_ext": "pdf", "raw_path": pdf_path})
                details.append({"url": url, "doc_dir": doc_dir, "status": status})

            # ── New scraped web/text doc: text is ready at import time ──────────
            else:
                if it["txt_path"] is None or not it["txt_path"].exists():
                    unsupported += 1          # .doc/.xls/other — no text, not OCRable
                    continue
                if not dest_md.exists():      # never clobber a reviewer's saved edit
                    dest_md.parent.mkdir(parents=True, exist_ok=True)
                    title = (it["title"] or "").strip()
                    head = f"# {title}\n\n" if title else ""   # heading -> chunk structure
                    body = it["txt_path"].read_text(encoding="utf-8")
                    # Web pages hit the SAME broken-font problem as PDFs (measured on
                    # finance.cg.gov.in: 3 pages carrying 3, 28 and 124 control chars),
                    # but they never pass through the PDF text-quality gate — the
                    # scraper's .txt is copied straight in. Unlike a PDF there is no
                    # image to fall back on, so OCR is not an option: strip the junk
                    # bytes so they cannot reach the index, and record that we did.
                    body, stripped = _strip_control_chars(body)
                    md_text = head + body.strip() + "\n"
                    dest_md.write_text(md_text, encoding="utf-8")
                    if stripped:
                        _mark_web_cleaned(doc_dir, stripped)
                else:
                    md_text = dest_md.read_text(encoding="utf-8")
                raw_ext = it["raw_ext"]
                raw_path = _copy_raw(f"{doc_dir}.{raw_ext}") if raw_ext else None
                registry.add({**base,
                    "pdf_path": None, "md_path": str(dest_md), "status": "processed",
                    "md_hash": hashlib.sha256(md_text.encode("utf-8")).hexdigest(),
                    "text_source": "extract", "source_type": "web",
                    "doc_kind": it["doc_kind"] or "html",
                    "raw_ext": raw_ext, "raw_path": raw_path})
                imported += 1
                details.append({"url": url, "doc_dir": doc_dir, "status": "processed"})

            existing_urls.add(url)
            by_dir[doc_dir] = {"doc_dir": doc_dir, "source_url": url}  # mark seen this run
        except Exception as e:
            errors += 1
            details.append({"url": url, "error": f"{type(e).__name__}: {e}"})

    out = {"ok": True, "corpus": str(corpus),
           "imported": imported, "processing": processing, "upgraded": upgraded,
           "skipped": skipped, "unsupported": unsupported, "errors": errors,
           "details": details}
    if processing:
        out["message"] = (f"{processing} scraped PDF(s) registered as 'processing' — they "
                          f"appear in the panel now and flip to 'processed' as OCR completes.")
    return out
