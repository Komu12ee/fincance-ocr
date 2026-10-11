"""Per-pipeline document registry (data/<pipeline>/documents.json).

Tracks every uploaded document through its lifecycle: uploaded -> processed (OCR
-> .md) -> verified (indexed into the RAG). Plain JSON (no native deps); all
writes are serialized with a lock (the Flask app touches this from request
handlers and background threads).
"""

import json
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # core/ on path
import pipeline_config as pc

_LOCK = threading.Lock()


def _registry_path() -> Path:
    return pc.DATA_ROOT / "documents.json"


def _load() -> list:
    p = _registry_path()
    if not p.exists():
        return []
    try:
        return json.loads(p.read_text(encoding="utf-8")).get("documents", [])
    except Exception:
        return []


def _save(docs: list):
    p = _registry_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"documents": docs}, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(p)


def all() -> list:
    with _LOCK:
        return _load()


def get(doc_id: str):
    with _LOCK:
        for d in _load():
            if d.get("id") == doc_id:
                return d
    return None


def add(record: dict) -> dict:
    """Register a document. Enforces the INGESTION.md provenance invariant:

    every doc must be traceable to something a human can open — a source URL, an
    uploaded PDF, or a stored raw snapshot. A doc with none of those produces
    answers that cannot be traced back to a source, which is the one thing this
    pipeline must never do.
    """
    if not (record.get("source_url") or record.get("pdf_path") or record.get("raw_path")):
        raise ValueError(
            f"refusing to register {record.get('doc_dir') or record.get('id')!r}: no "
            "source_url, pdf_path or raw_path (INGESTION.md provenance invariant)")
    with _LOCK:
        docs = _load()
        docs.append(record)
        _save(docs)
    return record


def update(doc_id: str, **fields) -> dict | None:
    with _LOCK:
        docs = _load()
        updated = None
        for d in docs:
            if d.get("id") == doc_id:
                d.update(fields)
                updated = d
                break
        if updated is not None:
            _save(docs)
        return updated


def get_by_doc_dir(doc_dir: str):
    with _LOCK:
        for d in _load():
            if d.get("doc_dir") == doc_dir:
                return d
    return None


def update_by_doc_dir(doc_dir: str, **fields) -> dict | None:
    """Update the record whose doc_dir matches (the scraper urlhash == OCR stage2 dir).

    Lets core/ocr_scanned.py flip a scraped PDF from 'processing' to 'processed' the
    instant its OCR text lands, so the Documents panel updates live instead of only
    after a full re-import. No-op (returns None) if the doc isn't registered yet."""
    with _LOCK:
        docs = _load()
        updated = None
        for d in docs:
            if d.get("doc_dir") == doc_dir:
                d.update(fields)
                updated = d
                break
        if updated is not None:
            _save(docs)
        return updated


def remove(doc_id: str) -> bool:
    with _LOCK:
        docs = _load()
        new = [d for d in docs if d.get("id") != doc_id]
        if len(new) == len(docs):
            return False
        _save(new)
        return True


def unverified() -> list:
    """Documents that finished OCR but are not yet verified."""
    return [d for d in all() if d.get("status") == "processed" and not d.get("verified")]
