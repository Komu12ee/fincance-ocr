"""Single-document orchestration for the verification workspace.

Reuses the existing stage scripts (OCR is heavy -> run as subprocesses) and the
query pipeline's already-loaded bge-m3 model + Qdrant client for in-process
embedding / upsert / delete. Only VERIFIED documents get embedded, and each
point is tagged with `doc_id` so a document's vectors can be replaced or removed
cleanly when its .md is edited (the chunker names every md "structured", so the
default `source` payload can't distinguish documents).
"""

import hashlib
import os
import shutil
import subprocess
import sys
import threading
import uuid
from pathlib import Path

CORE = Path(__file__).resolve().parent
sys.path.insert(0, str(CORE))
import pipeline_config as pc
import doc_registry as registry

SCRIPTS = CORE / "04_embeddings_and_kg" / "scripts"
STAGE1 = CORE / "01_preprocessing" / "run_stage1.py"
STAGE2 = CORE / "01_preprocessing" / "run_stage2.py"
CHUNK = CORE / "03_chunking" / "chunk_stage2_output.py"
PY = sys.executable

# Serialize heavy jobs (OCR / embedding) so concurrent uploads don't thrash CPU/models.
_JOB_LOCK = threading.Lock()


def md_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _run(args: list, env_extra: dict | None = None) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["PIPELINE"] = pc.PIPELINE
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    if env_extra:
        env.update({k: str(v) for k, v in env_extra.items()})   # e.g. STAGE2_SARVAM_MODE
    return subprocess.run([PY] + [str(a) for a in args], env=env,
                          capture_output=True, text=True, encoding="utf-8", errors="replace")


# ── Re-processing: produce structured.md a DIFFERENT way on demand ────────────
# The Documents panel lets a reviewer force how a doc's text is produced. Each
# method maps to the environment stage 2 reads (STAGE2_SARVAM_MODE); 'extract'
# skips OCR entirely and keeps the PDF's embedded text layer.
REPROCESS_METHODS = {
    "extract":    None,                              # embedded text layer, no OCR
    "ocr":        {"STAGE2_SARVAM_MODE": "off"},     # Docling/EasyOCR only
    "ocr_sarvam": {"STAGE2_SARVAM_MODE": "force"},   # Sarvam on every page
    "auto":       {"STAGE2_SARVAM_MODE": "auto"},    # Docling + Sarvam fallback (default)
}


def _locate_pdf(rec: dict) -> Path:
    """Find this document's source PDF wherever it currently lives."""
    dd = rec.get("doc_dir") or ""
    cands = []
    if rec.get("pdf_path"):
        cands.append(Path(rec["pdf_path"]))
    cands += [pc.paths.used_files / f"{dd}.pdf",
              pc.paths.input_pdfs / f"{dd}.pdf",
              pc.paths.scrape_raw / f"{dd}.pdf"]
    for p in cands:
        if p and p.exists():
            return p
    raise FileNotFoundError("no source PDF is stored for this document (cannot re-process)")


def _embedded_text(pdf: Path) -> str:
    """The PDF's own text layer, NFC-normalised. Used by the 'extract' method — for
    born-digital PDFs this is faster and cleaner than OCR; for scans it is empty."""
    import unicodedata
    import fitz
    try:
        fitz.TOOLS.mupdf_display_errors(False)
    except Exception:
        pass
    doc = fitz.open(str(pdf))
    try:
        txt = "\n".join(p.get_text("text") for p in doc)
    finally:
        doc.close()
    return unicodedata.normalize("NFC", txt).strip() + "\n"


def reprocess_document(doc_id: str, method: str = "auto"):
    """Re-produce a document's structured.md using the chosen method, then (if the
    doc was verified) re-index so the RAG reflects the new text. Runs in the
    background; sets status='processing' while it works."""
    if method not in REPROCESS_METHODS:
        method = "auto"
    with _JOB_LOCK:
        rec = registry.get(doc_id)
        if not rec:
            return
        dd = rec["doc_dir"]
        md_path = pc.paths.stage2 / dd / "structured.md"
        # md_path stays set so _reconcile_processing (which only promotes first-time
        # OCR, where md_path is null) never races this back to 'processed'.
        registry.update(doc_id, status="processing", error=None)
        try:
            pdf = _locate_pdf(rec)
            if method == "extract":
                md_path.parent.mkdir(parents=True, exist_ok=True)
                md_path.write_text(_embedded_text(pdf), encoding="utf-8")
                text_source = "extract"
            else:
                if method == "ocr_sarvam" and not pc.SARVAM_API_KEY:
                    registry.update(doc_id, status="error",
                                    error="Sarvam OCR requested but no Sarvam key is configured for this pipeline.")
                    return
                staged = pc.paths.input_pdfs / f"{dd}.pdf"
                staged.parent.mkdir(parents=True, exist_ok=True)
                staged.write_bytes(pdf.read_bytes())
                r1 = _run([STAGE1, staged])
                if r1.returncode != 0:
                    registry.update(doc_id, status="error", error=f"stage1 failed:\n{r1.stderr[-800:]}")
                    return
                r2 = _run([STAGE2, pc.paths.stage1 / dd, "--archive-source", "--force"],
                          env_extra=REPROCESS_METHODS[method])
                if r2.returncode != 0 or not md_path.exists():
                    registry.update(doc_id, status="error", error=f"stage2 failed:\n{r2.stderr[-800:]}")
                    return
                text_source = "ocr"

            text = md_path.read_text(encoding="utf-8")
            used = pc.paths.used_files / f"{dd}.pdf"
            registry.update(doc_id, status="processed", md_path=str(md_path),
                            md_hash=md_hash(text), text_source=text_source,
                            pdf_path=str(used) if used.exists() else rec.get("pdf_path"),
                            error=None)
        except Exception as e:
            registry.update(doc_id, status="error", error=f"{type(e).__name__}: {e}")
            return
    # Keep the RAG in sync: a verified doc whose text just changed must be re-embedded.
    fresh = registry.get(doc_id)
    if fresh and fresh.get("verified"):
        index_document(doc_id)


# ── OCR: PDF -> stage1 images -> stage2 structured.md ─────────────────────────
def process_document(doc_id: str):
    """Turn an uploaded PDF into text, following the SAME routing as the ingest
    script (core/ocr_scanned.py): if the PDF has a confidently-clean embedded text
    layer, keep it (fast, and higher fidelity than OCR for born-digital PDFs);
    otherwise OCR it (stage1 rasterise -> stage2 Docling/EasyOCR + Sarvam fallback).
    So an uploaded doc and a scraped doc are processed identically."""
    import ocr_scanned as gate   # shared quality gate + embedded-text reader
    with _JOB_LOCK:
        rec = registry.get(doc_id)
        if not rec:
            return
        dd = rec["doc_dir"]
        pdf_path = Path(rec["pdf_path"])
        md_path = pc.paths.stage2 / dd / "structured.md"
        try:
            # ── Gate: keep a clean embedded text layer instead of OCR'ing it ────
            try:
                text, pages = gate.embedded_text(pdf_path)
                keep, why = gate.text_quality(text, pages)
            except Exception:
                keep, text = False, ""
            if keep:
                md_path.parent.mkdir(parents=True, exist_ok=True)
                import unicodedata
                md_path.write_text(unicodedata.normalize("NFC", text).strip() + "\n", encoding="utf-8")
                md = md_path.read_text(encoding="utf-8")
                registry.update(doc_id, status="processed", md_path=str(md_path),
                                md_hash=md_hash(md), text_source="extract",
                                pdf_path=str(pdf_path), error=None)
                return

            # ── Otherwise OCR. --archive-source moves the source PDF to used_files;
            #    --force (re)processes this specific doc even if stale outputs exist.
            r1 = _run([STAGE1, pdf_path])
            if r1.returncode != 0:
                registry.update(doc_id, status="error", error=f"stage1 failed:\n{r1.stderr[-800:]}")
                return
            r2 = _run([STAGE2, pc.paths.stage1 / dd, "--archive-source", "--force"])
            if r2.returncode != 0:
                registry.update(doc_id, status="error", error=f"stage2 failed:\n{r2.stderr[-800:]}")
                return
            if not md_path.exists():
                registry.update(doc_id, status="error", error="OCR produced no structured.md")
                return

            text = md_path.read_text(encoding="utf-8")
            used_pdf = pc.paths.used_files / f"{dd}.pdf"   # stage2 moved the PDF here
            registry.update(doc_id, status="processed", md_path=str(md_path),
                            md_hash=md_hash(text), text_source="ocr",
                            pdf_path=str(used_pdf if used_pdf.exists() else pdf_path), error=None)
        except Exception as e:
            registry.update(doc_id, status="error", error=f"{type(e).__name__}: {e}")


# ── Indexing: chunk the md, embed, upsert into Qdrant (tagged with doc_id) ─────
def _load_rag():
    sys.path.insert(0, str(SCRIPTS))
    import rag_pipeline as rag
    rag.ensure_models_loaded()
    client = rag.ensure_qdrant_client()
    return rag, client


def _delete_by_doc(client, collection: str, doc_id: str):
    from qdrant_client.models import Filter, FieldCondition, MatchValue, FilterSelector
    if not client.collection_exists(collection):
        return
    client.delete(
        collection_name=collection,
        points_selector=FilterSelector(
            filter=Filter(must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))])
        ),
    )


def index_document(doc_id: str):
    """(Re)build this document's chunks and (re)embed them. Deletes any prior
    vectors for the same doc_id first, so an edit never leaves a stale copy."""
    with _JOB_LOCK:
        rec = registry.get(doc_id)
        if not rec:
            return
        try:
            # 1. Chunk this document's md (rebuilds from the current structured.md).
            #    Clear any previous chunk files FIRST — otherwise a shorter edit
            #    leaves stale higher-numbered chunks that get re-embedded as
            #    duplicates (two versions of the doc in the DB). Vectors are keyed
            #    by doc_id and deleted below, but the chunk FILES must be reset too.
            chunk_dir = pc.paths.chunks / rec["doc_dir"]
            if chunk_dir.exists():
                shutil.rmtree(chunk_dir, ignore_errors=True)
            rc = _run([CHUNK, "--document", rec["doc_dir"],
                       "--input", pc.paths.stage2, "--output", pc.paths.chunks])
            if rc.returncode != 0:
                registry.update(doc_id, indexed=False, error=f"chunking failed:\n{rc.stderr[-800:]}")
                return

            files = sorted(chunk_dir.glob("*_chunk_*.txt"))
            texts = [f.read_text(encoding="utf-8").strip() for f in files]
            texts_files = [(t, f) for t, f in zip(texts, files) if t]
            if not texts_files:
                registry.update(doc_id, indexed=False, error="no chunks produced")
                return

            rag, client = _load_rag()
            from qdrant_client.models import PointStruct, VectorParams, Distance
            coll = rag.COLLECTION_NAME
            if not client.collection_exists(coll):
                client.create_collection(coll, vectors_config=VectorParams(size=1024, distance=Distance.COSINE))

            # 2. Remove any previous vectors for this document, then embed fresh.
            _delete_by_doc(client, coll, doc_id)

            texts_only = [t for t, _ in texts_files]
            enc = rag.model.encode(texts_only, batch_size=8, max_length=1024)
            dense = enc["dense_vecs"]
            sparse = enc.get("lexical_weights", [None] * len(texts_only))

            points = []
            for i, (text, fpath) in enumerate(texts_files):
                payload = {
                    "text": text,
                    "source": rec["doc_dir"],
                    "doc_id": doc_id,
                    "chunk": str(i + 1),
                    "file": f"{rec['doc_dir']}/{fpath.name}",
                }
                s = sparse[i] if i < len(sparse) else None
                if s is not None:
                    try:
                        payload["sparse_embedding"] = {str(k): float(v) for k, v in dict(s).items()}
                    except Exception:
                        pass
                points.append(PointStruct(id=str(uuid.uuid4()), vector=dense[i].tolist(), payload=payload))

            client.upsert(collection_name=coll, points=points)
            registry.update(doc_id, indexed=True, chunk_count=len(points), error=None)
        except Exception as e:
            registry.update(doc_id, indexed=False, error=f"indexing error: {type(e).__name__}: {e}")


def remove_document_vectors(doc_id: str):
    with _JOB_LOCK:
        rec = registry.get(doc_id)
        if not rec:
            return
        try:
            rag, client = _load_rag()
            _delete_by_doc(client, rag.COLLECTION_NAME, doc_id)
            registry.update(doc_id, indexed=False, chunk_count=0)
        except Exception as e:
            registry.update(doc_id, error=f"remove error: {type(e).__name__}: {e}")
