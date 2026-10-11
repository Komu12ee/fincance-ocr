"""Central per-pipeline configuration for the shared RAG core.

One codebase serves multiple department pipelines (chips, fg, fg2). The active
pipeline is chosen by the ``PIPELINE`` environment variable (default ``chips``).
All per-pipeline settings live in the repo-root ``pipelines.json`` registry, and
secrets in the repo-root ``.env`` — nothing is hardcoded in the stage scripts.

Usage:
    from pipeline_config import PIPELINE, COLLECTION, SARVAM_API_KEY, paths
    input_dir = paths.input_pdfs          # data/<pipeline>/input_pdfs
    db_dir    = paths.db                  # data/<pipeline>/db
"""

import os
import json
from pathlib import Path

from dotenv import load_dotenv

# core/pipeline_config.py -> repo root is the parent of core/
CORE_DIR = Path(__file__).resolve().parent
REPO_ROOT = CORE_DIR.parent

load_dotenv(REPO_ROOT / ".env")

_REGISTRY = json.loads((REPO_ROOT / "pipelines.json").read_text(encoding="utf-8"))

PIPELINE = os.getenv("PIPELINE", "chips").strip().lower()
if PIPELINE not in _REGISTRY:
    raise ValueError(
        f"Unknown PIPELINE={PIPELINE!r}. Known pipelines: {list(_REGISTRY)}"
    )

_CFG = _REGISTRY[PIPELINE]

COLLECTION = _CFG["collection"]
NODE_PORT = _CFG["nodePort"]
FLASK_PORT = _CFG["flaskPort"]
SARVAM_API_KEY = os.getenv(_CFG["sarvamKeyEnv"], "")

# Base directory holding every pipeline's data. Defaults to <repo>/data for local
# dev; on a deployed server point RAG_DATA_ROOT at e.g. /data/rag_data so code and
# data live on separate volumes. Each pipeline gets <RAG_DATA_ROOT>/<dataRoot>.
# expanduser() so a leading ~ set in .env (where no shell expands it) resolves to
# $HOME instead of a literal directory named "~".
DATA_BASE = Path(
    os.path.expanduser(os.getenv("RAG_DATA_ROOT", str(REPO_ROOT / "data")))
).resolve()

# Active pipeline's data dir — absolute. PIPELINE_DATA_ROOT is a full-path override
# for ad-hoc/sandbox runs that bypasses DATA_BASE entirely.
DATA_ROOT = Path(
    os.path.expanduser(os.getenv("PIPELINE_DATA_ROOT", str(DATA_BASE / _CFG["dataRoot"])))
).resolve()


class paths:
    """Absolute filesystem locations for the active pipeline's data."""

    input_pdfs = DATA_ROOT / "input_pdfs"      # source PDFs to ingest
    downloads = DATA_ROOT / "downloads"        # department download buckets
    stage1 = DATA_ROOT / "stage1"              # preprocessed page images (transient)
    stage2 = DATA_ROOT / "stage2"              # OCR output (structured.md/json)
    used_files = DATA_ROOT / "used_files"      # processed source PDFs
    scrape_raw = DATA_ROOT / "scrape_raw"      # raw source snapshots of scraped web docs (html/pdf/json)
    optimization = DATA_ROOT / "optimization"  # cleaned/optimized text
    chunks = DATA_ROOT / "chunks"              # semantic chunks
    db = DATA_ROOT / "db"                      # local Qdrant vector store
    manifest = DATA_ROOT / "manifest.json"     # embeddings incremental manifest


def ensure_dirs():
    """Create all data directories for the active pipeline if missing."""
    for p in (
        paths.input_pdfs, paths.downloads, paths.stage1, paths.stage2,
        paths.used_files, paths.scrape_raw, paths.optimization, paths.chunks, paths.db,
    ):
        p.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    print(f"PIPELINE   = {PIPELINE}")
    print(f"DATA_ROOT  = {DATA_ROOT}")
    print(f"COLLECTION = {COLLECTION}")
    print(f"NODE_PORT  = {NODE_PORT}  FLASK_PORT = {FLASK_PORT}")
    print(f"SARVAM key = {'set' if SARVAM_API_KEY else 'MISSING'}")
    for name in ("input_pdfs", "stage1", "stage2", "used_files", "chunks", "db", "manifest"):
        print(f"  {name:12s} {getattr(paths, name)}")
