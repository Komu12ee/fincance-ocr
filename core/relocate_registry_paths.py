"""Relocate hardcoded file paths in each pipeline's documents.json.

The document registry (data/<pipeline>/documents.json) stores absolute paths for
each doc's source PDF, its OCR markdown, and (for scraped docs) its raw snapshot.
Those were written on the authoring machine, so they are absolute *and* OS-specific
(e.g. C:\\Users\\HP\\Desktop\\Rag2\\data\\fg2\\stage2\\...\\structured.md) and break
after copying to a Linux server, or wherever RAG_DATA_ROOT points.

This script rewrites `pdf_path`, `md_path`, and `raw_path` to correct absolute
paths for the CURRENT host: it keeps each path's tail *after* the pipeline data
root (e.g. `stage2/<doc_dir>/structured.md`, `used_files/<name>.pdf`) and re-roots
it under this machine's DATA_ROOT — the same location the app resolves via
pipeline_config (RAG_DATA_ROOT or <repo>/data, then /<dataRoot>).

Idempotent: running it again on already-correct paths is a no-op. Writes a
`documents.json.bak` before changing anything. Use --dry-run to preview.

Usage:
    python core/relocate_registry_paths.py            # fix every pipeline
    python core/relocate_registry_paths.py --dry-run  # preview only
    python core/relocate_registry_paths.py --pipeline fg2
"""

import argparse
import json
import os
from pathlib import Path

CORE_DIR = Path(__file__).resolve().parent
REPO_ROOT = CORE_DIR.parent

# Fields in each doc record that hold a filesystem path we may need to relocate.
PATH_FIELDS = ("pdf_path", "md_path", "raw_path")


def data_base() -> Path:
    """Same rule as pipeline_config.DATA_BASE."""
    return Path(os.getenv("RAG_DATA_ROOT", str(REPO_ROOT / "data"))).resolve()


def load_registry() -> dict[str, str]:
    """pipeline name -> dataRoot (from repo-root pipelines.json)."""
    reg = json.loads((REPO_ROOT / "pipelines.json").read_text(encoding="utf-8"))
    return {name: cfg["dataRoot"] for name, cfg in reg.items()}


def tail_after_data_root(raw_path: str, data_root_name: str) -> list[str] | None:
    """Return the path segments AFTER the pipeline's data-root directory.

    Handles both `\\` and `/` separators. Anchors on the `data/<dataRoot>/` pair
    when present (most reliable), else on the last bare `<dataRoot>/` segment.
    Returns None if the data root can't be located in the path.
    """
    parts = [seg for seg in raw_path.replace("\\", "/").split("/") if seg not in ("", ".")]
    anchor = None
    for k in range(1, len(parts)):
        if parts[k] == data_root_name and parts[k - 1] == "data":
            anchor = k  # last data/<name> wins
    if anchor is None:
        for k, seg in enumerate(parts):
            if seg == data_root_name:
                anchor = k
    if anchor is None:
        return None
    return parts[anchor + 1:]


def relocate_path(raw_path: str, data_root: Path, data_root_name: str) -> str | None:
    """Rebuild `raw_path` under this host's data_root. None = leave unchanged."""
    if not raw_path:
        return None
    tail = tail_after_data_root(raw_path, data_root_name)
    if not tail:
        return None
    return str(data_root.joinpath(*tail))


def process_pipeline(name: str, data_root_name: str, base: Path, dry_run: bool) -> None:
    data_root = base / data_root_name
    reg_path = data_root / "documents.json"
    if not reg_path.exists():
        print(f"[{name}] no documents.json ({reg_path}) — skipping")
        return

    data = json.loads(reg_path.read_text(encoding="utf-8"))
    docs = data.get("documents", [])
    changed = 0

    for doc in docs:
        for field in PATH_FIELDS:
            old = doc.get(field)
            if not old:
                continue
            new = relocate_path(old, data_root, data_root_name)
            if new and new != old:
                doc[field] = new
                changed += 1

    print(f"[{name}] {len(docs)} docs, {changed} path(s) rewritten "
          f"-> {data_root}" + ("  (dry-run)" if dry_run else ""))
    if changed and not dry_run:
        backup = reg_path.with_suffix(".json.bak")
        backup.write_text(reg_path.read_text(encoding="utf-8"), encoding="utf-8")
        reg_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[{name}] wrote {reg_path}  (backup: {backup.name})")


def main() -> None:
    ap = argparse.ArgumentParser(description="Relocate documents.json paths for this host.")
    ap.add_argument("--pipeline", help="Only this pipeline (default: all in pipelines.json)")
    ap.add_argument("--dry-run", action="store_true", help="Preview without writing")
    args = ap.parse_args()

    base = data_base()
    registry = load_registry()
    print(f"DATA_BASE = {base}  (RAG_DATA_ROOT {'set' if os.getenv('RAG_DATA_ROOT') else 'unset -> repo/data'})")

    targets = ({args.pipeline: registry[args.pipeline]}
               if args.pipeline else registry)
    if args.pipeline and args.pipeline not in registry:
        raise SystemExit(f"Unknown pipeline {args.pipeline!r}. Known: {list(registry)}")

    for name, data_root_name in targets.items():
        process_pipeline(name, data_root_name, base, args.dry_run)


if __name__ == "__main__":
    main()
