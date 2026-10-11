"""Register already-OCR'd stage2 documents into the GAD verify workspace as
UNVERIFIED. FOR LOCALLY-OCR'd PDFs ONLY — the source is the local PDF file.

  ⚠ Scraped content (web pages / downloaded PDFs) must NOT go through here — it
    would lose its source URL. Use the provenance-complete flow instead:
        core/ocr_scanned.py  (OCR the scraper's scanned PDFs)  →  scrape_import
    See INGESTION.md. This tool now ENFORCES the provenance invariant: a doc with
    no resolvable source PDF is SKIPPED (pass --allow-orphan to override, not
    recommended — the verify panel would show no source for it).

Discovers every stage2/<doc_dir>/structured.md not already in documents.json whose
source PDF exists (used_files / input_pdfs / downloads / <doc_dir>.pdf) and adds a
record. Verifying each in the UI is what finally chunks + embeds it into the RAG.

Idempotent (skips already-registered doc_dirs). Writes documents.json.bak before
changing anything. Dry-run by default.

Usage:
    PIPELINE=chips python core/register_unverified.py            # DRY RUN (default)
    PIPELINE=chips python core/register_unverified.py --apply    # write (source-checked)
    PIPELINE=chips python core/register_unverified.py --apply --allow-orphan
"""

import argparse
import hashlib
import json
import re
import shutil
import sys
import uuid
from datetime import datetime
from pathlib import Path

CORE = Path(__file__).resolve().parent
sys.path.insert(0, str(CORE))
import pipeline_config as pc
import doc_registry


def md_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def find_pdf(doc_dir: str):
    """Locate a source PDF for this doc, in the places the app can serve it from."""
    for base in (pc.paths.used_files, pc.paths.input_pdfs, pc.paths.downloads):
        p = base / f"{doc_dir}.pdf"
        if p.exists():
            return p
    return None


def build_record(doc_dir: str, md_path: Path, pdf: Path | None) -> dict:
    text = md_path.read_text(encoding="utf-8", errors="replace")
    doc_id = uuid.uuid4().hex
    return {
        "id": doc_id,
        "original_name": (pdf.name if pdf else f"{doc_dir}.pdf"),
        "doc_dir": doc_dir,
        "pdf_path": (str(pdf) if pdf else None),
        "md_path": str(md_path),
        "status": "processed",            # OCR is done; awaiting verification
        "verified": False, "verified_by": None, "verified_at": None,
        "uploaded_by": "ocr-import", "uploaded_at": datetime.now().isoformat(),
        "edited_by": None, "edited_at": None,
        "md_hash": md_hash(text),
        "indexed": False, "chunk_count": 0,
        "error": None,
        "source": "upload", "source_url": None,
        "doc_kind": "pdf", "raw_ext": None, "raw_path": None,
    }


# A scraper doc_dir is sha1(url)[:16] — 16 lowercase hex. Uploaded docs are
# "<slug>_<8hex>" and locally-OCR'd ones are PDF stems, so this matches only
# scraped content.
_SCRAPED_DIR = re.compile(r"^[0-9a-f]{16}$")


def discover_new(allow_scraped: bool = False):
    """Returns (existing, new, scraped_skipped).

    Scraped docs are skipped by default: they belong to core/scrape_import.py,
    which knows their source URL. Registering them here would stamp them
    `source="upload", source_url=None` — destroying provenance — and create a
    SECOND record for a doc scrape_import will also register.
    """
    existing = doc_registry.all()
    have_dirs = {d.get("doc_dir") for d in existing}
    have_md = {d.get("md_path") for d in existing}
    new, scraped_skipped = [], []
    if pc.paths.stage2.exists():
        for d in sorted(pc.paths.stage2.iterdir()):
            md = d / "structured.md"
            if not d.is_dir() or not md.exists():
                continue
            if d.name in have_dirs or str(md) in have_md:
                continue
            if _SCRAPED_DIR.match(d.name) and not allow_scraped:
                scraped_skipped.append(d.name)
                continue
            new.append(build_record(d.name, md, find_pdf(d.name)))
    return existing, new, scraped_skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="write documents.json (default: dry run)")
    ap.add_argument("--allow-orphan", action="store_true",
                    help="also register docs with NO source PDF (breaks the 'every doc has a "
                         "source' invariant — the verify panel shows no source). Not recommended.")
    ap.add_argument("--allow-scraped", action="store_true",
                    help="also register stage2 dirs that look like scraped content "
                         "(16-hex urlhash). Almost always wrong — it erases the source URL "
                         "and duplicates what scrape_import registers.")
    args = ap.parse_args()

    existing, discovered, scraped_skipped = discover_new(allow_scraped=args.allow_scraped)
    sourced = [r for r in discovered if r["pdf_path"]]
    orphans = [r for r in discovered if not r["pdf_path"]]
    new = discovered if args.allow_orphan else sourced

    print(f"pipeline           : {pc.PIPELINE}")
    print(f"data_root          : {pc.DATA_ROOT}")
    print(f"already registered : {len(existing)}")
    print(f"with a source PDF  : {len(sourced)}  (registerable)")
    print(f"NO source (orphan) : {len(orphans)}  -> " +
          ("WILL register (--allow-orphan)" if args.allow_orphan else
           "SKIPPED. Scraped docs? Use ocr_scanned.py + scrape_import (keeps the URL). See INGESTION.md."))
    if scraped_skipped:
        print(f"SCRAPED (16-hex)   : {len(scraped_skipped)}  -> SKIPPED. These belong to "
              "scrape_import, which keeps their source URL.")
        print(f"                     e.g. {', '.join(scraped_skipped[:3])}")
    for r in new[:5]:
        print(f"   e.g. {r['doc_dir']}  pdf={r['pdf_path']}")

    if not args.apply:
        print("\nDRY RUN — nothing written. Re-run with --apply to register these as UNVERIFIED.")
        return

    if not new:
        print("\nNothing to register (no source-bearing docs).")
        return

    reg = pc.DATA_ROOT / "documents.json"
    if reg.exists():
        bak = reg.with_suffix(".json.bak")
        shutil.copyfile(reg, bak)
        print(f"\nbackup written     : {bak}")

    docs = existing + new
    tmp = reg.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"documents": docs}, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(reg)
    print(f"registered {len(new)} docs as UNVERIFIED. documents.json now has {len(docs)}.")
    print("Open the GAD verify panel to review & verify them (verifying is what embeds each into the RAG).")


if __name__ == "__main__":
    main()
