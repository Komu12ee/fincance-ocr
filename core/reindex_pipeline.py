"""Re-embed all VERIFIED documents for a pipeline into its Qdrant collection.

Use when documents are marked verified but the collection is missing or empty —
e.g. after a data-path change (the ~ expansion fix), or if the vector store was
reset. Reuses the exact production indexing path (doc_pipeline.index_document), so
the result is identical to verifying each doc in the UI, just in one batch.

IMPORTANT: run with the stack STOPPED for this pipeline — the local Qdrant store is
single-writer, so a running Flask backend holds the lock. deploy/reindex.sh handles
the stop/start for you.

    PIPELINE=chips uv run python core/reindex_pipeline.py
    PIPELINE=chips uv run python core/reindex_pipeline.py --only-missing  # skip already-indexed
"""

import argparse
import sys
from pathlib import Path

CORE = Path(__file__).resolve().parent
sys.path.insert(0, str(CORE))
import pipeline_config as pc
import doc_registry as registry
import doc_pipeline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only-missing", action="store_true",
                    help="skip docs already marked indexed (default: re-embed all verified)")
    args = ap.parse_args()

    docs = registry.all()
    todo = [d for d in docs if d.get("verified") and d.get("md_path")]
    if args.only_missing:
        todo = [d for d in todo if not d.get("indexed")]

    print(f"[{pc.PIPELINE}] data_root={pc.DATA_ROOT}")
    print(f"[{pc.PIPELINE}] {len(todo)} verified doc(s) to index into collection '{pc.COLLECTION}'")
    if not todo:
        print("nothing to do.")
        return

    ok = 0
    for i, d in enumerate(todo, 1):
        doc_id = d.get("id")
        doc_pipeline.index_document(doc_id)          # chunk (single doc) + embed + upsert
        rec = registry.get(doc_id) or {}
        if rec.get("indexed"):
            ok += 1
            print(f"  [{i}/{len(todo)}] {d.get('doc_dir')}: {rec.get('chunk_count')} chunks")
        else:
            print(f"  [{i}/{len(todo)}] {d.get('doc_dir')}: FAILED — {rec.get('error')}")

    print(f"done: {ok}/{len(todo)} indexed.")
    try:
        rag, client = doc_pipeline._load_rag()
        if client.collection_exists(rag.COLLECTION_NAME):
            print(f"collection '{rag.COLLECTION_NAME}' now holds {client.count(rag.COLLECTION_NAME).count} points.")
    except Exception as e:
        print(f"(point-count check skipped: {type(e).__name__}: {e})")


if __name__ == "__main__":
    main()
