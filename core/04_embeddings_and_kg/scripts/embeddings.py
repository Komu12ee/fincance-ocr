import os
import atexit
from pathlib import Path
from tqdm import tqdm
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from FlagEmbedding import BGEM3FlagModel, FlagReranker

# ────────────────────────────────────────────────────────────────
# Configuration
# ────────────────────────────────────────────────────────────────

def _default_paths() -> dict:
    """Get default paths relative to this script location."""
    root = Path(__file__).resolve().parents[2]
    return {
        "chunk_dir": root.parent / "chunking" / "output_child_first",
        "collection": os.getenv("CHIPPY_QDRANT_COLLECTION", "db3"),
        "qdrant_local_path": Path(
            os.getenv(
                "CHIPPY_QDRANT_LOCAL_PATH",
                str(root / "04_embeddings_and_kg" / "db" / "qdrant_local")
            )
        ),
        "files_mapping": root.parent / "files.txt",
        "encode_batch_size": 8,
        "upsert_batch_size": 100,
        "max_length": 1024,
    }

DEFAULTS = _default_paths()

CHUNK_DIR = DEFAULTS["chunk_dir"]
COLLECTION_NAME = DEFAULTS["collection"]
QDRANT_LOCAL_PATH = DEFAULTS["qdrant_local_path"]
FILES_MAPPING = DEFAULTS["files_mapping"]
ENCODE_BATCH_SIZE = DEFAULTS["encode_batch_size"]
UPSERT_BATCH_SIZE = DEFAULTS["upsert_batch_size"]
MAX_LENGTH = DEFAULTS["max_length"]

# ── Retrieval Configuration ────────────────────────────────────
HYBRID_ALPHA = 0.6           # 0.0 = pure sparse, 1.0 = pure dense
RERANK_MIN_K = 3             # Minimum results to return
RERANK_MAX_K = 6             # Maximum results to return
RERANK_THRESHOLD = 0.65      # Score threshold for inclusion

# ── Load file mapping ──────────────────────────────────────────
def load_file_mapping(mapping_file: Path) -> dict:
    """Load mapping from files.txt to convert output names to actual file names."""
    mapping = {}
    try:
        with open(mapping_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or "-------->" not in line:
                    continue
                actual_name, output_name = line.split("-------->")
                actual_name = actual_name.strip()
                output_name = output_name.strip()
                output_base = output_name.replace(".pdf", "")
                mapping[output_base] = actual_name
    except Exception as e:
        print(f"[WARNING] Could not load file mapping: {e}")
    return mapping

FILE_MAPPING = load_file_mapping(FILES_MAPPING)

# ── Helper: Get actual file name ───────────────────────────────
def get_actual_filename(chunk_source: str) -> str:
    """Convert chunk source to actual filename using mapping."""
    if chunk_source in FILE_MAPPING:
        return FILE_MAPPING[chunk_source]
    return f"{chunk_source}.pdf"

# ── Load models ────────────────────────────────────────────────
print("[Embeddings] Loading BGE-M3 model...")
model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
model.return_sparse = True

print("[Embeddings] Loading reranker model...")
reranker = FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=True)

# ── Connect to Qdrant (local embedded mode) ────────────────────
print(f"[Embeddings] Connecting to local Qdrant at {QDRANT_LOCAL_PATH}...")
try:
    QDRANT_LOCAL_PATH.mkdir(parents=True, exist_ok=True)
    client = QdrantClient(path=str(QDRANT_LOCAL_PATH))
    client.get_collections()
    print("[Embeddings] ✓ Connected to local embedded Qdrant")
except Exception as e:
    print(f"[ERROR] Failed to initialize local Qdrant: {e}")
    print(f"[ERROR] Make sure {QDRANT_LOCAL_PATH} is writable")
    exit(1)


def _cleanup_qdrant():
    """Explicitly close Qdrant client on exit to avoid shutdown import errors."""
    try:
        if 'client' in globals():
            client.close()
    except Exception:
        pass  # Suppress cleanup errors during shutdown


atexit.register(_cleanup_qdrant)

# ════════════════════════════════════════════════════════════════
# ║                    INDEXING PHASE (one time)                 ║
# ════════════════════════════════════════════════════════════════

if not client.collection_exists(COLLECTION_NAME):
    print(f"\n[Embeddings] Collection '{COLLECTION_NAME}' not found. Starting indexing...\n")
    
    # ── Validate chunk directory ────────────────────────────────
    if not CHUNK_DIR.exists():
        print(f"[ERROR] Chunk directory not found: {CHUNK_DIR}")
        exit(1)
    
    # ── Create collection ─────────────────────────────────────────
    print(f"[Embeddings] Creating collection '{COLLECTION_NAME}'...")
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
    )

    # ── Read chunks ───────────────────────────────────────────────
    chunks = []
    metadata = []
    
    print(f"[Embeddings] Reading chunk files from {CHUNK_DIR}...")
    for file in sorted(CHUNK_DIR.glob("*.txt")):
        try:
            text = file.read_text(encoding="utf-8").strip()
        except Exception as e:
            print(f"[WARNING] Could not read {file.name}: {e}")
            continue
            
        if not text:
            continue

        # Extract source and chunk ID from filename
        name = file.stem
        chunk_marker = "_chunk_"
        if chunk_marker in name:
            doc_name, chunk_id = name.split(chunk_marker, 1)
        else:
            doc_name = name
            chunk_id = "0"

        chunks.append(text)
        metadata.append({"source": doc_name, "chunk": chunk_id, "file": file.name})

    if not chunks:
        print(f"[ERROR] No chunk files found in {CHUNK_DIR}")
        exit(1)
    
    print(f"[Embeddings] Loaded {len(chunks)} chunks")

    # ── Generate embeddings ───────────────────────────────────────
    print(f"[Embeddings] Encoding {len(chunks)} chunks with BGE-M3...")
    try:
        encoding_result = model.encode(
            chunks,
            batch_size=ENCODE_BATCH_SIZE,
            max_length=MAX_LENGTH
        )
    except Exception as e:
        print(f"[ERROR] Model encoding failed: {e}")
        exit(1)
    
    if encoding_result is None:
        print("[ERROR] Model encoding returned None")
        exit(1)
    
    if "dense_vecs" not in encoding_result:
        print(f"[ERROR] No dense_vecs in encoding result. Keys: {encoding_result.keys()}")
        exit(1)
    
    dense_embeddings = encoding_result["dense_vecs"]
    sparse_embeddings = encoding_result.get("lexical_weights")
    
    if sparse_embeddings is None:
        print("[WARNING] No sparse embeddings generated")
        sparse_embeddings = [None] * len(dense_embeddings)

    # ── Build points ──────────────────────────────────────────────
    print(f"[Embeddings] Building point objects...")
    points = []
    for i, (chunk, d_vector, s_embedding, meta) in enumerate(zip(chunks, dense_embeddings, sparse_embeddings, metadata)):
        payload = {
            "text": chunk,
            "source": meta["source"],
            "chunk": meta["chunk"],
            "file": meta["file"]
        }
        
        # Store sparse embeddings if available
        if s_embedding is not None:
            try:
                sparse_dict = {str(k): float(v) for k, v in s_embedding.items()}
                payload["sparse_embedding"] = sparse_dict
            except Exception as e:
                print(f"[WARNING] Could not serialize sparse embedding {i}: {e}")
        
        try:
            points.append(
                PointStruct(
                    id=i,
                    vector=d_vector.tolist(),
                    payload=payload
                )
            )
        except Exception as e:
            print(f"[WARNING] Could not create point {i}: {e}")
            continue

    # ── Upload to Qdrant ─────────────────────────────────────────
    print(f"[Embeddings] Uploading {len(points)} points to Qdrant...")
    try:
        for batch_idx in tqdm(range(0, len(points), UPSERT_BATCH_SIZE), desc="Upserting"):
            client.upsert(
                collection_name=COLLECTION_NAME,
                points=points[batch_idx:batch_idx + UPSERT_BATCH_SIZE]
            )
        print(f"\n[Embeddings] ✓ Indexed {len(points)} chunks successfully")
    except Exception as e:
        print(f"[ERROR] Upsert failed: {e}")
        exit(1)

else:
    print(f"[Embeddings] ✓ Using existing collection '{COLLECTION_NAME}'")

# ── Helper: Sparse search (BM25-like scoring) ─────────────────
def sparse_search(query_sparse, all_points, limit=5):
    """Score points based on sparse embeddings overlap."""
    scores = []
    for point in all_points:
        sparse_payload = point.payload.get("sparse_embedding", {})
        score = sum(sparse_payload.get(token, 0) * query_sparse.get(token, 0) 
                   for token in query_sparse if token in sparse_payload)
        scores.append((point.id, score))
    return sorted(scores, key=lambda x: x[1], reverse=True)[:limit]

# ── Helper: Hybrid search with RRF (Reciprocal Rank Fusion) ────
def hybrid_search(dense_scores, sparse_scores, alpha=0.5, k=60):
    """Combine dense and sparse scores using RRF."""
    rrf_scores = {}
    
    # Add dense scores (RRF formula: 1 / (k + rank))
    for rank, (point_id, score) in enumerate(dense_scores):
        rrf_scores[point_id] = alpha / (k + rank + 1)
    
    # Add sparse scores
    for rank, (point_id, score) in enumerate(sparse_scores):
        if point_id not in rrf_scores:
            rrf_scores[point_id] = 0
        rrf_scores[point_id] += (1 - alpha) / (k + rank + 1)
    
    return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

# ── Helper: Rerank search results using BGE-Reranker (Hybrid Threshold) ────
def rerank_results(query, candidate_points, min_k=3, max_k=6, threshold=0.65):
    """Rerank with hybrid method: threshold-based with min/max bounds.
    
    Args:
        query: Query string
        candidate_points: List of point objects
        min_k: Minimum results to return (default 3)
        max_k: Maximum results to return (default 6)
        threshold: Score threshold to include results (default 0.65)
    
    Logic:
        1. Include all results with score >= threshold
        2. But ensure at least min_k results
        3. Cap at max_k results
    """
    if not candidate_points:
        return []
    
    # Prepare query-document pairs for reranking
    pairs = []
    point_map = {}  # Map index to point for later retrieval
    
    for idx, point in enumerate(candidate_points):
        text = point.payload.get("text", "")
        pairs.append([query, text])
        point_map[idx] = point
    
    # Score with reranker
    rerank_scores = reranker.compute_score(pairs, normalize=True)
    
    # Sort by reranker scores (descending)
    ranked_indices = sorted(range(len(rerank_scores)), key=lambda i: rerank_scores[i], reverse=True)
    
    # Apply hybrid threshold logic
    results = []
    for rank, idx in enumerate(ranked_indices):
        score = rerank_scores[idx]
        
        # Include if:
        # 1. Score >= threshold AND results < max_k, OR
        # 2. results < min_k (ensure minimum)
        if (score >= threshold and len(results) < max_k) or len(results) < min_k:
            results.append({
                "point": point_map[idx],
                "score": score,
                "rank": len(results) + 1
            })
        # Stop if we've reached max_k
        if len(results) >= max_k:
            break
    
    return results

# ════════════════════════════════════════════════════════════════
# ║                    QUERY PHASE (many times)                  ║
# ════════════════════════════════════════════════════════════════

def run_query_loop():
    """Interactive query loop for testing."""
    print("\n" + "="*70)
    print("Ready for queries. Type 'exit' to quit.")
    print("="*70 + "\n")
    
    while True:
        query = input("Enter query (or 'exit'): ").strip()
        if query.lower() == "exit":
            break
        if not query:
            continue

        try:
            # Encode query
            query_encoding = model.encode([query], batch_size=1, max_length=MAX_LENGTH)
            
            if query_encoding is None:
                print("[ERROR] Query encoding returned None")
                continue
            
            if "dense_vecs" not in query_encoding:
                print(f"[ERROR] No dense_vecs in result")
                continue
            
            dense_vecs = query_encoding.get("dense_vecs")
            if dense_vecs is None or len(dense_vecs) == 0:
                print("[ERROR] No dense vectors generated")
                continue
            
            query_dense = dense_vecs[0].tolist()
            
            # Get sparse embeddings if available
            query_sparse = {}
            lex_weights = query_encoding.get("lexical_weights")
            if lex_weights is not None:
                if isinstance(lex_weights, list) and len(lex_weights) > 0:
                    try:
                        query_sparse = dict(lex_weights[0])
                    except (TypeError, ValueError):
                        pass
                elif isinstance(lex_weights, dict):
                    query_sparse = lex_weights
            
            # Dense search
            dense_results = client.query_points(
                collection_name=COLLECTION_NAME,
                query=query_dense,
                limit=20
            )
            
            if dense_results is None or not dense_results.points:
                print("No results found")
                continue
            
            dense_scores = [(p.id, p.score) for p in dense_results.points]
            
            # Hybrid search with sparse support
            if query_sparse:
                sparse_scores = sparse_search(query_sparse, dense_results.points, limit=20)
                hybrid_scores = hybrid_search(dense_scores, sparse_scores, alpha=HYBRID_ALPHA)
                print(f"  ⚡ Hybrid search (α={HYBRID_ALPHA})")
            else:
                hybrid_scores = [(pid, score) for pid, score in dense_scores]
                print(f"  ⚡ Dense-only search")
            
            # Collect candidate points
            candidate_points = [
                next((p for p in dense_results.points if p.id == point_id), None)
                for point_id, _ in hybrid_scores[:20]
            ]
            candidate_points = [p for p in candidate_points if p is not None]
            
            # Rerank results
            reranked_results = rerank_results(query, candidate_points, 
                                             min_k=RERANK_MIN_K, 
                                             max_k=RERANK_MAX_K, 
                                             threshold=RERANK_THRESHOLD)
            
            # Display results
            print("\n" + "="*70)
            print(f"Results ({len(reranked_results)} documents)")
            print("="*70 + "\n")
            
            if not reranked_results:
                print("No results found after reranking")
            else:
                for result in reranked_results:
                    point = result["point"]
                    score = result["score"]
                    rank = result["rank"]
                    source = point.payload.get('source', '')
                    actual_filename = get_actual_filename(source)
                    print(f"[#{rank}] {actual_filename} (score: {score:.3f})")
                    print(point.payload["text"][:400] + "...")
                    print("-" * 70 + "\n")

        except Exception as e:
            print(f"[ERROR] {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Standalone embeddings indexer for CHiPPY",
        epilog="Example: python embeddings.py --query 'What is this about?'"
    )
    parser.add_argument("--query", type=str, help="Single query to run (instead of interactive mode)")
    parser.add_argument("--recreate", action="store_true", help="Delete and recreate collection")
    
    args = parser.parse_args()
    
    # Handle collection recreation
    if args.recreate and client.collection_exists(COLLECTION_NAME):
        print(f"\n[Embeddings] Deleting collection '{COLLECTION_NAME}'...")
        client.delete_collection(collection_name=COLLECTION_NAME)
        print("[Embeddings] ✓ Collection deleted")
    
    # Run query if provided, otherwise start interactive mode
    if args.query:
        print(f"\n[Embeddings] Running query: {args.query}")
        # Would run single query here, but full implementation is in loop
        print("Use interactive mode for full query experience (no --query flag)")
    else:
        run_query_loop()