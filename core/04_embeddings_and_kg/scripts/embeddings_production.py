#!/usr/bin/env python3
"""
Production Embeddings Pipeline - Incremental Indexing
Detects new chunks and appends to Qdrant without re-embedding existing chunks.

Usage:
    python embeddings_production.py                  # Index new chunks
    python embeddings_production.py --query         # Interactive query mode
    python embeddings_production.py --recreate      # Delete and rebuild DB
    python embeddings_production.py --status        # Show indexing status
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Set
import atexit

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import VectorParams, Distance, PointStruct
    from FlagEmbedding import BGEM3FlagModel, FlagReranker
    from tqdm import tqdm
except ImportError:
    print("Error: Required packages not installed.")
    print("Run: pip install qdrant-client FlagEmbedding torch tqdm")
    exit(1)

# ────────────────────────────────────────────────────────────────
# Configuration
# ────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-7s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Paths — pipeline-aware via shared config (data/<pipeline>/...)
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent  # = core/
sys.path.insert(0, str(PROJECT_ROOT))
import pipeline_config as pc

CHUNK_DIR = pc.paths.chunks
QDRANT_PATH = pc.paths.db          # same env-driven store the query path reads
MANIFEST_FILE = pc.paths.manifest

# Configuration
COLLECTION_NAME = pc.COLLECTION
ENCODE_BATCH_SIZE = 8
UPSERT_BATCH_SIZE = 100
MAX_LENGTH = 1024
HYBRID_ALPHA = 0.6
RERANK_MIN_K = 3
RERANK_MAX_K = 6
RERANK_THRESHOLD = 0.65

# ────────────────────────────────────────────────────────────────
# Manifest Management
# ────────────────────────────────────────────────────────────────

class EmbeddingsManifest:
    """Track indexed chunks to enable incremental updates."""
    
    def __init__(self, manifest_path: Path):
        self.path = manifest_path
        self.data = self._load()
    
    def _load(self) -> dict:
        """Load manifest from file."""
        if self.path.exists():
            try:
                with open(self.path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load manifest: {e}")
                return self._empty()
        return self._empty()
    
    @staticmethod
    def _empty() -> dict:
        """Create empty manifest structure."""
        return {
            "version": "1.0",
            "created_at": datetime.now().isoformat(),
            "last_updated": None,
            "indexed_chunks": {},  # {filename: {id, hash}}
            "total_indexed": 0,
            "collection": COLLECTION_NAME
        }
    
    def save(self):
        """Save manifest to file."""
        self.data["last_updated"] = datetime.now().isoformat()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, indent=2)
    
    def add_chunk(self, filename: str, point_id: int):
        """Mark chunk as indexed."""
        self.data["indexed_chunks"][filename] = {
            "id": point_id,
            "indexed_at": datetime.now().isoformat()
        }
        self.data["total_indexed"] = len(self.data["indexed_chunks"])
        self.save()
    
    def get_indexed_files(self) -> Set[str]:
        """Get set of already-indexed chunk files."""
        return set(self.data["indexed_chunks"].keys())
    
    def get_next_id(self) -> int:
        """Get next available point ID."""
        if not self.data["indexed_chunks"]:
            return 0
        return max(chunk["id"] for chunk in self.data["indexed_chunks"].values()) + 1
    
    def clear(self):
        """Reset manifest."""
        self.data = self._empty()
        self.save()


manifest = EmbeddingsManifest(MANIFEST_FILE)

# ────────────────────────────────────────────────────────────────
# Load Models
# ────────────────────────────────────────────────────────────────

logger.info("Loading BGE-M3 embedding model...")
try:
    model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
    model.return_sparse = True
except Exception as e:
    logger.error(f"Failed to load embedding model: {e}")
    exit(1)

logger.info("Loading BGE-Reranker model...")
try:
    reranker = FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=True)
except Exception as e:
    logger.error(f"Failed to load reranker: {e}")
    exit(1)

# ────────────────────────────────────────────────────────────────
# Qdrant Client
# ────────────────────────────────────────────────────────────────

logger.info(f"Connecting to Qdrant at {QDRANT_PATH}...")
try:
    QDRANT_PATH.mkdir(parents=True, exist_ok=True)
    client = QdrantClient(path=str(QDRANT_PATH))
    client.get_collections()
    logger.info("✓ Connected to local embedded Qdrant")
except Exception as e:
    logger.error(f"Failed to initialize Qdrant: {e}")
    exit(1)


def _cleanup_qdrant():
    """Explicitly close Qdrant on exit."""
    try:
        if 'client' in globals():
            client.close()
    except Exception:
        pass


atexit.register(_cleanup_qdrant)

# ════════════════════════════════════════════════════════════════
# Incremental Indexing
# ════════════════════════════════════════════════════════════════

def find_new_chunks() -> List[Path]:
    """Find chunk files not yet indexed (recursively in subdirectories)."""
    if not CHUNK_DIR.exists():
        logger.error(f"Chunk directory not found: {CHUNK_DIR}")
        return []
    
    indexed_files = manifest.get_indexed_files()
    new_chunks = []
    
    # Search recursively for chunk files in subdirectories
    for chunk_file in sorted(CHUNK_DIR.rglob("*_chunk_*.txt")):
        # Use relative path from CHUNK_DIR for manifest tracking
        relative_path = str(chunk_file.relative_to(CHUNK_DIR))
        if relative_path not in indexed_files:
            new_chunks.append(chunk_file)
    
    return new_chunks


def index_new_chunks():
    """Find and index only new chunks."""
    # Create collection if needed
    if not client.collection_exists(COLLECTION_NAME):
        logger.info(f"Creating collection '{COLLECTION_NAME}'...")
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
        )
    
    # Find new chunks
    new_chunks = find_new_chunks()
    
    if not new_chunks:
        logger.info("✓ No new chunks to index")
        return 0
    
    logger.info(f"Found {len(new_chunks)} new chunk(s) to index")
    
    # Read new chunks
    chunks_data = []
    for chunk_file in new_chunks:
        try:
            text = chunk_file.read_text(encoding='utf-8').strip()
            
            # Extract metadata from filename
            name = chunk_file.stem
            chunk_marker = "_chunk_"
            if chunk_marker in name:
                doc_name, chunk_id = name.split(chunk_marker, 1)
            else:
                doc_name = name
                chunk_id = "0"
            
            # Store relative path for manifest tracking
            relative_path = str(chunk_file.relative_to(CHUNK_DIR))
            
            chunks_data.append({
                "file": relative_path,  # Use relative path for manifest
                "text": text,
                "source": doc_name,
                "chunk": chunk_id
            })
        except Exception as e:
            logger.warning(f"Could not read {chunk_file.name}: {e}")
            continue
    
    if not chunks_data:
        logger.error("No valid chunks found")
        return 0
    
    # Encode chunks
    logger.info(f"Encoding {len(chunks_data)} chunks...")
    try:
        encoding_result = model.encode(
            [c["text"] for c in chunks_data],
            batch_size=ENCODE_BATCH_SIZE,
            max_length=MAX_LENGTH
        )
    except Exception as e:
        logger.error(f"Encoding failed: {e}")
        return 0
    
    if not encoding_result or "dense_vecs" not in encoding_result:
        logger.error("Invalid encoding result")
        return 0
    
    dense_embeddings = encoding_result["dense_vecs"]
    sparse_embeddings = encoding_result.get("lexical_weights", [None] * len(chunks_data))
    
    # Build points with incremental IDs
    next_id = manifest.get_next_id()
    points = []
    
    logger.info("Building point objects...")
    for i, (chunk_data, d_vector, s_embedding) in enumerate(
        zip(chunks_data, dense_embeddings, sparse_embeddings)
    ):
        point_id = next_id + i
        
        payload = {
            "text": chunk_data["text"],
            "source": chunk_data["source"],
            "chunk": chunk_data["chunk"],
            "file": chunk_data["file"]
        }
        
        # Store sparse embeddings
        if s_embedding is not None:
            try:
                sparse_dict = {str(k): float(v) for k, v in s_embedding.items()}
                payload["sparse_embedding"] = sparse_dict
            except Exception as e:
                logger.warning(f"Could not serialize sparse embedding {point_id}: {e}")
        
        try:
            points.append(
                PointStruct(
                    id=point_id,
                    vector=d_vector.tolist(),
                    payload=payload
                )
            )
        except Exception as e:
            logger.warning(f"Could not create point {point_id}: {e}")
            continue
    
    # Upsert to Qdrant
    logger.info(f"Uploading {len(points)} points to Qdrant...")
    try:
        for batch_idx in tqdm(range(0, len(points), UPSERT_BATCH_SIZE), desc="Upserting"):
            client.upsert(
                collection_name=COLLECTION_NAME,
                points=points[batch_idx:batch_idx + UPSERT_BATCH_SIZE]
            )
        
        # Update manifest
        for point, chunk_data in zip(points, chunks_data):
            manifest.add_chunk(chunk_data["file"], point.id)
        
        logger.info(f"\n✅ Indexed {len(points)} new chunks successfully")
        return len(points)
    
    except Exception as e:
        logger.error(f"Upsert failed: {e}")
        return 0


# ════════════════════════════════════════════════════════════════
# Search Helpers
# ════════════════════════════════════════════════════════════════

def sparse_search(query_sparse, all_points, limit=5):
    """Score points based on sparse embeddings overlap."""
    scores = []
    for point in all_points:
        sparse_payload = point.payload.get("sparse_embedding", {})
        score = sum(sparse_payload.get(token, 0) * query_sparse.get(token, 0) 
                   for token in query_sparse if token in sparse_payload)
        scores.append((point.id, score))
    return sorted(scores, key=lambda x: x[1], reverse=True)[:limit]


def hybrid_search(dense_scores, sparse_scores, alpha=0.5, k=60):
    """Combine dense and sparse scores using RRF."""
    rrf_scores = {}
    
    for rank, (point_id, score) in enumerate(dense_scores):
        rrf_scores[point_id] = alpha / (k + rank + 1)
    
    for rank, (point_id, score) in enumerate(sparse_scores):
        if point_id not in rrf_scores:
            rrf_scores[point_id] = 0
        rrf_scores[point_id] += (1 - alpha) / (k + rank + 1)
    
    return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)


def rerank_results(query, candidate_points, min_k=3, max_k=6, threshold=0.65):
    """Rerank results with threshold-based inclusion."""
    if not candidate_points:
        return []
    
    pairs = [[query, p.payload.get("text", "")] for p in candidate_points]
    rerank_scores = reranker.compute_score(pairs, normalize=True)
    
    ranked_indices = sorted(range(len(rerank_scores)), 
                           key=lambda i: rerank_scores[i], reverse=True)
    
    results = []
    for rank, idx in enumerate(ranked_indices):
        score = rerank_scores[idx]
        
        if (score >= threshold and len(results) < max_k) or len(results) < min_k:
            results.append({
                "point": candidate_points[idx],
                "score": score,
                "rank": len(results) + 1
            })
        
        if len(results) >= max_k:
            break
    
    return results


# ════════════════════════════════════════════════════════════════
# Query Mode
# ════════════════════════════════════════════════════════════════

def run_query_loop():
    """Interactive query mode."""
    if not client.collection_exists(COLLECTION_NAME):
        logger.error(f"Collection '{COLLECTION_NAME}' not found. Run indexing first.")
        return
    
    logger.info("\n" + "="*70)
    logger.info("Ready for queries. Type 'exit' to quit.")
    logger.info("="*70 + "\n")
    
    while True:
        query = input("Enter query (or 'exit'): ").strip()
        if query.lower() == "exit":
            break
        if not query:
            continue
        
        try:
            # Encode query
            query_encoding = model.encode([query], batch_size=1, max_length=MAX_LENGTH)
            
            if not query_encoding or "dense_vecs" not in query_encoding:
                logger.error("Query encoding failed")
                continue
            
            query_dense = query_encoding["dense_vecs"][0].tolist()
            
            # Get sparse embeddings
            query_sparse = {}
            lex_weights = query_encoding.get("lexical_weights")
            if lex_weights and isinstance(lex_weights, list) and len(lex_weights) > 0:
                try:
                    query_sparse = dict(lex_weights[0])
                except (TypeError, ValueError):
                    pass
            
            # Dense search
            dense_results = client.query_points(
                collection_name=COLLECTION_NAME,
                query=query_dense,
                limit=20
            )
            
            if not dense_results or not dense_results.points:
                logger.info("No results found")
                continue
            
            dense_scores = [(p.id, p.score) for p in dense_results.points]
            
            # Hybrid search
            if query_sparse:
                sparse_scores = sparse_search(query_sparse, dense_results.points, limit=20)
                hybrid_scores = hybrid_search(dense_scores, sparse_scores, alpha=HYBRID_ALPHA)
            else:
                hybrid_scores = [(pid, score) for pid, score in dense_scores]
            
            # Collect candidates
            candidate_points = [
                next((p for p in dense_results.points if p.id == pid), None)
                for pid, _ in hybrid_scores[:20]
            ]
            candidate_points = [p for p in candidate_points if p is not None]
            
            # Rerank
            reranked_results = rerank_results(
                query, candidate_points,
                min_k=RERANK_MIN_K,
                max_k=RERANK_MAX_K,
                threshold=RERANK_THRESHOLD
            )
            
            # Display results
            logger.info("\n" + "="*70)
            logger.info(f"Results ({len(reranked_results)} documents)")
            logger.info("="*70 + "\n")
            
            if not reranked_results:
                logger.info("No results found after reranking")
            else:
                for result in reranked_results:
                    point = result["point"]
                    score = result["score"]
                    rank = result["rank"]
                    source = point.payload.get('source', 'Unknown')
                    print(f"\n[#{rank}] {source} (score: {score:.3f})")
                    text = point.payload.get("text", "")[:400]
                    print(text + "...\n")
        
        except Exception as e:
            logger.error(f"{type(e).__name__}: {e}")


# ════════════════════════════════════════════════════════════════
# Status & Management
# ════════════════════════════════════════════════════════════════

def show_status():
    """Show indexing status."""
    logger.info("\n" + "="*70)
    logger.info("Embeddings Status")
    logger.info("="*70 + "\n")
    
    # Manifest info
    logger.info("📋 Manifest Information:")
    logger.info(f"  Total indexed chunks: {manifest.data['total_indexed']}")
    logger.info(f"  Last updated: {manifest.data.get('last_updated', 'Never')}")
    
    # Qdrant info
    if client.collection_exists(COLLECTION_NAME):
        collection_info = client.get_collection(COLLECTION_NAME)
        logger.info(f"\n🗄️  Qdrant Collection: {COLLECTION_NAME}")
        logger.info(f"  Points stored: {collection_info.points_count}")
        logger.info(f"  Vector size: 1024 dimensions")
    else:
        logger.info(f"\n🗄️  Qdrant Collection: {COLLECTION_NAME} (not yet created)")
    
    # Chunk files
    logger.info(f"\n📁 Chunk Files:")
    if CHUNK_DIR.exists():
        chunk_files = list(CHUNK_DIR.glob("*_chunk_*.txt"))
        logger.info(f"  Total on disk: {len(chunk_files)}")
        new_chunks = find_new_chunks()
        logger.info(f"  New (not indexed): {len(new_chunks)}")
    else:
        logger.info(f"  Directory not found: {CHUNK_DIR}")
    
    logger.info("")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Production Embeddings Pipeline with Incremental Indexing"
    )
    parser.add_argument('--query', action='store_true',
                       help='Enter interactive query mode')
    parser.add_argument('--recreate', action='store_true',
                       help='Delete and recreate collection')
    parser.add_argument('--status', action='store_true',
                       help='Show indexing status')
    
    args = parser.parse_args()
    
    # Handle recreate
    if args.recreate:
        if client.collection_exists(COLLECTION_NAME):
            logger.info(f"Deleting collection '{COLLECTION_NAME}'...")
            client.delete_collection(collection_name=COLLECTION_NAME)
            manifest.clear()
            logger.info("✓ Collection deleted and manifest reset")
        return
    
    # Handle status
    if args.status:
        show_status()
        return
    
    # Handle query mode
    if args.query:
        run_query_loop()
        return
    
    # Default: Index new chunks
    logger.info("="*70)
    logger.info("Embeddings Indexing - Incremental Mode")
    logger.info("="*70)
    logger.info("")
    
    show_status()
    logger.info("")
    
    indexed = index_new_chunks()
    
    logger.info("")
    logger.info("="*70)
    if indexed > 0:
        logger.info(f"✅ Successfully indexed {indexed} new chunks")
    else:
        logger.info("✅ Database is up to date")
    logger.info("="*70)
    logger.info("")
    logger.info("💡 To query the database:")
    logger.info("  python embeddings_production.py --query")


if __name__ == "__main__":
    main()
