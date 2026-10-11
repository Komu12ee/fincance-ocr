#!/usr/bin/env python3
"""
Index new chunks from Stage 3 to Qdrant database.
Appends 14 new structured chunks to existing db3 collection.
"""

import json
import sys
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from FlagEmbedding import BGEM3FlagModel
from tqdm import tqdm

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CHUNK_DIR = PROJECT_ROOT / "03_chunking" / "output"
QDRANT_PATH = PROJECT_ROOT / "04_embeddings_and_kg" / "db" / "qdrant_local"
COLLECTION_NAME = "db3"

print("="*70)
print("Indexing New Chunks to Qdrant")
print("="*70)

# Load model
print("\n📦 Loading BGE-M3 embedding model...")
model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
print("✓ Model loaded")

# Connect to Qdrant
print("\n🗄️  Connecting to Qdrant...")
client = QdrantClient(path=str(QDRANT_PATH))
print("✓ Connected")

# Find new chunks (the ones from this stage created at timestamps)
print("\n🔍 Finding new chunks...")
all_chunks = sorted(CHUNK_DIR.rglob("*_chunk_*.txt"))
new_chunks = [f for f in all_chunks if 'structured_chunk' in f.name]

print(f"Found {len(new_chunks)} new chunks to index")

if not new_chunks:
    print("✅ No new chunks to index")
    sys.exit(0)

# Read chunks and prepare data
print("\n📖 Reading chunks...")
chunks_data = []
for chunk_file in new_chunks:
    try:
        text = chunk_file.read_text(encoding='utf-8').strip()
        # Get folder name (document name)
        doc_name = chunk_file.parent.name
        chunk_name = chunk_file.stem
        
        chunks_data.append({
            "file": str(chunk_file.relative_to(CHUNK_DIR)),
            "text": text,
            "source": doc_name,
            "chunk_name": chunk_name
        })
    except Exception as e:
        print(f"❌ Error reading {chunk_file}: {e}")
        continue

print(f"✓ Read {len(chunks_data)} chunks")

# Encode chunks
print("\n⚙️  Encoding chunks with BGE-M3...")
try:
    texts = [c["text"] for c in chunks_data]
    encoding_result = model.encode(texts, batch_size=8, max_length=1024)
    dense_embeddings = encoding_result["dense_vecs"]
    print(f"✓ Encoded {len(dense_embeddings)} chunks")
except Exception as e:
    print(f"❌ Encoding failed: {e}")
    sys.exit(1)

# Get next ID from existing collection
print("\n🔢 Getting next point ID...")
collection_info = client.get_collection(COLLECTION_NAME)
max_id = collection_info.points_count - 1 if collection_info.points_count > 0 else -1
next_id = max_id + 1

print(f"Last point ID: {max_id}")
print(f"Next ID: {next_id}")

# Build points
print("\n🔨 Building point objects...")
points = []
for i, (chunk_data, vector) in enumerate(zip(chunks_data, dense_embeddings)):
    point_id = next_id + i
    
    payload = {
        "text": chunk_data["text"],
        "source": chunk_data["source"],
        "chunk_name": chunk_data["chunk_name"],
        "file": chunk_data["file"]
    }
    
    try:
        point = PointStruct(
            id=point_id,
            vector=vector.tolist() if hasattr(vector, 'tolist') else vector,
            payload=payload
        )
        points.append(point)
    except Exception as e:
        print(f"❌ Error creating point {point_id}: {e}")
        continue

print(f"✓ Built {len(points)} points")

# Upsert to Qdrant
print("\n📤 Upserting to Qdrant...")
try:
    batch_size = 100
    for batch_idx in tqdm(range(0, len(points), batch_size), desc="Upserting"):
        batch = points[batch_idx:batch_idx + batch_size]
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=batch
        )
    
    print(f"\n✅ Successfully indexed {len(points)} new chunks!")
    
    # Verify
    updated_info = client.get_collection(COLLECTION_NAME)
    print(f"\n✓ Collection '{COLLECTION_NAME}' now has {updated_info.points_count} total points")
    
except Exception as e:
    print(f"❌ Upsert failed: {e}")
    sys.exit(1)

print("\n" + "="*70)
print("✅ Embeddings indexing complete!")
print("="*70)

client.close()
