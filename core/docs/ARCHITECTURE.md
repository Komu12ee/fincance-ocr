# CHiPPY Architecture Documentation

## System Overview

CHiPPY is a modular, sequential RAG pipeline where each stage processes documents and passes output to the next stage. This document describes the architecture, data flow, and technical specifications.

## Architecture Diagram

```
┌─────────────────────────┐
│   PDF INPUT FILES       │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│ [01] PREPROCESSING (chips-rag)      │
│ ┌───────────────────────────────┐   │
│ │ A. PDF to Images             │   │
│ │ B. Image Preprocessing       │   │
│ │    - Denoise                 │   │
│ │    - Deskew                  │   │
│ │    - Stamp Detection         │   │
│ │ C. Multiple OCR Backends     │   │
│ │    - Tesseract               │   │
│ │    - EasyOCR                 │   │
│ │    - RapidOCR                │   │
│ └───────────────────────────────┘   │
│         Output: OCR Text (.txt)     │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│ [02] OPTIMIZATION                   │
│ ┌───────────────────────────────┐   │
│ │ A. Spell Checking (spellv2.py)│  │
│ │ B. Text Normalization         │   │
│ │    (optimize.py)              │   │
│ │ C. Dictionary-based Fixes     │   │
│ │    - English dictionary       │   │
│ │    - Domain-specific terms    │   │
│ └───────────────────────────────┘   │
│     Output: Cleaned Text         │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│ [03] CHUNKING                       │
│ ┌───────────────────────────────┐   │
│ │ A. Intelligent Segmentation   │   │
│ │    (docling_chunker.py)       │   │
│ │ B. Metadata Preservation      │   │
│ │    - Document structure       │   │
│ │    - Reference tracking       │   │
│ │ C. Overlap Handling           │   │
│ └───────────────────────────────┘   │
│    Output: Structured Chunks   │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│ [04] EMBEDDINGS & KNOWLEDGE GRAPHS  │
│ ┌───────────────────────────────┐   │
│ │ A. Generate Vector Embeddings │   │
│ │    (FlagEmbedding / BGE)      │   │
│ │ B. Build Knowledge Graph      │   │
│ │    - Entity Extraction        │   │
│ │    - Relationship Mapping     │   │
│ │ C. Index in Vector DB         │   │
│ │    (Qdrant)                   │   │
│ └───────────────────────────────┘   │
│  Output: Indexed Vectors & KG   │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│ [05] WEB UI & RETRIEVAL             │
│ ┌───────────────────────────────┐   │
│ │ A. Flask Web Application      │   │
│ │ B. Query Processing           │   │
│ │    (rag_pipeline.py)          │   │
│ │ C. Hybrid Search              │   │
│ │    - Vector search            │   │
│ │    - Sparse search (keyword)  │   │
│ │ D. LLM Integration            │   │
│ │    (Ollama)                   │   │
│ │ E. Result Ranking             │   │
│ └───────────────────────────────┘   │
│      Output: User Interface     │
└─────────────────────────────────────┘
```

## Stage-by-Stage Details

### Stage 1: Preprocessing (chips-rag)

**Purpose**: Convert PDFs to searchable text with minimal information loss

**Key Components**:
- `run_stage1.py` - Entry point, orchestrates Stage 1
- `stage1_image_prep/` - Image processing pipeline
  - `pdf_to_image.py` - PDF → Image conversion
  - `denoise.py` - Noise reduction filters
  - `deskew.py` - Text alignment correction
  - `stamp_detector.py` - Overlaid text removal
- `stage2_ocr/` - OCR implementations
  - `docling_ocr.py` - Docling backend
  - Multiple OCR backends through `models.py`

**Configuration**: `stage1_image_prep/config.py` and `stage2_ocr/config.py`

**Input**: PDF files in `01_preprocessing/input_pdfs/`
**Output**: 
- Raw text: `01_preprocessing/stage1_output/*/`
- Processed text: `01_preprocessing/stage2_output/*/`

**Key Parameters**:
```python
# Preprocessing thresholds
DENOISE_STRENGTH = 1.5      # 0.5-3.0
DESKEW_CONFIDENCE = 0.7     # 0.5-1.0
OCR_BACKEND = 'tesseract'   # or 'easyocr', 'rapidocr'
```

### Stage 2: Optimization

**Purpose**: Clean and normalize extracted text

**Key Components**:
- `optimize.py` - Text normalization
  - Line ending fixes
  - Whitespace normalization
  - Character encoding fixes
- `spellv2.py` - Spell correction
  - Dictionary-based matching
  - Context-aware corrections
  - Support for domain-specific terms
- `dict/` - Dictionary resources
  - `hi_dict_2_updated.txt` - Main dictionary
  - `office.txt` - Office/Business terms

**Configuration**: Set in each Python file as constants

**Input**: OCR output from Stage 1
**Output**: Optimized text ready for chunking

**Performance**:
- Spell check: ~5-10ms per 1000 chars
- Normalization: ~1-2ms per 1000 chars

### Stage 3: Chunking

**Purpose**: Segment text into semantically meaningful chunks

**Key Component**:
- `docling_chunker.py` - Intelligent chunker
  - Respects document structure
  - Maintains sentence integrity
  - Preserves reference information

**Parameters**:
```python
MAX_CHUNK_SIZE = 512        # tokens
MIN_CHUNK_SIZE = 100        # tokens
OVERLAP = 50                # tokens
```

**Input**: Optimized text from Stage 2
**Output**: 
- Structured chunks with metadata
- Document structure preserved
- Reference tracking enabled

### Stage 4: Embeddings & Knowledge Graphs

**Purpose**: Create searchable vector indices and knowledge relationship maps

**Key Components in `scripts/`:
- `embeddings.py` - Vector embedding generation
  - Uses FlagEmbedding (BGE model)
  - Batch processing capable
  - GPU-accelerated (when available)
- `build_knowledge_graph.py` - KG construction
  - Entity extraction (spaCy)
  - Relationship identification
  - Graph serialization
- `kg_retriever.py` - KG-based retrieval
- `entity_extraction.py` - Named entity identification
- `integrated_retrieval.py` - Combined search layer
- `rag_pipeline.py` - RAG orchestration with Ollama
- `quickstart.py` - Quick setup and testing

**Configuration**: `kg_config.py`

**Data Storage**:
- Vector DB: Qdrant (in `db/` directory)
- KG JSON: `knowledge_graph.json`
- Embeddings cache: `data/`

**Example Configuration**:
```python
# kg_config.py
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
EMBEDDING_DIM = 768
VECTOR_DB_HOST = "localhost"
VECTOR_DB_PORT = 6333
CHUNK_SIZE = 512
```

**Input**: Chunks from Stage 3
**Output**:
- Indexed vectors in Qdrant
- Knowledge graph JSON
- Embeddings metadata

### Stage 5: Web UI & Retrieval

**Purpose**: User-facing interface for querying the knowledge base

**Components**:
- `app.py` - Flask web application
  - REST API endpoints
  - Session management
  - Result formatting
- `static/` - Frontend assets
  - `style.css` - Styling
  - `script.js` - Client-side logic
- `templates/` - HTML templates
  - `index.html` - Query interface

**Features**:
- Query input interface
- Result highlighting
- Source document tracking
- Search history

**Integration Points**:
- Connects to Stage 4 indices
- Optional Ollama integration for LLM responses
- Hybrid search (vector + keyword)

## Data Flow

```
Stage 1 Output Structure:
├── stage1_output/          (Raw OCR)
│   ├── document_01/
│   │   ├── page_01.txt
│   │   ├── page_02.txt
│   │   └── ...
│   └── ...
├── stage2_output/          (Enhanced OCR)
│   ├── document_01/
│   │   └── combined.txt
│   └── ...

Stage 2 Output Structure:
├── optimized/
│   ├── document_01_optimized.txt
│   └── ...

Stage 3 Output Structure:
├── chunks/
│   ├── document_01_chunks.json
│   └── ...

Stage 4 Output Structure:
├── vectors/                (Qdrant DB)
├── embeddings/             (Cached embeddings)
├── knowledge_graph.json    (KG data)

Stage 5 Output Structure:
└── Via Web UI only
```

## Configuration Management

### Environment-Specific Configs

**Development**:
```python
DEBUG = True
OCR_BACKENDS = ['tesseract', 'easyocr', 'rapidocr']  # Try all
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
```

**Production**:
```python
DEBUG = False
OCR_BACKENDS = ['tesseract']  # Single backend
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"  # Optimized model
BATCH_SIZE = 32
NUM_WORKERS = 4
```

### Logging

Configure in each stage's main script:
```python
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

## Performance Characteristics

| Stage | Input Size | Time | Memory | Notes |
|-------|-----------|------|--------|-------|
| Stage 1 | 100 pages | 30-60s | 2-4 GB | Variable by image quality |
| Stage 2 | 100k tokens | 10-20s | 1-2 GB | Linear to text size |
| Stage 3 | 100k tokens | 5-10s | 1 GB | Memory efficient |
| Stage 4 | 200 chunks | 20-40s | 3-6 GB | GPU: 5-10s, CPU: 20-40s |
| Stage 5 | Query | <1s | 200-500 MB | Query response time |

## Extension Points

### Adding a Custom OCR Backend
1. Create file: `01_preprocessing/stage2_ocr/custom_ocr.py`
2. Implement `NativeOCRProcessor` interface
3. Register in `models.py`
4. Update config to use it

### Custom Text Processor
1. Add preprocessing step in `02_optimization/`
2. Call from optimize.py pipeline
3. Update requirements if needed

### Custom Chunking Strategy
1. Extend `docling_chunker.py`
2. Implement alternative chunking logic
3. Return StandardChunk objects

### KG Extensions
1. Modify entity types in `kg_config.py`
2. Extend `entity_extraction.py`
3. Update `knowledge_graph.py` schema

## Error Handling

**Critical Failures**:
- Missing PDF: Skip and log
- OCR timeout: Fall back to alternative backend
- Out of memory: Switch to streaming mode
- DB connection failure: Queue for retry

**Recovery Mechanisms**:
- Automatic retry with exponential backoff
- Fallback backends for OCR
- Checkpoint system for long-running jobs
- Graceful degradation for missing features

## Security Considerations

- Input validation at each stage
- No sensitive data in logs
- Document access control via UI
- CORS policies for API endpoints
- Rate limiting on query interface

---

For more details, see stage-specific documentation files within each directory.
