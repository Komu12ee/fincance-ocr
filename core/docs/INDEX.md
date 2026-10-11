# CHiPPY Documentation Index

Complete documentation for the CHiPPY Complete High-Performance Pipeline.

## Getting Started

**New to CHiPPY?** Start here:
1. [README.md](../README.md) - Overview and features
2. [QUICKSTART.md](QUICKSTART.md) - 15-minute setup guide
3. [ARCHITECTURE.md](ARCHITECTURE.md) - System design and flow

## Deployment & Operations

- [SETUP_INSTRUCTIONS.md](SETUP_INSTRUCTIONS.md) - Installation, deployment, cloud setup
- Getting Help - Common issues and troubleshooting

## Pipeline Stages Documentation

### Stage 1: Preprocessing (PDF → Text)
- **Location**: `01_preprocessing/`
- **Purpose**: Extract text from PDFs using OCR
- **Entry Point**: `run_stage1.py`, `run_stage2.py`
- **Key Files**:
  - `stage1_image_prep/` - Image processing
  - `stage2_ocr/` - OCR engines
  - `config.py` - Configuration

### Stage 2: Optimization (Text Cleaning)
- **Location**: `02_optimization/`
- **Purpose**: Spell check and normalize text
- **Key Files**:
  - `optimize.py` - Text normalization
  - `spellv2.py` - Spell checking
  - `dict/` - Dictionary resources

### Stage 3: Chunking (Text Segmentation)
- **Location**: `03_chunking/`
- **Purpose**: Intelligently segment documents
- **Key Files**:
  - `docling_chunker.py` - Main chunking engine

### Stage 4: Embeddings & Knowledge Graphs
- **Location**: `04_embeddings_and_kg/scripts/`
- **Purpose**: Generate embeddings and build knowledge graphs
- **Key Files**:
  - `embeddings.py` - Vector embedding generation
  - `build_knowledge_graph.py` - KG construction
  - `rag_pipeline.py` - RAG with Ollama
  - `kg_config.py` - Configuration

### Stage 5: Web UI & Retrieval
- **Location**: `05_webui/`
- **Purpose**: Interactive query interface
- **Key Files**:
  - `app.py` - Flask application
  - `static/` - Frontend assets
  - `templates/` - HTML templates

## Configuration Reference

| Stage | Config File | Key Settings |
|-------|-----------|--------------|
| 1 | `01_preprocessing/stage1_image_prep/config.py` | DPI, OCR backend, image processing |
| 1 | `01_preprocessing/stage2_ocr/config.py` | OCR options, endpoints |
| 2 | `02_optimization/optimize.py` | Normalization rules |
| 2 | `02_optimization/spellv2.py` | Dictionary selection, algorithm |
| 4 | `04_embeddings_and_kg/scripts/kg_config.py` | Model, batch size, vector DB |
| 5 | `05_webui/app.py` | Port, debug mode, limits |

## Common Tasks

### Run Preprocessing
```bash
cd 01_preprocessing
python run_stage1.py
python run_stage2.py
```

### Build Knowledge Graph
```bash
cd 04_embeddings_and_kg/scripts
python build_knowledge_graph.py
python embeddings.py
```

### Launch Web UI
```bash
cd 05_webui
python app.py
# Open http://localhost:5000
```

### Run Full Pipeline
See SETUP_INSTRUCTIONS.md → Multi-Stage Pipeline Automation

## Performance Tuning

- **Faster Processing**: Reduce image DPI to 100-150
- **Better Quality**: Increase OCR backends, use GPU
- **Lower Memory**: Reduce batch sizes in configs
- **Faster Search**: Index smaller chunks

Detailed guidance in ARCHITECTURE.md → Performance Characteristics

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Tesseract not found | Install from https://github.com/UB-Mannheim/tesseract/wiki |
| CUDA out of memory | Reduce batch size, disable GPU |
| Port already in use | Kill existing process or use different port |
| spaCy model missing | `python -m spacy download en_core_web_sm` |

For more, see QUICKSTART.md → Common Issues & Solutions

## API Reference

### RAG Pipeline
```python
from 04_embeddings_and_kg.scripts.rag_pipeline import RAGPipeline

pipeline = RAGPipeline()
results = pipeline.query("Your question here")
```

### Chunking
```python
from 03_chunking.docling_chunker import Chunker

chunker = Chunker()
chunks = chunker.chunk_document(text)
```

### Embeddings
```python
from 04_embeddings_and_kg.scripts.embeddings import EmbeddingGenerator

embedder = EmbeddingGenerator()
vectors = embedder.encode(chunks)
```

## FAQs

**Q: Can I use a different LLM instead of Ollama?**
A: Yes, modify `rag_pipeline.py` to use your LLM endpoint.

**Q: What's the minimum system requirement?**
A: Python 3.9+, 4GB RAM, CPU is fine (GPU recommended).

**Q: How long does processing take?**
A: ~5-10 minutes for a 100-page PDF (CPU), ~1-2 min with GPU.

**Q: Can I run stages in parallel?**
A: Currently sequential. Parallelization possible with modifications.

**Q: How do I add custom dictionaries?**
A: Add .txt files to `02_optimization/dict/` and update `spellv2.py`.

## Development

### Contributing
See CONTRIBUTING.md (if exists)

### Unit Tests
```bash
pytest tests/  # if tests/ directory exists
```

### Code Quality
```bash
black --check .
flake8 .
mypy .
```

## Resources

- [Main README](../README.md)
- [Architecture Deep Dive](ARCHITECTURE.md)
- [Quick Start](QUICKSTART.md)
- [Installation & Deployment](SETUP_INSTRUCTIONS.md)

## Documentation Update Log

| Date | Changes |
|------|---------|
| 2026-03-17 | Initial documentation |

## Support & Contact

For issues:
1. Check relevant stage documentation
2. Review configuration files (all have comments)
3. Check error logs in each stage directory

---

**Last Updated**: March 2026  
**Documentation Version**: 1.0
