# CHiPPY Quick Start Guide

Get CHiPPY up and running in 15 minutes!

## Prerequisites

- **Python 3.9+** (tested on 3.9, 3.10, 3.11)
- **pip** or **conda**
- **Git** (optional, for cloning)
- **4 GB RAM minimum** (8 GB+ recommended)
- **GPU optional** (NVIDIA with CUDA for faster embeddings)

## System Requirements by OS

### Windows 10/11
- Visual C++ Build Tools (for some dependencies)
- Tesseract OCR (optional but recommended)

### macOS
- Command Line Tools: `xcode-select --install`
- Tesseract: `brew install tesseract`

### Linux (Ubuntu/Debian)
```bash
sudo apt-get update
sudo apt-get install tesseract-ocr python3-dev build-essential
```

## Installation (5 steps)

### Step 1: Navigate to CHiPPY
```bash
cd CHiPPY
```

### Step 2: Create Virtual Environment
```bash
# Using venv
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt

# Download spaCy model (required for NLP)
python -m spacy download en_core_web_sm
```

**Installation time**: Typically 5-10 minutes depending on internet speed.

### Step 4: Verify Installation
```bash
python -c "import spacy; print('spaCy OK')"
python -c "import cv2; print('OpenCV OK')"
python -c "import qdrant_client; print('Qdrant OK')"
```

### Step 5: (Optional) Install Ollama
For LLM-powered responses in queries:
1. Download from https://ollama.ai
2. Run: `ollama serve`
3. In another terminal: `ollama pull mistral` (or another model)

## First Run (10 minutes)

### Option A: Minimal Test

1. **Add a test PDF**
   - Place any PDF in `01_preprocessing/input_pdfs/`
   - Or use a downloaded sample

2. **Run preprocessing**
   ```bash
   cd 01_preprocessing
   python run_stage1.py
   ```
   - Should process and output extracted text
   - Check `stage1_output/` for results

3. **Check Stage 2 (advanced OCR)**
   ```bash
   python run_stage2.py
   ```
   - Produces enhanced OCR output in `stage2_output/`

### Option B: Full Pipeline Test

```bash
# From CHiPPY root directory

# 1. Preprocessing (30-60 sec for 1 small PDF)
cd 01_preprocessing && python run_stage1.py && python run_stage2.py && cd ..

# 2. Optimization (5-10 sec)
cd 02_optimization && python optimize.py && python spellv2.py && cd ..

# 3. Chunking (2-5 sec)
cd 03_chunking && python docling_chunker.py && cd ..

# 4. Embeddings & KG (10-20 sec)
cd 04_embeddings_and_kg/scripts
python build_knowledge_graph.py
python embeddings.py
cd ../..

# 5. Start Web UI
cd 05_webui
python app.py
```

Then open your browser to: **http://localhost:5000**

## Common Issues & Solutions

### Issue: "No module named 'spacy'"
```bash
pip install spacy
python -m spacy download en_core_web_sm
```

### Issue: "Tesseract not found"
**Windows**: Download installer from https://github.com/UB-Mannheim/tesseract/wiki
**macOS**: `brew install tesseract`
**Linux**: `sudo apt-get install tesseract-ocr`

### Issue: "CUDA out of memory"
Reduce batch size or use CPU:
```python
# In 04_embeddings_and_kg/scripts/kg_config.py
USE_GPU = False
BATCH_SIZE = 8  # Reduce from default
```

### Issue: "Port 5000 already in use"
Either:
- Kill existing process: `taskkill /F /IM python.exe` (Windows)
- Use different port: `python app.py --port 5001`

### Issue: "PermissionError" on macOS/Linux
```bash
chmod +x 01_preprocessing/run_stage1.py
chmod +x 01_preprocessing/run_stage2.py
```

## Project Structure Overview

```
CHiPPY/
├── 01_preprocessing/    ← Start here: PDF → Text
├── 02_optimization/     ← Clean up text
├── 03_chunking/         ← Split into chunks
├── 04_embeddings_and_kg/ ← Create indices
├── 05_webui/            ← Interactive interface
├── requirements.txt     ← All dependencies
└── docs/               ← Detailed documentation
```

## Data Flow Example

```
Input:  my_document.pdf
   ↓
[01] preprocessing → my_document.txt
   ↓
[02] optimization → my_document_clean.txt
   ↓
[03] chunking → 25 chunks
   ↓
[04] embeddings → vectors indexed, KG built
   ↓
[05] webui → Ready for queries!
```

## Configuration Quick Reference

### Stage 1: Preprocessing
**File**: `01_preprocessing/stage1_image_prep/config.py`
```python
DPI = 150                    # Image quality (100-300)
OCR_BACKEND = 'tesseract'   # 'easyocr' or 'rapidocr'
```

### Stage 2: Optimization
**File**: `02_optimization/optimize.py`
```python
NORMALIZE_WHITESPACE = True
FIX_ENCODING = True
```

### Stage 4: Embeddings
**File**: `04_embeddings_and_kg/scripts/kg_config.py`
```python
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
CHUNK_SIZE = 512
EMBEDDING_DIM = 768
```

### Stage 5: Web UI
**File**: `05_webui/app.py` (check Flask config)
```python
DEBUG = True           # False for production
PORT = 5000
MAX_UPLOAD_SIZE = 50   # MB
```

## Testing Your Installation

Run this Python script to verify everything:

```python
# test_installation.py
import sys

tests = []

def test(name):
    def decorator(func):
        def wrapper():
            try:
                func()
                tests.append((name, "✓ PASS"))
            except Exception as e:
                tests.append((name, f"✗ FAIL: {e}"))
        return wrapper
    return decorator

@test("spaCy")
def test_spacy():
    import spacy
    spacy.load("en_core_web_sm")

@test("OpenCV")
def test_cv2():
    import cv2
    assert cv2.__version__

@test("PyTorch")
def test_torch():
    import torch
    return torch.cuda.is_available()

@test("Qdrant")
def test_qdrant():
    from qdrant_client import QdrantClient
    return QdrantClient(":memory:")

@test("FlagEmbedding")
def test_embeddings():
    from FlagEmbedding import FlagModel
    # Note: This might download model on first run

# Run tests
for name, func in [(n, v) for n, v in locals().items() if callable(v) and n.startswith('test_')]:
    func()

print("\n".join(f"{name}: {status}" for name, status in tests))
```

Run with:
```bash
python test_installation.py
```

## Next Steps

After successful installation:

1. **Read the full docs**: See `docs/ARCHITECTURE.md`
2. **Explore stage configs**: Each stage has commented config files
3. **Try with your documents**: Place PDFs in `01_preprocessing/input_pdfs/`
4. **Query results**: Use the web UI at http://localhost:5000
5. **(Optional) Setup Ollama**: For LLM-powered responses

## Performance Tips

- **Faster OCR**: Use `rapidocr` instead of `tesseract`
- **Faster Embeddings**: Enable GPU in config
- **Parallel Processing**: Use multiple workers in batch operations
- **Production Mode**: Disable DEBUG, increase batch sizes

## Troubleshooting Resources

- **Stage 1**: Check `01_preprocessing/README.md`
- **Stage 4**: Check `04_embeddings_and_kg/scripts/SETUP_GUIDE.md`
- **Web UI**: Check `05_webui/README.md` (if exists)

## Getting Help

1. Check relevant stage documentation
2. Look at error logs in each stage directory
3. Review configuration files with comments
4. Check GitHub issues (if applicable)

## File Size Notes

The complete pipeline handles:
- **Single PDF**: Instant to 5 minutes (depending on pages)
- **100 documents**: 1-4 hours (fully parallel possible)
- **Growing knowledge base**: Add new documents incrementally

## Next: Running Your First Document

When ready:
```bash
# 1. Place PDF in 01_preprocessing/input_pdfs/{filename}.pdf
# 2. From CHiPPY root:
cd 01_preprocessing && python run_stage1.py
# 3. Check output:
cat stage1_output/{filename}/combined.txt
```

---

**Time to first usable system**: ~15 minutes  
**Time for first document**: ~2-5 minutes (depending on size)
