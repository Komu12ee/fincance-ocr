# CHiPPY Modular Configuration Guide

Complete guide to CHiPPY's dynamic, modular path and configuration system.

## Overview

CHiPPY now features a centralized configuration system (`config_manager.py`) that allows you to:

- ✅ **Auto-detect** CHiPPY base directory
- ✅ **Override paths** via command-line arguments
- ✅ **Configure via environment variables** (.env files)
- ✅ **Load from config files** (JSON)
- ✅ **Chain stages** automatically with intelligent I/O defaults
- ✅ **No hardcoded paths** in scripts

## Configuration Hierarchy

Configurations are applied in this order (later overrides earlier):

```
1. Config defaults (in config_manager.py)
   ↓
2. Environment variables (CHIPPY_*_INPUT, CHIPPY_*_OUTPUT)
   ↓
3. Configuration file (config.json, .env)
   ↓
4. Command-line arguments
```

## Quick Start: Using Each Stage

### Stage 1: Preprocessing

**Default behavior (auto-paths):**
```bash
cd 01_preprocessing
python run_stage1.py
# Looks for: input_pdfs/ → outputs to: stage1_output/
```

**Specify custom paths:**
```bash
python run_stage1.py /path/to/pdfs -o /path/to/output
```

**Show current configuration:**
```bash
python run_stage1.py --show-config
```

**Load from config file:**
```bash
python run_stage1.py --config ../config.example.json
```

### Stage 3: Chunking

**Default behavior:**
```bash
cd 03_chunking
python docling_chunker.py
# Looks for: ../02_optimization/output → outputs to: output/
```

**Process specific file:**
```bash
python docling_chunker.py --input document.md -o ./chunks
```

**With file mapping:**
```bash
python docling_chunker.py --mapping ../files.txt
```

**Show configuration:**
```bash
python docling_chunker.py --show-config
```

## Configuration Methods

### Method 1: Command-Line Arguments (Easiest)

```bash
# Override input
python script.py --input /new/path

# Override output
python script.py --output /results/

# Use config file
python script.py --config myconfig.json

# Show current paths
python script.py --show-config
```

**Stage-specific examples:**
```bash
# Preprocessing
python run_stage1.py /path/to/pdfs -o ./processed

# Chunking
python docling_chunker.py --input ./docs --output ./chunks --pattern "*.md"

# Embeddings
python build_knowledge_graph.py --input ./data --model "BAAI/bge-large-en-v1.5"
```

### Method 2: Environment Variables (.env file)

**Create `.env` in CHiPPY root:**
```bash
# Copy the example
cp .env.example .env
```

**Edit `.env` with your paths:**
```env
# Stage 1
CHIPPY_PREPROCESSING_INPUT=/path/to/pdfs
CHIPPY_PREPROCESSING_OUTPUT=/path/to/output

# Stage 3
CHIPPY_CHUNKING_INPUT=/optimized/docs
CHIPPY_CHUNKING_OUTPUT=/chunks/

# Stage 4
CHIPPY_EMBEDDINGS_INPUT=/data/
EMBEDDINGS_MODEL=BAAI/bge-large-en-v1.5
```

**Scripts automatically load from `.env`:**
```bash
cd CHiPPY
python 01_preprocessing/run_stage1.py
# Uses paths from .env
```

### Method 3: Configuration Files (JSON)

**Create config file (e.g., `production.json`):**
```json
{
  "base_dir": "/path/to/chippy",
  "stages": {
    "preprocessing": {
      "input_dir": "/data/pdfs",
      "output_dir": "/data/processed",
      "settings": {
        "dpi": 200,
        "ocr_backend": "easyocr"
      }
    },
    "chunking": {
      "input_dir": "/data/optimized",
      "output_dir": "/data/chunks"
    }
  }
}
```

**Use it:**
```bash
python stage.py --config production.json
```

### Method 4: Programmatic Usage (In Python Scripts)

```python
from config_manager import Config, PathManager

# Create config for a stage
config = Config(stage='chunking')

# Get paths
input_path = config.get_input_path()
output_path = config.get_output_path()
print(f"Input: {input_path}")
print(f"Output: {output_path}")

# Create directories
input_dir, output_dir = config.create_directories()

# Get list of files
files = PathManager.get_files(input_path, pattern='*.md')

# Display configuration
config.log_config()
```

## Directory Auto-Detection

CHiPPY automatically finds its base directory by:

1. **Checking environment variable**
   ```bash
   export CHIPPY_BASE_DIR=/path/to/CHiPPY
   ```

2. **Searching parent directories** for marker files (README.md, 01_preprocessing/)

3. **Using current directory** as fallback

**Force a specific base directory:**
```python
config = Config(stage='preprocessing', base_dir='/custom/path/CHiPPY')
```

## Stage-to-Stage Path Chaining

Stages automatically chain input/output paths:

```
[Stage 1: Preprocessing]
├── Input: 01_preprocessing/input_pdfs/
└── Output: 01_preprocessing/stage1_output/
           (and stage2_output/ after run_stage2.py)
           ↓
[Stage 2: Optimization]
├── Input: 01_preprocessing/stage2_output/  ← Automatic!
└── Output: 02_optimization/output/
           ↓
[Stage 3: Chunking]
├── Input: 02_optimization/output/  ← Automatic!
└── Output: 03_chunking/output/
           ↓
[Stage 4: Embeddings]
├── Input: 04_embeddings_and_kg/data/
└── Output: 04_embeddings_and_kg/db/
```

No manual path management needed!

## Advanced: Custom Path Resolution

### Override Just Input
```bash
python docling_chunker.py --input /custom/input --output-dir auto
```

### Override Just Output
```bash
python build_knowledge_graph.py --output ./my_vectors
```

### Relative to CHiPPY Base
```bash
# These are resolved relative to CHiPPY root
CHIPPY_CHUNKING_INPUT=../my_optimized_docs
CHIPPY_CHUNKING_OUTPUT=./my_chunks
```

## Configuration Priority Example

Given this setup:

```bash
# In .env
CHIPPY_CHUNKING_INPUT=/from/env

# Command line
python docling_chunker.py --input /from/cli
```

**Result**: Uses `/from/cli` (command-line wins)

---

Given this setup:

```bash
# .env
CHIPPY_CHUNKING_INPUT=/from/env

# No command-line override
python docling_chunker.py
```

**Result**: Uses `/from/env` (environment wins)

---

Given this setup:

```bash
# No .env, no command-line
python docling_chunker.py
```

**Result**: Uses default `02_optimization/output` (default auto-path)

## Troubleshooting Configuration

### Check current configuration:
```bash
python01_preprocessing/run_stage1.py --show-config
python 03_chunking/docling_chunker.py --show-config
```

### Verify paths exist:
```python
from config_manager import Config
config = Config(stage='chunking')
print(f"Input exists: {config.validate_input_exists()}")
```

### Debug path resolution:
```python
from config_manager import Config
config = Config(stage='chunking')
config.log_config()  # Prints: base_dir, stage_dir, input, output, etc
```

### Force base directory detection:
```bash
# If auto-detection fails, explicitly set:
export CHIPPY_BASE_DIR=/correct/path
python script.py
```

## Common Scenarios

### Scenario 1: All in one directory

```bash
mkdir -p /data/chippy/{inputs,outputs}

# Environment setup
CHIPPY_PREPROCESSING_INPUT=/data/chippy/inputs/pdfs
CHIPPY_PREPROCESSING_OUTPUT=/data/chippy/outputs/stage1

CHIPPY_OPTIMIZATION_INPUT=/data/chippy/outputs/stage1
CHIPPY_OPTIMIZATION_OUTPUT=/data/chippy/outputs/stage2

# Run stages
python 01_preprocessing/run_stage1.py
```

### Scenario 2: Network/Cloud paths

```bash
# For S3
CHIPPY_PREPROCESSING_INPUT=s3://bucket/inputs/
CHIPPY_PREPROCESSING_OUTPUT=s3://bucket/processed/

# For NFS
CHIPPY_PREPROCESSING_INPUT=/mnt/nfs/inputs/
CHIPPY_PREPROCESSING_OUTPUT=/mnt/nfs/outputs/
```

### Scenario 3: Per-project configurations

Create separate configs:

```bash
# project_a_config.json
{
  "preprocessing": {
    "input_dir": "/projects/a/pdfs",
    "output_dir": "/projects/a/output"
  }
}

# project_b_config.json
{
  "preprocessing": {
    "input_dir": "/projects/b/pdfs",
    "output_dir": "/projects/b/output"
  }
}

# Run for different projects
python 01_preprocessing/run_stage1.py --config project_a_config.json
python 01_preprocessing/run_stage1.py --config project_b_config.json
```

### Scenario 4: Development vs Production

**development.env:**
```env
ENVIRONMENT=development
DEBUG=true
CHIPPY_PREPROCESSING_INPUT=./test_data/pdfs
CHIPPY_PREPROCESSING_OUTPUT=./test_data/output
EMBEDDINGS_USE_GPU=false
```

**production.env:**
```env
ENVIRONMENT=production
DEBUG=false
CHIPPY_PREPROCESSING_INPUT=/data/production/pdfs
CHIPPY_PREPROCESSING_OUTPUT=/data/production/output
EMBEDDINGS_USE_GPU=true
```

```bash
# For development
cp development.env .env
python 01_preprocessing/run_stage1.py

# For production
cp production.env .env
python 01_preprocessing/run_stage1.py
```

## API Reference

### Config Class

```python
from config_manager import Config

# Create config
config = Config(stage='chunking')

# Get paths
input_path = config.get_input_path()        # Returns Path object
input_str = config.get_input_path(as_str=True)  # Returns string

output_path = config.get_output_path()
stage_dir = config.get_stage_dir()

# Create directories
input_dir, output_dir = config.create_directories()

# Validation
exists = config.validate_input_exists()

# Get data files
data_file = config.get_data_path(filename='myfile.txt')

# Export configuration
config_dict = config.to_dict()

# Log for debugging
config.log_config()
```

### PathManager Class

```python
from config_manager import PathManager

# Create directories
PathManager.ensure_dirs('/path1', '/path2', '/path3')

# Get files
files = PathManager.get_files('/data', pattern='*.md')
files_recursive = PathManager.get_files('/data', pattern='**/*.md', recursive=True)

# Get stem (filename without extension)
name = PathManager.get_stem('document.pdf')  # Returns 'document'

# Get files for a stage
files = PathManager.get_files_by_stage(config, '.md')
```

## Environment Variables Reference

All CHiPPY environment variables:

```bash
# Base configuration
CHIPPY_BASE_DIR=                    # Override CHiPPY root directory

# Stage-specific I/O
CHIPPY_PREPROCESSING_INPUT=
CHIPPY_PREPROCESSING_OUTPUT=
CHIPPY_OPTIMIZATION_INPUT=
CHIPPY_OPTIMIZATION_OUTPUT=
CHIPPY_CHUNKING_INPUT=
CHIPPY_CHUNKING_OUTPUT=
CHIPPY_EMBEDDINGS_INPUT=
CHIPPY_EMBEDDINGS_OUTPUT=

# Feature settings
EMBEDDINGS_USE_GPU=false
EMBEDDINGS_MODEL=BAAI/bge-m3
FLASK_ENV=development
OLLAMA_API_URL=http://localhost:11434
```

## Best Practices

✅ **DO**:
- Use environment variables for production deployments
- Create separate `.env` files per environment
- Use `--show-config` to verify paths before processing
- Store configuration files in version control (without secrets)

❌ **DON'T**:
- Hardcode paths in scripts (use Config instead)
- Commit `.env` files with sensitive data
- Mix configuration methods (pick one for simplicity)
- Forget to create output directories

## Migration Guide

If you have existing scripts with hardcoded paths:

**Before:**
```python
input_dir = r"D:\Code\temp\paddleocr_test\out\clean"
output_dir = r"D:\Code\temp\chunking\output4"

# Process files...
```

**After:**
```python
from config_manager import Config

config = Config(stage='chunking')
input_dir = config.get_input_path(as_str=True)
output_dir = config.get_output_path(as_str=True)

# Process files... (same code)
```

## Support & Debugging

**Check if environment variables are loaded:**
```bash
python -c "import os; print(os.getenv('CHIPPY_BASE_DIR'))"
```

**Test config loading:**
```bash
python -c "from config_manager import Config; c = Config(stage='chunking'); c.log_config()"
```

**Validate paths in Python:**
```python
from config_manager import Config
config = Config(stage='chunking')
print(f"Input: {config.get_input_path()}", config.validate_input_exists())
```

---

**Configuration System Version**: 1.0  
**Last Updated**: March 2026  
**Compatible Stages**: All (01-05)
