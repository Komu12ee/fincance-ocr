# CHiPPY - Setup and Deployment Instructions

## Deployment Scenarios

### 1. Local Development Setup

```bash
# Clone/navigate to repository
cd CHiPPY

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Run tests
pytest tests/  # if tests directory exists

# Start development
cd 01_preprocessing
python run_stage1.py --help
```

### 2. Docker Deployment

Create `Dockerfile`:
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY . .

RUN apt-get update && apt-get install -y tesseract-ocr
RUN pip install --no-cache-dir -r requirements.txt
RUN python -m spacy download en_core_web_sm

EXPOSE 5000
CMD ["python", "05_webui/app.py"]
```

Build and run:
```bash
docker build -t chippy:latest .
docker run -p 5000:5000 -v $(pwd)/data:/app/data chippy:latest
```

### 3. Server/Production Setup

#### Prerequisites
- Ubuntu 20.04/22.04 or similar
- 4GB+ RAM
- 20GB+ free disk space

#### Installation Steps

```bash
# 1. System dependencies
sudo apt-get update
sudo apt-get install -y python3.11 python3.11-venv python3-pip
sudo apt-get install -y tesseract-ocr build-essential

# 2. Clone and setup
git clone <repo-url> chippy
cd chippy
python3.11 -m venv venv
source venv/bin/activate

# 3. Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# 4. Create required directories
mkdir -p 01_preprocessing/input_pdfs
mkdir -p 01_preprocessing/stage1_output
mkdir -p 01_preprocessing/stage2_output
mkdir -p 04_embeddings_and_kg/data
mkdir -p 04_embeddings_and_kg/db

# 5. Setup as systemd service (optional)
sudo tee /etc/systemd/system/chippy.service > /dev/null <<EOL
[Unit]
Description=CHiPPY RAG Pipeline
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$(pwd)
ExecStart=$(pwd)/venv/bin/python 05_webui/app.py
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
EOL

sudo systemctl enable chippy
sudo systemctl start chippy
```

### 4. Multi-Stage Pipeline Automation

#### Automating the full pipeline:

Create `run_full_pipeline.sh`:
```bash
#!/bin/bash

echo "CHiPPY Full Pipeline Execution"
echo "=============================="

# Configuration
INPUT_PDF="${1:-.}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs/$TIMESTAMP"
mkdir -p "$LOG_DIR"

# Stage 1: Preprocessing
echo "[$(date)] Starting Stage 1: Preprocessing..."
cd 01_preprocessing
python run_stage1.py >> "$LOG_DIR/stage1.log" 2>&1
python run_stage2.py >> "$LOG_DIR/stage2.log" 2>&1
cd ..

# Stage 2: Optimization
echo "[$(date)] Starting Stage 2: Optimization..."
cd 02_optimization
python optimize.py >> "$LOG_DIR/optimize.log" 2>&1
python spellv2.py >> "$LOG_DIR/spellcheck.log" 2>&1
cd ..

# Stage 3: Chunking
echo "[$(date)] Starting Stage 3: Chunking..."
cd 03_chunking
python docling_chunker.py >> "$LOG_DIR/chunking.log" 2>&1
cd ..

# Stage 4: Embeddings & KG
echo "[$(date)] Starting Stage 4: Embeddings & Knowledge Graph..."
cd 04_embeddings_and_kg/scripts
python build_knowledge_graph.py >> "$LOG_DIR/kg.log" 2>&1
python embeddings.py >> "$LOG_DIR/embeddings.log" 2>&1
cd ../..

echo "[$(date)] Pipeline completed! Logs in $LOG_DIR"
```

Usage:
```bash
chmod +x run_full_pipeline.sh
./run_full_pipeline.sh
```

### 5. Cloud Deployment (AWS Example)

#### Using AWS Lambda + S3 + ECS

1. **S3 Bucket Setup**
```bash
aws s3 mb s3://chippy-documents
aws s3 mb s3://chippy-results
```

2. **ECR Repository**
```bash
aws ecr create-repository --repository-name chippy
docker tag chippy:latest <account>.dkr.ecr.<region>.amazonaws.com/chippy:latest
docker push <account>.dkr.ecr.<region>.amazonaws.com/chippy:latest
```

3. **ECS Task Definition** (task-definition.json)
```json
{
  "family": "chippy",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "2048",
  "memory": "4096",
  "containerDefinitions": [
    {
      "name": "chippy",
      "image": "<account>.dkr.ecr.<region>.amazonaws.com/chippy:latest",
      "portMappings": [{"containerPort": 5000}],
      "environment": [
        {"name": "S3_BUCKET", "value": "chippy-documents"},
        {"name": "RESULTS_BUCKET", "value": "chippy-results"}
      ]
    }
  ]
}
```

Deploy:
```bash
aws ecs register-task-definition --cli-input-json file://task-definition.json
```

## Performance Optimization

### CPU Optimization

```python
# In each stage, set:
import os
os.environ['OMP_NUM_THREADS'] = '8'  # Match CPU cores
os.environ['OPENBLAS_NUM_THREADS'] = '8'
```

### Memory Optimization

```python
# Reduce batch sizes in kg_config.py:
BATCH_SIZE = 4          # Reduced from 32
NUM_WORKERS = 1         # Reduced from 4
CACHE_SIZE = '1GB'      # Limit cache
```

### GPU Optimization

```python
# In kg_config.py:
USE_GPU = True
DEVICE = 'cuda:0'
BATCH_SIZE = 64         # Can be larger with GPU
```

## Monitoring & Logging

### Setup Structured Logging

Create `setup_logging.py`:
```python
import logging
import json
from datetime import datetime

def setup_logging(name, log_file=None):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    # JSON formatter
    class JSONFormatter(logging.Formatter):
        def format(self, record):
            return json.dumps({
                'timestamp': datetime.now().isoformat(),
                'level': record.levelname,
                'logger': record.name,
                'message': record.getMessage()
            })
    
    if log_file:
        handler = logging.FileHandler(log_file)
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)
    
    return logger
```

### Monitoring Endpoints

In `05_webui/app.py`, add:
```python
@app.route('/health')
def health():
    return {'status': 'healthy', 'timestamp': datetime.now().isoformat()}

@app.route('/metrics')
def metrics():
    return {
        'documents_processed': count_docs(),
        'vectors_indexed': count_vectors(),
        'memory_usage': get_memory_usage()
    }
```

## Scaling Strategies

### Horizontal Scaling (Multiple Servers)

1. **Shared Vector DB**: Use Qdrant Cloud or self-hosted cluster
2. **Shared Storage**: NFS or S3 for documents
3. **Load Balancer**: nginx in front of multiple web UI instances

### Vertical Scaling (Single Server)

1. Increase GPU memory allocation
2. Increase batch sizes
3. Use larger embedding models
4. Enable multi-processing

## Backup & Recovery

### Backup Strategy

```bash
# Backup knowledge graph and embeddings
tar -czf backup_$(date +%Y%m%d).tar.gz \
    04_embeddings_and_kg/db/ \
    04_embeddings_and_kg/data/

# Upload to remote storage
aws s3 cp backup_$(date +%Y%m%d).tar.gz s3://chippy-backups/
```

### Recovery Procedure

```bash
# Restore from backup
aws s3 cp s3://chippy-backups/backup_<date>.tar.gz .
tar -xzf backup_<date>.tar.gz
```

## Security Setup

### API Security

```python
# In 05_webui/app.py
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

limiter = Limiter(app, key_func=get_remote_address)

@app.route('/query', methods=['POST'])
@limiter.limit("100 per hour")
def query():
    # Rate limited endpoint
    pass
```

### Environment Variables

Create `.env`:
```
FLASK_ENV=production
DATABASE_URL=postgresql://...
API_KEY=<secure-key>
ALLOWED_HOSTS=example.com
```

Load in code:
```python
from dotenv import load_dotenv
import os
load_dotenv()
api_key = os.getenv('API_KEY')
```

## Troubleshooting Deployment

### Port Already in Use
```bash
# Find and kill process
lsof -i :5000
kill -9 <PID>
```

### Out of Memory
```bash
# Monitor memory
watch -n 1 'free -h'

# Reduce batch size and retry
```

### Slow Embeddings
```bash
# Check GPU usage
nvidia-smi

# If not using GPU, enable it in config
```

### Database Connection Issues
```bash
# Check Qdrant status
python -c "from qdrant_client import QdrantClient; QdrantClient(url='http://localhost:6333').get_collections()"
```

## Maintenance Schedule

| Task | Frequency | Time |
|------|-----------|------|
| Update dependencies | Monthly | 30 min |
| Clean old logs | Weekly | 5 min |
| Backup databases | Daily | 10 min |
| Security updates | As needed | 1 hour |
| Performance review | Monthly | 1 hour |

## Upgrade Process

```bash
# 1. Backup current state
./backup.sh

# 2. Update code
git pull origin main

# 3. Update dependencies
pip install --upgrade -r requirements.txt

# 4. Run tests
pytest tests/

# 5. Restart service
sudo systemctl restart chippy
```

---

For additional support, refer to stage-specific documentation or contact the development team.
