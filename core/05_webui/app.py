"""
RAG Pipeline Web UI
Simple Flask-based interface for document question-answering
"""

import os
import sys
import importlib.util
import json
import time
import threading
import uuid
from flask import Flask, request, jsonify, send_file
from datetime import datetime
from pathlib import Path
from werkzeug.utils import secure_filename

for stream_name in ('stdout', 'stderr'):
    stream = getattr(sys, stream_name, None)
    if hasattr(stream, 'reconfigure'):
        stream.reconfigure(encoding='utf-8', errors='replace')



# Add parent directory to path for imports
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent  # = core/
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / '04_embeddings_and_kg' / 'scripts'))
import pipeline_config as pc

# Dynamically find RAG pipeline module - works for both local and Docker
def _find_rag_module():
    """Find and import RAG pipeline module from various possible locations."""
    # Reuse an already-imported instance (e.g. loaded by the document-indexing
    # path) so there is exactly ONE rag_pipeline module and ONE Qdrant client in
    # the process. Re-executing it would open a second client on the same local
    # store and fail with "already accessed by another instance".
    existing = sys.modules.get('rag_pipeline')
    if existing is not None and hasattr(existing, 'retrieve_context'):
        return existing

    _candidate_paths = [
        PROJECT_ROOT / '04_embeddings_and_kg' / 'scripts' / 'rag_pipeline.py',
    ]

    for path in _candidate_paths:
        path = Path(path)
        if path.exists():
            print(f"[Web UI] Found RAG module at: {path}")
            spec = importlib.util.spec_from_file_location("rag_pipeline", str(path))
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                sys.modules["rag_pipeline"] = module
                spec.loader.exec_module(module)
                return module
    
    raise ImportError("rag_pipeline.py not found in expected locations")


def _build_initialize_pipeline(rag_module):
    """Create a compatibility initializer when rag_pipeline.py does not expose one."""
    def _initialize_pipeline():
        status = {
            'initialized': False,
            'qdrant_connected': False,
            'collection_exists': False,
            'embeddings_loaded': False,
            'error': None,
        }

        client = getattr(rag_module, 'client', None)
        ensure_qdrant_client = getattr(rag_module, 'ensure_qdrant_client', None)
        collection_name = getattr(rag_module, 'COLLECTION_NAME', None)
        if collection_name is None and hasattr(rag_module, 'CFG'):
            collection_name = rag_module.CFG.get('collection')

        try:
            if client is None and callable(ensure_qdrant_client):
                client = ensure_qdrant_client()

            if client is None:
                status['error'] = 'RAG client is not initialized'
                return status

            client.get_collections()
            status['qdrant_connected'] = True
        except Exception as e:
            status['error'] = f'Qdrant connection failed: {e}'
            return status

        try:
            if collection_name and client.collection_exists(collection_name):
                status['collection_exists'] = True
            else:
                status['error'] = f"Collection '{collection_name}' does not exist" if collection_name else 'Collection name unavailable'
                return status
        except Exception as e:
            status['error'] = f'Collection check failed: {e}'
            return status

        status['embeddings_loaded'] = True
        status['initialized'] = True
        return status

    return _initialize_pipeline


def _build_get_db_status(rag_module):
    """Create a compatibility DB status helper when rag_pipeline.py does not expose one."""
    def _get_db_status():
        status = {
            'db_connected': False,
            'collection_exists': False,
            'collection_name': None,
            'points_count': 0,
            'error': None,
        }

        client = getattr(rag_module, 'client', None)
        ensure_qdrant_client = getattr(rag_module, 'ensure_qdrant_client', None)
        collection_name = getattr(rag_module, 'COLLECTION_NAME', None)
        if collection_name is None and hasattr(rag_module, 'CFG'):
            collection_name = rag_module.CFG.get('collection')
        status['collection_name'] = collection_name

        try:
            if client is None and callable(ensure_qdrant_client):
                client = ensure_qdrant_client()

            if client is None:
                status['error'] = 'RAG client is not initialized'
                return status

            client.get_collections()
            status['db_connected'] = True
        except Exception as e:
            status['error'] = f'Cannot connect to Qdrant: {e}'
            return status

        try:
            if collection_name and client.collection_exists(collection_name):
                status['collection_exists'] = True
                collection_info = client.get_collection(collection_name)
                status['points_count'] = collection_info.points_count
            else:
                status['error'] = f"Collection '{collection_name}' not found" if collection_name else 'Collection name unavailable'
        except Exception as e:
            status['error'] = f'Collection check failed: {e}'

        return status

    return _get_db_status


_rag_module = None
retrieve_context = None
generate_answer = None
get_actual_filename = lambda chunk_source: f'{chunk_source}.pdf'
initialize_pipeline = None
get_db_status = None
RAG_AVAILABLE = False
_rag_import_error = None


def _load_rag_module():
    """Import the heavy RAG module on demand so the Flask server can start first."""
    global _rag_module, retrieve_context, generate_answer, get_actual_filename
    global initialize_pipeline, get_db_status, RAG_AVAILABLE, _rag_import_error

    if _rag_module is not None:
        return _rag_module

    try:
        _rag_module = _find_rag_module()

        retrieve_context = _rag_module.retrieve_context
        generate_answer = _rag_module.generate_answer
        get_actual_filename = getattr(_rag_module, 'get_actual_filename', lambda chunk_source: f'{chunk_source}.pdf')
        initialize_pipeline = getattr(_rag_module, 'initialize_pipeline', None) or _build_initialize_pipeline(_rag_module)
        get_db_status = getattr(_rag_module, 'get_db_status', None) or _build_get_db_status(_rag_module)
        RAG_AVAILABLE = True
        _rag_import_error = None
        return _rag_module
    except Exception as e:
        print(f"Warning: Could not import RAG pipeline: {e}")
        _rag_module = None
        retrieve_context = None
        generate_answer = None
        initialize_pipeline = None
        get_db_status = None
        RAG_AVAILABLE = False
        _rag_import_error = str(e)
        return None

# Initialize Flask app
app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100 MB max upload

# Document verification workspace (upload -> OCR -> verify -> gated indexing)
import doc_registry
import doc_pipeline

def _bg(fn, *args):
    threading.Thread(target=fn, args=args, daemon=True).start()

# Store settings
pipeline_initialized = False
qdrant_retry_attempted = False
kg_enabled = True
num_results = 3


def _is_qdrant_connection_failure(error_text):
    """Return True when the error likely indicates Qdrant connectivity failure."""
    if not error_text:
        return False

    msg = str(error_text).lower()
    keywords = [
        "qdrant connection failed",
        "cannot connect to qdrant",
        "connection refused",
        "failed to connect",
    ]
    return any(k in msg for k in keywords)


def _retry_initialize_once_on_qdrant_failure(error_text):
    """Attempt a single re-initialization after a Qdrant connection failure."""
    global qdrant_retry_attempted, pipeline_initialized

    if qdrant_retry_attempted:
        return None
    if not _is_qdrant_connection_failure(error_text):
        return None
    if not RAG_AVAILABLE or initialize_pipeline is None:
        return None

    qdrant_retry_attempted = True
    print("[Web UI] Qdrant connection failed. Retrying pipeline initialization once...")

    retry_result = initialize_pipeline()
    if retry_result.get('initialized'):
        pipeline_initialized = True
        print("[Web UI] One-time retry succeeded.")
    else:
        print(f"[Web UI] One-time retry failed: {retry_result.get('error')}")

    return retry_result

@app.route('/api/health', methods=['GET'])
def health():
    """Check system health"""
    return jsonify({
        'status': 'ok',
        'rag_pipeline': 'available',
        'rag_module_loaded': _rag_module is not None,
        'pipeline_initialized': pipeline_initialized,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/init', methods=['POST'])
def init():
    """Initialize RAG pipeline and verify database connection"""
    global pipeline_initialized
    
    if _load_rag_module() is None or initialize_pipeline is None:
        return jsonify({
            'success': False,
            'error': _rag_import_error or 'RAG pipeline not available. Check imports and configuration.',
            'details': {}
        }), 503
    
    try:
        print("\n[Web UI] Initializing RAG pipeline...")
        init_result = initialize_pipeline()
        
        if init_result.get('initialized'):
            pipeline_initialized = True
            print("[Web UI] RAG pipeline initialization successful")
            return jsonify({
                'success': True,
                'message': 'RAG pipeline initialized successfully',
                'details': init_result,
            }), 200

        retry_result = _retry_initialize_once_on_qdrant_failure(init_result.get('error'))
        if retry_result and retry_result.get('initialized'):
            return jsonify({
                'success': True,
                'message': 'RAG pipeline initialized successfully after one retry',
                'details': retry_result,
                'retried_once': True,
            }), 200

        else:
            print(f"[Web UI] RAG pipeline initialization failed: {init_result.get('error')}")
            return jsonify({
                'success': False,
                'error': init_result.get('error', 'Initialization failed'),
                'details': init_result,
                'retried_once': retry_result is not None,
                'retry_details': retry_result,
            }), 400
            
    except Exception as e:
        print(f"[Web UI] Error during initialization: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': f'Initialization error: {str(e)}',
            'details': {}
        }), 500

@app.route('/api/db-status', methods=['GET'])
def db_status():
    """Check database connection and collection status"""
    
    if _rag_module is None:
        return jsonify({
            'success': True,
            'error': 'RAG module has not been loaded yet',
            'db_connected': False,
            'collection_exists': False,
            'collection_name': None,
            'points_count': 0
        }), 200
    
    try:
        status = get_db_status()

        if not status.get('db_connected', False):
            retry_result = _retry_initialize_once_on_qdrant_failure(status.get('error'))
            if retry_result and retry_result.get('initialized'):
                status = get_db_status()

        return jsonify({
            'success': True,
            'db_connected': status.get('db_connected', False),
            'collection_exists': status.get('collection_exists', False),
            'collection_name': status.get('collection_name'),
            'points_count': status.get('points_count', 0),
            'error': status.get('error'),
            'retried_once': qdrant_retry_attempted,
        }), 200
        
    except Exception as e:
        print(f"[Web UI] Error checking DB status: {e}")
        return jsonify({
            'success': False,
            'error': f'DB status check failed: {str(e)}',
            'db_connected': False,
            'collection_exists': False
        }), 500

def _format_results(context_results, query_text):
    """Shape retrieved chunks for the frontend (PDF name + highlighted excerpt).

    Shared by the single-shot path and thinking mode so both render identically in
    the sources drawer.
    """
    formatted = []
    query_words = [w for w in query_text.lower().split() if len(w) > 3]
    excerpt_fn = getattr(_rag_module, 'extract_highlighted_excerpt', None)

    for result in context_results:
        point = result.get('point', {})
        payload = getattr(point, 'payload', None) or {}
        source = payload.get('source', '')
        text = payload.get('text', '')

        actual_pdf = _rag_module.get_actual_filename(source) if _rag_module else source
        # Readable label + the ORIGINAL source URL. Most documents here are scraped web
        # pages (finance.cg.gov.in) with no local PDF on disk, so the drawer must open
        # the live source instead of fetching used_files/<hash>.pdf (which 404s).
        label, source_url = actual_pdf, None
        if _rag_module and hasattr(_rag_module, 'source_label'):
            try:
                label, source_url = _rag_module.source_label(source)
            except Exception:
                pass
        if excerpt_fn:
            excerpt = excerpt_fn(text, query_words, max_length=250)
        else:
            excerpt = text[:250] + "..." if len(text) > 250 else text

        formatted.append({
            'rank': result.get('rank', 0),
            'source': source,
            'actual_pdf': actual_pdf,
            'label': label,
            'source_url': source_url,
            'score': result.get('score', 0),
            'text': text,
            'excerpt': excerpt,
            'parent_id': result.get('parent_id', ''),
            # Thinking mode pulls in chunks adjacent to a hit for continuity; label
            # them so the drawer can show they are supporting rather than matched.
            'supporting': 'neighbor_of' in result,
        })
    return formatted


# Thinking mode lives next to rag_pipeline and is imported lazily, so a failure to
# load it (experimental module) degrades to normal query instead of 503-ing the app.
_thinking_module = None
_thinking_import_error = None


def _load_thinking_module():
    global _thinking_module, _thinking_import_error
    if _thinking_module is not None or _thinking_import_error is not None:
        return _thinking_module
    try:
        if _load_rag_module() is None:
            raise RuntimeError('RAG pipeline unavailable')
        import thinking_mode                      # same dir as rag_pipeline (already on sys.path)
        _thinking_module = thinking_mode
    except Exception as e:
        _thinking_import_error = f'{type(e).__name__}: {e}'
        print(f'[Web UI] thinking mode unavailable: {_thinking_import_error}')
    return _thinking_module


@app.route('/api/query', methods=['POST'])
def query():
    """Process a query.

    Body: {query, num_results?, thinking?}
    `thinking: true` runs the multi-hop orchestrator and additionally returns a
    `trace` (what it searched, what it found, how precedence was decided).
    """
    query_start_time = time.time()

    if _load_rag_module() is None or retrieve_context is None or generate_answer is None:
        return jsonify({
            'success': False,
            'error': _rag_import_error or 'RAG pipeline not available. Check imports and configuration.',
            'query': ''
        }), 503

    data = request.get_json()
    query_text = data.get('query', '').strip()
    num_context = data.get('num_results', num_results)
    thinking = bool(data.get('thinking'))

    if not query_text:
        return jsonify({
            'success': False,
            'error': 'Query cannot be empty',
            'query': ''
        }), 400

    # ── Thinking mode: multi-hop retrieve/assess, with a reasoning trace ─────
    if thinking:
        tm = _load_thinking_module()
        if tm is None:
            return jsonify({
                'success': False,
                'error': f'Thinking mode unavailable: {_thinking_import_error}',
                'query': query_text, 'results': []
            }), 503
        try:
            print(f"\n🧠 [FLASK] Thinking mode: {query_text}")
            out = tm.deep_query(query_text)
            total_time = time.time() - query_start_time
            print(f"⏱️ [FLASK] Thinking mode total {total_time:.2f}s\n")
            return jsonify({
                'success': True,
                'query': query_text,
                'answer': out['answer'],
                'results': _format_results(out['results'], query_text),
                'result_count': len(out['results']),
                'thinking': True,
                'trace': out['trace'],
                'hops': out['hops'],
                'timeline': out['timeline'],
                'execution_time': f"{total_time:.2f}s",
            }), 200
        except Exception as e:
            print(f"[Web UI] Thinking mode error: {e}")
            import traceback
            traceback.print_exc()
            return jsonify({
                'success': False,
                'error': f'Thinking mode failed: {e}',
                'query': query_text, 'results': []
            }), 500

    try:
        print(f"\n⏱️ [FLASK] Total request start")
        print(f"[Web UI] Processing query: {query_text}")
        
        # Step 1: Retrieve context (parent chunks)
        print(f"[Web UI] Retrieving context...")
        retrieval_start = time.time()
        context_results = retrieve_context(query_text, num_context=num_context)
        retrieval_time = time.time() - retrieval_start
        print(f"⏱️ [FLASK] Retrieval completed in {retrieval_time:.2f}s")
        
        if context_results is None or len(context_results) == 0:
            return jsonify({
                'success': False,
                'error': 'No context documents found for this query',
                'query': query_text,
                'results': []
            }), 200
        
        # Step 2: Generate answer
        print(f"[Web UI] Generating answer...")
        answer_start = time.time()
        answer = generate_answer(query_text, context_results)
        answer_time = time.time() - answer_start
        print(f"⏱️ [FLASK] Answer generation completed in {answer_time:.2f}s")
        
        formatted_results = _format_results(context_results, query_text)

        total_time = time.time() - query_start_time
        print(f"[Web UI] Query processed successfully")
        print(f"⏱️ [FLASK] TOTAL PIPELINE TIME: {total_time:.2f}s\n")
        
        return jsonify({
            'success': True,
            'query': query_text,
            'answer': answer,
            'results': formatted_results,
            'result_count': len(formatted_results),
            'execution_time': f"{total_time:.2f}s"
        }), 200
    
    except Exception as e:
        print(f"[Web UI] Error processing query: {e}")
        import traceback
        traceback.print_exc()
        
        return jsonify({
            'success': False,
            'error': f'Error processing query: {str(e)}',
            'query': query_text,
            'results': []
        }), 500

@app.route('/api/settings', methods=['GET', 'POST'])
def settings():
    """Get or update settings"""
    global kg_enabled, num_results
    
    if request.method == 'GET':
        return jsonify({
            'kg_enabled': kg_enabled,
            'num_results': num_results
        }), 200
    
    elif request.method == 'POST':
        data = request.get_json()
        
        if 'kg_enabled' in data:
            kg_enabled = data['kg_enabled']
        if 'num_results' in data:
            num_results = max(1, min(10, data['num_results']))  # Clamp between 1-10
        
        return jsonify({
            'success': True,
            'kg_enabled': kg_enabled,
            'num_results': num_results
        }), 200

@app.route('/api/examples', methods=['GET'])
def examples():
    """Return example queries"""
    examples_list = [
        "What approval was given in the recent meeting?",
        "Who leads the committee?",
        "What are the key financial decisions?",
        "Summarize the meeting agenda",
        "What are the next action items?",
        "What entities are mentioned in the documents?",
        "What was discussed about budget allocation?",
        "Tell me about the committee members",
    ]
    
    return jsonify({
        'examples': examples_list
    }), 200

@app.route('/01_preprocessing/used_files/<filename>', methods=['GET'])
def serve_pdf(filename):
    """Serve PDF files from used_files directory"""
    import os
    
    # Security: Only allow PDF files and alphanumeric filenames
    if not filename.endswith('.pdf'):
        return jsonify({'error': 'Only PDF files are allowed'}), 403
    
    # Prevent directory traversal attacks
    if '..' in filename or '/' in filename:
        return jsonify({'error': 'Invalid filename'}), 403
    
    pdf_path = pc.paths.used_files / filename

    if not pdf_path.exists():
        return jsonify({'error': f'PDF not found: {filename}'}), 404
    
    try:
        return send_file(str(pdf_path), mimetype='application/pdf')
    except Exception as e:
        print(f"[Web UI] Error serving PDF {filename}: {e}")
        return jsonify({'error': f'Error serving PDF: {str(e)}'}), 500


# ════════════════════════════════════════════════════════════════════
# Document verification workspace
# Upload PDF -> background OCR -> .md -> human verify -> gated RAG indexing.
# Only VERIFIED documents are embedded; editing a verified .md re-indexes it.
# ════════════════════════════════════════════════════════════════════

def _reviewer(source):
    """Extract the reviewer/uploader name from a request body/form."""
    return (source.get('verified_by') or source.get('uploaded_by')
            or source.get('edited_by') or source.get('user') or 'unknown')


def _reconcile_processing():
    """Flip scraped PDFs from 'processing' to 'processed' once their OCR text lands.

    scrape_import registers scraped PDFs immediately as 'processing' (so they show in
    the panel with the source PDF openable, without waiting for the whole OCR pass).
    OCR then writes stage2/<doc_dir>/structured.md but does NOT touch the registry —
    so this app stays the single registry writer and there is no cross-process race
    during the long OCR run. On each panel load we cheaply check the handful of
    'processing' docs and promote any whose OCR output now exists."""
    try:
        for d in doc_registry.all():
            # Only promote FIRST-time OCR of a scraped PDF (md_path still null). A
            # re-process keeps md_path set and manages its own status, so this never
            # races it back to 'processed' with stale text.
            if d.get('status') != 'processing' or not d.get('doc_dir') or d.get('md_path'):
                continue
            md = pc.paths.stage2 / d['doc_dir'] / 'structured.md'
            if not md.exists():
                continue
            fields = {'status': 'processed', 'md_path': str(md),
                      'md_hash': doc_pipeline.md_hash(md.read_text(encoding='utf-8')),
                      'error': None}
            used = pc.paths.used_files / f"{d['doc_dir']}.pdf"
            if used.exists():   # OCR's --archive-source moved the source PDF here
                fields['pdf_path'] = str(used)
            doc_registry.update(d['id'], **fields)
    except Exception as e:  # reconcile is best-effort; never fail the list call
        print(f"[Web UI] processing-reconcile skipped: {e}")


# The panel filters and badges by OCR method, which means every row needs it — but
# _processing_summary reads a JSON sidecar per document, and the registry runs to
# thousands. Memoised on (doc_dir, status): a confidence sidecar is written once by
# stage2 and never changes afterwards, so the only thing that can invalidate an entry
# is the document moving out of 'processing'.
_method_cache = {}


def _method_for(rec):
    """(method, label) for a document — cheap enough to call for every row."""
    key = (rec.get('doc_dir'), rec.get('status'), rec.get('text_source'))
    hit = _method_cache.get(key)
    if hit is None:
        summary = _processing_summary(rec)
        hit = (summary.get('method'), summary.get('label'))
        _method_cache[key] = hit
    return hit


@app.route('/api/documents', methods=['GET'])
def documents_list():
    _reconcile_processing()   # promote scraped PDFs whose OCR has since completed
    docs = []
    for rec in doc_registry.all():
        method, label = _method_for(rec)
        # Surfaced on the row itself so the reviewer can see "this was Sarvam" while
        # triaging, and filter to it, without opening each document first.
        docs.append({**rec, 'ocr_method': method, 'ocr_label': label})
    return jsonify({'success': True, 'documents': docs}), 200


@app.route('/api/documents/<doc_id>/spellcheck', methods=['GET'])
def documents_spellcheck(doc_id):
    """Words in this document's markdown that a reviewer should check.

    Read-only: returns character offsets + ranked replacement candidates. Nothing is
    rewritten here — the reviewer picks a suggestion and the edit goes through the
    existing PUT /md, so every change stays an explicit human action.
    """
    rec = doc_registry.get(doc_id)
    if not rec:
        return jsonify({'success': False, 'error': 'Document not found'}), 404
    md_path = rec.get('md_path')
    if not md_path or not Path(md_path).is_file():
        return jsonify({'success': False, 'error': 'This document has no extracted text yet.'}), 400

    try:
        import spellcheck
    except Exception as e:
        return jsonify({'success': False, 'error': f'Spellcheck unavailable: {e}'}), 503

    try:
        text = Path(md_path).read_text(encoding='utf-8', errors='replace')
        # Page scores let the UI point at the weakest pages; absent for extracted text.
        pages = []
        d = pc.paths.stage2 / (rec.get('doc_dir') or '')
        if d.is_dir():
            for f in d.glob('*_confidence.json'):
                try:
                    pages = (json.loads(f.read_text(encoding='utf-8')) or {}).get('pages') or []
                    break
                except Exception:
                    continue
        result = spellcheck.find_issues(text, page_scores=pages)
        result['success'] = True
        return jsonify(result), 200
    except Exception as e:
        print(f'[Web UI] spellcheck failed for {doc_id}: {e}')
        return jsonify({'success': False, 'error': f'Spellcheck failed: {e}'}), 500


@app.route('/api/documents/upload', methods=['POST'])
def documents_upload():
    f = request.files.get('file')
    if f is None or not f.filename:
        return jsonify({'success': False, 'error': 'A PDF or Markdown file is required'}), 400
    name = f.filename.lower()
    is_md = name.endswith('.md') or name.endswith('.markdown')
    if not (name.endswith('.pdf') or is_md):
        return jsonify({'success': False, 'error': 'Only PDF or Markdown (.md) files are allowed'}), 400

    doc_id = uuid.uuid4().hex

    # ── Markdown upload: the file IS the text, so skip OCR entirely. Land it exactly
    #    where OCR would have written structured.md and register straight to
    #    'processed'. Verifying it in the panel then chunks + embeds it like any other
    #    document — a fast, high-fidelity way to add a curated reference note.
    if is_md:
        import unicodedata
        safe = secure_filename(f.filename) or 'reference.md'
        stem = Path(safe).stem or 'reference'
        doc_dir = f"{stem}_{doc_id[:8]}"
        md_path = pc.paths.stage2 / doc_dir / 'structured.md'
        md_path.parent.mkdir(parents=True, exist_ok=True)
        raw = f.read().decode('utf-8', errors='replace')
        md_text = unicodedata.normalize('NFC', raw).strip() + '\n'
        md_path.write_text(md_text, encoding='utf-8')
        record = {
            'id': doc_id,
            'original_name': f.filename,
            'doc_dir': doc_dir,
            'pdf_path': None,
            'md_path': str(md_path),
            'status': 'processed',            # text is ready; no OCR needed
            'verified': False, 'verified_by': None, 'verified_at': None,
            'uploaded_by': request.form.get('uploaded_by', 'unknown'),
            'uploaded_at': datetime.now().isoformat(),
            'edited_by': None, 'edited_at': None,
            'md_hash': doc_pipeline.md_hash(md_text),
            'indexed': False, 'chunk_count': 0,
            'error': None,
            'source': 'upload',
            'source_type': 'markdown',
            'source_url': None,
            'text_source': 'markdown',
            'doc_kind': 'markdown',
            'raw_ext': 'md',
            'raw_path': None,
        }
        doc_registry.add(record)
        return jsonify({'success': True, 'id': doc_id, 'document': record}), 201

    safe = secure_filename(f.filename) or 'document.pdf'
    stem = Path(safe).stem or 'document'
    doc_dir = f"{stem}_{doc_id[:8]}"

    pc.paths.input_pdfs.mkdir(parents=True, exist_ok=True)
    pdf_path = pc.paths.input_pdfs / f"{doc_dir}.pdf"
    f.save(str(pdf_path))

    record = {
        'id': doc_id,
        'original_name': f.filename,
        'doc_dir': doc_dir,
        'pdf_path': str(pdf_path),
        'md_path': None,
        'status': 'processing',
        'verified': False, 'verified_by': None, 'verified_at': None,
        'uploaded_by': request.form.get('uploaded_by', 'unknown'),
        'uploaded_at': datetime.now().isoformat(),
        'edited_by': None, 'edited_at': None,
        'md_hash': None,
        'indexed': False, 'chunk_count': 0,
        'error': None,
        # Provenance — an upload's source IS the PDF file (viewable via /pdf); it
        # has no source URL. text_source becomes 'ocr' once process_document runs.
        'source': 'upload',
        'source_type': 'pdf',
        'source_url': None,
        'text_source': 'ocr',
        'doc_kind': 'pdf',
        'raw_ext': None,
        'raw_path': None,
    }
    doc_registry.add(record)
    _bg(doc_pipeline.process_document, doc_id)
    return jsonify({'success': True, 'id': doc_id, 'document': record}), 201


@app.route('/api/documents/<doc_id>/pdf', methods=['GET'])
def documents_pdf(doc_id):
    rec = doc_registry.get(doc_id)
    if not rec:
        return jsonify({'error': 'Document not found'}), 404
    # pdf_path can be null (e.g. OCR-imported docs that kept no source PDF) — skip
    # it rather than Path(None), which would raise and 500. Fall back to the
    # conventional on-disk locations before reporting a clean 404.
    candidates = []
    if rec.get('pdf_path'):
        candidates.append(Path(rec['pdf_path']))
    candidates.append(pc.paths.used_files / f"{rec['doc_dir']}.pdf")
    candidates.append(pc.paths.input_pdfs / f"{rec['doc_dir']}.pdf")
    for p in candidates:
        if p.exists():
            return send_file(str(p), mimetype='application/pdf')
    return jsonify({'error': 'No source PDF available for this document'}), 404


_RAW_MIME = {
    'pdf': 'application/pdf',
    'html': 'text/html', 'htm': 'text/html',
    # Serve structured/text snapshots as plain text so they render (not download)
    # inside the review iframe instead of triggering a save dialog.
    'json': 'text/plain; charset=utf-8', 'xml': 'text/plain; charset=utf-8',
    'txt': 'text/plain; charset=utf-8',
}


@app.route('/api/documents/<doc_id>/raw', methods=['GET'])
def documents_raw(doc_id):
    """Serve a scraped document's raw source snapshot (the content that produced
    its markdown). PDFs render in the browser viewer, HTML renders, JSON/XML/text
    display as plain text — so the reviewer sees exactly what was fetched."""
    rec = doc_registry.get(doc_id)
    if not rec:
        return jsonify({'error': 'Document not found'}), 404
    raw_path = rec.get('raw_path')
    if not raw_path or not Path(raw_path).exists():
        return jsonify({'error': 'No raw snapshot for this document'}), 404
    ext = (rec.get('raw_ext') or Path(raw_path).suffix.lstrip('.')).lower()
    return send_file(str(raw_path), mimetype=_RAW_MIME.get(ext, 'text/plain; charset=utf-8'))


@app.route('/api/documents/<doc_id>/md', methods=['GET'])
def documents_md_get(doc_id):
    rec = doc_registry.get(doc_id)
    if not rec:
        return jsonify({'error': 'Document not found'}), 404
    md_path = rec.get('md_path')
    if not md_path or not Path(md_path).exists():
        return jsonify({'success': False, 'status': rec.get('status'),
                        'error': 'Document is still processing or has no output yet'}), 200
    return jsonify({'success': True, 'markdown': Path(md_path).read_text(encoding='utf-8'),
                    'document': rec}), 200


@app.route('/api/documents/<doc_id>/md', methods=['PUT'])
def documents_md_put(doc_id):
    rec = doc_registry.get(doc_id)
    if not rec:
        return jsonify({'error': 'Document not found'}), 404
    if not rec.get('md_path'):
        return jsonify({'success': False, 'error': 'Document has no .md yet'}), 400

    data = request.get_json(silent=True) or {}
    md = data.get('markdown')
    if md is None:
        return jsonify({'success': False, 'error': 'markdown field is required'}), 400

    Path(rec['md_path']).write_text(md, encoding='utf-8')
    updated = doc_registry.update(doc_id, md_hash=doc_pipeline.md_hash(md),
                                  edited_by=_reviewer(data), edited_at=datetime.now().isoformat())

    reindexing = bool(rec.get('verified'))
    if reindexing:
        # Keep verified; re-propagate the edit through chunking + embedding now.
        _bg(doc_pipeline.index_document, doc_id)
    return jsonify({'success': True, 'reindexing': reindexing, 'document': updated}), 200


def _processing_summary(rec: dict) -> dict:
    """How this document's markdown was produced — text extraction vs OCR, and for
    OCR the per-page Docling/Sarvam split (read live from the stage2 confidence
    sidecar). Drives the method badge in the panel and works for old docs too."""
    out = {'text_source': rec.get('text_source'), 'method': None, 'label': None,
           'pages': None, 'docling_pages': 0, 'sarvam_pages': 0,
           'can_reprocess': bool(rec.get('doc_dir'))}
    if rec.get('status') == 'processing':
        out['label'] = 'Processing…'
        return out
    if rec.get('text_source') == 'extract':
        src = 'Scraped web text' if rec.get('source_type') == 'web' else 'Embedded PDF text'
        out['method'] = 'extract'
        out['label'] = f'Text extraction ({src})'
        out['can_reprocess'] = rec.get('source_type') != 'web'  # web docs have no PDF to OCR
        return out
    # OCR: read the per-page source split from stage2/<doc_dir>/*_confidence.json
    d = pc.paths.stage2 / (rec.get('doc_dir') or '')
    sidecar = None
    if d.is_dir():
        for f in d.glob('*_confidence.json'):
            try:
                sidecar = json.loads(f.read_text(encoding='utf-8'))
                break
            except Exception:
                continue
    # No confidence sidecar means the OCR pipeline never ran on this document. If
    # ocr_scanned kept the PDF's own text layer instead, it leaves extraction.json —
    # report that truthfully rather than inheriting the registry's historical "ocr"
    # label, which put an OCR badge on text no engine had touched.
    if sidecar is None and (d / 'extraction.json').is_file():
        try:
            marker = json.loads((d / 'extraction.json').read_text(encoding='utf-8'))
        except Exception:
            marker = {}
        if marker.get('method') == 'text_layer':
            out['method'] = 'extract'
            out['label'] = 'Text extraction (Embedded PDF text — no OCR was run)'
            return out

    pages = (sidecar or {}).get('pages') or []
    sarvam = sum(1 for p in pages if p.get('source') == 'sarvam')
    docling = sum(1 for p in pages if p.get('source') == 'docling')
    out.update(pages=len(pages) or None, docling_pages=docling, sarvam_pages=sarvam)
    if pages:
        if sarvam and docling:
            out['method'], out['label'] = 'auto', f'OCR — Docling + Sarvam ({sarvam}/{len(pages)} pages)'
        elif sarvam:
            out['method'], out['label'] = 'ocr_sarvam', 'OCR — Sarvam (all pages)'
        else:
            out['method'], out['label'] = 'ocr', 'OCR — Docling / EasyOCR'
    else:
        out['label'] = 'OCR' if rec.get('text_source') == 'ocr' else 'Unknown'
    return out


@app.route('/api/documents/<doc_id>/processing', methods=['GET'])
def documents_processing(doc_id):
    rec = doc_registry.get(doc_id)
    if not rec:
        return jsonify({'error': 'Document not found'}), 404
    return jsonify({'success': True, 'processing': _processing_summary(rec)}), 200


@app.route('/api/documents/<doc_id>/reprocess', methods=['POST'])
def documents_reprocess(doc_id):
    """Force re-processing of a document by a chosen method:
    'extract' (embedded PDF text) | 'ocr' (Docling only) | 'ocr_sarvam' (Sarvam every
    page) | 'auto' (Docling + Sarvam fallback). Runs in the background; the panel
    shows 'processing' until it finishes, then the new markdown/method appear."""
    rec = doc_registry.get(doc_id)
    if not rec:
        return jsonify({'error': 'Document not found'}), 404
    method = (request.get_json(silent=True) or {}).get('method', 'auto')
    if method not in doc_pipeline.REPROCESS_METHODS:
        return jsonify({'success': False,
                        'error': f'Unknown method {method!r}. Use one of: '
                                 f'{", ".join(doc_pipeline.REPROCESS_METHODS)}'}), 400
    if rec.get('source_type') == 'web':
        return jsonify({'success': False,
                        'error': 'This is scraped web text with no source PDF — nothing to OCR.'}), 400
    _bg(doc_pipeline.reprocess_document, doc_id, method)
    return jsonify({'success': True, 'method': method, 'status': 'processing'}), 202


@app.route('/api/documents/<doc_id>/verify', methods=['POST'])
def documents_verify(doc_id):
    rec = doc_registry.get(doc_id)
    if not rec:
        return jsonify({'error': 'Document not found'}), 404
    if not rec.get('md_path'):
        return jsonify({'success': False, 'error': 'Document is not processed yet'}), 400

    data = request.get_json(silent=True) or {}
    updated = doc_registry.update(doc_id, verified=True, verified_by=_reviewer(data),
                                  verified_at=datetime.now().isoformat())
    _bg(doc_pipeline.index_document, doc_id)   # only verified docs get embedded
    return jsonify({'success': True, 'document': updated}), 200


@app.route('/api/documents/verify-bulk', methods=['POST'])
def documents_verify_bulk():
    """Verify many documents at once. Body: {"ids": [...]} or {"all_pending": true}.
    Marks each processed-but-unverified doc verified, then indexes them in ONE
    background worker (index_document serializes on a lock, so a single sequential
    thread is correct — hundreds of threads would just contend)."""
    data = request.get_json(silent=True) or {}
    reviewer = _reviewer(data)
    docs = doc_registry.all()
    if data.get('all_pending'):
        targets = [d['id'] for d in docs if d.get('status') == 'processed' and not d.get('verified')]
    else:
        wanted = set(data.get('ids') or [])
        targets = [d['id'] for d in docs
                   if d['id'] in wanted and d.get('status') == 'processed' and not d.get('verified')]
    now = datetime.now().isoformat()
    for doc_id in targets:
        doc_registry.update(doc_id, verified=True, verified_by=reviewer, verified_at=now)

    def _index_all(ids):
        for i in ids:
            doc_pipeline.index_document(i)
    if targets:
        _bg(_index_all, list(targets))
    return jsonify({'success': True, 'verified': len(targets)}), 200


@app.route('/api/documents/<doc_id>/unverify', methods=['POST'])
def documents_unverify(doc_id):
    rec = doc_registry.get(doc_id)
    if not rec:
        return jsonify({'error': 'Document not found'}), 404
    updated = doc_registry.update(doc_id, verified=False, verified_by=None, verified_at=None)
    _bg(doc_pipeline.remove_document_vectors, doc_id)   # pull it back out of the RAG
    return jsonify({'success': True, 'document': updated}), 200


@app.route('/api/documents/<doc_id>', methods=['DELETE'])
def documents_delete(doc_id):
    """Discard a document from the review workspace: remove its RAG vectors (if any)
    and drop it from the registry, clearing it from the queue. Source/OCR files on
    disk are left untouched (a later re-import can bring it back) — this is a
    'reject', not a hard file delete."""
    rec = doc_registry.get(doc_id)
    if not rec:
        return jsonify({'error': 'Document not found'}), 404
    # Remove vectors BEFORE deregistering (the remover looks the record up by id).
    if rec.get('indexed'):
        try:
            doc_pipeline.remove_document_vectors(doc_id)
        except Exception as e:
            print(f"[Web UI] delete: vector removal failed for {doc_id}: {e}")
    doc_registry.remove(doc_id)
    return jsonify({'success': True, 'id': doc_id}), 200


if __name__ == '__main__':
    print("\n" + "="*70)
    print("CHiPS-RAG Pipeline (Internal Backend)")
    print("="*70)
    print("\nRAG Pipeline Status: Deferred (loaded on demand)")
    print(f"Environment: {os.getenv('ENVIRONMENT', 'development')}")
    print("\nThis Flask server is INTERNAL ONLY.")
    print("Authentication is handled by Express.js at :3000")
    
    # INTERNAL ONLY: the Node UI proxies to us over loopback, and this backend has
    # NO auth of its own (auth lives at the Express layer). Bind to 127.0.0.1 by
    # default so :500x is never exposed on a public interface even if the firewall
    # is misconfigured. Override with FLASK_HOST only if you know you need it.
    flask_host = os.getenv('FLASK_HOST', '127.0.0.1')
    flask_port = int(os.getenv('FLASK_PORT', '5000'))
    flask_debug = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    
    print(f"\nStarting Flask server on http://{flask_host}:{flask_port} (debug: {flask_debug})")
    print("Press Ctrl+C to stop the server\n")
    
    app.run(debug=flask_debug, host=flask_host, port=flask_port, use_reloader=False)
