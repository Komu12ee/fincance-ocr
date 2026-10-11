import os
import json
import atexit
import re
import time
import sys
from pathlib import Path
import requests
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from FlagEmbedding import BGEM3FlagModel   # FlagReranker imported lazily — see load_reranker()

# Shared per-pipeline config (core/ is parents[2] of this script)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import pipeline_config as pc

# fp16 is a win on CUDA but a slowdown on CPU — pick it based on the actual device.
try:
    import torch
    _USE_FP16 = torch.cuda.is_available()
except Exception:
    _USE_FP16 = False

for stream_name in ("stdout", "stderr"):
    stream = getattr(sys, stream_name, None)
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

# ── Timing Utilities ───────────────────────────────────────────
_pipeline_start = None
_stage_times = {}

def _mark_time(stage_name):
    """Mark the current time for a stage and print elapsed time."""
    global _pipeline_start, _stage_times
    current_time = time.time()
    
    if _pipeline_start is None:
        _pipeline_start = current_time
        _stage_times[stage_name] = current_time
        print(f"⏱️ [{stage_name}] STARTED")
    else:
        if stage_name in _stage_times:
            elapsed = current_time - _stage_times[stage_name]
            print(f"⏱️ [{stage_name}] COMPLETED in {elapsed:.2f}s")
        _stage_times[stage_name] = current_time

def _print_timing_summary():
    """Print timing summary for entire pipeline."""
    global _pipeline_start
    if _pipeline_start is None:
        return
    total = time.time() - _pipeline_start
    print(f"\n{'='*70}")
    print(f"⏱️ TOTAL PIPELINE TIME: {total:.2f} seconds")
    print(f"{'='*70}\n")

# ── Configuration ──────────────────────────────────────────────
def _get_config():
    """Load configuration with environment variable overrides."""
    qdrant_mode = os.getenv("QDRANT_MODE", "local").lower()
    return {
        "chunk_dir": Path(os.getenv("CHIPPY_CHUNK_DIR", str(pc.paths.chunks))),
        "collection": os.getenv("CHIPPY_QDRANT_COLLECTION", pc.COLLECTION),
        "qdrant_mode": qdrant_mode,
        "qdrant_local_path": Path(
            os.getenv("CHIPPY_QDRANT_LOCAL_PATH", str(pc.paths.db))
        ),
        "qdrant_host": os.getenv("QDRANT_HOST", "localhost"),
        "qdrant_port": int(os.getenv("QDRANT_PORT", "6333")),
        "qdrant_api_key": os.getenv("QDRANT_API_KEY", None),
        "qdrant_timeout": int(os.getenv("QDRANT_TIMEOUT", "60")),
        "encode_batch_size": 8,
        "max_length": 1024,
        "qdrant_connect_retries": int(os.getenv("CHIPPY_QDRANT_CONNECT_RETRIES", "3")),
        "qdrant_connect_delay": float(os.getenv("CHIPPY_QDRANT_CONNECT_DELAY", "1.5")),
    }

CFG = _get_config()
CHUNK_DIR = CFG["chunk_dir"]
COLLECTION_NAME = CFG["collection"]
ENCODE_BATCH_SIZE = CFG["encode_batch_size"]
MAX_LENGTH = CFG["max_length"]

# ──────────────────────────────────────────────────────────────
# GROQ API Configuration (COMMENTED OUT - kept for future use)
# ──────────────────────────────────────────────────────────────
# GROQ_API_KEY = os.getenv("GROQ_API_KEY", None)
# GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
# GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# ──────────────────────────────────────────────────────────────
# SARVAM AI API Configuration (ACTIVE)
# ──────────────────────────────────────────────────────────────
# Per-pipeline key comes from the registry-selected env var (see pipeline_config / .env).
# SARVAM_API_KEY env still wins if explicitly set (e.g. launcher override).
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY") or pc.SARVAM_API_KEY
SARVAM_MODEL = os.getenv("SARVAM_MODEL", "sarvam-105b")  # Sarvam AI's 105B model
SARVAM_API_URL = "https://api.sarvam.ai/v1/chat/completions"

# Chat generation can use the existing Sarvam service or a GPU VM exposing an
# OpenAI-compatible API. The production SDC VM's tunnel listens on port 8000.
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "sarvam").strip().lower()
LLM_BASE_URL = (os.getenv("LLM_BASE_URL") or "http://127.0.0.1:8000/v1").strip().rstrip("/")
LLM_MODEL = (os.getenv("LLM_MODEL") or "").strip()
LLM_API_KEY = (os.getenv("LLM_API_KEY") or "").strip()
_gpu_model = None

# ── Retrieval Configuration ────────────────────────────────────
HYBRID_ALPHA = 0.6           # 0.0 = pure sparse, 1.0 = pure dense (0.6 = 60% dense, 40% sparse)
USE_MULTI_QUERY = True       # Enable multi-query retrieval for better coverage
DENSE_CANDIDATE_LIMIT = 60   # Dense hits fetched per query variation (sparse re-scores only these)
MAX_CHUNKS_PER_SOURCE = 3    # Cap chunks contributed by any single document (0 = no cap)
MAX_ANSWER_TOKENS = 4096     # Output cap; Hindi answers need roughly 2x the tokens of English

# ── Reranking (OFF by default) ─────────────────────────────────
# Measured on this corpus: the bge-m3 Hindi embeddings already score 0.86+, so
# cross-encoder reranking bought <5% quality for ~62s per query. It stays available
# but must be opted into — and, critically, the 2.2GB bge-reranker-v2-m3 model is
# only downloaded/loaded when it is actually switched on (see load_reranker). It used
# to load on every startup regardless, costing 2.2GB of RAM per pipeline (6.6GB across
# the three) for code whose only call site was commented out.
#
#   RERANK_ENABLED=1 uv run python app.py     # turn it back on, no code edit
RERANK_ENABLED = os.getenv("RERANK_ENABLED", "").strip().lower() in ("1", "true", "yes", "on")
RERANK_MIN_K = 3             # Minimum results to return
RERANK_MAX_K = 6             # Maximum results to return
RERANK_THRESHOLD = 0.65      # Score threshold for inclusion

# ── Helper: Get file number from chunk source ───────────────────
def extract_file_number(chunk_source):
    """Extract file number from chunk source (e.g., 'output_corrected2' → 2)."""
    import re
    match = re.search(r'output_corrected(\d+)', chunk_source)
    if match:
        return int(match.group(1))
    return None

# ── Helper: Get actual file name from chunk metadata ───────────
def get_actual_filename(chunk_source):
    """Convert output_corrected* to actual file name using file numbers (file1.pdf, file2.pdf, etc.)."""
    file_num = extract_file_number(chunk_source)
    if file_num:
        return f"file{file_num}.pdf"
    return chunk_source + ".pdf"  # Fallback

# ── Helper: Extract highlighted excerpt from chunk text ─────────
def extract_highlighted_excerpt(chunk_text, query_words, max_length=300):
    """Extract the most relevant part of chunk text containing query words.
    
    Args:
        chunk_text: Full chunk text
        query_words: List of important words from the query
        max_length: Max length of excerpt
    
    Returns:
        Highlighted excerpt with query words in context
    """
    sentences = chunk_text.split('. ')
    best_sentences = []
    
    for sentence in sentences:
        sentence_lower = sentence.lower()
        if any(word.lower() in sentence_lower for word in query_words if len(word) > 3):
            best_sentences.append(sentence.strip())
    
    if best_sentences:
        excerpt = '. '.join(best_sentences[:2])  # Take first 2 matching sentences
    else:
        excerpt = chunk_text[:max_length]
    
    # Truncate if too long
    if len(excerpt) > max_length:
        excerpt = excerpt[:max_length].rsplit(' ', 1)[0] + '...'
    
    return excerpt.strip()

# Loaded on first retrieval, never at import time — importing this module must stay
# cheap so the Flask app boots fast and `--help`/health checks don't pay for a model.
model = None
reranker = None
_models_loaded = False


def ensure_models_loaded():
    """Load the embedding model on first use. Idempotent."""
    global model, _models_loaded
    if _models_loaded:
        return

    t0 = time.time()
    print("Loading embedding model bge-m3 (deferred)...")
    model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=_USE_FP16)
    model.return_sparse = True
    print(f"✓ Embedding model ready in {time.time() - t0:.1f}s (fp16={_USE_FP16})")

    _models_loaded = True


def load_reranker():
    """The cross-encoder, loaded only if RERANK_ENABLED — it is 2.2GB.

    Separate from ensure_models_loaded on purpose: the embedding model is required
    for every query, the reranker is optional and expensive. Returns None when
    reranking is off, which is what makes the default path cost nothing."""
    global reranker
    if not RERANK_ENABLED:
        return None
    if reranker is None:
        from FlagEmbedding import FlagReranker   # deferred: import pulls in the model class
        t0 = time.time()
        print("Loading reranker bge-reranker-v2-m3 (RERANK_ENABLED)...")
        reranker = FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=_USE_FP16)
        print(f"✓ Reranker ready in {time.time() - t0:.1f}s")
    return reranker

client = None


def _connect_qdrant_once():
    """Create a Qdrant client using the configured local or remote mode."""
    if CFG["qdrant_mode"] == "remote":
        print(f"Connecting to remote Qdrant at {CFG['qdrant_host']}:{CFG['qdrant_port']}...")
        qdrant = QdrantClient(
            host=CFG["qdrant_host"],
            port=CFG["qdrant_port"],
            api_key=CFG["qdrant_api_key"],
            timeout=CFG["qdrant_timeout"],
        )
        qdrant.get_collections()
        print(f"✓ Connected to remote Qdrant at {CFG['qdrant_host']}:{CFG['qdrant_port']}")
        return qdrant

    print(f"Connecting to local Qdrant at {CFG['qdrant_local_path']}...")
    CFG["qdrant_local_path"].mkdir(parents=True, exist_ok=True)
    qdrant = QdrantClient(path=str(CFG["qdrant_local_path"]))
    qdrant.get_collections()
    print("✓ Connected to local embedded Qdrant")
    return qdrant


def ensure_qdrant_client():
    """Create the module-level Qdrant client on demand with a short retry for stale locks."""
    global client
    if client is not None:
        return client

    last_error = None
    for attempt in range(1, CFG["qdrant_connect_retries"] + 1):
        try:
            client = _connect_qdrant_once()
            return client
        except Exception as e:
            last_error = e
            message = str(e)
            if "already accessed by another instance" in message and attempt < CFG["qdrant_connect_retries"]:
                wait_seconds = CFG["qdrant_connect_delay"] * attempt
                print(
                    f"[RAG] Local Qdrant is still locked; retrying in {wait_seconds:.1f}s "
                    f"({attempt}/{CFG['qdrant_connect_retries']})..."
                )
                time.sleep(wait_seconds)
                continue
            break

    raise RuntimeError(
        f"Unable to initialize Qdrant in {CFG['qdrant_mode']} mode: {last_error}"
    ) from last_error


def _cleanup_qdrant():
    """Explicitly close Qdrant client on exit to avoid shutdown import errors."""
    try:
        if client is not None:
            client.close()
    except Exception:
        pass  # Suppress cleanup errors during shutdown


atexit.register(_cleanup_qdrant)

# ── Report the selected chat provider ────────────────────────
if LLM_PROVIDER == "gpu":
    print(f"✓ GPU LLM configured at {LLM_BASE_URL} (model: {LLM_MODEL or 'auto-discover'})")
elif LLM_PROVIDER != "sarvam":
    print(f"⚠ Unknown LLM_PROVIDER={LLM_PROVIDER!r}; use 'sarvam' or 'gpu'.")
elif not SARVAM_API_KEY:
    print("⚠ WARNING: SARVAM_API_KEY not set. Set it via environment variable.")
    print("  Add to your terminal: $env:SARVAM_API_KEY='your-api-key-here'")
else:
    print(f"✓ Sarvam AI configured with model: {SARVAM_MODEL}")

# # ── Validate Groq Configuration (COMMENTED OUT) ──────────────
# if not GROQ_API_KEY:
#     print("⚠ WARNING: GROQ_API_KEY not set. Set it via environment variable.")
#     print("  Add to your terminal: $env:GROQ_API_KEY='your-api-key-here'")
# else:
#     print(f"✓ Groq API configured with model: {GROQ_MODEL}")

# ── Helper: Sparse search ──────────────────────────────────────
def sparse_search(query_sparse, all_points, limit=5):
    """Score points based on sparse embeddings overlap."""
    scores = []
    for point in all_points:
        sparse_payload = point.payload.get("sparse_embedding", {})
        score = sum(sparse_payload.get(token, 0) * query_sparse.get(token, 0) 
                   for token in query_sparse if token in sparse_payload)
        scores.append((point.id, score))
    return sorted(scores, key=lambda x: x[1], reverse=True)[:limit]

# ── Helper: Pull the chunk that follows a retrieved chunk ──────
def expand_with_neighbors(qdrant, results, max_extra=None):
    """Append the chunk immediately after each retrieved chunk, same document.

    The chunker splits these minutes on headings, which puts "AGENDA # 7" and its
    "DECISION TAKEN" into two separate chunks — and the decision chunk is often
    tiny (17 tokens for "the Committee deferred the agenda Point"). Those short
    chunks carry no distinctive terms, so retrieval reliably finds the proposal
    and misses the outcome, and the answer reports a deferred item as approved.
    Following the chunk pointer costs one filtered lookup and repairs the pairing
    without re-chunking or re-embedding anything.
    """
    # Expand every hit by default. Capping this made it first-come-first-served over
    # the result list, so the neighbour that actually carried the decision was often
    # skipped in favour of neighbours of higher-ranked but less relevant chunks.
    if max_extra is None:
        max_extra = len(results)

    have = {(r["point"].payload.get("source"), str(r["point"].payload.get("chunk"))) for r in results}

    # Work out every neighbour we want BEFORE touching the DB, then fetch them all in
    # one filtered scroll (an OR over the wanted source/chunk pairs). This used to be
    # one round trip per result — 8 sequential queries for a default 8-hit answer.
    wanted = []          # [(source, next_chunk, parent_result)]
    for r in results:
        if len(wanted) >= max_extra:
            break
        payload = r["point"].payload
        source = payload.get("source")
        try:
            next_chunk = str(int(payload.get("chunk")) + 1)
        except (TypeError, ValueError):
            continue
        if (source, next_chunk) in have:
            continue
        have.add((source, next_chunk))       # also dedupes within this batch
        wanted.append((source, next_chunk, r))

    if not wanted:
        return results

    try:
        found, _ = qdrant.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=qmodels.Filter(should=[
                qmodels.Filter(must=[
                    qmodels.FieldCondition(key="source", match=qmodels.MatchValue(value=src)),
                    qmodels.FieldCondition(key="chunk", match=qmodels.MatchValue(value=chunk)),
                ])
                for src, chunk, _ in wanted
            ]),
            limit=len(wanted),
            with_payload=True,
            with_vectors=False,
        )
    except Exception as e:
        print(f"  ⚠ Neighbor lookup failed ({len(wanted)} wanted): {e}")
        return results

    # Match fetched points back to the hit that asked for them, so the supporting
    # chunk inherits a score derived from its parent.
    by_key = {(p.payload.get("source"), str(p.payload.get("chunk"))): p for p in found}
    extras = []
    for source, next_chunk, parent in wanted:
        point = by_key.get((source, next_chunk))
        if point is None:
            continue
        extras.append({
            "point": point,
            "score": parent["score"] * 0.5,   # supporting context, not a primary hit
            "rank": len(results) + len(extras) + 1,
            "neighbor_of": f"{source}#{parent['point'].payload.get('chunk')}",
        })

    if extras:
        print(f"  🔗 Added {len(extras)} adjacent chunk(s) for decision/outcome continuity "
              f"(1 batched lookup)")
    return results + extras


# ── Helper: Detect the language the answer should be written in ─
# Hindi written in Roman script ("mujhe chutti ke niyam batao") is extremely common
# from these users, and a Devanagari-vs-Latin character count calls it English —
# so the user asked in Hinglish and got a formal English answer back. These tokens
# are Hindi function/question words that are rare or absent in English prose, so
# their presence in a Latin-script query is a reliable Hinglish signal.
# Deliberately EXCLUDED as too ambiguous in English: the, me, par, or, is, do, to,
# he, we, so, ka/ki/ke (too short, appear inside other tokens).
_HINGLISH_TOKENS = {
    "kya", "kyu", "kyun", "kyon", "kaise", "kaisa", "kaisi", "kaun", "kaunsa", "konsa",
    "kaunsi", "konsi", "kab", "kahan", "kitna", "kitne", "kitni", "hai", "hain", "hoga",
    "hogi", "honge", "hona", "mujhe", "mera", "meri", "mere", "hume", "humein", "aap",
    "aapka", "aapko", "aapki", "chahiye", "batao", "bataye", "bataiye", "batana",
    "nahi", "nahin", "karna", "karne", "karta", "karti", "karte", "sakta", "sakti",
    "sakte", "raha", "rahi", "rahe", "gaya", "gayi", "diya", "kiya", "milega", "milegi",
    "milta", "jankari", "niyam", "adhikari", "karmchari", "chutti", "vetan", "aavedan",
    "prakriya", "yadi", "lekin", "aur", "wala", "wale", "wali", "hoti", "hota", "hote",
}
_HINGLISH_MIN_HITS = 2      # 2+ markers: one stray loanword should not flip an English query


def detect_query_language(query):
    """Return 'hi' (Devanagari), 'hinglish' (Roman-script Hindi), or 'en'.

    The corpus is mixed Hindi/English and bge-m3 retrieves across both, so the
    retrieved chunks are a poor signal for what language the *user* wants back.
    Left to itself the LLM mirrors the chunks, which means an English question
    about a Hindi document comes back in Hindi (and vice versa). Decide from the
    question instead.
    """
    devanagari = sum(1 for ch in query if 'ऀ' <= ch <= 'ॿ')
    latin = sum(1 for ch in query if ch.isascii() and ch.isalpha())
    if devanagari > latin:
        return "hi"
    words = set(re.findall(r"[a-z]+", query.lower()))
    if len(words & _HINGLISH_TOKENS) >= _HINGLISH_MIN_HITS:
        # A romanized-Hindi question ("2016 me konsa niyam add hua") IS asked in Hindi,
        # just typed in Roman. For a government audience answer in proper Devanagari
        # Hindi rather than Roman-script Hinglish — the demo wants clean Hindi/English,
        # not a mixed register. (Flip this to "hinglish" to restore Roman replies.)
        return "hi"
    return "en"


# How each detected language is enforced. The model reliably drifts into the language
# of the retrieved chunks, so the rule is stated THREE times per request: in the system
# prompt, as the last line of the user message, and as an explicit first-sentence
# constraint. Repetition is what actually holds it — a single rule buried above several
# thousand tokens of Hindi context loses.
_LANGUAGE_RULES = {
    "hi": {
        "label": "Hindi (Devanagari)",
        "rule": ("LANGUAGE (ABSOLUTE): The user wrote in Hindi. Write EVERY sentence of "
                 "your answer in Hindi using Devanagari script — including headings, "
                 "labels and explanations — even though most source documents are in "
                 "English. Translate what you quote into Hindi, keeping the original "
                 "English alongside exact quotes. Keep official names, order numbers, "
                 "department names and filenames in their original form. Do NOT answer "
                 "in English. Do NOT switch language part-way."),
        "closing": "उत्तर पूरी तरह हिंदी (देवनागरी) में दें, चाहे दस्तावेज़ किसी भी भाषा में हों।",
    },
    "hinglish": {
        "label": "Hinglish (Hindi written in Roman/English script)",
        "rule": ("LANGUAGE (ABSOLUTE): The user wrote in Hinglish — Hindi using Roman "
                 "letters, e.g. 'mujhe chutti ke niyam batao'. Reply in the SAME style: "
                 "Hindi sentences written in Roman script, NOT Devanagari and NOT formal "
                 "English. Example of the required style: 'Is order ke according, 15 din "
                 "ki casual leave allowed hai.' Keep technical terms, official names, "
                 "order numbers and filenames as they are. Do NOT reply in Devanagari. "
                 "Do NOT reply in pure English."),
        "closing": "Jawab Hinglish (Roman script) mein hi dijiye, Devanagari ya pure English mein nahi.",
    },
    "en": {
        "label": "English",
        "rule": ("LANGUAGE (ABSOLUTE): The user wrote in English. Write EVERY sentence of "
                 "your answer in English — including headings and explanations — even "
                 "though many source documents are in Hindi. Translate any Hindi you "
                 "quote into English, keeping the original Hindi alongside exact quotes. "
                 "Do NOT answer in Hindi. Do NOT switch language part-way."),
        "closing": "Answer entirely in English, regardless of the language of the source documents.",
    },
}


def answer_language_ok(answer, expected_lang):
    """Does `answer` appear to be written in `expected_lang`?

    Prompting alone does not hold. With a couple of dozen Hindi passages in context
    the model mirrors the sources and answers in Hindi even when the rule says
    English three times over — observed on a real query. So the result is CHECKED
    rather than trusted, and the caller retries once on a mismatch.

    Majority script, not mere presence: an English answer legitimately quotes
    Devanagari, so a few Hindi words must not count as a violation.
    """
    if not answer:
        return True
    dev = sum(1 for ch in answer if 'ऀ' <= ch <= 'ॿ')
    latin = sum(1 for ch in answer if ch.isascii() and ch.isalpha())
    if expected_lang == "hi":
        return dev >= latin
    # en and hinglish are both Latin-script outputs; Hinglish is scored as Latin
    # here, so this only catches the "asked in English, answered in Devanagari" flip.
    return latin >= dev


def language_directive(query):
    """(lang, rule, closing_line, label) for the language the answer must use.

    One place decides this, so the single-shot answer and every thinking-mode step
    enforce language identically.
    """
    lang = detect_query_language(query)
    spec = _LANGUAGE_RULES.get(lang, _LANGUAGE_RULES["en"])
    return lang, spec["rule"], spec["closing"], spec["label"]


# ── Helper: Expand query into multiple perspectives ────────────
def expand_query(original_query):
    """Generate multiple query variations to improve retrieval coverage.
    
    Uses keyword expansion and perspective shifts:
    - Original query
    - Query + context keywords (approval, implementation, decision, etc.)
    - Query + document type keywords (agenda, minutes, meeting, etc.)
    - Query with synonyms
    """
    variations = [original_query]  # Always include original
    
    # Add context-specific variations for government documents
    context_keywords = ["approval", "decision", "implementation", "status", "progress"]
    doc_keywords = ["meeting", "agenda", "minutes", "committee", "approval"]
    
    for keyword in context_keywords:
        if keyword not in original_query.lower():
            variations.append(f"{original_query} {keyword}")
    
    for keyword in doc_keywords:
        if keyword not in original_query.lower():
            variations.append(f"{original_query} {keyword}")
    
    # Add detail-focused variations
    variations.append(f"{original_query} details implementation")
    variations.append(f"{original_query} decision taken")
    
    # Remove duplicates while preserving order
    seen = set()
    unique_variations = []
    for v in variations:
        v_lower = v.lower()
        if v_lower not in seen:
            seen.add(v_lower)
            unique_variations.append(v)
    
    return unique_variations[:5]  # Cap at 5 variations to avoid excessive querying

# ── Helper: Encode queries (one model call for the whole batch) ─
def encode_queries(queries):
    """Encode every query variation in ONE model.encode call.

    bge-m3 has a fixed per-call cost (tokenise, move to device, forward pass setup)
    that dwarfs the marginal cost of extra rows, so encoding 5 variations together
    is far cheaper than 5 separate calls. Returns [(dense_list, sparse_dict), ...]
    aligned with `queries`; entries that failed to encode are None.
    """
    ensure_models_loaded()
    if not queries:
        return []

    t0 = time.time()
    enc = model.encode(queries, batch_size=len(queries), max_length=MAX_LENGTH)
    if not enc or "dense_vecs" not in enc:
        return [None] * len(queries)

    dense_vecs = enc.get("dense_vecs")
    if dense_vecs is None or len(dense_vecs) == 0:
        return [None] * len(queries)

    # lexical_weights is a per-row list; older/edge builds hand back a bare dict for
    # a single row, so normalise both shapes into "one sparse dict per query".
    lex = enc.get("lexical_weights")
    if isinstance(lex, dict):
        lex = [lex]
    elif not isinstance(lex, list):
        lex = []

    out = []
    for i in range(len(queries)):
        if i >= len(dense_vecs):
            out.append(None)
            continue
        try:
            sparse = dict(lex[i]) if i < len(lex) and lex[i] is not None else {}
        except (TypeError, ValueError):
            sparse = {}
        out.append((dense_vecs[i].tolist(), sparse))

    print(f"  🧮 Encoded {len(queries)} query variation(s) in one batch ({time.time() - t0:.2f}s)")
    return out


# ── Helper: Single query retrieval ─────────────────────────────
def perform_single_retrieval(query):
    """Retrieve for ONE query (the USE_MULTI_QUERY=False path)."""
    try:
        qdrant = ensure_qdrant_client()
        encoded = encode_queries([query])
        if not encoded or encoded[0] is None:
            return None
        query_dense, query_sparse = encoded[0]

        # Dense search. The sparse arm only re-scores points that dense already
        # returned, so this limit is also the ceiling on what sparse can ever
        # rescue — keep it well above num_context.
        dense_results = qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=query_dense,
            limit=DENSE_CANDIDATE_LIMIT,
            with_payload=True,
        )

        if dense_results is None or not dense_results.points:
            return None

        return {
            "dense_results": dense_results,
            "dense_scores": [(p.id, p.score) for p in dense_results.points],
            "query_sparse": query_sparse
        }

    except Exception as e:
        print(f"  Single retrieval error: {e}")
        return None

# ── Helper: Multi-query retrieval ──────────────────────────────
def multi_query_retrieval(query):
    """Retrieve results using multiple query variations and merge them.
    
    Benefits:
    - Captures different aspects of the query
    - Better coverage of semantically related documents
    - More robust to query phrasing variations
    """
    _mark_time("MULTI_QUERY_RETRIEVAL")
    ensure_models_loaded()
    
    if not USE_MULTI_QUERY:
        # Fall back to single query
        result = perform_single_retrieval(query)
        if result is None:
            return None, []
        return result["dense_results"], result["dense_scores"], result["query_sparse"]
    
    # Expand query into multiple variations
    query_variations = expand_query(query)
    print(f"  🔍 Searching with {len(query_variations)} query variations...")

    # ONE encode call for every variation, then ONE batched Qdrant round trip —
    # this used to be N encodes and N searches issued strictly in sequence.
    encoded = encode_queries(query_variations)
    live = [(i, e) for i, e in enumerate(encoded) if e is not None]
    if not live:
        print("  Error: No query variation could be encoded.")
        return None, [], {}

    qdrant = ensure_qdrant_client()
    try:
        t0 = time.time()
        responses = qdrant.query_batch_points(
            collection_name=COLLECTION_NAME,
            requests=[
                qmodels.QueryRequest(query=dense, limit=DENSE_CANDIDATE_LIMIT, with_payload=True)
                for _, (dense, _sparse) in live
            ],
        )
        print(f"  🔎 Batched {len(live)} dense search(es) into one call ({time.time() - t0:.2f}s)")
    except Exception as e:
        # Any batch-API incompatibility falls back to the original sequential path
        # rather than failing the query outright.
        print(f"  ⚠ Batched search unavailable ({e}); falling back to sequential")
        responses = None

    all_dense_results = {}   # {point_id: point}
    aggregated_scores = {}   # {point_id: weighted score sum}
    all_sparse_queries = {}

    def _absorb(i, points, scored, sparse):
        """Merge one variation's hits. Earlier variations weigh more (1/(i+1))."""
        for point in points:
            all_dense_results[point.id] = point
        for point_id, score in scored:
            aggregated_scores[point_id] = aggregated_scores.get(point_id, 0) + score * (1.0 / (i + 1))
        return sparse or all_sparse_queries

    if responses is not None:
        for (i, (_dense, sparse)), resp in zip(live, responses):
            points = getattr(resp, "points", None) or []
            all_sparse_queries = _absorb(i, points, [(p.id, p.score) for p in points], sparse)
    else:
        for i, q_variant in enumerate(query_variations):
            r = perform_single_retrieval(q_variant)
            if r is None:
                continue
            all_sparse_queries = _absorb(i, r["dense_results"].points,
                                         r["dense_scores"], r["query_sparse"])

    if not all_dense_results:
        print("  Error: No results from multi-query retrieval.")
        return None, [], {}

    # Lightweight stand-in matching the .points shape the callers expect.
    class MergedResults:
        __slots__ = ("points",)

        def __init__(self, points):
            self.points = points

    dense_results = MergedResults(list(all_dense_results.values()))

    _mark_time("MULTI_QUERY_RETRIEVAL")
    return dense_results, list(aggregated_scores.items()), all_sparse_queries

# ── Helper: Rerank search results (Hybrid Threshold) ──────────
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
    _mark_time("RERANKING")
    
    if not candidate_points:
        return []
    
    # Prepare query-document pairs for reranking
    pairs = []
    point_map = {}
    
    for idx, point in enumerate(candidate_points):
        text = point.payload.get("text", "")
        pairs.append([query, text])
        point_map[idx] = point
    
    # Score with reranker (load_reranker so this is safe to call directly)
    print(f"  📊 Reranking {len(pairs)} candidates...")
    rerank_scores = load_reranker().compute_score(pairs, normalize=True)
    
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
    
    _mark_time("RERANKING")
    return results

# ── Helper: Hybrid search using RRF ───────────────────────────
def hybrid_search(dense_scores, sparse_scores, alpha=0.5, k=60):
    """Combine dense and sparse scores using RRF (Reciprocal Rank Fusion)."""
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

# ── Helper: Retrieve context ───────────────────────────────────
def retrieve_context(query, num_context=8):
    """Retrieve context chunks for a query via hybrid (dense+sparse) search.

    Args:
        query: Search query
        num_context: Number of context results to return

    Returns:
        List of result dicts with 'point', 'score', 'rank'
    """
    _mark_time("RETRIEVE_CONTEXT")
    try:
        # Step 1: Perform embedding-based retrieval
        print("🔍 Retrieving context...")
        
        # Multi-query retrieval (or single-query fallback)
        dense_results, aggregated_scores, query_sparse = multi_query_retrieval(query)
        
        if dense_results is None or not dense_results.points:
            print("Error: No results from retrieval.")
            return None
        
        # Sort aggregated scores
        dense_scores = sorted(aggregated_scores, key=lambda x: x[1], reverse=True)
        
        # Sparse search (if available)
        if query_sparse:
            sparse_scores = sparse_search(query_sparse, dense_results.points, limit=DENSE_CANDIDATE_LIMIT)
            # Use configurable HYBRID_ALPHA
            hybrid_scores = hybrid_search(dense_scores, sparse_scores, alpha=HYBRID_ALPHA)
            print(f"  ⚡ Using hybrid search (α={HYBRID_ALPHA}: {int(HYBRID_ALPHA*100)}% dense, {int((1-HYBRID_ALPHA)*100)}% sparse)")
        else:
            hybrid_scores = [(pid, score) for rank, (pid, score) in enumerate(dense_scores)]
            print(f"  ⚡ Using dense-only search (sparse embeddings unavailable)")
        
        # Index the points once by id. Both the candidate collection below and the
        # result assembly further down look points up by id inside a loop; doing that
        # with a linear scan over dense_results.points made retrieval O(hits x points)
        # — with multi-query merging several hundred points that is tens of thousands
        # of comparisons per query, for what is a dict lookup.
        points_by_id = {p.id: p for p in dense_results.points}

        # Collect candidate points (top hybrid results) for optional reranking
        candidate_points = [
            points_by_id[point_id]
            for point_id, _ in hybrid_scores[:DENSE_CANDIDATE_LIMIT]
            if point_id in points_by_id
        ]
        
        # Optional cross-encoder rerank. Off by default (see RERANK_ENABLED); when on,
        # it reorders the hybrid candidates and returns directly.
        if RERANK_ENABLED and load_reranker() is not None:
            print("🔄 Reranking results (RERANK_ENABLED)...")
            reranked = rerank_results(query, candidate_points,
                                      min_k=RERANK_MIN_K,
                                      max_k=RERANK_MAX_K,
                                      threshold=RERANK_THRESHOLD)
            _mark_time("RETRIEVE_CONTEXT")
            return reranked[:num_context]

        print("⚡ Using hybrid search results directly (reranking disabled)")

        # Convert hybrid scores to result format.
        # A per-source cap keeps one large document from monopolising the context
        # window: EB 56 alone is 319 of the 471 chunks in the chips collection, so
        # without this the top-5 is frequently 5/5 EB 56 and answers get built from
        # whichever document is biggest rather than whichever is relevant.
        results = []
        per_source_count = {}
        rank = 0
        for point_id, score in hybrid_scores:
            if len(results) >= num_context:
                break
            point = points_by_id.get(point_id)
            if point:
                source = point.payload.get("source", "")
                if MAX_CHUNKS_PER_SOURCE and per_source_count.get(source, 0) >= MAX_CHUNKS_PER_SOURCE:
                    continue
                per_source_count[source] = per_source_count.get(source, 0) + 1
                rank += 1
                results.append({
                    "point": point,
                    "score": score,
                    "rank": rank,
                })

        results = expand_with_neighbors(ensure_qdrant_client(), results)

        _mark_time("RETRIEVE_CONTEXT")
        return results
    
    except Exception as e:
        print(f"Retrieval error: {e}")
        import traceback
        traceback.print_exc()
        _mark_time("RETRIEVE_CONTEXT")
        return None

# ── Helper: One place that talks to the LLM ────────────────────
class LLMError(RuntimeError):
    """Raised when the LLM call fails. Callers decide whether that is fatal."""


def _llm_target(timeout):
    """Resolve the chat URL, model and auth for the selected provider only."""
    global _gpu_model
    headers = {"Content-Type": "application/json"}
    if LLM_PROVIDER == "sarvam":
        if not SARVAM_API_KEY:
            raise LLMError("SARVAM_API_KEY not configured")
        headers["Authorization"] = f"Bearer {SARVAM_API_KEY}"
        return "Sarvam API", SARVAM_API_URL, SARVAM_MODEL, headers
    if LLM_PROVIDER != "gpu":
        raise LLMError(f"Unknown LLM_PROVIDER={LLM_PROVIDER!r}; use 'sarvam' or 'gpu'.")
    if not LLM_BASE_URL.startswith(("http://", "https://")):
        raise LLMError("LLM_BASE_URL must be an http:// or https:// API base URL ending in /v1")

    # A GPU server may require its own key; never send a Sarvam key to the VM.
    if LLM_API_KEY:
        headers["Authorization"] = f"Bearer {LLM_API_KEY}"
    model = LLM_MODEL or _gpu_model
    if not model:
        try:
            response = requests.get(f"{LLM_BASE_URL}/models", headers=headers,
                                    timeout=min(timeout, 30))
        except requests.exceptions.RequestException as e:
            raise LLMError(f"GPU model discovery failed at {LLM_BASE_URL}/models: {e}") from e
        if response.status_code != 200:
            raise LLMError(f"GPU model discovery returned HTTP {response.status_code}: {response.text}")
        try:
            models = response.json()["data"]
            if not isinstance(models, list):
                raise ValueError("data must be a model list")
            ids = list(dict.fromkeys(m["id"] for m in models
                                     if isinstance(m, dict) and isinstance(m.get("id"), str) and m["id"]))
        except (ValueError, KeyError, TypeError) as e:
            raise LLMError("GPU /models returned an invalid response; set LLM_MODEL explicitly") from e
        if len(ids) != 1:
            raise LLMError(f"GPU /models returned {len(ids)} models; set LLM_MODEL to an exact model ID")
        model = _gpu_model = ids[0]
        print(f"✓ GPU LLM discovered model: {model}")
    return "GPU LLM", f"{LLM_BASE_URL}/chat/completions", model, headers


def call_llm(system, user, temperature=0.2, max_tokens=None, timeout=300):
    """POST one chat completion to the selected provider and return the text.

    Extracted so every caller — the single-shot answer, and each reasoning step of
    thinking mode — shares one implementation of auth, timeouts, truncation
    detection and error shape. Raises LLMError rather than returning an error
    string, so an orchestrator can catch and degrade instead of quietly embedding
    "Error: ..." into a user-facing answer.

    Returns (text, truncated).
    """
    llm_name, api_url, model_name, headers = _llm_target(timeout)

    # sarvam-105b is a REASONING model: it writes its chain of thought into
    # `reasoning_content` and the actual reply into `content`. If max_tokens runs out
    # while it is still reasoning, `content` comes back as JSON null — note that
    # .get("content", default) does NOT save you there, because the key exists with a
    # None value, and that None then propagates into `answer += ...` as a TypeError.
    # An empty reply is never useful and the only thing that recovers it is more room,
    # so when the budget was exhausted mid-reasoning we retry ONCE with a bigger cap
    # before giving up. Fast common case, self-healing on the hard case.
    budget = max_tokens or MAX_ANSWER_TOKENS
    last_reasoning = 0
    # Sarvam intermittently resets the connection (ConnectionResetError 104) or returns
    # a 5xx under load — observed live. These are transient, so retry the request a few
    # times with a short backoff before surfacing an error to the user; a single blip
    # must not fail a demo query. NETWORK_TRIES counts network/5xx attempts; the inner
    # empty-content path additionally retries once with a bigger budget.
    NETWORK_TRIES = 4
    net_attempt = 0
    attempt = 0
    while True:
        try:
            response = requests.post(
                api_url,
                headers=headers,
                json={
                    "model": model_name,
                    "messages": [{"role": "system", "content": system},
                                 {"role": "user", "content": user}],
                    # Low temperature: grounded extraction from official records, not
                    # creative writing. At 0.7 the model drifted on facts and language.
                    "temperature": temperature,
                    "max_tokens": budget,
                },
                timeout=timeout,
            )
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            net_attempt += 1
            if net_attempt < NETWORK_TRIES:
                time.sleep(1.5 * net_attempt)          # 1.5s, 3s, 4.5s backoff
                print(f"  ⚠ {llm_name} network error ({type(e).__name__}); retry {net_attempt}/{NETWORK_TRIES - 1}...")
                continue
            kind = "timed out" if isinstance(e, requests.exceptions.Timeout) else "cannot be reached"
            raise LLMError(f"{llm_name} {kind} after {net_attempt} attempts: {e}") from e

        if response.status_code >= 500:
            net_attempt += 1
            if net_attempt < NETWORK_TRIES:
                time.sleep(1.5 * net_attempt)
                print(f"  ⚠ {llm_name} {response.status_code}; retry {net_attempt}/{NETWORK_TRIES - 1}...")
                continue
        if response.status_code != 200:
            msg = response.text
            try:
                msg = response.json().get("error", {}).get("message", msg)
            except Exception:
                pass
            raise LLMError(f"{llm_name} {response.status_code}: {msg}")

        choice = (response.json().get("choices") or [{}])[0]
        message = choice.get("message") or {}
        truncated = choice.get("finish_reason") == "length"

        text = message.get("content") or ""
        if text.strip():
            return text, truncated

        # Empty content. A refusal is terminal; retrying will not change it.
        if message.get("refusal"):
            raise LLMError(f"model refused: {message['refusal']}")
        last_reasoning = len((message.get("reasoning_content") or "").strip())
        if truncated and attempt == 0:
            attempt += 1
            budget = min(budget * 2, 8192)
            print(f"  ⚠ Reasoning used the whole budget, empty reply — "
                  f"retrying with max_tokens={budget}...")
            continue
        break

    raise LLMError(
        f"model produced no answer within max_tokens={budget} — it spent the whole "
        f"budget reasoning ({last_reasoning} chars of reasoning_content). "
        f"Raise MAX_ANSWER_TOKENS.")


# ── Helper: Generate answer with Llama 3.3 70B via Groq API ─────
# ── Readable source labels + clean chunk text ──────────────────
# The vector store keys a chunk by its doc_dir (a urlhash). On its own that is
# meaningless to a reader — "5f369c50ead865ab.pdf" tells an officer nothing — so we
# resolve it back to the document's real title/URL via the pipeline registry.
_reg_cache = {"mtime": None, "idx": {}}


def _registry_index():
    """{doc_dir: record} from this pipeline's documents.json, reloaded on change."""
    reg = pc.DATA_ROOT / "documents.json"
    try:
        m = reg.stat().st_mtime
    except OSError:
        return {}
    if _reg_cache["mtime"] != m:
        try:
            docs = json.loads(reg.read_text(encoding="utf-8"))
            docs = docs if isinstance(docs, list) else docs.get("documents", [])
            _reg_cache["idx"] = {d.get("doc_dir"): d for d in docs if d.get("doc_dir")}
            _reg_cache["mtime"] = m
        except Exception:
            pass
    return _reg_cache["idx"]


def source_label(chunk_source):
    """(display_name, url) for a chunk's source — human-readable, never a bare hash.

    Prefers the document's title; falls back to a decoded URL path segment; only then
    to the hash. This is what a reader sees in the Sources list and what the model is
    told to refer to, so it must read like a document name, not an id.
    """
    rec = _registry_index().get(chunk_source, {}) or {}
    url = rec.get("source_url") or None
    name = (rec.get("original_name") or "").strip()
    if name and not name.lower().startswith(("http://", "https://")):
        return name, url
    if url:
        from urllib.parse import unquote, urlparse
        parts = [x for x in urlparse(url).path.split("/") if x]
        return (unquote(parts[-1]) if parts else urlparse(url).netloc), url
    return f"{chunk_source}.pdf", None


def clean_chunk_text(text):
    """Strip the chunk-file metadata header so only real document content reaches the
    LLM. Chunks are stored as `# Chunk N / Document: / Pages: / Tokens: / Headings:`
    then a `---` separator then the content. Left in, the model parrots that header
    into its answer ("Document: <hash> (Chunk 6)"). The header is short, so a `---`
    far into the text is a real horizontal rule and is left alone.
    """
    marker = '\n---\n'
    idx = text.find(marker)
    if 0 <= idx <= 400:
        return text[idx + len(marker):].strip()
    return (text or "").strip()


# A doc_dir is a 16-hex urlhash; the model sometimes echoes it as "<hash>.pdf".
# Match 8+ hex so shorter leaks are caught too. It is an internal id, not a title.
# NB: match only the token (and hugging parens) — never the surrounding spaces, or
# removal would fuse the words on either side ("document X lists" -> "documentlists").
_HASH_PDF = re.compile(r'\(?\b[0-9a-f]{8,}\.pdf\b\)?', re.I)

# A "Sources:" / "स्रोत:" style heading sitting on its own line — the start of a
# source list the model tacked on despite being told not to.
_SOURCES_HEADING = re.compile(
    r'\n[ \t]*(?:[*_#>\-]+[ \t]*)*'
    r'(?:sources?(?:\s+used)?|references?|source\s+documents?|स्रोत|स्रोतों|संदर्भ)'
    r'[ \t]*:?[ \t]*(?:\*\*|__)?[ \t]*(?:\n|$)',
    re.I)


def strip_model_source_refs(answer):
    """Keep document identifiers OUT of the answer prose — they belong only in the
    Sources list we append ourselves (what the UI shows under "View sources").

    sarvam-105b, despite the prompt, often (a) tacks its own "Sources used:" list of
    raw urlhashes onto the end and (b) drops a hash like `be302bf58abc3a98.pdf` into
    a sentence. Both read as gibberish to an official. We remove them here so the
    single, readable Sources list is the only place a document is named.
    """
    if not answer:
        return answer

    # 1. Cut a trailing source list the model wrote — but only when the block after
    #    the heading actually looks like one (bulleted, or naming .pdf files), so a
    #    legitimate "Source of funds:" heading in a budget answer is never touched.
    for m in _SOURCES_HEADING.finditer(answer):
        tail = answer[m.end():]
        if '.pdf' in tail.lower() or _HASH_PDF.search(tail) or re.search(r'^\s*[-*•]', tail, re.M):
            answer = answer[:m.start()]
            break

    # 2. Strip any bare urlhash filename left in the prose.
    answer = _HASH_PDF.sub('', answer)

    # 2b. Strip inline "[Source 3]", "(Source 3)", "Source 3:" citation tags. The
    #     synthesis prompt numbers the evidence as [Source N] and the model cites them
    #     back despite being told not to; a reader does not have that numbering.
    answer = re.sub(r'[\(\[]?\bSources?\s+\d+\b[\)\]]?\s*:?', '', answer, flags=re.I)

    # 3. Tidy the punctuation/whitespace the removals leave behind.
    answer = re.sub(r'\(\s*\)', '', answer)                       # empty ()
    answer = re.sub(r'\bdocuments?\s+(?=[,.;:)])', '', answer, flags=re.I)  # "document ." -> ""
    answer = re.sub(r'[ \t]{2,}', ' ', answer)                    # doubled spaces
    answer = re.sub(r'[ \t]+([,.;:)])', r'\1', answer)            # space before punctuation
    answer = re.sub(r'\n{3,}', '\n\n', answer)                    # collapse blank runs
    return answer.strip()


# ── Always-in-context reference note ──────────────────────────────────────────
# A curated, authoritative note (real CG AI-Mission / E&IT / recent-budget facts) that
# is prepended to EVERY answer's context, so a retrieval miss on those topics cannot
# blank the answer. The model is told to answer from it FIRST, then the retrieved docs.
_REFERENCE_CACHE = {"mtime": None, "text": ""}


def _reference_path():
    env = os.getenv("REFERENCE_CONTEXT_PATH")
    if env:
        return Path(env)
    # repo root: .../core/04_embeddings_and_kg/scripts/rag_pipeline.py -> parents[3]
    return Path(__file__).resolve().parents[3] / "CG_EIT_AI_Budget_Reference.md"


def reference_context():
    """The curated reference note's text, cached and reloaded on file change ('' if
    the file is absent, so this is always safe to call)."""
    p = _reference_path()
    try:
        m = p.stat().st_mtime
    except OSError:
        return ""
    if _REFERENCE_CACHE["mtime"] != m:
        try:
            _REFERENCE_CACHE["text"] = p.read_text(encoding="utf-8").strip()
            _REFERENCE_CACHE["mtime"] = m
        except Exception:
            _REFERENCE_CACHE["text"] = ""
    return _REFERENCE_CACHE["text"]


REFERENCE_HEADER = (
    "=== PRIORITY REFERENCE (curated & authoritative) ===\n"
    "Answer from THIS material first when it addresses the question (e.g. the CG AI Mission, "
    "Electronics & IT department, or recent state budgets); it holds verified figures. Use "
    "the retrieved documents below for anything it does not cover. Treat these figures as "
    "valid, in-scope, grounded facts.\n"
    "PRESENTATION RULES for your answer:\n"
    "- NEVER mention 'the reference note', 'the provided material', 'Section 4', section "
    "numbers, or where a fact came from. The reader only wants the facts, stated plainly.\n"
    "- For a straightforward factual question, do NOT add a 'Why this applies' explanation or "
    "restate your sourcing — just give the answer.\n"
    "- Do NOT add caveats that are unrelated to what was asked.\n")


def generate_answer(query, context_results):
    """Generate answer using the selected chat provider with retrieved context.
    
    Enhanced with knowledge graph information when available:
    - Mentions key entities found in results
    - Uses entity relationships for better context
    - Improves answer grounding with KG data
    - Shows source PDFs and highlighted excerpts
    """
    _mark_time("ANSWER_GENERATION")
    
    if not context_results:
        return "No context found to generate an answer."
    
    if LLM_PROVIDER == "sarvam" and not SARVAM_API_KEY:
        return "Error: SARVAM_API_KEY not configured. Please set the environment variable SARVAM_API_KEY."
    
    # Build the context. Two things matter here for answer quality:
    #  - clean_chunk_text strips the "# Chunk N / Document / Tokens / Headings"
    #    scaffolding so the model sees prose, not chunk metadata it will echo back.
    #  - each document is labelled with its READABLE title (source_label), so if the
    #    model names a document it uses the title, never a hash like 5f369c50....pdf.
    context_parts = []
    sources = []          # [(label, url)] deduped, in order of first appearance
    seen_sources = set()

    for i, r in enumerate(context_results, 1):
        source = r['point'].payload.get('source', '')
        label, url = source_label(source)
        text = clean_chunk_text(r['point'].payload.get('text', ''))

        if source not in seen_sources:
            seen_sources.add(source)
            sources.append((label, url))

        context_parts.append(f"--- Document {i}: {label} ---\n{text}")

    context_text = "\n\n".join(context_parts)

    # Prepend the always-in-context reference note so topics that retrieval misses
    # (AI Mission, E&IT, recent budgets) are still answerable from verified figures.
    _ref = reference_context()
    if _ref:
        context_text = f"{REFERENCE_HEADER}\n{_ref}\n\n=== RETRIEVED DOCUMENTS ===\n{context_text}"

    lang, language_rule_text, closing_line, answer_in = language_directive(query)
    language_rule = f"0. {language_rule_text}"

    system_content = f"""You are a government records assistant. You answer an official who
needs to act on your answer and has NOT read the source documents. Write the answer they
would want: clear, self-contained, and grounded only in the provided documents.

{language_rule}

HOW TO WRITE THE ANSWER
- Lead with a direct, plain-language answer to the question in the first sentence or two.
- Then give the supporting detail, organised for skimming: short paragraphs, and a bullet
  or numbered list when enumerating items, dates, posts, or steps.
- Use a bold lead-in for a section only when it genuinely helps a reader scan; do not force
  structure onto a short answer.
- Explain, do not transcribe. Turn the source content into readable prose. Never copy chunk
  scaffolding, table dumps, or raw fragments into the answer.

GROUNDING — THIS OVERRIDES EVERY OTHER INSTRUCTION
- The provided documents are the ONLY thing you know. You have NO background knowledge of
  Chhattisgarh, its budget, schemes, laws, departments or officials. If a fact is not in
  the documents, you do not know it — full stop.
- NEVER state a number, amount, rupee figure, percentage, date, section/rule number,
  demand number, scheme or department name unless those exact characters appear in the
  documents. Do not calculate, estimate, round, infer, or "recall" a value. A confident,
  plausible figure that is not in the documents is the single worst thing you can do.
- If the documents do not contain what was asked, say so in one plain sentence, name what
  is missing, and STOP. Do not soften the gap with general context or a plausible-sounding
  narrative. "The documents provided do not state X" is a CORRECT, valued answer.
- Test every sentence before writing it: can you point to the exact document it came from?
  If not, delete it. Prefer a short, certain answer over a long, impressive one.
- Refer to a document by its SUBJECT or title within a sentence when it aids the reader
  (e.g. "the 27 July 2026 seniority circular"). Do NOT write filenames, hashes, "Document 3",
  or "Chunk 6" in the answer — a Sources list is added automatically below.

JUDGEMENT
- Distinguish a PROPOSAL from a DECISION. Agenda notes state what was proposed; only the
  "DECISION TAKEN" / "निर्णय" text states what was approved. Never report a proposal
  as approved; if an item was deferred or dropped, say so.
- These documents span years and supersede one another. When they disagree about the current
  state of something, lead with the most recent and say plainly that the older position was
  later changed, rather than presenting it as still true.
- Do NOT append your own list of sources or references — that is added for you."""
    
    # The language directive is repeated here, immediately after the question. A rule
    # buried in the system prompt loses to several thousand tokens of Hindi context;
    # restating it as the last thing the model reads is what actually holds.
    user_message = f"""Documents:
{context_text}

Question: {query}

Write a clear, self-contained answer for an official, grounded only in the documents above.
{closing_line}
Your FIRST sentence must already be in {answer_in}, and must directly answer the question."""
    
    try:
        selected_model = SARVAM_MODEL if LLM_PROVIDER == "sarvam" else LLM_MODEL or _gpu_model or 'auto-discover'
        print(f"\n🤖 Generating answer with {LLM_PROVIDER} ({selected_model})...")
        answer, truncated = call_llm(system_content, user_message, temperature=0.2)

        # Keep document ids/hashes out of the prose; they belong only in the Sources
        # list below (the UI's "View sources"). Do this before we append our own list.
        answer = strip_model_source_refs(answer)

        # Surface truncation instead of silently returning a half-written answer.
        if truncated:
            print(f"  ⚠ Answer hit the {MAX_ANSWER_TOKENS}-token cap and was cut off mid-sentence.")
            answer += "\n\n_(Answer truncated — output token limit reached.)_"

        # ONE clean sources list, appended by us (the model is told not to write its
        # own). Readable titles, with the URL as a markdown link when we have one.
        heading = "स्रोत" if lang == "hi" else "Sources"
        if sources:
            _lines = [f"- [{lbl}]({u})" if u else f"- {lbl}" for lbl, u in sources]
            answer += f"\n\n**{heading}:**\n" + "\n".join(_lines)

        _mark_time("ANSWER_GENERATION")
        return answer

    except LLMError as e:
        _mark_time("ANSWER_GENERATION")
        return f"Error generating answer: {e}"
    except Exception as e:
        _mark_time("ANSWER_GENERATION")
        return f"Error generating answer: {e}"
    
    # ── COMMENTED OUT GROQ CODE (for future use) ──────────────────
    # try:
    #     print("\n🤖 Generating answer with Groq (Llama 3.3 70B)...")
    #     response = requests.post(
    #         GROQ_API_URL,
    #         headers={
    #             "Authorization": f"Bearer {GROQ_API_KEY}",
    #             "Content-Type": "application/json"
    #         },
    #         json={
    #             "model": GROQ_MODEL,
    #             "messages": [
    #                 {"role": "system", "content": system_content},
    #                 {"role": "user", "content": user_message}
    #             ],
    #             "temperature": 0.7,
    #             "max_tokens": 2048,
    #         },
    #         timeout=300  # 5 minute timeout
    #     )
    #     
    #     if response.status_code != 200:
    #         error_msg = response.text
    #         try:
    #             error_data = response.json()
    #             error_msg = error_data.get("error", {}).get("message", error_msg)
    #         except:
    #             pass
    #         return f"Error from Groq API: {response.status_code} - {error_msg}"
    #     
    #     result = response.json()
    #     answer = result.get("choices", [{}])[0].get("message", {}).get("content", "No response generated")
    #     
    #     # Append source information to the answer
    #     answer += f"\n\n**Sources used:**\n" + "\n".join([f"• {pdf}" for pdf in source_references])
    #     
    #     return answer
    # 
    # except requests.exceptions.ConnectionError:
    #     return f"Error: Cannot connect to Groq API. Check your internet connection and GROQ_API_KEY."
    # except requests.exceptions.Timeout:
    #     return "Error: Groq API request timed out. Please try again."
    # except Exception as e:
    #     return f"Error generating answer: {e}"

# ── Main RAG Pipeline ──────────────────────────────────────────
def rag_query(query):
    """Full RAG pipeline: retrieve context → generate answer.
    
    Steps:
    1. Retrieve context using embeddings + optional KG enhancement
    2. Rerank with BGE-Reranker
    3. Generate answer using Qwen via Ollama
    """
    global _pipeline_start, _stage_times
    _pipeline_start = None
    _stage_times = {}
    
    print(f"\n📝 Query: {query}\n")
    
    # Step 1: Retrieve context
    context_results = retrieve_context(query, num_context=8)
    
    if context_results is None:
        print("Failed to retrieve context.")
        return
    
    # Display retrieved context with KG information
    print(f"\n📚 Retrieved {len(context_results)} context documents:\n")
    for result in context_results:
        source = result['point'].payload.get('source', '')
        actual_filename = get_actual_filename(source)
        embedding_score = result.get('score', result.get('embedding_score', 0))
        
        # Show scores
        score_info = f"Embedding: {embedding_score:.4f}"

        print(f"[Rank {result['rank']}] {actual_filename}")
        print(f"  Scores: {score_info}")
        
        # Show entities if available
        if "entities" in result and result.get("entities"):
            entities = result["entities"][:5]
            print(f"  Entities: {', '.join(entities)}")
        
        print(f"  {result['point'].payload['text'][:200]}...\n")
    
    # Step 2: Generate answer with KG awareness
    answer = generate_answer(query, context_results)
    
    print("\n" + "=" * 70)
    print(answer)
    print("=" * 70)
    
    _print_timing_summary()
    
    return answer

# ── Interactive Loop ───────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 70)
    print(f"🚀 RAG Pipeline: bge-m3 hybrid retrieval + {LLM_PROVIDER}")
    print("=" * 70)
    
    retrieval_mode = "Hybrid (dense + sparse) + rerank" if RERANK_ENABLED else "Hybrid (dense + sparse)"
    print(f"\nRetrieval Mode: {retrieval_mode}")
    print(f"Multi-Query: {'Enabled' if USE_MULTI_QUERY else 'Disabled'}")
    print(f"Reranker: BGE-Reranker v2-M3")
    print(f"LLM: Qwen (via Ollama)")
    print("\nType 'exit' to quit, 'help' for commands.\n")
    print("=" * 70 + "\n")
    
    while True:
        query = input("Enter your question: ").strip()
        
        if query.lower() == "exit":
            print("\nGoodbye! 👋\n")
            break
        elif query.lower() == "help":
            print("\n" + "=" * 70)
            print("COMMANDS:")
            print("=" * 70)
            print("  help          - Show this help message")
            print("  config        - Show current configuration")
            print("  exit          - Exit the program")
            print("\nOtherwise, enter any question about your documents.")
            print("=" * 70 + "\n")
            continue
        elif query.lower() == "config":
            print("\n" + "=" * 70)
            print("CURRENT CONFIGURATION:")
            print("=" * 70)
            print(f"  Hybrid Alpha:          {HYBRID_ALPHA}")
            print(f"  Multi-Query:           {USE_MULTI_QUERY}")
            print(f"  Dense candidates:      {DENSE_CANDIDATE_LIMIT}")
            print(f"  Max chunks/source:     {MAX_CHUNKS_PER_SOURCE}")
            print(f"  Reranking:             {'ON' if RERANK_ENABLED else 'OFF (set RERANK_ENABLED=1)'}")
            if RERANK_ENABLED:
                print(f"    Rerank Min/Max K:    {RERANK_MIN_K} / {RERANK_MAX_K}")
                print(f"    Rerank Threshold:    {RERANK_THRESHOLD}")
            print("=" * 70 + "\n")
            continue
        elif not query:
            continue
        
        try:
            rag_query(query)
        except KeyboardInterrupt:
            print("\n\nInterrupted by user.\n")
        except Exception as e:
            print(f"Error processing query: {e}\n")
