"""Thinking mode — multi-hop retrieval with an inspectable trace.

WHY THIS EXISTS
Single-shot RAG retrieves once and answers. That is right for "what is the sanctioned
strength of X", and wrong for the questions this corpus actually gets asked:

  "Notice A says one thing and notice B says another — which applies to my case?"
  "Is this officer eligible under the current rules?"
  "What changed between the 2019 order and now?"

Those need more than one look at the corpus. You cannot retrieve for a question you
have not realised you need to ask yet: you first find the 2019 order, and only then
learn there is a 2023 amendment worth retrieving. So this module runs a loop —
retrieve, read what came back, decide what is still missing, retrieve again — for a
bounded number of hops, then answers from everything it gathered.

THE LOOP
    PLAN        break the question into sub-questions              (LLM)
    RETRIEVE    hybrid search for each, merged and deduped         (deterministic)
    ASSESS      what is answered, what is missing, what to ask next(LLM)
      ...repeat RETRIEVE/ASSESS while there are gaps and hops left...
    TIMELINE    order the evidence by date, flag supersession      (deterministic)
    SYNTHESIZE  answer, with precedence rules made explicit        (LLM)

EVERY STEP IS RECORDED. `deep_query` returns a `trace`: an ordered list of what it
did, why, what came back and how long it took. The UI renders it as an expandable
panel. That is deliberate — an answer about which government order applies is only
useful if you can check the reasoning, and a black box that says "notice B applies"
is worse than useless in an official setting.

WHAT IS DETERMINISTIC ON PURPOSE
Date extraction and supersession detection are regex, not LLM. They are the load
bearing part of a precedence answer, they are cheap, and — most importantly — they
are auditable: the trace shows the exact date string matched and the exact phrase
("in supersession of", "अधिक्रमण में") that triggered the ruling. An LLM asked to
"figure out which is newer" gives you a confident sentence with nothing to check.
The LLM is used where judgment is genuinely needed: decomposing the question,
noticing gaps, and writing the final answer.

USAGE
    from thinking_mode import deep_query
    out = deep_query("Which leave rules apply to a deputationist?")
    out["answer"]           # str
    out["trace"]            # [{step, kind, title, detail, elapsed_s}, ...]
    out["results"]          # the retrieved chunks, same shape as retrieve_context
"""

import json
import re
import time
from datetime import date

import rag_pipeline as rp

# ── Tunables ────────────────────────────────────────────────────────────────
# Hops are the expensive dimension: each one costs a retrieval plus an LLM call.
# 3 is enough for "find the rule -> find its amendment -> check the exception";
# beyond that the model tends to re-ask what it already has.
MAX_HOPS = 3
SUB_QUESTIONS_PER_HOP = 3      # asked per hop; kept small so each stays specific
CHUNKS_PER_SUB_QUESTION = 6    # retrieved per sub-question before dedupe
MAX_EVIDENCE_CHUNKS = 24       # hard cap fed to synthesis, keeps the prompt bounded
SNIPPET_CHARS = 400            # per-chunk text shown in the trace (not to the LLM)
# sarvam-105b spends part of its budget reasoning before it writes the answer, so the
# 4096 default cut multi-part synthesis answers off mid-sentence. Give the final step
# room for reasoning AND a complete reply.
SYNTH_MAX_TOKENS = 8192

# The configured model is a REASONING model: it burns tokens on chain-of-thought
# before emitting anything in `content`. A budget sized for the visible JSON (a few
# hundred tokens) is spent entirely on reasoning and returns content=None. These
# budgets cover thinking AND the reply — do not shrink them to "just enough JSON".
REASONING_TOKENS = 3000


# ── Supersession vocabulary ─────────────────────────────────────────────────
# Phrases that mark one order acting on another. Split by direction because they
# mean opposite things for precedence:
#   SUPERSEDES_*  this document REPLACES an earlier one   -> this one wins
#   AMENDS_*      this document MODIFIES an earlier one   -> read both together
#   CONTINUES_*   this document EXTENDS an earlier one    -> earlier still stands
# Hindi and English are both present because this corpus mixes them within a
# single order (an English body under a Hindi header is routine).
SUPERSEDES_PATTERNS = [
    r"in supersession of", r"supersession", r"supersede[sd]?", r"superseding",
    r"in cancellation of", r"hereby cancell?ed", r"stands? cancell?ed",
    r"rescind(?:ed|s)?", r"withdrawn with effect", r"repeal(?:ed|s)?",
    r"अधिक्रमण", r"निरस्त", r"निरस्तीकरण", r"रद्द",
]
AMENDS_PATTERNS = [
    r"in partial modification", r"partially modif(?:y|ied)", r"amendment to",
    r"amend(?:ed|ment|s)?", r"modif(?:y|ied|ication)", r"substituted? for",
    r"read as", r"आंशिक संशोधन", r"संशोधन", r"संशोधित",
]
CONTINUES_PATTERNS = [
    r"in continuation of", r"further to", r"in reference to", r"in pursuance of",
    r"क्रम में", r"के संदर्भ में", r"अनुक्रम में",
]

_REL_KINDS = (
    ("supersedes", SUPERSEDES_PATTERNS),
    ("amends", AMENDS_PATTERNS),
    ("continues", CONTINUES_PATTERNS),
)


# ── Date extraction ─────────────────────────────────────────────────────────
_MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}
_HI_MONTHS = {
    "जनवरी": 1, "फरवरी": 2, "मार्च": 3, "अप्रैल": 4, "मई": 5, "जून": 6,
    "जुलाई": 7, "अगस्त": 8, "सितंबर": 9, "सितम्बर": 9, "अक्टूबर": 10,
    "नवंबर": 11, "नवम्बर": 11, "दिसंबर": 12, "दिसम्बर": 12,
}

# Numeric d/m/y — govt documents are day-first, so 03/04/2024 is 3 April.
_RE_NUMERIC = re.compile(r"\b(\d{1,2})[./-](\d{1,2})[./-]((?:19|20)\d{2})\b")
_RE_TEXT_DMY = re.compile(
    r"\b(\d{1,2})\s*(?:st|nd|rd|th)?[\s.,-]+"
    r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*[\s.,-]+((?:19|20)\d{2})\b", re.I)
_RE_TEXT_MDY = re.compile(
    r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*[\s.,-]+"
    r"(\d{1,2})\s*(?:st|nd|rd|th)?[\s.,-]+((?:19|20)\d{2})\b", re.I)
_RE_HI = re.compile(r"\b(\d{1,2})\s+(" + "|".join(_HI_MONTHS) + r")\s+((?:19|20)\d{2})\b")
_RE_YEAR = re.compile(r"\b((?:19|20)\d{2})\b")


def _safe_date(y, m, d):
    try:
        return date(int(y), int(m), int(d))
    except (ValueError, TypeError):
        return None


def extract_dates(text):
    """Every date found in `text`, as (date, matched_string), newest last.

    Day-first for numeric forms (Indian government convention). A bare year is only
    used when nothing more precise is present — it becomes 1 Jan of that year and is
    marked as such by its short matched string, so the trace shows the imprecision
    rather than hiding it.
    """
    found = []
    for m in _RE_NUMERIC.finditer(text):
        d = _safe_date(m.group(3), m.group(2), m.group(1))
        if d:
            found.append((d, m.group(0)))
    for m in _RE_TEXT_DMY.finditer(text):
        d = _safe_date(m.group(3), _MONTHS[m.group(2).lower()[:3]], m.group(1))
        if d:
            found.append((d, m.group(0)))
    for m in _RE_TEXT_MDY.finditer(text):
        d = _safe_date(m.group(3), _MONTHS[m.group(1).lower()[:3]], m.group(2))
        if d:
            found.append((d, m.group(0)))
    for m in _RE_HI.finditer(text):
        d = _safe_date(m.group(3), _HI_MONTHS[m.group(2)], m.group(1))
        if d:
            found.append((d, m.group(0)))
    if not found:
        for m in _RE_YEAR.finditer(text):
            d = _safe_date(m.group(1), 1, 1)
            if d:
                found.append((d, m.group(1)))
    return sorted(set(found))


def find_relations(text):
    """Supersession/amendment phrases present in `text`.

    Returns [(kind, matched_phrase), ...] where kind is supersedes|amends|continues.
    The matched phrase is kept so the trace can quote the exact words that drove the
    precedence call instead of asserting it.
    """
    low = text.lower()
    out = []
    for kind, patterns in _REL_KINDS:
        for pat in patterns:
            m = re.search(pat, low, re.I)
            if m:
                out.append((kind, m.group(0)))
                break          # one hit per kind is enough to flag the document
    return out


# ── Trace ───────────────────────────────────────────────────────────────────
class Trace:
    """Ordered record of what thinking mode did. This is the product, not a log."""

    def __init__(self, on_step=None):
        self.steps = []
        self._on_step = on_step

    def add(self, kind, title, detail=None, started=None):
        step = {
            "step": len(self.steps) + 1,
            "kind": kind,
            "title": title,
            "detail": detail or {},
            "elapsed_s": round(time.time() - started, 2) if started else 0.0,
        }
        self.steps.append(step)
        print(f"  🧠 [{step['step']}] {kind.upper()}: {title} ({step['elapsed_s']}s)")
        if self._on_step:
            try:
                self._on_step(step)
            except Exception:
                pass           # a broken listener must never break the query
        return step


# ── LLM plumbing ────────────────────────────────────────────────────────────
def _json_from_llm(text):
    """Parse JSON out of an LLM reply, tolerating prose and code fences.

    Models wrap JSON in ```json blocks, prefix it with "Here is...", or add a
    trailing note. Rather than demand perfection from the prompt, extract the first
    balanced {...} and parse that. Returns None if nothing parses — callers treat
    that as "no structured answer" and fall back.
    """
    if not text:
        return None
    fence = re.search(r"```(?:json)?\s*(.+?)```", text, re.S)
    if fence:
        text = fence.group(1)
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i, ch in enumerate(text[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    return None
    return None


PLANNER_SYSTEM = """You plan how to search a corpus of Indian government documents
(orders, circulars, notices, office memoranda, meeting minutes) in Hindi and English.

Break the user's question into focused sub-questions that can each be answered by
retrieving a passage. Think about which ARCHETYPE the question is, because each needs
different sub-questions:

- PRECEDENCE ("which notice applies", "is this still in force"): retrieve the rule
  itself, AND anything that amends/supersedes/cancels it, AND its effective date.
- COMPARISON ("A vs B", "what changed"): retrieve each side separately.
- ELIGIBILITY ("does X qualify"): retrieve the criteria, the exceptions, and any
  clarification issued later.
- PROCEDURE ("how do I", "what is the process"): retrieve the steps and the
  authority competent to approve.
- AGGREGATION ("list all", "how many"): retrieve each category separately.
- FACTUAL (single lookup): one or two sub-questions is enough — do not pad.

Rules:
- Write sub-questions in the SAME language as the user's question.
- Each sub-question must stand alone (no "it", "that order").
- Prefer the document's own vocabulary over the user's paraphrase.
- Do not invent document numbers or dates you have not seen.

Reply with ONLY JSON:
{"archetype": "<one of the above>", "reasoning": "<one sentence>",
 "sub_questions": ["...", "..."]}"""

ASSESS_SYSTEM = """You are checking whether retrieved passages actually answer a
question, and deciding what to search for next.

Be strict. "Related to the topic" is not "answers the question". In particular:
- If the passages give a rule but nothing about whether it was later amended,
  superseded or cancelled, that is a GAP for any question about what currently applies.
- If they mention another order by number or date without its contents, that is a GAP.
- If they state a proposal but not the decision taken, that is a GAP.
- If a date or authority the question depends on is missing, that is a GAP.

Do NOT invent follow-up questions when the evidence is genuinely sufficient — say so
and stop. Unnecessary hops cost the user time.

Reply with ONLY JSON:
{"sufficient": true|false,
 "answered": ["what the evidence establishes"],
 "gaps": ["what is still missing"],
 "next_questions": ["specific retrievable question", "..."]}
next_questions must be [] when sufficient is true."""


def _plan(question, trace):
    """Decompose the question. Falls back to the question itself on any failure."""
    t0 = time.time()
    try:
        raw, _ = rp.call_llm(PLANNER_SYSTEM, f"Question: {question}",
                             temperature=0.1, max_tokens=REASONING_TOKENS)
        parsed = _json_from_llm(raw) or {}
    except rp.LLMError as e:
        trace.add("plan", "Planner unavailable — searching the question as asked",
                  {"error": str(e), "sub_questions": [question]}, t0)
        return [question], "unknown"

    subs = [s for s in (parsed.get("sub_questions") or []) if isinstance(s, str) and s.strip()]
    subs = subs[:SUB_QUESTIONS_PER_HOP] or [question]
    archetype = parsed.get("archetype", "unknown")
    trace.add("plan", f"Recognised a {archetype} question, split into {len(subs)} searches",
              {"archetype": archetype, "reasoning": parsed.get("reasoning", ""),
               "sub_questions": subs}, t0)
    return subs, archetype


def _retrieve(questions, seen_ids, trace, hop):
    """Retrieve for each sub-question; return only chunks not already gathered."""
    t0 = time.time()
    fresh, per_q = [], []
    for q in questions:
        try:
            hits = rp.retrieve_context(q, num_context=CHUNKS_PER_SUB_QUESTION) or []
        except Exception as e:
            per_q.append({"question": q, "found": 0, "error": str(e)})
            continue
        new = 0
        for h in hits:
            pid = getattr(h.get("point"), "id", None)
            if pid is None or pid in seen_ids:
                continue
            seen_ids.add(pid)
            fresh.append(h)
            new += 1
        per_q.append({"question": q, "found": len(hits), "new": new})

    trace.add("retrieve", f"Hop {hop}: searched {len(questions)} question(s), "
                          f"{len(fresh)} new passage(s)",
              {"queries": per_q, "new_chunks": [_describe(h) for h in fresh]}, t0)
    return fresh


def _describe(hit):
    """Compact, human-readable summary of one retrieved chunk for the trace."""
    p = hit.get("point")
    payload = getattr(p, "payload", {}) or {}
    source = payload.get("source", "")
    text = payload.get("text", "") or ""
    return {
        "source": rp.get_actual_filename(source) if source else "?",
        "chunk": payload.get("chunk"),
        "score": round(float(hit.get("score", 0)), 4),
        "supporting": "neighbor_of" in hit,
        "snippet": text[:SNIPPET_CHARS] + ("..." if len(text) > SNIPPET_CHARS else ""),
    }


def _assess(question, evidence, trace, hop):
    """Ask whether we can answer yet, and what to retrieve next."""
    t0 = time.time()
    packed = _pack_evidence(evidence, limit=14)
    user = (f"Question: {question}\n\nRetrieved passages:\n{packed}\n\n"
            f"Can this question be answered fully from these passages?")
    try:
        raw, _ = rp.call_llm(ASSESS_SYSTEM, user, temperature=0.1, max_tokens=REASONING_TOKENS)
        parsed = _json_from_llm(raw) or {}
    except rp.LLMError as e:
        trace.add("assess", "Could not assess coverage — answering with what we have",
                  {"error": str(e)}, t0)
        return True, []

    sufficient = bool(parsed.get("sufficient"))
    gaps = [g for g in (parsed.get("gaps") or []) if isinstance(g, str)]
    nxt = [q for q in (parsed.get("next_questions") or []) if isinstance(q, str) and q.strip()]
    nxt = [] if sufficient else nxt[:SUB_QUESTIONS_PER_HOP]

    title = ("Evidence is sufficient — ready to answer" if sufficient
             else f"Found {len(gaps)} gap(s), searching {len(nxt)} more")
    trace.add("assess", f"Hop {hop}: {title}",
              {"sufficient": sufficient, "answered": parsed.get("answered") or [],
               "gaps": gaps, "next_questions": nxt}, t0)
    return sufficient, nxt


def _timeline(evidence, trace):
    """Deterministically order the evidence and flag supersession language.

    This is what turns "here are two conflicting notices" into "the later one
    supersedes the earlier, and here is the phrase that says so".
    """
    t0 = time.time()
    docs = {}
    for hit in evidence:
        payload = getattr(hit.get("point"), "payload", {}) or {}
        source = payload.get("source", "")
        if not source:
            continue
        text = payload.get("text", "") or ""
        rec = docs.setdefault(source, {
            "source": rp.get_actual_filename(source), "dates": [], "relations": [],
        })
        rec["dates"].extend(extract_dates(text))
        rec["relations"].extend(find_relations(text))

    rows = []
    for rec in docs.values():
        dates = sorted(set(rec["dates"]))
        rels = sorted(set(rec["relations"]))
        rows.append({
            "source": rec["source"],
            "latest_date": dates[-1][0].isoformat() if dates else None,
            "date_evidence": dates[-1][1] if dates else None,
            "all_dates": [d.isoformat() for d, _ in dates][:6],
            "relations": [{"kind": k, "phrase": p} for k, p in rels],
        })

    # Newest first; undated documents last (they cannot win a precedence argument).
    rows.sort(key=lambda r: (r["latest_date"] is not None, r["latest_date"] or ""), reverse=True)

    flagged = [r for r in rows if r["relations"]]
    dated = [r for r in rows if r["latest_date"]]
    if flagged:
        title = f"{len(dated)} dated document(s); {len(flagged)} contain supersession language"
    elif dated:
        title = f"{len(dated)} dated document(s), newest first"
    else:
        title = "No dates found in the evidence — precedence cannot be established"
    trace.add("timeline", title, {"documents": rows}, t0)
    return rows


def _pack_evidence(evidence, limit=MAX_EVIDENCE_CHUNKS):
    """Render chunks for an LLM prompt, labelled by source so it can cite them."""
    parts = []
    for i, hit in enumerate(evidence[:limit], 1):
        payload = getattr(hit.get("point"), "payload", {}) or {}
        # Label with the READABLE title/URL segment, never the raw urlhash — the model
        # echoes whatever label it is given, and a hash in the answer reads as gibberish.
        source = rp.source_label(payload.get("source", ""))[0] or "?"
        text = rp.clean_chunk_text(payload.get("text", ""))
        tag = " (adjacent context)" if "neighbor_of" in hit else ""
        parts.append(f"[Source {i}: {source}]{tag}\n{text}")
    return "\n\n".join(parts)


def _render_timeline(rows):
    """Timeline as text for the synthesis prompt."""
    if not rows:
        return "(no dated documents)"
    lines = []
    for r in rows:
        when = r["latest_date"] or "undated"
        rel = ""
        if r["relations"]:
            rel = "  [" + "; ".join(f'{x["kind"]}: "{x["phrase"]}"' for x in r["relations"]) + "]"
        lines.append(f"- {r['source']} — {when}{rel}")
    return "\n".join(lines)


SYNTH_SYSTEM = """You answer questions from Indian government documents, for staff who
will act on your answer. Accuracy and traceability matter more than fluency.

{language_rule}

GROUNDING — THIS OVERRIDES EVERY OTHER INSTRUCTION
- The numbered sources are the ONLY thing you know. You have NO background knowledge of
  Chhattisgarh, its budget, schemes, laws, departments or officials. If a fact is not in
  the sources, you do not know it — full stop.
- NEVER state a number, amount, rupee figure, percentage, date, section/rule number,
  demand number, scheme or department name unless those exact characters appear in a
  source. Do not calculate, estimate, round, infer, or "recall" a value. A confident,
  plausible figure that is not in the sources is the single worst thing you can do here.
- If the sources do not contain what was asked, say so in one plain sentence, name what
  is missing, and STOP. Do not soften the gap with general context or a plausible-sounding
  narrative. "The documents provided do not state the allocation for X" is a CORRECT,
  valued answer — not a failure.
- Test every sentence before writing it: can you point to the exact source it came from?
  If not, delete it. Prefer a short, certain answer over a long, impressive one.

RULES
1. Refer to a source by its SUBJECT or title within the sentence when it helps the reader
   (e.g. "the 27 July 2026 seniority circular"). Do NOT write filenames, hashes, or
   "[Source N]" tags in the answer — a Sources list is added automatically below.
2. PRECEDENCE. A document timeline is provided, newest first, with any supersession
   language quoted. When documents conflict:
   - the most recent one states the current position;
   - if a document says it supersedes/cancels an earlier one, the earlier one no
     longer applies — say so plainly and name both;
   - if it says it partially modifies/amends, BOTH apply — the original as changed;
     spell out what changed;
   - if it says it is in continuation of an earlier one, the earlier one still stands.
   State which document governs and WHY, quoting the phrase that settles it.
3. Distinguish a PROPOSAL from a DECISION. Agenda notes say what was proposed; only
   the "DECISION TAKEN"/"निर्णय" text says what was approved. Never report a proposal
   as approved; if something was deferred, say so.
4. If dates are missing and precedence therefore cannot be settled, say that outright
   rather than assuming an order.
5. Where the question was broken into parts, answer each part — including parts the
   sources cannot answer, which you mark as not covered.

STRUCTURE
- Lead with the direct answer.
- Then "Why this applies" — the precedence/date reasoning, when relevant.
- Then any caveats, conflicts or gaps.
Be concise. Do not pad."""


def _synthesize(question, evidence, timeline_rows, plan_archetype, trace):
    """Final answer over everything gathered."""
    t0 = time.time()
    # Shared with the single-shot path so both enforce language identically —
    # including Hinglish, which a Devanagari-vs-Latin check misses.
    lang, language_rule, closing_line, answer_in = rp.language_directive(question)

    # Always-in-context reference note first: verified CG AI-Mission / E&IT / recent-budget
    # facts, so those questions are answerable even when retrieval surfaces nothing on them.
    _ref = rp.reference_context()
    ref_block = f"{rp.REFERENCE_HEADER}\n{_ref}\n\n" if _ref else ""

    user = (f"Question: {question}\n\n"
            f"Question type: {plan_archetype}\n\n"
            f"{ref_block}"
            f"DOCUMENT TIMELINE (newest first):\n{_render_timeline(timeline_rows)}\n\n"
            f"RETRIEVED SOURCES:\n{_pack_evidence(evidence)}\n\n"
            f"{closing_line}\n"
            f"Your FIRST sentence must already be in {answer_in}.\n"
            f"Answer the question directly.")
    system = SYNTH_SYSTEM.format(language_rule=language_rule)
    try:
        answer, truncated = rp.call_llm(system, user, temperature=0.2,
                                        max_tokens=SYNTH_MAX_TOKENS)
    except rp.LLMError as e:
        trace.add("synthesize", f"Answer generation failed: {e}", {"error": str(e)}, t0)
        return f"Error generating answer: {e}"

    # Prompting alone does not hold: with two dozen Hindi passages in context the
    # model mirrors the sources and answers in Hindi even though the rule says
    # English in the system prompt, before the question, and as a first-sentence
    # constraint (observed on a real query). So the output is CHECKED, and one retry
    # is made with the instruction restated as the very last thing it reads. The
    # retry is recorded in the trace rather than hidden — if language enforcement is
    # costing an extra LLM call, that should be visible.
    retried = language_ok = False
    if rp.answer_language_ok(answer, lang):
        language_ok = True
    else:
        # Re-synthesising with a sterner instruction does NOT work — measured: the
        # model answered in Hindi again, because it is still reading 25 Hindi
        # passages while being asked to reason AND control language at once.
        # Translating the finished answer is a much narrower task with no competing
        # context, and it complies. Cheaper too: no need to re-read the evidence.
        retried = True
        trace.add("assess", f"Answer came back in the wrong language — translating to {answer_in}",
                  {"expected": answer_in}, t0)
        try:
            translated, truncated = rp.call_llm(
                (f"You are a translator. Translate the user's text into {answer_in}. "
                 f"Translate EVERYTHING, including headings and bullet labels. Preserve "
                 f"markdown structure, [Source N: filename] citations, document numbers, "
                 f"dates and numerals exactly as they appear. Output only the translation, "
                 f"with no preamble."),
                answer, temperature=0.1, max_tokens=SYNTH_MAX_TOKENS)
            if translated.strip():
                answer = translated
                language_ok = rp.answer_language_ok(answer, lang)
        except rp.LLMError:
            pass        # keep the original answer rather than failing the whole query

    # Keep document ids/hashes out of the prose — they belong only in the Sources
    # list below (the UI's "View sources"), the same rule as the single-shot path.
    answer = rp.strip_model_source_refs(answer)

    if truncated:
        answer += "\n\n_(Answer truncated — output token limit reached.)_"

    # ONE readable Sources list: deduped in order of first appearance, each a
    # markdown link when we have a URL — never a raw urlhash filename.
    sources, seen = [], set()
    for hit in evidence[:MAX_EVIDENCE_CHUNKS]:
        payload = getattr(hit.get("point"), "payload", {}) or {}
        s = payload.get("source", "")
        if not s or s in seen:
            continue
        seen.add(s)
        sources.append(rp.source_label(s))         # (label, url)
    if sources:
        heading = "स्रोत" if lang == "hi" else "Sources"
        _lines = [f"- [{lbl}]({u})" if u else f"- {lbl}" for lbl, u in sources]
        answer += f"\n\n**{heading}:**\n" + "\n".join(_lines)

    # Report what actually happened, not what was intended: if the answer is still
    # not in the requested language after translation, say so rather than claiming
    # success. A wrong-language answer the user can see flagged is recoverable; a
    # wrong-language answer labelled correct is not.
    if language_ok:
        lang_note = f"Answered in {answer_in}" + (" (after translation)" if retried else "")
    else:
        lang_note = f"⚠ could not force {answer_in} — answer is in the sources' language"
    trace.add("synthesize", f"{lang_note}, from {len(sources)} document(s)",
              {"sources": [lbl for lbl, _ in sources], "truncated": truncated,
               "language": answer_in, "language_retry": retried,
               "language_ok": language_ok}, t0)
    return answer


# ── Entry point ─────────────────────────────────────────────────────────────
def deep_query(question, max_hops=MAX_HOPS, on_step=None):
    """Answer `question` with multi-hop retrieval, returning the reasoning trace.

    Args:
        question: the user's question.
        max_hops: retrieve/assess rounds before answering with what is gathered.
        on_step: optional callable invoked with each trace step as it completes
                 (for streaming a live view to the UI).

    Returns:
        {answer, results, trace, hops, evidence_count, timeline}
    """
    started = time.time()
    trace = Trace(on_step=on_step)
    print(f"\n🧠 Thinking mode: {question}")

    sub_questions, archetype = _plan(question, trace)

    evidence, seen_ids = [], set()
    hop = 0
    pending = sub_questions
    while hop < max_hops and pending:
        hop += 1
        evidence.extend(_retrieve(pending, seen_ids, trace, hop))
        if not evidence:
            break
        if hop >= max_hops:
            trace.add("assess", f"Hop limit ({max_hops}) reached — answering with "
                                f"{len(evidence)} passage(s)", {"hop_limit": max_hops})
            break
        sufficient, pending = _assess(question, evidence, trace, hop)
        if sufficient or not pending:
            break

    if not evidence:
        # Retrieval found nothing — but the always-in-context reference note may still
        # answer (AI Mission / E&IT / recent budgets). Synthesise from it before giving
        # up; only return the hard "no documents" when there is no reference either.
        if rp.reference_context():
            trace.add("synthesize", "No passages retrieved — answering from the reference note", {})
            answer = _synthesize(question, [], [], archetype, trace)
            return {"answer": answer, "results": [], "trace": trace.steps, "hops": hop,
                    "evidence_count": 0, "timeline": [], "elapsed_s": round(time.time() - started, 2)}
        trace.add("synthesize", "No passages retrieved — cannot answer", {})
        return {"answer": "No relevant documents were found for this question.",
                "results": [], "trace": trace.steps, "hops": hop,
                "evidence_count": 0, "timeline": []}

    # Best evidence first, so the synthesis cap keeps the strongest passages.
    evidence.sort(key=lambda h: float(h.get("score", 0)), reverse=True)
    timeline_rows = _timeline(evidence, trace)
    answer = _synthesize(question, evidence, timeline_rows, archetype, trace)

    total = round(time.time() - started, 2)
    print(f"🧠 Thinking mode done in {total}s over {hop} hop(s), {len(evidence)} passage(s)")
    return {
        "answer": answer,
        "results": evidence[:MAX_EVIDENCE_CHUNKS],
        "trace": trace.steps,
        "hops": hop,
        "evidence_count": len(evidence),
        "timeline": timeline_rows,
        "elapsed_s": total,
    }
