"""Flag OCR words a human should look at, and offer ranked replacements.

WHY THIS EXISTS
OCR on Devanagari government scans is good, not perfect, and the errors it makes are
the dangerous kind: a plausible-looking word that quietly changes what an order says.
The verify workspace already shows the source page beside the extracted text, but a
reviewer reading a two-page circular has no way to know WHICH of 400 words to doubt.
This narrows that to the handful worth checking.

WHAT IT DOES NOT DO
It never rewrites anything. It reports candidates; a human picks. That distinction is
the whole point — automated OCR post-correction is known to produce "contextually
plausible but factually incorrect" fixes that read fine and pass unnoticed, which in a
seniority list or a reservation rule is worse than leaving the error visible. Ranked
suggestions with a person choosing is the safe shape for the same technology.

WHY NOT PER-WORD OCR CONFIDENCE
That would be the obvious signal, but stage2 records confidence per PAGE only
(page_number / ocr_score / source in *_confidence.json) — there is no per-word score to
read. Page score is still used as a HINT: a low-scoring page's words are surfaced
first, since that is where errors cluster.

HOW A WORD BECOMES SUSPECT (in priority order)
  broken_encoding  contains C0 control characters — the PDF's font mapped conjuncts to
                   non-text codepoints. Certain damage, not a guess.
  orphan_matra     begins with a combining vowel sign, which is structurally impossible
                   in Devanagari — the consonant it attached to was dropped.
  unknown_word     absent from a 90k-entry Hindi frequency dictionary.

Suggestions come from SymSpell (edit distance + corpus frequency), so a proposal is
only offered when a real, more-common word sits close by.

DELIBERATELY NOT FLAGGED, because on this corpus they are noise rather than signal:
tokens containing digits (order numbers, dates, section refs — where a "correction"
would be actively harmful), Latin-script tokens (the dictionary is Hindi; every English
word would light up), and 1-2 character tokens (too many valid particles).
"""

import functools
import os
import re
import unicodedata
from pathlib import Path

CORE = Path(__file__).resolve().parent
DEFAULT_DICT = CORE / "02_optimization" / "dict" / "hi_dict_2_updated.txt"

# Suggestion search. 2 is the practical ceiling for SymSpell on a 90k dictionary:
# beyond it the candidate list fills with unrelated words of similar shape.
MAX_EDIT_DISTANCE = 2
MAX_SUGGESTIONS = 5

# Tokens shorter than this are skipped — Hindi has many valid 1-2 char particles
# (के, का, है, में) and flagging them buries the real errors.
MIN_WORD_LEN = 3

# A page scoring below this is reported as low-confidence so the UI can surface it
# first. Matches the range seen in real confidence files (0.45-0.80).
LOW_PAGE_CONFIDENCE = 0.60

_DEV_MATRAS = set("ािीुूृॄॅॆेैॉॊोौ्ंःँ़ॢॣ")
_DEV_RANGE = ("ऀ", "ॿ")

# Word = a run of Devanagari (letters + combining marks), which MAY contain the C0
# control characters a broken PDF font emits mid-word ("छ\x02ीसगढ़"). Including them
# matters: with a Devanagari-only pattern the control byte splits the token in two, so
# the pieces get reported as an orphaned matra and a too-short fragment, and the
# broken_encoding branch can never fire at all. Tab/newline/CR are excluded — they are
# real separators, not damage. Latin and digits stay out entirely (see module docstring),
# and a token must START with Devanagari so a run of control bytes alone is not a word.
_CTRL = "\x00-\x08\x0b\x0c\x0e-\x1f\x7f"
_WORD_RE = re.compile(rf"[ऀ-ॿ][ऀ-ॿ{_CTRL}]*")


def _is_devanagari(ch):
    return _DEV_RANGE[0] <= ch <= _DEV_RANGE[1]


def has_control_chars(s):
    return any((ord(c) < 0x20 and c not in "\t\n\r") or ord(c) == 0x7F for c in s)


def starts_with_matra(s):
    return bool(s) and s[0] in _DEV_MATRAS


@functools.lru_cache(maxsize=4)
def _load_symspell(dict_path):
    """Load the frequency dictionary once per process (~1s, ~90k entries).

    Cached because the verify panel calls this per document and re-reading the
    dictionary each time would dominate the request.
    """
    from symspellpy import SymSpell

    sym = SymSpell(max_dictionary_edit_distance=MAX_EDIT_DISTANCE, prefix_length=7)
    # term<TAB>frequency, one per line. separator="\t" is NOT optional: symspellpy
    # defaults to a space, and on a tab-separated file it then fails to parse every
    # single line — while still returning True. The symptom is a dictionary that
    # loads "successfully" with word_count == 0, so every word looks misspelled and
    # no suggestion is ever produced. Verified: without it 0 words load, with it 90026.
    if not sym.load_dictionary(str(dict_path), term_index=0, count_index=1,
                               encoding="utf-8", separator="\t"):
        raise RuntimeError(f"could not load spelling dictionary: {dict_path}")
    if not sym.word_count:
        raise RuntimeError(
            f"spelling dictionary {dict_path} loaded 0 words — check the separator/format")
    return sym


def dictionary_path():
    return Path(os.getenv("SPELL_DICT", str(DEFAULT_DICT)))


def suggest(word, sym=None, limit=MAX_SUGGESTIONS):
    """Ranked replacements for `word`, closest-and-most-frequent first."""
    from symspellpy import Verbosity

    sym = sym or _load_symspell(str(dictionary_path()))
    out = []
    for s in sym.lookup(word, Verbosity.CLOSEST, max_edit_distance=MAX_EDIT_DISTANCE):
        if s.term == word:
            continue
        out.append({"term": s.term, "distance": s.distance, "frequency": s.count})
        if len(out) >= limit:
            break
    return out


def _known(word, sym):
    """Is the word in the dictionary as-is? (distance 0 lookup)"""
    from symspellpy import Verbosity

    for s in sym.lookup(word, Verbosity.TOP, max_edit_distance=0):
        if s.term == word:
            return True
    return False


def find_issues(text, page_scores=None, max_issues=400):
    """Words in `text` a reviewer should check.

    Args:
        text: the extracted markdown.
        page_scores: optional [{page_number, ocr_score, source}, ...] from stage2's
            *_confidence.json — used only to report which pages were weak.
        max_issues: hard cap so a badly-damaged document cannot produce a payload
            (or a UI) with thousands of highlights.

    Returns:
        {issues: [...], counts: {...}, truncated: bool, low_confidence_pages: [...]}

    Each issue carries `start`/`end` character offsets into `text`, so the frontend
    highlights exactly what was analysed instead of re-tokenising and drifting.
    """
    # NOT normalised as a whole. NFC composition changes Devanagari character counts
    # (nukta forms collapse), so offsets computed on a normalised copy do not line up
    # with the text the caller holds — measured: 87 of 115 offsets pointed at the
    # wrong span. Normalisation is applied PER WORD for the dictionary lookup instead,
    # which is where it actually matters, leaving offsets anchored to the real text.
    text = text or ""
    sym = _load_symspell(str(dictionary_path()))

    issues = []
    counts = {"broken_encoding": 0, "orphan_matra": 0, "unknown_word": 0}
    truncated = False
    # A misread repeats throughout a document ("नवा" appears on every page of a
    # letterhead). Both the dictionary check and the far more expensive suggestion
    # search are memoised per distinct word — this is the difference between a
    # response the reviewer waits for and one that feels instant.
    verdict_cache = {}
    suggestion_cache = {}

    for m in _WORD_RE.finditer(text):
        word = m.group(0)
        if len(word) < MIN_WORD_LEN:
            continue

        reason = verdict_cache.get(word, ...)
        if reason is ...:
            if has_control_chars(word):
                reason = "broken_encoding"
            elif starts_with_matra(word):
                reason = "orphan_matra"
            elif _known(unicodedata.normalize("NFC", word), sym):
                reason = None                 # clean — remembered so we skip it fast
            else:
                reason = "unknown_word"
            verdict_cache[word] = reason
        if reason is None:
            continue

        if len(issues) >= max_issues:
            truncated = True
            break

        counts[reason] += 1
        if word not in suggestion_cache:
            suggestion_cache[word] = suggest(unicodedata.normalize("NFC", word), sym)
        issues.append({
            "word": word,
            "start": m.start(),
            "end": m.end(),
            "reason": reason,
            "suggestions": suggestion_cache[word],
        })

    low_pages = []
    for p in (page_scores or []):
        score = p.get("ocr_score")
        if isinstance(score, (int, float)) and score < LOW_PAGE_CONFIDENCE:
            low_pages.append({"page": p.get("page_number"), "score": round(score, 3),
                              "source": p.get("source")})

    return {
        "issues": issues,
        "counts": counts,
        "total": len(issues),
        "distinct_words": len(suggestion_cache),
        "truncated": truncated,
        "low_confidence_pages": low_pages,
    }


if __name__ == "__main__":   # quick manual check: python core/spellcheck.py <file>
    import json
    import sys

    src = Path(sys.argv[1]).read_text(encoding="utf-8", errors="replace")
    print(json.dumps(find_issues(src), ensure_ascii=False, indent=2)[:3000])
