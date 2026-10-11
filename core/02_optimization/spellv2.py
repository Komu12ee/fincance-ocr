from symspellpy import SymSpell, Verbosity
from pathlib import Path
import re
import os

# ── Config ────────────────────────────────────────────────────────────────────
# Environment-based paths for Docker compatibility
_SCRIPT_DIR = Path(__file__).parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
INPUT_MD    = Path(os.getenv("SPELL_INPUT", str(_PROJECT_ROOT / "02_optimization" / "output" / "output_corrected1.md")))
OUTPUT_MD   = Path(os.getenv("SPELL_OUTPUT", str(_PROJECT_ROOT / "02_optimization" / "output" / "output_corrected1_spell.md")))
DICT_PATH   = Path(os.getenv("SPELL_DICT", str(_SCRIPT_DIR / "dict" / "hi_dict_2_updated.txt")))
OUTPUT_DICT = Path(os.getenv("SPELL_DICT_OUT", str(_SCRIPT_DIR / "dict" / "hi_dict_2_updated.txt")))
DRY_RUN_LOG = Path(os.getenv("SPELL_LOG", str(_PROJECT_ROOT / "02_optimization" / "output" / "corrections_log.txt")))

FREQ_INCREMENT = 500

# Only correct words within this length range
MIN_WORD_LEN = 3
MAX_WORD_LEN = 6

# Edit distance scaled by word length
def max_edit_for_len(n: int) -> int:
    if n <= 4: return 1
    return 2

# Suggestion must be this many times more frequent than the input word
MIN_FREQ_RATIO_ED1 = 4    # edit distance 1
MIN_FREQ_RATIO_ED2 = 50   # edit distance 2

# If True: log corrections but don't write output files
DRY_RUN = False

DEVANAGARI_RE = re.compile(r'[\u0900-\u097F]+')
MD_SKIP       = (':','॰','#', '|', '```', '<!--', '>', '---')

# ── Known OCR error patterns (applied before SymSpell) ───────────────────────
# Add your own as you discover them: (compiled_regex, replacement_string)
OCR_PATTERNS = [
    # example: (re.compile(r'ा(?=\s|$)'), '')  # trailing matra artefact
]


# ── Helpers ───────────────────────────────────────────────────────────────────
def is_structural(line: str) -> bool:
    s = line.strip()
    return (not s
            or s.startswith(tuple(MD_SKIP))
            or re.match(r'^\s*[-*]\s', line)
            or re.match(r'^\s*\d+\.\s', line))


def apply_ocr_patterns(word: str) -> str:
    for pattern, replacement in OCR_PATTERNS:
        word = pattern.sub(replacement, word)
    return word


# ── Load dictionary (word -> frequency) ──────────────────────────────────────
def load_dict(dict_path: str) -> dict:
    freq_dict = {}
    for line in Path(dict_path).read_text(encoding='utf-8-sig').splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split('\t')
        if len(parts) == 2:
            word, freq = parts[0].strip(), parts[1].strip()
            try:
                freq_dict[word] = int(freq)
            except ValueError:
                pass
    print(f"  Dictionary loaded: {len(freq_dict):,} words")
    return freq_dict


# ── Build SymSpell ────────────────────────────────────────────────────────────
def build_symspell(freq_dict: dict) -> SymSpell:
    sym = SymSpell(max_dictionary_edit_distance=2)
    for word, freq in freq_dict.items():
        sym.create_dictionary_entry(word, freq)
    print(f"  SymSpell ready  : {sym.word_count:,} words")
    return sym


# ── Core correction logic ─────────────────────────────────────────────────────
def try_correct(sym: SymSpell, freq_dict: dict, word: str) -> tuple[str, str]:
    """
    Returns (corrected_word, skip_reason).
    skip_reason is empty string if a correction was made.
    Skips if:
      - word already in dictionary
      - length outside [MIN_WORD_LEN, MAX_WORD_LEN]
      - no suggestion found
      - suggestion frequency ratio too low
    """
    wlen = len(word)

    if wlen < MIN_WORD_LEN or wlen > MAX_WORD_LEN:
        return word, f"length {wlen} out of range"

    if word in freq_dict:
        return word, "already in dictionary"

    max_ed = max_edit_for_len(wlen)

    suggestions = sym.lookup(
        word,
        Verbosity.CLOSEST,
        max_edit_distance=max_ed,
        include_unknown=False
    )

    if not suggestions:
        return word, "no suggestion"

    best = suggestions[0]

    if best.distance == 0:
        return word, "exact match"

    # Frequency gate — suggestion must be significantly more common
    word_freq       = max(freq_dict.get(word, 0), 1)
    suggestion_freq = freq_dict.get(best.term, 1)
    required_ratio  = MIN_FREQ_RATIO_ED1 if best.distance == 1 else MIN_FREQ_RATIO_ED2

    if suggestion_freq < word_freq * required_ratio:
        return word, (f"freq ratio too low "
                      f"({suggestion_freq} / {word_freq} "
                      f"= {suggestion_freq / word_freq:.1f}x < {required_ratio}x required)")

    return best.term, ""


def correct_line(
    line: str,
    sym: SymSpell,
    freq_dict: dict,
    log_lines: list
) -> tuple[str, list[str]]:
    """Returns (corrected_line, all_devanagari_words_after_correction)."""
    if is_structural(line):
        return line, []

    result, offset = line, 0
    all_words = []

    for m in DEVANAGARI_RE.finditer(line):
        word = m.group()
        word = apply_ocr_patterns(word)

        corrected, skip_reason = try_correct(sym, freq_dict, word)

        if corrected != word:
            start   = m.start() + offset
            end     = m.end()   + offset
            result  = result[:start] + corrected + result[end:]
            offset += len(corrected) - len(word)
            log_lines.append(f"CORRECTED  {word:<20} →  {corrected}")
            all_words.append(corrected)
        else:
            # Only log non-trivial skips (useful for tuning)
            if skip_reason and skip_reason not in (
                "already in dictionary", "exact match", f"length {len(word)} out of range"
            ):
                log_lines.append(f"SKIPPED    {word:<20}    ({skip_reason})")
            all_words.append(word)

    return result, all_words


# ── Step 1: Spell check ───────────────────────────────────────────────────────
def run_spell_check(sym: SymSpell, freq_dict: dict) -> list[str]:
    print("\n── Step 1: Spell Check ─────────────────────────────────────────")
    lines     = Path(INPUT_MD).read_text(encoding='utf-8').splitlines(keepends=True)
    log_lines = []
    corrected_lines = []
    all_words = []
    fixes = 0

    for line in lines:
        fixed, words = correct_line(line, sym, freq_dict, log_lines)
        if fixed != line:
            fixes += 1
        corrected_lines.append(fixed)
        all_words.extend(words)

    Path(DRY_RUN_LOG).write_text('\n'.join(log_lines), encoding='utf-8')
    print(f"  Correction log  : {DRY_RUN_LOG}  ({len(log_lines)} entries)")

    if DRY_RUN:
        print(f"  DRY RUN — output file not written")
    else:
        Path(OUTPUT_MD).write_text(''.join(corrected_lines), encoding='utf-8')
        print(f"  Lines corrected : {fixes}")
        print(f"  Output saved    : {OUTPUT_MD}")

    return all_words


# ── Step 2: Update frequencies ────────────────────────────────────────────────
def update_frequencies(freq_dict: dict, doc_words: list[str]) -> dict:
    print("\n── Step 2: Update Frequencies ──────────────────────────────────")
    updated = 0
    for word in doc_words:
        if word in freq_dict:
            freq_dict[word] += FREQ_INCREMENT
            updated += 1
    print(f"  Words boosted   : {updated:,} (by +{FREQ_INCREMENT} each)")
    return freq_dict


# ── Save dictionary ───────────────────────────────────────────────────────────
def save_dict(freq_dict: dict, out_path: str):
    sorted_entries = sorted(freq_dict.items(), key=lambda x: x[1], reverse=True)
    Path(out_path).write_text(
        ''.join(f"{w}\t{f}\n" for w, f in sorted_entries),
        encoding='utf-8'
    )
    print(f"  Dict saved      : {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("── Hindi OCR Corrector + Dictionary Updater ────────────────────")
    if DRY_RUN:
        print("  *** DRY RUN MODE — no files will be written ***")

    freq_dict = load_dict(DICT_PATH)
    sym       = build_symspell(freq_dict)

    doc_words = run_spell_check(sym, freq_dict)

    if not DRY_RUN:
        freq_dict = update_frequencies(freq_dict, doc_words)
        save_dict(freq_dict, OUTPUT_DICT)

    print("\n── Done ─────────────────────────────────────────────────────────")


if __name__ == '__main__':
    main()