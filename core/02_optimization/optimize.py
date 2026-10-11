"""
docling_md_optimizer.py
=======================
General-purpose cleaner for Markdown produced by Docling from Hindi/English PDFs.
Fixes:
  - Floating single-character / numeric artifact lines
  - Broken table span-rows (repeated text across all columns)
  - Out-of-order list numbering
  - Duplicate blank lines / trailing whitespace
  - Page-break comment normalization
  - Mixed script whitespace issues
  - Stray OCR noise lines (very short, no alphanumeric meaning)
  - Redundant header deduplication
  - Misformatted Markdown headers
"""

import re
import sys
from pathlib import Path


# ════════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ════════════════════════════════════════════════════════════════════════════════

# Unicode ranges
DEVANAGARI = r'\u0900-\u097F'
LATIN       = r'A-Za-z'
DIGITS      = r'0-9'

def is_meaningful(text: str) -> bool:
    """True if line has enough real alphanumeric content to keep."""
    stripped = text.strip()
    # Must contain at least 2 alphanumeric / Devanagari characters
    alnum = re.findall(rf'[{DEVANAGARI}{LATIN}{DIGITS}]', stripped)
    return len(alnum) >= 2


def is_table_row(line: str) -> bool:
    s = line.strip()
    return s.startswith('|') and s.endswith('|') and s.count('|') >= 2


def is_separator_row(line: str) -> bool:
    """Markdown table separator: |---|---|"""
    s = line.strip()
    if not is_table_row(s):
        return False
    cells = [c.strip() for c in s.strip('|').split('|')]
    return all(re.fullmatch(r':?-+:?', c) or c == '' for c in cells)


def normalize_cells(row: str) -> list[str]:
    return [c.strip() for c in row.strip().strip('|').split('|')]


# ════════════════════════════════════════════════════════════════════════════════
# PASSES
# ════════════════════════════════════════════════════════════════════════════════

def pass_normalize_endings(md: str) -> str:
    """Unified line endings, collapse 3+ blank lines."""
    md = md.replace('\r\n', '\n').replace('\r', '\n')
    md = re.sub(r'\n{3,}', '\n\n', md)
    return md


def pass_page_comments(md: str) -> str:
    """Normalize <!-- Page N --> comments → clean section dividers."""
    # Keep them but ensure they're surrounded by blank lines
    md = re.sub(
        r'\s*<!--\s*[Pp]age\s*(\d+)\s*-->\s*',
        r'\n\n<!-- page \1 -->\n\n',
        md
    )
    md = re.sub(r'\n{3,}', '\n\n', md)
    return md


def pass_artifact_lines(lines: list[str]) -> list[str]:
    """
    Remove lines that are clearly OCR/extraction artifacts:
      - Single characters alone on a line (e.g. "6", "A<", "-00")
      - Pure punctuation lines
      - Page numbers
      - Lines with only non-alphanumeric content
    Preserves: table rows, headers, code blocks, blank lines.
    """
    cleaned = []
    in_code  = False

    for line in lines:
        stripped = line.strip()

        # Track code blocks — never touch them
        if stripped.startswith('```'):
            in_code = not in_code
            cleaned.append(line)
            continue
        if in_code:
            cleaned.append(line)
            continue

        # Always keep blank lines, table rows, headers, html comments
        if (not stripped
                or is_table_row(stripped)
                or stripped.startswith('#')
                or stripped.startswith('<!--')):
            cleaned.append(line)
            continue

        # Drop lines that are just page numbers: "84", "- 1 -", "Page 3"
        if re.fullmatch(r'\s*[-—]*\s*(Page\s*)?\d{1,4}\s*[-—]*\s*', stripped, re.I):
            continue

        # Drop very short lines with no real content (≤ 3 chars, no Devanagari/Latin)
        if len(stripped) <= 3 and not re.search(rf'[{DEVANAGARI}{LATIN}]', stripped):
            continue

        # Drop lines that are only symbols / punctuation / stray OCR chars
        # e.g. "A<", "-00", "li9", "lि9"  — heuristic: < 4 chars, not meaningful
        if len(stripped) < 5 and not is_meaningful(stripped):
            continue

        cleaned.append(line)

    return cleaned


def pass_fix_headers(lines: list[str]) -> list[str]:
    """
    - Normalize ## spacing
    - Remove bold wrapping inside headers
    - Ensure blank line before/after headers
    - Deduplicate consecutive identical headers
    """
    result      = []
    prev_header = None

    for line in lines:
        stripped = line.strip()

        m = re.match(r'^(#{1,6})\s*(.*)', stripped)
        if m:
            hashes  = m.group(1)
            content = m.group(2).strip()

            # Remove bold: **Title** → Title
            content = re.sub(r'^\*\*(.+?)\*\*$', r'\1', content)
            # Remove trailing punctuation artifacts
            content = content.rstrip(':-')

            line = f'{hashes} {content}'.rstrip()

            # Deduplicate: skip if same as previous header
            if line == prev_header:
                continue
            prev_header = line

            # Blank line before header
            if result and result[-1].strip():
                result.append('')
        else:
            prev_header = None

        result.append(line)

    return result


def pass_fix_tables(lines: list[str]) -> list[str]:
    """
    Collects table blocks and repairs:
      - Missing separator row after header
      - Span rows (all cells identical) — collapse to a label row
      - Uneven column counts — pad to max
    """
    result = []
    i      = 0

    while i < len(lines):
        if is_table_row(lines[i]):
            # Collect contiguous table lines
            block = []
            while i < len(lines) and (is_table_row(lines[i]) or lines[i].strip() == ''):
                if lines[i].strip():
                    block.append(lines[i])
                i += 1
            result.extend(_repair_table(block))
            result.append('')
        else:
            result.append(lines[i])
            i += 1

    return result


def _repair_table(rows: list[str]) -> list[str]:
    if not rows:
        return rows

    parsed   = [normalize_cells(r) for r in rows]
    max_cols = max(len(r) for r in parsed)

    # Pad rows
    for row in parsed:
        while len(row) < max_cols:
            row.append('')

    # Identify span rows: all non-empty cells have same value
    def is_span_row(row):
        vals = [c for c in row if c]
        return len(vals) > 1 and len(set(vals)) == 1

    # Identify separator rows
    def is_sep(row):
        return all(re.fullmatch(r':?-+:?', c) or c == '' for c in row)

    # Rebuild — convert span rows to single merged label rows
    rebuilt      = []
    header_done  = False
    sep_inserted = False

    for idx, row in enumerate(parsed):
        if is_span_row(row):
            # Collapse span row → bold label spanning full width
            label = row[0]
            # Emit as a merged "section label" row: | **LABEL** | | | ... |
            merged = ['**' + label + '**'] + [''] * (max_cols - 1)
            rebuilt.append('| ' + ' | '.join(merged) + ' |')

            # If this was the very first row, insert sep after it
            if not header_done and not sep_inserted:
                rebuilt.append('| ' + ' | '.join(['---'] * max_cols) + ' |')
                sep_inserted = True
                header_done  = True
            continue

        if is_sep(row):
            if not sep_inserted:
                rebuilt.append('| ' + ' | '.join(['---'] * max_cols) + ' |')
                sep_inserted = True
            continue

        rebuilt.append('| ' + ' | '.join(row) + ' |')

        # Insert separator after first non-span, non-sep row (the header)
        if not header_done and not sep_inserted:
            rebuilt.append('| ' + ' | '.join(['---'] * max_cols) + ' |')
            sep_inserted = True
            header_done  = True

    return rebuilt


def pass_fix_lists(lines: list[str]) -> list[str]:
    """
    - Normalize exotic bullets (–, —, •, ▪) → -
    - Fix out-of-order numbered lists (reset counter per uninterrupted block)
    - Fix "1)" → "1." style
    """
    result       = []
    list_counter = 0
    in_list      = False

    for line in lines:
        stripped = line.strip()

        # Normalize exotic bullets
        normed = re.sub(r'^(\s*)[–—•▪➤►▶]\s+', r'\1- ', line)
        if normed != line:
            line    = normed
            stripped = line.strip()

        # Fix "1)" → "1."
        line    = re.sub(r'^(\s*)(\d+)\)\s+', r'\1\2. ', line)
        stripped = line.strip()

        # Renumber ordered lists sequentially within a block
        m = re.match(r'^(\s*)(\d+)\.\s+(.*)', line)
        if m:
            indent  = m.group(1)
            content = m.group(3)
            if not in_list:
                list_counter = 0
                in_list      = True
            list_counter += 1
            line = f'{indent}{list_counter}. {content}'
        else:
            if in_list and stripped and not stripped.startswith('-'):
                in_list      = False
                list_counter = 0

        result.append(line)

    return result


def pass_dedup_paragraphs(lines: list[str]) -> list[str]:
    """
    Remove duplicate non-table, non-header paragraphs.
    Uses normalized comparison (collapse whitespace, lowercase).
    Only deduplicates lines with meaningful length (>30 chars).
    """
    seen   = set()
    result = []

    for line in lines:
        stripped  = line.strip()
        normalized = re.sub(r'\s+', ' ', stripped).lower()

        if (len(normalized) > 30
                and not is_table_row(stripped)
                and not stripped.startswith('#')
                and not stripped.startswith('<!--')):
            if normalized in seen:
                continue
            seen.add(normalized)

        result.append(line)

    return result


def pass_clean_whitespace(lines: list[str]) -> list[str]:
    """
    - Strip trailing whitespace
    - Fix broken hyphenation: word-\nnext → wordnext
    - Collapse multiple internal spaces (outside tables/code)
    """
    result   = []
    in_code  = False

    for line in lines:
        if line.strip().startswith('```'):
            in_code = not in_code

        if not in_code:
            line = line.rstrip()

            if not is_table_row(line):
                line = re.sub(r'(?<!\|)  +(?!\|)', ' ', line)

            # Merge broken hyphenation from previous line
            if (result
                    and result[-1].endswith('-')
                    and line
                    and line[0].islower()):
                result[-1] = result[-1][:-1] + line.lstrip()
                continue

        result.append(line)

    return result


def pass_final_blank_lines(md: str) -> str:
    """Ensure max 1 blank line between content blocks."""
    md = re.sub(r'\n{3,}', '\n\n', md)
    # Remove blank lines immediately after page comments
    md = re.sub(r'(<!-- page \d+ -->)\n{2,}', r'\1\n', md)
    return md


# ════════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ════════════════════════════════════════════════════════════════════════════════

def optimize(md: str) -> str:
    # String-level passes
    md = pass_normalize_endings(md)
    md = pass_page_comments(md)

    # Line-level passes
    lines = md.split('\n')
    lines = pass_artifact_lines(lines)
    lines = pass_fix_headers(lines)
    lines = pass_fix_tables(lines)
    lines = pass_fix_lists(lines)
    lines = pass_dedup_paragraphs(lines)
    lines = pass_clean_whitespace(lines)

    # Final string-level cleanup
    md = '\n'.join(lines)
    md = pass_final_blank_lines(md)
    return md.strip()


def optimize_file(src: str, dst: str = None) -> str:
    src_path = Path(src)
    raw      = src_path.read_text(encoding='utf-8')
    result   = optimize(raw)

    dst_path = Path(dst) if dst else src_path.with_stem(src_path.stem + '_clean')
    dst_path.write_text(result, encoding='utf-8')

    reduction = 100 * (1 - len(result) / max(len(raw), 1))
    print(f"Input  : {src_path}  ({len(raw):,} chars)")
    print(f"Output : {dst_path}  ({len(result):,} chars)")
    print(f"Reduced: {reduction:.1f}%")
    return result


# ════════════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    if len(sys.argv) >= 2:
        src = sys.argv[1]
        dst = sys.argv[2] if len(sys.argv) > 2 else None
        optimize_file(src, dst)
    else:
        # fallback test paths (Docker-compatible)
        from pathlib import Path
        script_dir = Path(__file__).parent
        project_root = script_dir.parent
        test_input = project_root / "02_optimization" / "output" / "output_corrected1.md"
        test_output = project_root / "02_optimization" / "output" / "output_corrected1_optimized.md"
        if test_input.exists():
            optimize_file(str(test_input), str(test_output))
        else:
            print(f"No test file found at {test_input}. Provide input and output paths as arguments.")

    