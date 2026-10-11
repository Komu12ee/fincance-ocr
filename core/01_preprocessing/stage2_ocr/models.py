"""Data models for structured document output from OCR.

These models represent the structured content that Docling extracts —
headings, paragraphs, tables, lists — with full metadata about where
each element came from (page, position, role).
"""

from dataclasses import dataclass, field
from enum import Enum


class ElementType(str, Enum):
    """Type of structural element in the document."""
    TITLE = "title"
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST_ITEM = "list_item"
    TABLE = "table"
    HEADER = "header"          # page header / letterhead
    FOOTER = "footer"          # page footer
    PAGE_NUMBER = "page_number"
    FORMULA = "formula"
    UNKNOWN = "unknown"


@dataclass
class BoundingBox:
    """Bounding box of an element on the page (in pixels at source DPI)."""
    x: float
    y: float
    w: float
    h: float


@dataclass
class TableCell:
    """A single cell in a table."""
    row: int
    col: int
    row_span: int = 1
    col_span: int = 1
    text: str = ""
    is_header: bool = False


@dataclass
class Table:
    """A fully reconstructed table with rows, columns, and cell contents."""
    num_rows: int = 0
    num_cols: int = 0
    cells: list[TableCell] = field(default_factory=list)

    def to_markdown(self) -> str:
        """Render the table as a Markdown table."""
        if not self.cells or self.num_rows == 0 or self.num_cols == 0:
            return ""

        # Build grid
        grid = [["" for _ in range(self.num_cols)] for _ in range(self.num_rows)]
        for cell in self.cells:
            if 0 <= cell.row < self.num_rows and 0 <= cell.col < self.num_cols:
                grid[cell.row][cell.col] = cell.text.strip()

        lines = []
        for row_idx, row in enumerate(grid):
            line = "| " + " | ".join(row) + " |"
            lines.append(line)
            # Add separator after first row (header)
            if row_idx == 0:
                sep = "| " + " | ".join("---" for _ in row) + " |"
                lines.append(sep)

        return "\n".join(lines)

    def to_plain_text(self) -> str:
        """Render the table as tab-separated text."""
        if not self.cells or self.num_rows == 0:
            return ""

        grid = [["" for _ in range(self.num_cols)] for _ in range(self.num_rows)]
        for cell in self.cells:
            if 0 <= cell.row < self.num_rows and 0 <= cell.col < self.num_cols:
                grid[cell.row][cell.col] = cell.text.strip()

        return "\n".join("\t".join(row) for row in grid)


@dataclass
class DocumentElement:
    """A single structural element extracted from the document."""
    element_type: ElementType
    text: str
    page_num: int              # zero-indexed page number
    bbox: BoundingBox | None = None
    level: int = 0             # heading level (1, 2, 3...) if heading
    table: Table | None = None  # populated only for TABLE elements
    confidence: float | None = None  # OCR confidence if available

    def to_dict(self) -> dict:
        """Serialize to a plain dict for JSON output."""
        d = {
            "type": self.element_type.value,
            "text": self.text,
            "page": self.page_num + 1,
        }
        if self.bbox:
            d["bbox"] = {
                "x": round(self.bbox.x, 1),
                "y": round(self.bbox.y, 1),
                "width": round(self.bbox.w, 1),
                "height": round(self.bbox.h, 1),
            }
        if self.level > 0:
            d["level"] = self.level
        if self.table:
            d["table"] = {
                "rows": self.table.num_rows,
                "cols": self.table.num_cols,
                "cells": [
                    {
                        "row": c.row, "col": c.col,
                        "row_span": c.row_span, "col_span": c.col_span,
                        "text": c.text, "is_header": c.is_header,
                    }
                    for c in self.table.cells
                ],
            }
        return d


@dataclass
class PageOCRResult:
    """OCR result for a single page."""
    page_num: int
    elements: list[DocumentElement] = field(default_factory=list)
    raw_text: str = ""  # full plain text of the page
    confidence: float | None = None  # average OCR box confidence

    @property
    def tables(self) -> list[DocumentElement]:
        return [e for e in self.elements if e.element_type == ElementType.TABLE]

    @property
    def headings(self) -> list[DocumentElement]:
        return [e for e in self.elements if e.element_type in (ElementType.TITLE, ElementType.HEADING)]


@dataclass
class DocumentOCRResult:
    """Full OCR result for an entire document."""
    source_pdf: str
    total_pages: int
    pages: list[PageOCRResult] = field(default_factory=list)

    @property
    def all_elements(self) -> list[DocumentElement]:
        elements = []
        for page in self.pages:
            elements.extend(page.elements)
        return elements

    @property
    def all_tables(self) -> list[DocumentElement]:
        return [e for e in self.all_elements if e.element_type == ElementType.TABLE]

    @property
    def full_text(self) -> str:
        return "\n\n".join(page.raw_text for page in self.pages if page.raw_text)

    def to_markdown(self) -> str:
        """Render the entire document as structured Markdown.

        Uses Docling's native markdown (stored in raw_text) which has the
        best text quality and properly formatted tables.
        """
        parts = []
        for page in self.pages:
            parts.append(f"<!-- Page {page.page_num + 1} -->")
            if page.raw_text.strip():
                parts.append(page.raw_text)
            parts.append("")  # blank line between pages
        return "\n\n".join(parts)

    def to_dict(self) -> dict:
        """Serialize to a plain dict for JSON output."""
        all_elems = self.all_elements
        all_tbls = self.all_tables
        return {
            "source_pdf": self.source_pdf,
            "total_pages": self.total_pages,
            "total_elements": len(all_elems),
            "total_tables": len(all_tbls),
            "total_text_chars": len(self.full_text),
            "pages": [
                {
                    "page_num": p.page_num + 1,
                    "text": p.raw_text,
                    "confidence": None if p.confidence is None else round(p.confidence, 3),
                    "num_elements": len(p.elements),
                    "elements": [e.to_dict() for e in p.elements],
                }
                for p in self.pages
            ],
        }
