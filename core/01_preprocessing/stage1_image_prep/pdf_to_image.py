"""Convert each page of a PDF to a high-resolution image (450 DPI).

Uses PyMuPDF (fitz) which is a self-contained library — no external
poppler installation required, works cleanly on Windows / Colab / Linux.
"""

from __future__ import annotations

import math
import fitz  # PyMuPDF
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

from .config import DPI


def render_pdf_page(page: fitz.Page, dpi: int = DPI,
                    max_pixels: int | None = None) -> fitz.Pixmap:
    """Render one open PDF page as RGB, optionally bounding inference memory.

    Shared by the existing BGR helpers and remote GPU OCR. Call on the thread
    owning the document; PyMuPDF page operations must not run concurrently.
    """
    if dpi <= 0 or (max_pixels is not None and max_pixels <= 0):
        raise ValueError("dpi and max_pixels must be positive")
    zoom = dpi / 72
    pixels = page.rect.width * page.rect.height * zoom * zoom
    if max_pixels is not None and pixels > max_pixels:
        zoom *= math.sqrt(max_pixels / pixels)
    return page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), colorspace=fitz.csRGB, alpha=False)


def pdf_to_images(pdf_path: str | Path) -> list[np.ndarray]:
    """Convert every page of a PDF to a list of BGR numpy arrays at 450 DPI.

    Parameters
    ----------
    pdf_path : path to the PDF file.

    Returns
    -------
    List of numpy arrays (H, W, 3) in BGR color order, one per page.
    """
    import numpy as np

    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc = fitz.open(str(pdf_path))
    images = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        pix = render_pdf_page(page)
        # Convert to numpy array — PyMuPDF gives RGB
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, 3
        )
        # Convert RGB → BGR for OpenCV compatibility
        img_bgr = img[:, :, ::-1].copy()
        images.append(img_bgr)

    doc.close()
    return images


def pdf_page_to_image(pdf_path: str | Path, page_num: int) -> np.ndarray:
    """Convert a single page of a PDF to a BGR numpy array at 450 DPI.

    Parameters
    ----------
    pdf_path : path to the PDF file.
    page_num : zero-indexed page number.

    Returns
    -------
    Numpy array (H, W, 3) in BGR color order.
    """
    import numpy as np

    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc = fitz.open(str(pdf_path))
    if page_num < 0 or page_num >= len(doc):
        doc.close()
        raise ValueError(
            f"Page {page_num} out of range. PDF has {len(doc)} pages."
        )

    page = doc[page_num]
    pix = render_pdf_page(page)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
        pix.height, pix.width, 3
    )
    img_bgr = img[:, :, ::-1].copy()
    doc.close()
    return img_bgr


def get_page_count(pdf_path: str | Path) -> int:
    """Return the number of pages in a PDF."""
    doc = fitz.open(str(pdf_path))
    count = len(doc)
    doc.close()
    return count
