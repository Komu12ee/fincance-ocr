"""Configuration constants for Stage 1 — Image Preparation."""

from pathlib import Path

# ---------- PDF-to-Image ----------
DPI = 450  # Preserve tiny punctuation for OCR
IMAGE_FORMAT = "png"  # Lossless — no artifacts introduced

# ---------- Deskew ----------
# Maximum angle correction (degrees). Pages tilted beyond this are likely
# landscape or rotated 90°, not just slightly skewed from scanning.
MAX_SKEW_ANGLE = 15.0
# Minimum angle to bother correcting — skip tiny angles to preserve dots.
MIN_SKEW_ANGLE = 1.5

# ---------- Denoise ----------
# Connected components with area < this are treated as noise speckles.
# Set low to preserve punctuation and Devanagari micro-components.
MIN_COMPONENT_AREA = 3

# ---------- Stamp / Annotation Detection ----------
# HSV ranges for common government stamp colors.
# Each entry: (name, lower_hsv, upper_hsv)
STAMP_COLOR_RANGES = [
    # Red stamps (wraps around H=0 in HSV, so two ranges)
    ("red_low",  (0, 70, 50),   (10, 255, 255)),
    ("red_high", (160, 70, 50), (180, 255, 255)),
    # Blue stamps / ink pads
    ("blue",     (90, 50, 50),  (130, 255, 255)),
    # Purple / violet stamps
    ("purple",   (130, 50, 50), (160, 255, 255)),
]
# Fraction of page area a colored blob must exceed to count as a stamp.
STAMP_MIN_AREA_RATIO = 0.002
# Fraction of page area above which a blob is probably a colored background,
# not a stamp.
STAMP_MAX_AREA_RATIO = 0.25

# ---------- Output ----------
DEFAULT_OUTPUT_DIR = Path("stage1_output")
