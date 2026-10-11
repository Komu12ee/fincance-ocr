#!/usr/bin/env python3
"""Short FG-only command for running OCR on the GPU VM's local API."""

from pathlib import Path
import sys

from gpu_ocr_pdf import main


if __name__ == "__main__":
    # Later CLI options can override these defaults when needed.
    raise SystemExit(main([
        "--base-url", "http://127.0.0.1:8001/v1", "--model", "ocr-primary",
        "--workers", "1", "--timeout", "300",
        "--prompt-file", str(Path(__file__).with_name("fg_ocr_prompt.txt")),
        *sys.argv[1:],
    ]))
