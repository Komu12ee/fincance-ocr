#!/usr/bin/env python3
"""Recursively OCR PDFs with a remote vision API and create searchable PDFs.

PDF operations stay on the main thread: PyMuPDF is not thread safe. Only remote
HTTP inference runs concurrently. Originals are never modified.
"""

from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import html
import json
import logging
import math
import os
from pathlib import Path
import re
import sys
import tempfile
import time
from urllib.parse import quote

import pymupdf as fitz
import requests
from dotenv import load_dotenv
from fg_ocr_status import (
    OCRTracker, atomic_json, cache_directory, is_complete, output_stat, print_status,
    read_json, source_signature, status_snapshot,
)

LOG = logging.getLogger("gpu_ocr")
PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "core" / "01_preprocessing"))
from stage1_image_prep.pdf_to_image import render_pdf_page
from stage2_ocr.models import (
    BoundingBox, DocumentElement, DocumentOCRResult, ElementType, PageOCRResult,
)
DEFAULT_PROMPT = """Perform OCR on this document page. Transcribe ALL visible text,
including headers, footers, tables, stamps and handwritten text when readable.
Preserve the original language, especially Hindi (Devanagari) and English.
Do not translate, summarize, explain, or invent missing text.
Return ONLY a JSON object in this form:
{"blank": false, "lines": [{"text": "one line of text", "bbox": [x1,y1,x2,y2]}]}
List lines in reading order. Coordinates must be normalized to 0..1000, with
origin at the top left of the supplied image. Each bbox encloses its line.
For tables, transcribe each cell separately, in row order, with its own bbox.
For a truly blank page return {"blank": true, "lines": []}.
If JSON coordinates are unsupported, return only the complete transcription.
"""
CACHE_VERSION = 2
# PDF writer fixes must rebuild outputs while retaining expensive GPU OCR pages.
TEXT_LAYER_VERSION = 2


def discover_pdfs(root: Path, output: Path):
    """Include mixed-case extensions, exclude output subtrees and symlinks."""
    for folder, dirs, names in os.walk(root, followlinks=False):
        dirs[:] = sorted(
            d for d in dirs
            if not (Path(folder) / d).is_symlink()
            and (Path(folder) / d).resolve() != output
        )
        for name in sorted(names):
            path = Path(folder) / name
            if path.suffix.lower() == ".pdf" and not path.is_symlink():
                yield path


def approximate_lines(text: str) -> dict:
    """Text-only models cannot provide accurate selection/highlight geometry."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        raise ValueError("Empty OCR response; refusing to mark this page complete")
    step = 920 / len(lines)
    return {
        "blank": False,
        "positioning": "approximate",
        "lines": [
            {"text": line, "bbox": [40, 40 + i * step, 960, 40 + (i + 1) * step]}
            for i, line in enumerate(lines)
        ],
    }


def parse_ocr(text: str) -> dict:
    text = text.strip()
    if text.startswith("```") and text.endswith("```"):
        text = re.sub(r"^```[^\n]*\n", "", text)
        text = re.sub(r"\n?```$", "", text).strip()
    if not text:
        raise ValueError("The OCR model returned no text")
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        # A malformed/truncated JSON response must not become searchable JSON.
        if text.startswith(("{", "[")):
            raise ValueError("Invalid OCR JSON; try --prompt-file with a text-only prompt")
        return approximate_lines(text)
    if isinstance(value, str):
        return approximate_lines(value)
    if not isinstance(value, dict):
        raise ValueError("OCR JSON must be an object containing lines")
    lines = value.get("lines")
    if not isinstance(lines, list):
        if isinstance(value.get("text"), str):
            return approximate_lines(value["text"])
        raise ValueError("OCR JSON is missing the lines array")
    if not lines:
        if value.get("blank") is True:
            return {"blank": True, "positioning": "model", "lines": []}
        raise ValueError("No OCR lines and page was not explicitly marked blank")
    if value.get("blank") is True:
        raise ValueError("OCR response marks a page blank but also contains text")
    clean = []
    missing_boxes = False
    for line in lines:
        if not isinstance(line, dict) or not isinstance(line.get("text"), str):
            raise ValueError("Each OCR line must have a text string")
        if not line["text"].strip():
            continue
        box = line.get("bbox")
        if box is None:
            missing_boxes = True
        else:
            if not isinstance(box, list) or len(box) != 4:
                raise ValueError("Each bbox must contain four normalized coordinates")
            if not all(isinstance(n, (float, int)) and not isinstance(n, bool)
                       and math.isfinite(n) for n in box):
                raise ValueError("Invalid bbox coordinates")
            if not (0 <= box[0] < box[2] <= 1000 and 0 <= box[1] < box[3] <= 1000):
                raise ValueError("bbox coordinates must be ordered and within 0..1000")
        clean.append({"text": line["text"].strip(), "bbox": box})
    if not clean:
        raise ValueError("OCR returned only whitespace")
    if missing_boxes:
        return approximate_lines("\n".join(line["text"] for line in clean))
    return {"blank": False, "positioning": "model", "lines": clean}


class GPUClient:
    def __init__(self, args, prompt: str):
        self.url = args.base_url.rstrip("/")
        self.model = args.model
        self.timeout = args.timeout
        self.retries = args.retries
        self.max_tokens = args.max_tokens
        self.prompt = prompt
        key = os.environ.get("GPU_OCR_API_KEY")
        self.headers = {"Authorization": f"Bearer {key}"} if key else {}

    def raise_for_status(self, response: requests.Response) -> None:
        """Keep the server's rejection reason, rather than just its HTTP code."""
        try:
            response.raise_for_status()
        except requests.HTTPError as error:
            try:
                body = response.json()
            except ValueError:
                body = response.text
            detail = body.get("error", body) if isinstance(body, dict) else body
            if isinstance(detail, dict):
                # vLLM/OpenAI-compatible errors expose these fields without images.
                detail = "; ".join(
                    f"{name}: {detail[name]}" for name in ("message", "type", "param", "code")
                    if detail.get(name) is not None
                ) or json.dumps(detail, ensure_ascii=False)
            detail = re.sub(r"data:image/[^\s\"']+", "[image data omitted]", str(detail))
            key = os.environ.get("GPU_OCR_API_KEY")
            if key:
                detail = detail.replace(key, "[API key omitted]")
            detail = " ".join(detail.split())[:4000] or "empty error response"
            message = f"GPU API returned HTTP {response.status_code}: {detail}"
            if response.status_code == 400:
                message += f" (model={self.model}, max_tokens={self.max_tokens})"
            raise requests.HTTPError(message, response=response, request=response.request) from error

    def check(self) -> None:
        response = requests.get(f"{self.url}/models", headers=self.headers, timeout=(10, 30))
        self.raise_for_status(response)
        models = [item.get("id") for item in response.json().get("data", [])]
        if self.model not in models:
            raise ValueError(f"Model {self.model!r} is unavailable; server models: {models}")
        LOG.info("GPU endpoint ready: %s (model=%s)", self.url, self.model)

    def ocr(self, image: bytes, telemetry: list[dict] | None = None) -> dict:
        payload = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": self.prompt},
                {"type": "image_url", "image_url": {
                    "url": "data:image/png;base64," + base64.b64encode(image).decode("ascii")
                }},
            ]}],
        }
        for attempt in range(self.retries + 1):
            started = time.monotonic()
            stats = {"attempt": attempt + 1}
            try:
                response = requests.post(
                    f"{self.url}/chat/completions", headers=self.headers,
                    json=payload, timeout=(10, self.timeout),
                )
                stats["http_status"] = response.status_code
                self.raise_for_status(response)
                body = response.json()
                usage = body.get("usage") or {}
                if not isinstance(usage, dict):
                    usage = {}
                stats["usage"] = {
                    key: usage[key] for key in ("prompt_tokens", "completion_tokens", "total_tokens")
                    if isinstance(usage.get(key), int) and usage[key] >= 0
                }
                choices = body.get("choices", [])
                if not choices:
                    raise ValueError("GPU response contains no completion choices")
                choice = choices[0]
                reason = choice.get("finish_reason")
                stats["finish_reason"] = reason
                if reason and reason != "stop":
                    raise ValueError(
                        f"Incomplete OCR (finish_reason={reason}); for length, raise --max-tokens"
                    )
                content = choice.get("message", {}).get("content")
                if isinstance(content, list):
                    content = "\n".join(
                        item.get("text", "") for item in content if isinstance(item, dict)
                    )
                if not isinstance(content, str):
                    raise ValueError("GPU response has no text content")
                return parse_ocr(content)
            except requests.HTTPError as error:
                if error.response.status_code not in (408, 429, 500, 502, 503, 504):
                    raise
                last_error = error
            except (requests.ConnectionError, requests.Timeout, ValueError) as error:
                last_error = error
            finally:
                stats["elapsed_seconds"] = round(time.monotonic() - started, 3)
                if telemetry is not None:
                    telemetry.append(stats)
                LOG.info("GPU OCR attempt %s: %.1fs | HTTP %s | output tokens %s | finish %s",
                         attempt + 1, stats["elapsed_seconds"], stats.get("http_status", "no response"),
                         stats.get("usage", {}).get("completion_tokens", "unknown"),
                         stats.get("finish_reason", "unknown"))
            if attempt < self.retries:
                delay = min(30, 2 ** (attempt + 1))
                LOG.warning("OCR request failed (%s); retry in %ss", last_error, delay)
                time.sleep(delay)
        raise RuntimeError(f"OCR failed after {self.retries + 1} attempts: {last_error}")


def page_result(ocr: dict, number: int, width: int, height: int) -> PageOCRResult:
    """Adapt GPU lines to the existing Stage 2 pixel-coordinate data models."""
    elements = []
    for line in ocr["lines"]:
        x1, y1, x2, y2 = line["bbox"]
        elements.append(DocumentElement(
            element_type=ElementType.PARAGRAPH, text=line["text"], page_num=number,
            bbox=BoundingBox(x=x1 * width / 1000, y=y1 * height / 1000,
                             w=(x2 - x1) * width / 1000, h=(y2 - y1) * height / 1000),
        ))
    return PageOCRResult(page_num=number, elements=elements,
                         raw_text="\n".join(line["text"] for line in ocr["lines"]))


def add_text_layer(page, result: PageOCRResult, image_width: int, image_height: int) -> None:
    """Unicode text with explicit Hindi font, invisible PDF rendering mode 3.

    Helvetica fallback chooses a missing glyph for common-script Devanagari
    punctuation such as U+0964/U+0965 (danda), yielding NUL on extraction.
    Selecting the bundled Devanagari font explicitly preserves those characters.
    """
    latin_font = fitz.Font("helv")
    hindi_font = fitz.Font(script=getattr(fitz, "UCDN_SCRIPT_DEVANAGARI", 9))
    writer = fitz.TextWriter(page.rect)
    for element in result.elements:
        font = hindi_font if any("\u0900" <= char <= "\u097f" for char in element.text) else latin_font
        # Embedded newlines are placed individually within the model's box.
        parts = element.text.splitlines()
        box = element.bbox
        x = box.x * page.rect.width / image_width
        y = box.y * page.rect.height / image_height
        width = box.w * page.rect.width / image_width
        height = box.h * page.rect.height / image_height / len(parts)
        for index, text in enumerate(parts):
            if not text.strip():
                continue
            size = min(12, height / 1.7, width / max(font.text_length(text, fontsize=1), 1))
            writer.append((x, y + index * height + size * 1.1), text, font=font, fontsize=size)
    if result.elements:
        writer.write_text(page, render_mode=3)


def verify_text_layer(page, expected: str, number: int, cache: Path) -> None:
    """Require an exact Unicode round-trip, retaining diagnostics on failure."""
    extracted = page.get_text()
    wanted = "".join(expected.split())
    actual = "".join(extracted.split())
    diagnostic = cache / f"text-layer-failure-{number + 1:06d}.json"
    if wanted == actual:
        diagnostic.unlink(missing_ok=True)
        return
    position = next((i for i, pair in enumerate(zip(wanted, actual))
                     if pair[0] != pair[1]), min(len(wanted), len(actual)))
    codepoint = lambda s: f"U+{ord(s[position]):04X}" if position < len(s) else "end of text"
    difference = f"character {position + 1}: expected {codepoint(wanted)}, extracted {codepoint(actual)}"
    atomic_json(diagnostic, {
        "page": number + 1, "difference": difference,
        "expected": expected, "extracted": extracted,
        "pymupdf_version": str(getattr(fitz, "__version__", fitz.VersionBind)),
    })
    raise ValueError(
        f"Text-layer verification failed on page {number + 1} ({difference}); diagnostics: {diagnostic}"
    )


def ocr_config(args, prompt: str) -> dict:
    return {
        "version": CACHE_VERSION,
        "base_url": args.base_url.rstrip("/"), "model": args.model,
        "dpi": args.dpi, "max_pixels": args.max_pixels, "max_tokens": args.max_tokens,
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
    }


def signature(source: Path, relative: Path, args, prompt: str) -> dict:
    return source_signature(source, relative, ocr_config(args, prompt))


def process_pdf(source: Path, relative: Path, output: Path, cache: Path,
                client: GPUClient, pool: ThreadPoolExecutor, args, prompt: str,
                tracker: OCRTracker | None = None) -> str:
    destination = output / relative
    json_output = destination.with_name(destination.name + ".ocr.json")
    md_output = destination.with_name(destination.name + ".ocr.md")
    sig = signature(source, relative, args, prompt)
    if not args.force and is_complete(source, relative, output, sig, TEXT_LAYER_VERSION):
        return "skipped"
    destination.parent.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    page_cache_valid = not args.force and read_json(cache / "source.json") == sig
    # Invalidate page records before attempting a different source/configuration.
    if not page_cache_valid:
        for old in cache.glob("page-*.json"):
            old.unlink()
        atomic_json(cache / "source.json", sig)
    fd, temporary = tempfile.mkstemp(prefix=".ocr-", suffix=".pdf", dir=destination.parent)
    os.close(fd)
    titles = []
    try:
        with fitz.open(source) as original, fitz.open() as result:
            if original.needs_pass:
                raise ValueError("Encrypted PDF requires a password; unlock a copy before OCR")
            if not original.page_count:
                raise ValueError("PDF has no pages")
            structured = DocumentOCRResult(source_pdf=str(source), total_pages=original.page_count)
            # Render and publish pages in order; retain at most --workers page images.
            for start in range(0, original.page_count, args.workers):
                batch = []
                for number in range(start, min(start + args.workers, original.page_count)):
                    page = original[number]
                    pixmap = render_pdf_page(page, args.dpi, args.max_pixels)
                    image = pixmap.tobytes("png")
                    width, height = pixmap.width, pixmap.height
                    del pixmap
                    record = cache / f"page-{number + 1:06d}.json"
                    saved = read_json(record) if page_cache_valid else None
                    if saved is not None:
                        # Validate cached OCR rather than trusting arbitrary JSON.
                        try:
                            checked = parse_ocr(json.dumps(saved, ensure_ascii=False))
                            checked["positioning"] = saved.get("positioning", checked["positioning"])
                            saved = checked
                        except (ValueError, AttributeError):
                            saved = None
                    future = None if saved is not None else pool.submit(client.ocr, image)
                    batch.append((number, image, width, height, record, saved, future))
                for number, image, width, height, record, saved, future in batch:
                    if tracker:
                        tracker.page(number + 1, original.page_count,
                                     "using cached OCR" if saved is not None else "waiting for GPU OCR")
                    ocr = saved if saved is not None else future.result()
                    if saved is None:
                        atomic_json(record, ocr)
                    if ocr["positioning"] == "approximate":
                        LOG.warning("%s page %s: model supplied no boxes; approximate highlights", relative, number + 1)
                    target = result.new_page(width=original[number].rect.width,
                                             height=original[number].rect.height)
                    target.insert_image(target.rect, stream=image)
                    structured_page = page_result(ocr, number, width, height)
                    structured.pages.append(structured_page)
                    add_text_layer(target, structured_page, width, height)
                    verify_text_layer(target, structured_page.raw_text, number, cache)
                    if tracker:
                        tracker.page(number + 1, original.page_count, "text layer verified",
                                     verified=True, cached=saved is not None)
                    heading = ocr["lines"][0]["text"][:70] if ocr["lines"] else "Blank page"
                    titles.append(f"Page {number + 1} - {heading}")
                    LOG.info("%s: page %s/%s%s", relative, number + 1, original.page_count,
                             " (cached)" if saved is not None else "")
            result.set_metadata(original.metadata)
            toc = original.get_toc(simple=True)
            if toc:
                toc.append([1, "OCR page navigation", 1])
            level = 2 if toc else 1
            toc.extend([level, title, number + 1] for number, title in enumerate(titles))
            result.set_toc(toc)
            result.set_pagemode("UseOutlines")
            labels = original.get_page_labels()
            if labels:
                result.set_page_labels(labels)
            result.subset_fonts()
            if tracker:
                tracker.page(original.page_count, original.page_count, "saving searchable PDF")
            result.save(temporary, garbage=4, deflate=True)
            with fitz.open(temporary) as check:
                if check.page_count != original.page_count:
                    raise ValueError("Output page count differs from the original")
        # Avoid publishing results if a scraper updated the input during OCR.
        if signature(source, relative, args, prompt) != sig:
            raise ValueError("Input PDF changed during OCR; rerun to process the new version")
        atomic_json(json_output, structured.to_dict())
        md_output.write_text(structured.to_markdown(), encoding="utf-8")
        os.replace(temporary, destination)
        atomic_json(cache / "completed.json", {
            "signature": sig, "output": output_stat(destination),
            "text_layer_version": TEXT_LAYER_VERSION,
        })
        return "processed"
    finally:
        Path(temporary).unlink(missing_ok=True)


def write_index(output: Path) -> None:
    links = []
    for path in discover_pdfs(output, output / ".ocr-cache"):
        if path.name.startswith(".ocr-"):
            continue
        relative = path.relative_to(output).as_posix()
        links.append(f'<li><a href="{quote(relative)}">{html.escape(relative)}</a></li>')
    content = (
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        '<title>Searchable OCR PDFs</title>'
        '<style>body{font:16px system-ui;max-width:1100px;margin:40px auto;padding:0 20px}'
        'li{margin:10px 0}input{padding:10px;width:90%}</style>'
        '<h1>Searchable OCR PDFs</h1>'
        f'<p>{len(links)} PDFs. Open a PDF and use Ctrl+F to search its text. '
        'Use its bookmarks to navigate pages.</p>'
        '<input type="search" placeholder="Filter file paths" aria-label="Filter file paths">'
        '<ul>' + "\n".join(links) + '</ul>'
        '<script>document.querySelector("input").addEventListener("input",e=>{'
        'const q=e.target.value.toLowerCase();document.querySelectorAll("li").forEach('
        'li=>li.hidden=!li.textContent.toLowerCase().includes(q))})</script></html>'
    )
    (output / "index.html").write_text(content, encoding="utf-8")


def main(argv=None) -> int:
    # Load only the FG OCR client's explicit core/.env, preserving shell overrides.
    load_dotenv(Path(__file__).resolve().parent / ".env", override=False)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", type=Path, default=PROJECT / "finance_cg_data")
    parser.add_argument("-o", "--output", type=Path, help="Default: INPUT_NAME_ocr beside input")
    parser.add_argument("--base-url", default=os.getenv("GPU_OCR_BASE_URL", "http://127.0.0.1:8000/v1"))
    parser.add_argument("--model", default=os.getenv("GPU_OCR_MODEL", "ocr-primary"))
    parser.add_argument("--workers", type=int, default=2, help="Concurrent GPU requests (default: 2)")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--max-pixels", type=int, default=6_000_000, help="Maximum rendered pixels per page")
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--timeout", type=float, default=180, help="Per-request read timeout in seconds")
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0, help="Limit discovered PDFs; 0 = all")
    parser.add_argument("--force", action="store_true", help="Ignore caches and rebuild outputs")
    parser.add_argument("--dry-run", action="store_true", help="List PDFs without contacting the GPU")
    parser.add_argument("--prompt-file", type=Path, help="UTF-8 OCR prompt override for your vision model")
    status_options = parser.add_mutually_exclusive_group()
    status_options.add_argument("--status", action="store_true", help="Count all PDFs and show FG OCR status without GPU requests")
    status_options.add_argument("--watch", nargs="?", type=float, const=3,
                                help="Watch FG status every N seconds (default: 3); no GPU requests")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if min(args.workers, args.dpi, args.max_pixels, args.max_tokens, args.timeout) <= 0:
        parser.error("workers, dpi, max-pixels, max-tokens and timeout must be positive")
    if min(args.limit, args.retries) < 0:
        parser.error("limit and retries cannot be negative")
    if args.watch is not None and (not math.isfinite(args.watch) or args.watch < 1):
        parser.error("watch interval must be at least one second")
    source = args.input.resolve()
    output = (args.output or source.with_name(source.name + "_ocr")).resolve()
    if not source.is_dir():
        parser.error(f"Input directory does not exist: {source}")
    if output == source or source.is_relative_to(output):
        parser.error("Output must be a separate directory, not the input or its ancestor")
    paths = list(discover_pdfs(source, output))
    LOG.info("Found %s PDFs recursively in %s", len(paths), source)
    if args.status or args.watch is not None:
        # Existing tracking snapshots include the settings used by the last run.
        config = ocr_config(args, DEFAULT_PROMPT)
        try:
            while True:
                if args.watch is not None and sys.stdout.isatty() and os.getenv("TERM") != "dumb":
                    print("\033[2J\033[H", end="")
                print_status(status_snapshot(source, output, paths, config, TEXT_LAYER_VERSION))
                if args.watch is None:
                    return 0
                time.sleep(args.watch)
                paths = list(discover_pdfs(source, output))
        except KeyboardInterrupt:
            return 0
    all_paths = paths
    if args.limit:
        paths = paths[:args.limit]
    if args.dry_run:
        for path in paths:
            print(path.relative_to(source))
        return 0
    prompt = args.prompt_file.read_text(encoding="utf-8") if args.prompt_file else DEFAULT_PROMPT
    if not prompt.strip():
        parser.error("OCR prompt cannot be empty")
    tracker = OCRTracker(source, output, all_paths, ocr_config(args, prompt), TEXT_LAYER_VERSION, len(paths))
    tracker.show()
    if not paths:
        tracker.finish_run("completed")
        return 0
    client = GPUClient(args, prompt)
    tracker.set_state("checking_gpu")
    try:
        client.check()
    except (requests.RequestException, ValueError) as error:
        tracker.gpu("unavailable", str(error))
        tracker.finish_run("gpu_unavailable")
        LOG.error("GPU endpoint check failed: %s. Start the SSH tunnel on THIS processing VM.", error)
        return 2
    output.mkdir(parents=True, exist_ok=True)
    tracker.gpu("ready")
    tracker.set_state("running")
    counts = {"processed": 0, "skipped": 0, "failed": 0}
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for position, path in enumerate(paths, 1):
                relative = path.relative_to(source)
                cache = cache_directory(output, relative)
                LOG.info("PDF %s/%s: %s", position, len(paths), relative)
                try:
                    tracker.begin(relative, position)
                    status = process_pdf(path, relative, output, cache, client, pool, args, prompt, tracker)
                    counts[status] += 1
                    tracker.finish_document(relative, status)
                    LOG.info("%s: %s", status, relative)
                except Exception as error:
                    counts["failed"] += 1
                    tracker.finish_document(relative, "failed", str(error))
                    LOG.error("Failed %s: %s", relative, error)
                    with (output / "errors.jsonl").open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps({"file": relative.as_posix(), "error": str(error)},
                                                ensure_ascii=False) + "\n")
    except KeyboardInterrupt:
        tracker.finish_run("interrupted")
        LOG.warning("Interrupted; completed pages are cached. Run the same command to resume.")
        return 130
    except Exception:
        tracker.finish_run("stopped")
        raise
    else:
        tracker.finish_run("completed_with_errors" if counts["failed"] else "completed")
    finally:
        write_index(output)
    LOG.info("Finished: %s. Navigation index: %s", counts, output / "index.html")
    return 1 if counts["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
