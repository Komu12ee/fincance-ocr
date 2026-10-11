"""Contract/integration checks using a fake vision API, never the real GPU."""

import base64
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gpu_ocr_pdf as ocr
import pymupdf as fitz
from stage1_image_prep.pdf_to_image import pdf_page_to_image


LINES = [
    {"text": "हिन्दी वित्त विभाग Finance Department", "bbox": [80, 100, 920, 180]},
    {"text": "Order 123 / 2026", "bbox": [80, 220, 920, 300]},
]


def make_pdf(path, pages=1):
    path.parent.mkdir(parents=True, exist_ok=True)
    with fitz.open() as doc:
        for number in range(pages):
            page = doc.new_page(width=360, height=480)
            page.insert_text((30, 65), "Finance Department", fontsize=16)
            page.insert_text((30, 130), f"Original page {number + 1}", fontsize=12)
            page.draw_rect(fitz.Rect(25, 25, 335, 455), color=(0.1, 0.3, 0.7))
            if number % 2:
                page.set_cropbox(fitz.Rect(10, 10, 350, 470))
                page.set_rotation(90)
        doc.set_toc([[1, "Original section", 1]])
        doc.set_metadata({"title": "OCR verification fixture"})
        doc.save(path)


class OCRIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.source = self.root / "input"
        self.source.mkdir()
        self.output = self.root / "output"
        self.calls = 0
        self.fail_on = None
        self.mode = "json"
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def respond(self, body):
                data = json.dumps(body, ensure_ascii=False).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                if self.path != "/v1/models":
                    self.send_error(404)
                    return
                self.respond({"data": [{"id": "ocr-primary"}]})

            def do_POST(self):
                if self.path != "/v1/chat/completions":
                    self.send_error(404)
                    return
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                owner.assertEqual(payload["model"], "ocr-primary")
                uri = payload["messages"][0]["content"][1]["image_url"]["url"]
                owner.assertTrue(base64.b64decode(uri.split(",", 1)[1]).startswith(b"\x89PNG"))
                owner.calls += 1
                content = json.dumps({"blank": False, "lines": LINES}, ensure_ascii=False)
                if owner.mode == "text":
                    content = "\n".join(line["text"] for line in LINES)
                if owner.mode == "blank":
                    content = '{"blank": true, "lines": []}'
                reason = "length" if owner.calls == owner.fail_on else "stop"
                self.respond({"choices": [{"message": {"content": content}, "finish_reason": reason}]})

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.server.server_port}/v1"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.tmp.cleanup()

    def run_ocr(self, *extra):
        return ocr.main([
            str(self.source), "-o", str(self.output), "--base-url", self.base,
            "--workers", "1", "--retries", "0", *extra,
        ])

    def test_nested_names_unicode_bookmarks_structure_and_resume(self):
        first = self.source / "one" / "same.pdf"
        second = self.source / "two" / "same.PdF"
        make_pdf(first, 2)
        make_pdf(second)
        before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in (first, second)}
        self.assertEqual(self.run_ocr("--workers", "2"), 0)
        self.assertEqual(self.calls, 3)
        for source in (first, second):
            target = self.output / source.relative_to(self.source)
            self.assertTrue(target.is_file())
            self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), before[source])
            with fitz.open(source) as original, fitz.open(target) as result:
                self.assertEqual(result.page_count, original.page_count)
                self.assertEqual(result.metadata["title"], original.metadata["title"])
                self.assertEqual(result.get_toc()[0], [1, "Original section", 1])
                for number, page in enumerate(result):
                    self.assertEqual(page.rect, original[number].rect)
                    self.assertIn(LINES[0]["text"], page.get_text())
                    self.assertTrue(page.search_for("वित्त"))
                    self.assertTrue(page.search_for("Order 123"))
                    # Invisible layer must not alter the rebuilt image visually.
                    original_pixels = ocr.render_pdf_page(original[number], 200, 6_000_000)
                    output_pixels = ocr.render_pdf_page(page, 200, 6_000_000)
                    self.assertEqual(original_pixels.samples, output_pixels.samples)
                page_bookmarks = [row for row in result.get_toc() if row[1].startswith("Page ")]
                self.assertEqual(len(page_bookmarks), result.page_count)
            structured = json.loads(target.with_name(target.name + ".ocr.json").read_text(encoding="utf-8"))
            self.assertEqual(structured["total_pages"], 2 if source == first else 1)
            self.assertIn(LINES[0]["text"], structured["pages"][0]["text"])
            self.assertIn("bbox", structured["pages"][0]["elements"][0])
            self.assertTrue(target.with_name(target.name + ".ocr.md").is_file())
        index = (self.output / "index.html").read_text(encoding="utf-8")
        self.assertIn('href="one/same.pdf"', index)
        self.assertIn('href="two/same.PdF"', index)
        self.assertEqual(self.run_ocr(), 0)
        self.assertEqual(self.calls, 3)  # No repeated GPU requests.
        self.assertEqual(self.run_ocr("--force"), 0)
        self.assertEqual(self.calls, 6)

    def test_failed_page_is_not_published_and_resume_reuses_first_page(self):
        make_pdf(self.source / "scan.pdf", 2)
        self.fail_on = 2
        self.assertEqual(self.run_ocr(), 1)
        self.assertFalse((self.output / "scan.pdf").exists())
        self.assertTrue((self.output / "errors.jsonl").is_file())
        self.assertEqual(len(list(self.output.rglob("page-*.json"))), 1)
        self.fail_on = None
        self.assertEqual(self.run_ocr(), 0)
        self.assertEqual(self.calls, 3)  # Only the failed page needed a new request.

    def test_text_only_and_blank_page_models(self):
        make_pdf(self.source / "scan.pdf")
        self.mode = "text"
        self.assertEqual(self.run_ocr(), 0)
        cached = json.loads(next(self.output.rglob("page-*.json")).read_text(encoding="utf-8"))
        self.assertEqual(cached["positioning"], "approximate")
        self.mode = "blank"
        self.assertEqual(self.run_ocr("--force"), 0)
        with fitz.open(self.output / "scan.pdf") as doc:
            self.assertEqual(doc[0].get_text(), "")

    def test_output_inside_input_is_excluded_and_bad_cache_is_recovered(self):
        make_pdf(self.source / "scan.pdf")
        self.output = self.source / "results"
        self.assertEqual(self.run_ocr(), 0)
        self.assertEqual(list(ocr.discover_pdfs(self.source, self.output)), [self.source / "scan.pdf"])
        (self.output / "scan.pdf").unlink()
        cache = next(self.output.rglob("page-*.json"))
        cache.write_text('{"lines": "corrupted"}', encoding="utf-8")
        self.assertEqual(self.run_ocr(), 0)
        self.assertEqual(self.calls, 2)

    def test_legacy_stage1_single_page_bgr_helper(self):
        path = self.source / "legacy.pdf"
        make_pdf(path)
        image = pdf_page_to_image(path, 0)
        self.assertEqual(image.shape[2], 3)
        self.assertGreater(image.shape[0], 480)  # Existing 450-DPI default.


class OCRResponseTests(unittest.TestCase):
    def test_rejects_empty_truncated_and_invalid_geometry(self):
        for value in ("", '{"lines": []}', '{"lines": [', '{"lines":[{"text":"x","bbox":[0,0,1001,1]}]}'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                ocr.parse_ocr(value)

    def test_accepts_json_fences_and_text_without_boxes(self):
        value = ocr.parse_ocr('```json\n' + json.dumps({"lines": LINES}) + '\n```')
        self.assertEqual(value["positioning"], "model")
        self.assertEqual(ocr.parse_ocr('{"lines":[{"text":"hello"}]}')["positioning"], "approximate")


if __name__ == "__main__":
    unittest.main()
