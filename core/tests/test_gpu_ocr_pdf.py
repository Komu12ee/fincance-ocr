"""Contract/integration checks using a fake vision API, never the real GPU."""

import base64
from contextlib import redirect_stdout
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import io
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gpu_ocr_pdf as ocr
import pymupdf as fitz
from stage1_image_prep.pdf_to_image import pdf_page_to_image


LINES = [
    {"text": "हिन्दी वित्त विभाग। Finance Department ॥", "bbox": [80, 100, 920, 180]},
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
        self.http_error = None
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def respond(self, body, status=200):
                data = json.dumps(body, ensure_ascii=False).encode("utf-8")
                self.send_response(status)
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
                if owner.http_error:
                    status, error = owner.http_error
                    self.respond(error, status)
                    return
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
        self.console = io.StringIO()
        with redirect_stdout(self.console):
            return ocr.main([
                str(self.source), "-o", str(self.output), "--base-url", self.base,
                "--workers", "1", "--retries", "0", *extra,
            ])

    def read_status(self):
        return json.loads((self.output / "ocr_status.json").read_text(encoding="utf-8"))

    def test_http_400_retains_server_reason_and_does_not_retry_or_publish(self):
        make_pdf(self.source / "scan.pdf")
        message = "Input has 10000 tokens and max_tokens=8192 exceeds max_model_len=16384"
        self.http_error = (400, {"error": {"message": message, "type": "BadRequestError",
                                          "param": "max_tokens", "code": 400}})
        self.assertEqual(self.run_ocr("--retries", "2"), 1)
        self.assertEqual(self.calls, 1)
        self.assertFalse((self.output / "scan.pdf").exists())
        error = json.loads((self.output / "errors.jsonl").read_text(encoding="utf-8"))["error"]
        self.assertIn("HTTP 400", error)
        self.assertIn(message, error)
        self.assertIn("param: max_tokens", error)
        self.assertIn("max_tokens=8192", error)
        self.assertIn(message, self.read_status()["last_error"])

    def test_transient_http_error_still_retries_with_server_reason(self):
        make_pdf(self.source / "scan.pdf")
        self.http_error = (503, {"error": {"message": "GPU temporarily unavailable"}})
        with patch("gpu_ocr_pdf.time.sleep") as sleep:
            self.assertEqual(self.run_ocr("--retries", "1"), 1)
        self.assertEqual(self.calls, 2)
        sleep.assert_called_once()
        self.assertIn("GPU temporarily unavailable", self.read_status()["last_error"])

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
        failed_status = self.read_status()
        self.assertEqual(failed_status["counts"], {"total": 1, "done": 0, "failed": 1, "pending": 0})
        self.assertEqual(failed_status["state"], "completed_with_errors")
        self.assertIn("scan.pdf:", failed_status["last_error"])
        self.assertFalse((self.output / "scan.pdf").exists())
        self.assertTrue((self.output / "errors.jsonl").is_file())
        self.assertEqual(len(list(self.output.rglob("page-*.json"))), 1)
        self.fail_on = None
        self.assertEqual(self.run_ocr(), 0)
        self.assertEqual(self.calls, 3)  # Only the failed page needed a new request.
        self.assertEqual(self.read_status()["counts"], {"total": 1, "done": 1, "failed": 0, "pending": 0})
        self.assertEqual(self.read_status()["run"]["pages_cached"], 1)

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

    def test_writer_upgrade_reuses_gpu_cache(self):
        make_pdf(self.source / "scan.pdf")
        self.assertEqual(self.run_ocr(), 0)
        completed = next(self.output.rglob("completed.json"))
        data = json.loads(completed.read_text(encoding="utf-8"))
        data.pop("text_layer_version")  # Completion marker from the older writer.
        completed.write_text(json.dumps(data), encoding="utf-8")
        self.assertEqual(self.run_ocr(), 0)
        self.assertEqual(self.calls, 1)
        updated = json.loads(completed.read_text(encoding="utf-8"))
        self.assertEqual(updated["text_layer_version"], ocr.TEXT_LAYER_VERSION)

    def test_hindi_punctuation_search_after_save_and_subset(self):
        text = "हिन्दी वित्त विभाग। स्वीकृत राशि ₹१२३॥ ॒॑॓॔"
        data = {"lines": [{"text": text, "bbox": [20, 900, 980, 999]}]}
        with fitz.open() as doc:
            page = doc.new_page()
            result = ocr.page_result(data, 0, 1000, 1000)
            ocr.add_text_layer(page, result, 1000, 1000)
            ocr.verify_text_layer(page, text, 0, self.root)
            doc.subset_fonts()
            encoded = doc.tobytes(garbage=4, deflate=True)
        with fitz.open(stream=encoded, filetype="pdf") as saved:
            self.assertEqual(saved[0].get_text().strip(), text)
            self.assertTrue(saved[0].search_for("विभाग।"))
            self.assertTrue(saved[0].search_for("१२३॥"))

    def test_failed_text_verification_retains_diagnostics(self):
        with fitz.open() as doc:
            page = doc.new_page()
            page.insert_text((30, 60), "actual")
            with self.assertRaisesRegex(ValueError, "expected U\\+0065"):
                ocr.verify_text_layer(page, "expected", 0, self.root)
        diagnostic = self.root / "text-layer-failure-000001.json"
        saved = json.loads(diagnostic.read_text(encoding="utf-8"))
        self.assertEqual(saved["expected"], "expected")
        self.assertEqual(saved["extracted"].strip(), "actual")

    def test_tracker_counts_full_corpus_with_limit_and_status_is_read_only(self):
        make_pdf(self.source / "one" / "same.pdf", 2)
        make_pdf(self.source / "two" / "same.PdF")
        self.assertEqual(self.run_ocr("--limit", "1"), 0)
        status = self.read_status()
        self.assertEqual(status["counts"], {"total": 2, "done": 1, "failed": 0, "pending": 1})
        self.assertEqual(status["run"]["selected"], 1)
        self.assertEqual(status["run"]["pages_verified"], 2)
        self.assertEqual(status["state"], "completed")
        self.assertIsNone(status["current"])
        before = (self.output / "ocr_status.json").read_bytes()
        with patch("gpu_ocr_pdf.requests.get") as get:
            self.assertEqual(self.run_ocr("--status"), 0)
            get.assert_not_called()
        self.assertEqual(self.calls, 2)
        self.assertIn("-- FG OCR STATUS --", self.console.getvalue())
        self.assertEqual((self.output / "ocr_status.json").read_bytes(), before)
        self.assertEqual(self.run_ocr(), 0)
        self.assertEqual(self.read_status()["counts"]["done"], 2)
        self.assertEqual(self.read_status()["run"]["skipped"], 1)
        self.assertEqual(self.calls, 3)

    def test_count_status_before_first_run_and_watch_without_gpu(self):
        make_pdf(self.source / "one.pdf")
        make_pdf(self.source / "two.PDF")
        with patch("gpu_ocr_pdf.requests.get") as get:
            self.assertEqual(self.run_ocr("--status"), 0)
            get.assert_not_called()
        self.assertFalse(self.output.exists())
        self.assertIn("NOT STARTED", self.console.getvalue())
        self.assertEqual(self.calls, 0)
        with patch("gpu_ocr_pdf.time.sleep", side_effect=KeyboardInterrupt), patch("gpu_ocr_pdf.requests.get") as get:
            self.assertEqual(self.run_ocr("--watch", "1"), 0)
            get.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_changed_source_and_missing_output_are_pending_in_status(self):
        path = self.source / "scan.pdf"
        make_pdf(path)
        self.assertEqual(self.run_ocr(), 0)
        default_config = self.read_status()["ocr_config"]
        path.write_bytes(path.read_bytes() + b"\n")
        status = ocr.status_snapshot(self.source, self.output, [path], default_config, ocr.TEXT_LAYER_VERSION)
        self.assertEqual(status["counts"]["done"], 0)
        self.assertEqual(status["counts"]["pending"], 1)
        self.assertEqual(self.run_ocr(), 0)
        (self.output / "scan.pdf").unlink()
        status = ocr.status_snapshot(self.source, self.output, [path], default_config, ocr.TEXT_LAYER_VERSION)
        self.assertEqual(status["counts"]["pending"], 1)

    def test_tracker_persists_current_page_while_waiting(self):
        path = self.source / "scan.pdf"
        make_pdf(path, 2)
        config = {"base_url": self.base, "model": "ocr-primary", "version": ocr.CACHE_VERSION}
        tracker = ocr.OCRTracker(self.source, self.output, [path], config, ocr.TEXT_LAYER_VERSION, 1)
        tracker.begin(Path("scan.pdf"), 1)
        tracker.page(1, 2, "waiting for GPU OCR")
        status = self.read_status()
        self.assertEqual(status["current"]["file"], "scan.pdf")
        self.assertEqual(status["current"]["page"], 1)
        self.assertEqual(status["current"]["total_pages"], 2)
        self.assertEqual(status["current"]["stage"], "waiting for GPU OCR")
        self.assertEqual(status["counts"]["pending"], 1)

    def test_gpu_unavailable_is_tracked_without_marking_all_pdfs_failed(self):
        make_pdf(self.source / "scan.pdf")
        with patch("gpu_ocr_pdf.GPUClient.check", side_effect=ValueError("GPU offline")):
            self.assertEqual(self.run_ocr(), 2)
        status = self.read_status()
        self.assertEqual(status["state"], "gpu_unavailable")
        self.assertEqual(status["gpu_state"], "unavailable")
        self.assertEqual(status["counts"], {"total": 1, "done": 0, "failed": 0, "pending": 1})

    def test_latest_failures_persist_outside_a_limited_run(self):
        make_pdf(self.source / "a.pdf")
        make_pdf(self.source / "b.pdf")
        self.fail_on = 2
        self.assertEqual(self.run_ocr(), 1)
        self.assertEqual(self.read_status()["counts"], {"total": 2, "done": 1, "failed": 1, "pending": 0})
        self.fail_on = None
        self.assertEqual(self.run_ocr("--limit", "1"), 0)
        self.assertEqual(self.read_status()["counts"]["failed"], 1)
        self.assertEqual(self.run_ocr(), 0)
        self.assertEqual(self.read_status()["counts"]["failed"], 0)


class OCRResponseTests(unittest.TestCase):
    def test_plain_http_error_body_is_bounded_and_redacts_image_data_and_key(self):
        import requests
        from types import SimpleNamespace
        args = SimpleNamespace(base_url="http://localhost/v1", model="ocr-primary",
                               timeout=180, retries=0, max_tokens=4096)
        response = requests.Response()
        response.status_code = 400
        response._content = b"Image disabled. secret-key data:image/png;base64,ABCDEF " + b"x" * 5000
        with patch.dict("os.environ", {"GPU_OCR_API_KEY": "secret-key"}):
            client = ocr.GPUClient(args, "OCR")
            with self.assertRaises(requests.HTTPError) as caught:
                client.raise_for_status(response)
        self.assertIn("Image disabled", str(caught.exception))
        self.assertNotIn("secret-key", str(caught.exception))
        self.assertNotIn("ABCDEF", str(caught.exception))
        self.assertLess(len(str(caught.exception)), 4200)
        self.assertIs(caught.exception.response, response)

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
