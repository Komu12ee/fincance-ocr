"""Checks that VM diagnostics retain errors without uploading OCR content."""

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import fg_vm_diagnostics as diagnostics
import requests


class DiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.repo = Path(self.tmp.name)
        self.source = self.repo / "finance_cg_data"
        self.source.mkdir()
        self.output = self.repo / "finance_cg_data_ocr"
        self.output.mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def test_redaction_removes_keys_credentials_and_image_payloads(self):
        value = {"error": [f"{self.repo} secret-key Bearer abc123 ghp_exampleToken",
                           "https://user:password@example.test/v1?token=private",
                           "data:image/png;base64,SECRETIMAGE"]}
        result = json.dumps(diagnostics.redact(value, self.repo, ("secret-key",)))
        for secret in (str(self.repo), "secret-key", "abc123", "ghp_exampleToken",
                       "user:password", "private", "SECRETIMAGE"):
            self.assertNotIn(secret, result)

    def test_collection_excludes_ocr_transcripts_env_and_unknown_fields(self):
        (self.repo / "core").mkdir()
        (self.repo / "core" / ".env").write_text("OTHER_SECRET=DO_NOT_UPLOAD")
        (self.output / "ocr_status.json").write_text(json.dumps({
            "state": "completed_with_errors", "last_error": "HTTP 400: image input disabled",
            "unexpected_transcript": "PRIVATE OCR TEXT", "ocr_config": {"max_tokens": 4096},
        }))
        (self.output / "errors.jsonl").write_text(json.dumps({
            "file": "example.pdf", "error": "HTTP 400: image input disabled",
            "transcript": "PRIVATE OCR TEXT",
        }) + "\n")
        cache = self.output / ".ocr-cache" / "abc"
        cache.mkdir(parents=True)
        (cache / "text-layer-failure-000001.json").write_text(json.dumps({
            "page": 1, "difference": "expected U+0964, extracted U+0000",
            "expected": "PRIVATE OCR TEXT", "extracted": "PRIVATE EXTRACTED TEXT",
        }))
        response = MagicMock()
        response.json.return_value = {"data": [{"id": "ocr-primary", "max_model_len": 16384,
                                                "unknown": "DO_NOT_UPLOAD"}]}
        with patch.object(diagnostics, "git", return_value="test-commit"), \
                patch.object(diagnostics.requests, "get", return_value=response):
            result = diagnostics.collect(self.repo, self.source, self.output, False)
        encoded = json.dumps(result)
        self.assertIn("image input disabled", encoded)
        self.assertIn("U+0964", encoded)
        self.assertNotIn("PRIVATE", encoded)
        self.assertNotIn("DO_NOT_UPLOAD", encoded)
        self.assertEqual(result["settings"]["max_tokens"], 4096)
        self.assertNotIn("probe", result)

    def test_probe_records_actual_http_error_without_writing_ocr_outputs(self):
        response = MagicMock()
        response.json.return_value = {"data": [{"id": "ocr-primary"}]}
        before = list(self.output.rglob("*"))
        with patch.object(diagnostics, "git", return_value="test-commit"), \
                patch.object(diagnostics.requests, "get", return_value=response), \
                patch.object(diagnostics.ocr, "discover_pdfs", return_value=[self.source / "scan.pdf"]), \
                patch.object(diagnostics.ocr.fitz, "open"), \
                patch.object(diagnostics.ocr, "render_pdf_page", return_value=MagicMock(width=400, height=600)), \
                patch.object(diagnostics.ocr.GPUClient, "ocr", side_effect=requests.HTTPError(
                    "HTTP 400: image input disabled")):
            result = diagnostics.collect(self.repo, self.source, self.output, True)
        self.assertEqual(result["probe"]["state"], "failed")
        self.assertIn("image input disabled", result["probe"]["error"])
        self.assertEqual(list(self.output.rglob("*")), before)

    def test_push_stages_single_report_in_isolated_checkout(self):
        report = self.repo / "fg-vm-example.json"
        report.write_text('{"scope":"fg"}')
        calls = []

        def fake_git(repo, *args):
            calls.append((repo, args))
            if args == ("remote", "get-url", "origin"):
                return "git@github-finance-ocr:Komu12ee/fincance-ocr.git"
            if args[:1] == ("config",) and len(args) == 2:
                return "test-user"
            if args[:1] == ("add",):
                self.assertEqual(args, ("add", "--force", "--", "diagnostics/fg-vm-example.json"))
                self.assertEqual((repo / args[-1]).read_bytes(), report.read_bytes())
            return ""

        with patch.object(diagnostics, "git", side_effect=fake_git):
            diagnostics.push_report(self.repo, report, "fg-ocr-diagnostics-test")
        for repo, args in calls:
            if args[0] in ("add", "checkout", "push", "fetch", "init", "-c"):
                self.assertNotEqual(repo, self.repo)
        push = [args for repo, args in calls if args[0] == "push"]
        self.assertEqual(push, [("push", "origin", "HEAD:refs/heads/fg-ocr-diagnostics-test")])

    def test_push_rejects_other_repository_before_creating_checkout(self):
        with patch.object(diagnostics, "git", return_value="https://github.com/other/repo.git"), \
                patch.object(diagnostics.tempfile, "TemporaryDirectory") as temporary:
            with self.assertRaisesRegex(ValueError, "approved"):
                diagnostics.push_report(self.repo, self.repo / "report.json", "test")
            temporary.assert_not_called()

    def test_real_git_upload_leaves_working_branch_and_staged_files_untouched(self):
        original_git = diagnostics.git
        original_git(self.repo, "init", "--quiet", "--initial-branch=main")
        original_git(self.repo, "config", "user.name", "Test")
        original_git(self.repo, "config", "user.email", "test@example.test")
        (self.repo / "README.md").write_text("FG code")
        (self.repo / ".gitignore").write_text("/diagnostics/fg-vm-*.json\n")
        original_git(self.repo, "add", "--", "README.md", ".gitignore")
        original_git(self.repo, "-c", "core.hooksPath=/dev/null", "-c", "commit.gpgsign=false",
                     "commit", "--quiet", "-m", "Initial code")
        remote = self.repo / "local-remote.git"
        original_git(self.repo, "init", "--quiet", "--bare", str(remote))
        original_git(self.repo, "push", str(remote), "main")
        original_git(self.repo, "remote", "add", "origin", "https://github.com/Komu12ee/fincance-ocr.git")
        (self.repo / "local-secret.txt").write_text("DO NOT UPLOAD")
        original_git(self.repo, "add", "--", "local-secret.txt")
        report = self.repo / "diagnostics" / "fg-vm-example.json"
        report.parent.mkdir()
        report.write_text('{"scope":"fg"}')
        before = original_git(self.repo, "status", "--porcelain")
        commit = original_git(self.repo, "rev-parse", "HEAD")

        def local_git(repo, *args):
            # Redirect the isolated checkout's origin to a local bare test repo.
            if args[:3] == ("remote", "add", "origin"):
                args = (*args[:3], str(remote))
            return original_git(repo, *args)

        with patch.object(diagnostics, "git", side_effect=local_git):
            diagnostics.push_report(self.repo, report, "fg-ocr-diagnostics-test")
        self.assertEqual(original_git(self.repo, "status", "--porcelain"), before)
        self.assertEqual(original_git(self.repo, "rev-parse", "HEAD"), commit)
        self.assertEqual(original_git(self.repo, "branch", "--show-current"), "main")
        uploaded = original_git(remote, "ls-tree", "-r", "--name-only", "fg-ocr-diagnostics-test")
        self.assertIn("diagnostics/fg-vm-example.json", uploaded)
        self.assertNotIn("local-secret.txt", uploaded)
        self.assertEqual(original_git(remote, "rev-parse", "main"), commit)


if __name__ == "__main__":
    unittest.main()
