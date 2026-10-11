#!/usr/bin/env python3
"""Collect FG OCR diagnostics; optionally probe one page and push a report branch."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import platform
import re
import shutil
import socket
import subprocess
import tempfile
from types import SimpleNamespace
import uuid

import requests
from dotenv import load_dotenv

import gpu_ocr_pdf as ocr

PROJECT = Path(__file__).resolve().parent.parent


def git(repo: Path, *args: str) -> str:
    result = subprocess.run(["git", "-C", str(repo), *args], capture_output=True,
                            text=True, encoding="utf-8", errors="replace", timeout=120)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "Git command failed")
    return result.stdout.strip()


def redact(value, repo: Path, secrets: tuple[str, ...] = ()):
    """Apply redaction recursively before writing or uploading any report."""
    if isinstance(value, dict):
        return {key: redact(item, repo, secrets) for key, item in value.items()}
    if isinstance(value, list):
        return [redact(item, repo, secrets) for item in value]
    if not isinstance(value, str):
        return value
    for secret in secrets:
        if secret:
            value = value.replace(secret, "[REDACTED]")
    for path, label in ((str(repo), "<project>"), (str(Path.home()), "<home>")):
        value = value.replace(path, label)
    value = re.sub(r"data:image/[^\s\"']+", "[image data omitted]", value)
    value = re.sub(r"(?i)(bearer\s+)[^\s\"']+", r"\1[REDACTED]", value)
    value = re.sub(r"\b(?:gh[pousr]_[A-Za-z0-9_]+|github_pat_[A-Za-z0-9_]+|sk-[A-Za-z0-9_-]+)\b",
                   "[REDACTED]", value)
    value = re.sub(r"(https?://)[^/@\s]+@", r"\1[REDACTED]@", value)
    value = re.sub(r"(?i)([?&](?:api_key|token|key|password)=)[^&\s]+", r"\1[REDACTED]", value)
    return value[:8000]


def tail_errors(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open("rb") as stream:
        size = stream.seek(0, 2)
        stream.seek(max(0, size - 128_000))
        lines = stream.read().decode("utf-8", errors="replace").splitlines()
    records = []
    for line in lines:
        try:
            item = json.loads(line)
        except ValueError:
            continue
        if isinstance(item, dict):
            records.append({key: item[key] for key in ("file", "error") if key in item})
    return records[-30:]


def collect(repo: Path, source: Path, output: Path, probe: bool) -> dict:
    saved = ocr.read_json(output / "ocr_status.json")
    saved = saved if isinstance(saved, dict) else {}
    config = saved.get("ocr_config") or {}
    args = SimpleNamespace(
        base_url=config.get("base_url", os.getenv("GPU_OCR_BASE_URL", "http://127.0.0.1:8000/v1")),
        model=config.get("model", os.getenv("GPU_OCR_MODEL", "ocr-primary")),
        dpi=config.get("dpi", 200), max_pixels=config.get("max_pixels", 6_000_000),
        max_tokens=config.get("max_tokens", 8192), timeout=180, retries=0,
    )
    packages = {}
    for name in ("PyMuPDF", "requests", "python-dotenv"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = "not installed"
    paths = list(ocr.discover_pdfs(source, output))
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "scope": "fg", "python": platform.python_version(), "platform": platform.platform(),
        "packages": packages, "input": str(source), "output": str(output),
        "pdf_count": len(paths), "free_disk_gib": round(shutil.disk_usage(source).free / 1024**3, 2),
        "settings": vars(args),
        "git": {"commit": git(repo, "rev-parse", "HEAD"),
                "branch": git(repo, "branch", "--show-current"),
                "ocr_code_changes": git(repo, "status", "--porcelain", "--", "core/gpu_ocr_pdf.py",
                                        "core/fg_ocr_status.py", "core/fg_vm_diagnostics.py")},
        "status": {key: saved[key] for key in ("state", "counts", "run", "current", "last_error",
                                              "started_at", "updated_at", "finished_at", "gpu_state")
                   if key in saved},
        "recent_errors": tail_errors(output / "errors.jsonl"),
        "text_layer_failures": [],
    }
    # Do not include expected/extracted OCR text or page cache contents.
    for path in sorted((output / ".ocr-cache").glob("*/text-layer-failure-*.json"))[-20:]:
        item = ocr.read_json(path)
        if isinstance(item, dict):
            report["text_layer_failures"].append({
                "cache_file": str(path.relative_to(output)),
                **{key: item[key] for key in ("page", "difference", "pymupdf_version") if key in item},
            })
    client = ocr.GPUClient(args, ocr.DEFAULT_PROMPT)
    try:
        response = requests.get(f"{client.url}/models", headers=client.headers, timeout=(10, 30))
        client.raise_for_status(response)
        report["gpu_models"] = [
            {key: model[key] for key in ("id", "root", "max_model_len") if key in model}
            for model in response.json().get("data", []) if isinstance(model, dict)
        ]
    except (requests.RequestException, ValueError) as error:
        report["gpu_models_error"] = str(error)
    if probe:
        report["probe"] = {"note": "One source page sent to the configured GPU; no PDFs or OCR text uploaded to GitHub."}
        try:
            if not paths:
                raise ValueError("No FG PDFs found")
            report["probe"]["file"] = str(paths[0].relative_to(source))
            report["probe"]["page"] = 1
            with ocr.fitz.open(paths[0]) as doc:
                pixmap = ocr.render_pdf_page(doc[0], args.dpi, args.max_pixels)
                report["probe"]["image"] = {"width": pixmap.width, "height": pixmap.height}
                result = client.ocr(pixmap.tobytes("png"))
            report["probe"].update({"state": "success", "line_count": len(result["lines"]),
                                     "positioning": result["positioning"]})
        except Exception as error:
            report["probe"].update({"state": "failed", "error": str(error)})
    return redact(report, repo, (os.getenv("GPU_OCR_API_KEY", ""),))


def push_report(repo: Path, report: Path, branch: str) -> None:
    """Commit only the report in an isolated checkout; leave the VM branch alone."""
    remote = git(repo, "remote", "get-url", "origin")
    allowed = re.fullmatch(
        r"(?:https://github\.com/|git@(?:github\.com|github-finance-ocr):)"
        r"Komu12ee/fincance-ocr(?:\.git)?", remote, flags=re.IGNORECASE,
    )
    if not allowed:
        raise ValueError("origin must be the approved Komu12ee/fincance-ocr GitHub repository")
    identity = {}
    for name, fallback in (("name", "FG OCR Diagnostics"),
                           ("email", "fg-ocr-diagnostics@users.noreply.github.com")):
        try:
            identity[name] = git(repo, "config", "user." + name) or fallback
        except RuntimeError:
            identity[name] = fallback
    with tempfile.TemporaryDirectory(prefix="fg-ocr-diagnostics-") as folder:
        checkout = Path(folder)
        git(checkout, "init", "--quiet")
        git(checkout, "remote", "add", "origin", remote)
        git(checkout, "fetch", "--depth", "1", "origin", "main")
        git(checkout, "checkout", "--quiet", "-b", branch, "FETCH_HEAD")
        for name, value in identity.items():
            git(checkout, "config", "user." + name, value)
        relative = Path("diagnostics") / report.name
        destination = checkout / relative
        destination.parent.mkdir(exist_ok=True)
        shutil.copyfile(report, destination)
        git(checkout, "add", "--force", "--", relative.as_posix())
        git(checkout, "-c", "core.hooksPath=/dev/null", "-c", "commit.gpgsign=false",
            "commit", "--quiet", "-m", "Add sanitized FG VM OCR diagnostics")
        git(checkout, "push", "origin", f"HEAD:refs/heads/{branch}")


def main(argv=None) -> int:
    load_dotenv(PROJECT / "core" / ".env", override=False)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", type=Path, default=PROJECT / "finance_cg_data")
    parser.add_argument("-o", "--output", type=Path, help="Same OCR output directory used by the job")
    parser.add_argument("--probe", action="store_true", help="Send the first PDF's first page to the GPU once")
    parser.add_argument("--push", action="store_true", help="Upload the sanitized report to a new GitHub branch")
    args = parser.parse_args(argv)
    source = args.input.resolve()
    output = (args.output or source.with_name(source.name + "_ocr")).resolve()
    if not source.is_dir():
        parser.error(f"Input directory does not exist: {source}")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    host = hashlib.sha256(socket.gethostname().encode()).hexdigest()[:8]
    suffix = f"{timestamp}-{host}-{uuid.uuid4().hex[:8]}"
    report = PROJECT / "diagnostics" / f"fg-vm-{suffix}.json"
    try:
        data = collect(PROJECT, source, output, args.probe)
        ocr.atomic_json(report, data)
        print(f"Sanitized report saved: {report}")
        if args.push:
            branch = f"fg-ocr-diagnostics-{suffix}"
            push_report(PROJECT, report, branch)
            print(f"Uploaded report branch: {branch}")
            print(f"https://github.com/Komu12ee/fincance-ocr/blob/{branch}/diagnostics/{report.name}")
            print("Send this branch name or link in the chat so it can be fetched and inspected.")
        else:
            print("Nothing uploaded. Add --push to upload a new report.")
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
        print(f"Diagnostics failed: {redact(str(error), PROJECT, (os.getenv('GPU_OCR_API_KEY', ''),))}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
