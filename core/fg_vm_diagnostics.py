#!/usr/bin/env python3
"""Collect FG OCR diagnostics; optionally probe one page and push a report branch."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from datetime import datetime, timezone
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import socket
import subprocess
import tempfile
import time
from types import SimpleNamespace
from urllib.parse import urlsplit, urlunsplit
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


def select_prompt(config: dict, prompt_file: Path | None = None) -> tuple[str | None, dict]:
    previous = config.get("prompt_sha256")
    candidates = [("default JSON prompt", ocr.DEFAULT_PROMPT)]
    bundled = PROJECT / "core" / "fg_ocr_prompt.txt"
    if bundled.is_file():
        candidates.append(("bundled FG text prompt", bundled.read_text(encoding="utf-8")))
    if prompt_file:
        candidates = [("explicit prompt file", prompt_file.read_text(encoding="utf-8"))]
    for label, text in candidates:
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if prompt_file or previous == digest or (not previous and label == "default JSON prompt"):
            return text, {"source": label, "sha256": digest,
                          "matches_last_run": digest == previous if previous else None}
    return None, {"source": "unknown custom prompt", "matches_last_run": False,
                  "error": "Cannot reproduce the last prompt; pass --prompt-file with the same file."}


def runtime_snapshot(client: ocr.GPUClient) -> dict:
    """Read GPU utilization and content-free server metrics; never control jobs."""
    sample = {"utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    executable = shutil.which("nvidia-smi")
    if executable:
        try:
            result = subprocess.run([
                executable, "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,power.draw",
                "--format=csv,noheader,nounits",
            ], capture_output=True, text=True, timeout=10)
            sample["gpu_csv_columns"] = ["index", "name", "utilization_percent", "used_memory_mib",
                                         "total_memory_mib", "power_watts"]
            sample["gpu_csv"] = result.stdout.strip()[:2000] if result.returncode == 0 else "unavailable"
        except (OSError, subprocess.TimeoutExpired):
            sample["gpu_csv"] = "unavailable"
    else:
        sample["gpu_csv"] = "nvidia-smi unavailable on this host"
    parts = urlsplit(client.url)
    metrics_url = urlunsplit((parts.scheme, parts.netloc, "/metrics", "", ""))
    try:
        response = requests.get(metrics_url, headers=client.headers, timeout=(3, 3))
        client.raise_for_status(response)
        selected = {
            "num_requests_running", "num_requests_waiting", "generation_tokens_total", "prompt_tokens_total",
            "request_success_total", "kv_cache_usage_perc", "gpu_cache_usage_perc",
            "request_queue_time_seconds_sum", "request_queue_time_seconds_count",
            "request_prefill_time_seconds_sum", "request_prefill_time_seconds_count",
            "request_decode_time_seconds_sum", "request_decode_time_seconds_count",
            "time_to_first_token_seconds_sum", "time_to_first_token_seconds_count",
        }
        metrics = []
        for line in response.text[:2_000_000].splitlines():
            match = re.fullmatch(r"vllm:([a-z_]+)(?:\{(.*)\})?\s+(\S+)(?:\s+\S+)?", line)
            if match and match[1] in selected:
                try:
                    value = float(match[3])
                except ValueError:
                    continue
                if not math.isfinite(value):
                    continue
                labels = dict(re.findall(r'(model_name|engine|finished_reason)="([^"\n]*)"', match[2] or ""))
                metrics.append({"name": match[1], "value": value, **labels})
        sample["vllm_metrics"] = metrics
    except (requests.RequestException, ValueError) as error:
        sample["metrics_error"] = str(error)
    return sample


def collect(repo: Path, source: Path, output: Path, probe: bool,
            prompt_file: Path | None = None) -> dict:
    saved = ocr.read_json(output / "ocr_status.json")
    saved = saved if isinstance(saved, dict) else {}
    config = saved.get("ocr_config") or {}
    prompt, prompt_info = select_prompt(config, prompt_file)
    args = SimpleNamespace(
        base_url=config.get("base_url", os.getenv("GPU_OCR_BASE_URL", "http://127.0.0.1:8000/v1")),
        model=config.get("model", os.getenv("GPU_OCR_MODEL", "ocr-primary")),
        dpi=config.get("dpi", 200), max_pixels=config.get("max_pixels", 6_000_000),
        max_tokens=config.get("max_tokens", 8192), timeout=300, retries=0,
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
        "settings": vars(args), "prompt": prompt_info,
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
    client = ocr.GPUClient(args, prompt or ocr.DEFAULT_PROMPT)
    try:
        response = requests.get(f"{client.url}/models", headers=client.headers, timeout=(10, 30))
        client.raise_for_status(response)
        report["gpu_models"] = [
            {key: model[key] for key in ("id", "root", "max_model_len") if key in model}
            for model in response.json().get("data", []) if isinstance(model, dict)
        ]
    except (requests.RequestException, ValueError) as error:
        report["gpu_models_error"] = str(error)
    report["runtime_samples"] = [runtime_snapshot(client)]
    if probe:
        report["probe"] = {"note": "One source page sent to the configured GPU; no PDFs or OCR text uploaded to GitHub."}
        started = time.monotonic()
        telemetry = []
        try:
            if prompt is None:
                raise ValueError(prompt_info["error"])
            if not paths:
                raise ValueError("No FG PDFs found")
            report["probe"]["file"] = str(paths[0].relative_to(source))
            report["probe"]["page"] = 1
            print("FG probe: rendering page 1...", flush=True)
            render_started = time.monotonic()
            with ocr.fitz.open(paths[0]) as doc:
                pixmap = ocr.render_pdf_page(doc[0], args.dpi, args.max_pixels)
                report["probe"]["image"] = {"width": pixmap.width, "height": pixmap.height}
                image = pixmap.tobytes("png")
            report["probe"]["render_seconds"] = round(time.monotonic() - render_started, 3)
            print(f"FG probe: requesting OCR with {prompt_info['source']}...", flush=True)
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(client.ocr, image, telemetry)
                while True:
                    try:
                        result = future.result(timeout=10)
                        break
                    except FutureTimeout:
                        if future.done():
                            raise
                        print(f"FG probe: waiting {time.monotonic() - started:.0f}s; sampling GPU and queue...",
                              flush=True)
                        report["runtime_samples"].append(runtime_snapshot(client))
            report["probe"].update({"state": "success", "line_count": len(result["lines"]),
                                     "positioning": result["positioning"]})
        except Exception as error:
            report["probe"].update({"state": "failed", "error": str(error)})
        finally:
            report["probe"]["elapsed_seconds"] = round(time.monotonic() - started, 3)
            report["probe"]["requests"] = telemetry
            report["runtime_samples"].append(runtime_snapshot(client))
    return redact(report, repo, (os.getenv("GPU_OCR_API_KEY", ""),))


def github_repository(remote: str) -> str:
    """Recognize the two FG repository names used by the processing VMs."""
    allowed = re.fullmatch(
        r"(?:https://github\.com/|git@(?:github\.com|github-finance-ocr):)"
        r"(?P<repository>Komu12ee/(?:fincance-ocr|finance-ocr))(?:\.git)?",
        remote, flags=re.IGNORECASE,
    )
    if not allowed:
        raise ValueError("origin must be the approved Komu12ee/fincance-ocr or Komu12ee/finance-ocr GitHub repository")
    return allowed.group("repository")


def push_report(repo: Path, report: Path, branch: str) -> str:
    """Commit only the report in an isolated checkout; leave the VM branch alone."""
    remote = git(repo, "remote", "get-url", "origin")
    repository = github_repository(remote)
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
    return f"https://github.com/{repository}/blob/{branch}/diagnostics/{report.name}"


def main(argv=None) -> int:
    load_dotenv(PROJECT / "core" / ".env", override=False)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", type=Path, default=PROJECT / "finance_cg_data")
    parser.add_argument("-o", "--output", type=Path, help="Same OCR output directory used by the job")
    parser.add_argument("--probe", action="store_true", help="Send the first PDF's first page to the GPU once")
    parser.add_argument("--push", action="store_true", help="Upload the sanitized report to a new GitHub branch")
    parser.add_argument("--prompt-file", type=Path, help="Same custom prompt file used by the OCR job")
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
        print("Collecting FG diagnostics and GPU/server metrics...", flush=True)
        data = collect(PROJECT, source, output, args.probe, args.prompt_file)
        ocr.atomic_json(report, data)
        print(f"Sanitized report saved: {report}")
        if args.push:
            branch = f"fg-ocr-diagnostics-{suffix}"
            url = push_report(PROJECT, report, branch)
            print(f"Uploaded report branch: {branch}")
            print(url)
            print("Send this branch name or link in the chat so it can be fetched and inspected.")
        else:
            print("Nothing uploaded. Add --push to upload a new report.")
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
        print(f"Diagnostics failed: {redact(str(error), PROJECT, (os.getenv('GPU_OCR_API_KEY', ''),))}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
