"""Persistent FG OCR progress, readable without contacting the GPU server."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import tempfile

STATUS_FILE = "ocr_status.json"


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".ocr-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False)
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def record(path: Path) -> dict:
    value = read_json(path)
    return value if isinstance(value, dict) else {}


def output_stat(path: Path) -> dict:
    stat = path.stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def cache_directory(output: Path, relative: Path) -> Path:
    digest = hashlib.sha256(relative.as_posix().encode("utf-8")).hexdigest()
    return output / ".ocr-cache" / digest


def source_signature(source: Path, relative: Path, config: dict) -> dict:
    stat = source.stat()
    return {**config, "source": relative.as_posix(),
            "source_size": stat.st_size, "source_mtime_ns": stat.st_mtime_ns}


def is_complete(source: Path, relative: Path, output: Path, sig: dict, writer_version: int) -> bool:
    completed = record(cache_directory(output, relative) / "completed.json")
    destination = output / relative
    try:
        return (
            completed.get("signature") == sig
            and completed.get("text_layer_version") == writer_version
            and completed.get("output") == output_stat(destination)
            and destination.with_name(destination.name + ".ocr.json").is_file()
            and destination.with_name(destination.name + ".ocr.md").is_file()
        )
    except OSError:
        return False


def scan_documents(source: Path, output: Path, paths: list[Path],
                   config: dict, writer_version: int) -> dict[str, str]:
    """Count current results, not historical entries in the append-only error log."""
    states = {}
    for path in paths:
        relative = path.relative_to(source)
        name = relative.as_posix()
        try:
            sig = source_signature(path, relative, config)
        except OSError:
            states[name] = "pending"
            continue
        progress = record(cache_directory(output, relative) / "progress.json")
        if progress.get("signature") == sig and progress.get("state") in ("running", "failed"):
            states[name] = "failed" if progress["state"] == "failed" else "pending"
        elif is_complete(path, relative, output, sig, writer_version):
            states[name] = "done"
        else:
            states[name] = "pending"
    return states


def counts_for(states: dict[str, str]) -> dict:
    counts = Counter(states.values())
    return {"total": len(states), "done": counts["done"],
            "failed": counts["failed"], "pending": counts["pending"]}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def process_token(pid: int) -> str | None:
    # Reading /proc is safe; os.kill(pid, 0) is unsafe on Windows.
    try:
        content = Path(f"/proc/{pid}/stat").read_text()
        return content[content.rfind(")") + 2:].split()[19]
    except (OSError, IndexError):
        return None


def stopped_without_checkpoint(snapshot: dict) -> bool:
    pid = snapshot.get("pid")
    token = snapshot.get("process_token")
    if (snapshot.get("host") != socket.gethostname() or not pid or not token
            or not Path("/proc").is_dir()):
        return False
    return process_token(pid) != token


def status_snapshot(source: Path, output: Path, paths: list[Path],
                    default_config: dict, writer_version: int) -> dict:
    snapshot = record(output / STATUS_FILE)
    if snapshot.get("input") != str(source):
        snapshot = {}
    config = snapshot.get("ocr_config", default_config)
    snapshot.update({"input": str(source), "output": str(output),
                     "counts": counts_for(scan_documents(source, output, paths, config, writer_version))})
    snapshot.setdefault("gpu_url", config["base_url"])
    if snapshot.get("state") in ("counted", "checking_gpu", "running") and stopped_without_checkpoint(snapshot):
        snapshot["state"] = "stopped_without_checkpoint"
    return snapshot


def duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, rest = divmod(seconds, 3600)
    minutes, seconds = divmod(rest, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def print_status(snapshot: dict) -> None:
    counts = snapshot["counts"]
    total, done = counts["total"], counts["done"]
    ratio = done / total if total else 0
    filled = int(30 * ratio)
    bar = "#" * filled + "-" * (30 - filled)
    run = snapshot.get("run", {})
    current = snapshot.get("current") or {}
    elapsed = "--"
    if snapshot.get("started_at"):
        start = datetime.fromisoformat(snapshot["started_at"])
        end = datetime.fromisoformat(snapshot["finished_at"]) if snapshot.get("finished_at") else datetime.now(timezone.utc)
        elapsed = duration((end - start).total_seconds())
    disk_path = Path(snapshot["output"])
    while not disk_path.exists() and disk_path != disk_path.parent:
        disk_path = disk_path.parent
    try:
        free = f"{shutil.disk_usage(disk_path).free / 1024 ** 3:.1f} GiB"
    except OSError:
        free = "unavailable"
    print("\n-- FG OCR STATUS --")
    print(f"Input:  {snapshot['input']}")
    print(f"Output: {snapshot['output']}")
    print(f"{'Corpus':<8} {'PDFs present':>12} {'OCR done':>10} {'OCR failed':>12} {'Pending':>10}")
    print(f"{'fg':<8} {total:>12} {done:>10} {counts['failed']:>12} {counts['pending']:>10}")
    print(f"Progress: [{bar}] {ratio:.1%}")
    print(f"Last run: {snapshot.get('state', 'not_started').replace('_', ' ').upper()}")
    if run:
        print(f"This run: selected {run['selected']} | checked {run['checked']} | "
              f"newly done {run['processed']} | already done {run['skipped']} | failed {run['failed']}")
        print(f"Pages verified this run: {run['pages_verified']} (cached: {run['pages_cached']})")
    if current:
        print(f"Current PDF [{current['position']}/{run['selected']}]: {current['file']}")
        print(f"Current page: {current.get('page', 0)}/{current.get('total_pages') or '?'} | "
              f"{current.get('stage', 'opening PDF')}")
    else:
        print("Current PDF: --")
    print(f"GPU API: {snapshot.get('gpu_url', '--')} | last check: {snapshot.get('gpu_state', 'not checked')}")
    if snapshot.get("gpu_checked_at"):
        print(f"GPU checked (UTC): {snapshot['gpu_checked_at']}")
    print(f"Elapsed: {elapsed} | Free disk: {free}")
    print(f"Updated (UTC): {snapshot.get('updated_at', '--')}")
    if snapshot.get("last_error"):
        print(f"Last error: {snapshot['last_error']}")
    print(flush=True)


class OCRTracker:
    """Single-writer tracking; total counts cover the entire recursively found corpus."""

    def __init__(self, source: Path, output: Path, paths: list[Path],
                 config: dict, writer_version: int, selected: int):
        self.source, self.output, self.config = source, output, config
        self.states = scan_documents(source, output, paths, config, writer_version)
        self.snapshot = {
            "schema_version": 1, "corpus": "fg", "input": str(source), "output": str(output),
            "ocr_config": config, "started_at": utc_now(), "state": "counted",
            "pid": os.getpid(), "host": socket.gethostname(), "process_token": process_token(os.getpid()),
            "gpu_url": config["base_url"], "gpu_state": "not checked", "current": None,
            "run": {"selected": selected, "checked": 0, "processed": 0, "skipped": 0,
                    "failed": 0, "pages_verified": 0, "pages_cached": 0},
        }
        self.save()

    def save(self) -> None:
        self.snapshot["counts"] = counts_for(self.states)
        self.snapshot["updated_at"] = utc_now()
        atomic_json(self.output / STATUS_FILE, self.snapshot)

    def show(self) -> None:
        print_status(self.snapshot)

    def gpu(self, state: str, error: str | None = None) -> None:
        self.snapshot["gpu_state"] = state
        self.snapshot["gpu_checked_at"] = utc_now()
        if error:
            self.snapshot["last_error"] = error
        self.save()

    def set_state(self, state: str) -> None:
        self.snapshot["state"] = state
        self.save()

    def begin(self, relative: Path, position: int) -> None:
        name = relative.as_posix()
        self.states[name] = "pending"
        self.snapshot["current"] = {"file": name, "position": position,
                                    "page": 0, "total_pages": None, "stage": "opening PDF"}
        self.document_record = cache_directory(self.output, relative) / "progress.json"
        self.document_signature = {}
        self.document_signature = source_signature(self.source / relative, relative, self.config)
        self.save_document("running")

    def save_document(self, state: str, error: str | None = None) -> None:
        atomic_json(self.document_record, {
            "signature": self.document_signature, "state": state, "updated_at": utc_now(),
            "current": self.snapshot["current"], "error": error,
        })
        self.save()

    def page(self, number: int, total: int, stage: str, verified: bool = False,
             cached: bool = False) -> None:
        self.snapshot["current"].update({"page": number, "total_pages": total, "stage": stage})
        if verified:
            self.snapshot["run"]["pages_verified"] += 1
            self.snapshot["run"]["pages_cached"] += int(cached)
        self.save_document("running")

    def finish_document(self, relative: Path, status: str, error: str | None = None) -> None:
        self.states[relative.as_posix()] = "failed" if status == "failed" else "done"
        run = self.snapshot["run"]
        run["checked"] += 1
        run[status] += 1
        if error:
            self.snapshot["last_error"] = f"{relative.as_posix()}: {error}"
        self.save_document("failed" if error else "done", error)
        self.snapshot["current"] = None
        self.save()
        self.show()

    def finish_run(self, state: str) -> None:
        self.snapshot["state"] = state
        self.snapshot["finished_at"] = utc_now()
        self.save()
        self.show()
