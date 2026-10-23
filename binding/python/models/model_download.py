"""Pinned recipes, local Hub cache inspection and a streamed HF CLI primitive.

Hem owns model publication and inventory; these helpers never import old state.
"""
from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time

from . import model_store
from .recipe_types import HFSource
from .. import hem_display


def cache_root() -> Path:
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"]).expanduser().absolute()
    if os.environ.get("HUGGINGFACE_HUB_CACHE"):
        return Path(os.environ["HUGGINGFACE_HUB_CACHE"]).expanduser().absolute()
    home = os.environ.get("HF_HOME")
    if home:
        return Path(home).expanduser().absolute() / "hub"
    return Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))).expanduser().absolute() / "huggingface" / "hub"


def _cache_path(source: HFSource) -> Path:
    """Name the immutable snapshot locator; never resolve it to a mutable copy."""
    return (cache_root() / ("models--" + source.repo_id.replace("/", "--")) /
            "snapshots" / source.revision / source.filename)


def source_status(source: HFSource) -> dict:
    path = _cache_path(source)
    cached = model_store.readable_file(path)
    return dict(repo_id=source.repo_id, revision=source.revision, filename=source.filename,
                cached=cached, cache_path=str(path),
                size_bytes=path.stat().st_size if cached else None)


def command(source: HFSource) -> list[str]:
    return ['hf', 'download', source.repo_id, source.filename, '--revision',
            source.revision, '--cache-dir', str(cache_root())]


def _redact(value: str) -> str:
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        token = os.environ.get(key)
        if token:
            value = value.replace(token, "[REDACTED]")
    return re.sub(r"hf_[A-Za-z0-9]+", "[REDACTED]", value)


def _run_hf(command: list[str]) -> None:
    """Stream and drain both CLI channels to stderr without retaining output."""
    hem_display.emit(hem_display.command(command, secrets=[
        argument for argument in command if _redact(argument) != argument]))
    environment = os.environ.copy()
    environment["HF_HUB_DISABLE_TELEMETRY"] = "1"
    environment["HF_HUB_DISABLE_PROGRESS_BARS"] = "0"
    # HF otherwise hides byte progress when its stderr is piped to our streamer.
    environment["TQDM_POSITION"] = "-1"
    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   text=True, env=environment, errors="replace")
    except FileNotFoundError as exc:
        raise model_store.ModelImportError("Hugging Face CLI 'hf' is required for downloads; install huggingface_hub with its CLI, then run 'hf auth login' for gated repositories. See the setup documentation for Python 3.9.") from exc
    last_update = time.monotonic()
    finished = threading.Event()
    def heartbeat():
        while not finished.wait(5):
            if time.monotonic() - last_update >= 5:
                _progress("Waiting for Hugging Face CLI…")
    def pump(stream):
        nonlocal last_update
        pending = []
        while True:
            character = stream.read(1)
            if not character:
                break
            last_update = time.monotonic()
            pending.append(character)
            if character in "\n\r" or len(pending) >= 16384:
                line = _redact("".join(pending))
                sys.stderr.write(line)
                sys.stderr.flush()
                pending = []
        if pending:
            line = _redact("".join(pending))
            sys.stderr.write(line)
            sys.stderr.flush()
        stream.close()
    workers = [threading.Thread(target=pump, args=(process.stdout,), daemon=True),
               threading.Thread(target=pump, args=(process.stderr,), daemon=True)]
    monitor = threading.Thread(target=heartbeat, daemon=True)
    monitor.start()
    for worker in workers:
        worker.start()
    try:
        returncode = process.wait()
    except BaseException:
        process.terminate()
        process.wait()
        raise
    finally:
        finished.set()
        monitor.join()
        for worker in workers:
            worker.join()
    if returncode:
        raise model_store.ModelImportError(f"hf download failed (exit {returncode}); see the streamed error above. Check repository access with 'hf auth login', pinned filenames, cache permissions, and connectivity.")


def _progress(message: str) -> None:
    print(message, file=sys.stderr, flush=True)
