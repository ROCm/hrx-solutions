"""Shared per-user state and locking helpers for Loom setup tools."""

from __future__ import annotations

from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import tempfile
from typing import Any, BinaryIO, Iterator, Optional


def state_dir() -> Path:
    """Return the Loom state directory, creating it when needed."""
    path = Path.home() / ".loom"
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.chmod(0o700)
    return path


@contextmanager
def state_lock() -> Iterator[None]:
    """Serialize state reconciliation and updates across Loom commands.

    Callers should hold this only while accessing state files, not while
    waiting for a server or performing network probes.
    """
    directory = state_dir()
    lock_path = directory / "state.lock"
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    os.fchmod(descriptor, 0o600)
    with os.fdopen(descriptor, "r+") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


@contextmanager
def private_append_log(path: Path) -> Iterator[BinaryIO]:
    """Open an append-only log with owner-only permissions."""
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.parent.chmod(0o700)
    descriptor = os.open(path, os.O_CREAT | os.O_WRONLY | os.O_APPEND, 0o600)
    os.fchmod(descriptor, 0o600)
    with os.fdopen(descriptor, "ab", buffering=0) as stream:
        yield stream


def read_json(path: Path, default: Any = None) -> Any:
    """Read JSON from ``path`` or return ``default`` when it is absent."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return default


def write_json_atomic(path: Path, value: Any) -> None:
    """Write JSON using a same-directory temporary file and atomic replace."""
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.parent.chmod(0o700)
    temporary: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
        temporary.chmod(0o600)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
