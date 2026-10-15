"""Non-creating state/cache paths for direct native catalog translation."""
from __future__ import annotations

import os
from pathlib import Path


def state_root() -> Path:
    """Resolve Loom-owned state without creating it (evaluated per call)."""
    override = os.environ.get("LOOM_SERVE_DIR")
    return Path(override).expanduser().absolute() if override else Path.home() / ".loom"


def runtime_cache_dir() -> Path:
    """Resolve generated catalog/deployment cache without creating it."""
    if os.environ.get("LOOM_SERVE_DIR"):
        return state_root() / "cache"
    return Path.home() / ".cache" / "hrx-solutions"
