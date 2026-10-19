#!/usr/bin/env python3
"""Compatibility entry point for release publishing."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from build_tools.release import publish_release as _implementation


if __name__ == "__main__":
    raise SystemExit(_implementation.main())

# Preserve the historical import path, including module-level patch points.
sys.modules[__name__] = _implementation
