"""Compatibility entrypoint for the model catalog tools."""
from __future__ import annotations

import sys

try:
    from .model_tools import loom_catalog as _implementation
except ImportError:
    from model_tools import loom_catalog as _implementation

if __name__ == "__main__":
    _implementation.main()
else:
    sys.modules[__name__] = _implementation
