"""Compatibility import for :mod:`build_tools.ci_tools.local_ci_source`."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from build_tools.ci_tools import local_ci_source as _implementation
sys.modules[__name__] = _implementation
