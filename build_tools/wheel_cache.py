"""Compatibility CLI for :mod:`build_tools.ci_tools.wheel_cache`."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from build_tools.ci_tools import wheel_cache as _implementation

if __name__ == '__main__':
    raise SystemExit(_implementation.main())
else:
    sys.modules[__name__] = _implementation
