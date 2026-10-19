"""Compatibility entrypoint for :mod:`build_tools.ci_tools.local_ci`."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from build_tools.ci_tools import local_ci as _implementation

if __name__ == '__main__':
    raise SystemExit(_implementation.main())
else:
    sys.modules[__name__] = _implementation
