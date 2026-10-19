"""Compatibility entrypoint for :mod:`build_tools.ci_tools.local_ci_runtime`."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from build_tools.ci_tools import local_ci_runtime as _implementation

if __name__ == '__main__':
    if len(sys.argv) != 3:
        raise SystemExit('Usage: local_ci_runtime.py PRIVATE_STAGE_DIRECTORY REPORT_PATH')
    print(_implementation.validate_stage(Path(sys.argv[1]), Path(sys.argv[2])))
else:
    sys.modules[__name__] = _implementation
