#!/usr/bin/env python3
"""Compatibility entrypoint for build_tools.wheel_tools.bazel_compiler_inputs."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from build_tools.wheel_tools.bazel_compiler_inputs import *  # noqa: F401,F403

if __name__ == "__main__":
    raise SystemExit(main())
