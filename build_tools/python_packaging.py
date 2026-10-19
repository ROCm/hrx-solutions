#!/usr/bin/env python3
"""Compatibility entrypoint for build_tools.wheel_tools.python_packaging."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from build_tools.wheel_tools.python_packaging import *  # noqa: F401,F403
