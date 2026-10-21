#!/usr/bin/env python3
"""Compatibility entrypoint for the installed-wheel check."""
import runpy
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
runpy.run_module("build_tools.validation.check_installed_wheel", run_name="__main__")
