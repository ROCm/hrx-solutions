#!/usr/bin/env python3
"""Setuptools entry point for the Bazel-built HRX Solutions wheel."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from setuptools import setup

from build_tools.wheel_tools.python_packaging import (
    BazelBuildPy,
    BinaryDistribution,
    Py3PlatformWheel,
    package_version,
)


setup(
    version=package_version(),
    distclass=BinaryDistribution,
    cmdclass={
        "build_py": BazelBuildPy,
        "bdist_wheel": Py3PlatformWheel,
    },
)
