# Copyright 2026 The HRX Authors
# Licensed under the MIT License.
# SPDX-License-Identifier: MIT
"""Shared source-test setup and discovery, invoked from the repository root.

Callers own venv creation, environment cleaning, runtime fixtures and logging.
Each action replaces this process so exits and cancellation reach pip/unittest.
"""
import argparse
import os
from pathlib import Path
import site
import sys

REQUIREMENTS = ('setuptools>=77', 'wheel>=0.45', 'packaging>=24.2', 'build>=1.2',
                'textual>=0.47,<2')
DISCOVERY_DIRECTORIES = ('build_tools', 'tests')


def verify_provenance():
    import setuptools
    import packaging
    import textual
    import build

    prefix = Path(sys.prefix).resolve()
    assert sys.prefix != sys.base_prefix and not site.ENABLE_USER_SITE
    assert 'include-system-site-packages = false' in (prefix / 'pyvenv.cfg').read_text()
    assert not os.environ.get('PYTHONPATH') and not os.environ.get('PYTHONHOME')
    modules = (setuptools, packaging, textual, build)
    assert all(Path(module.__file__).resolve().is_relative_to(prefix) for module in modules)
    print(sys.executable, *(module.__file__ for module in modules))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('action', choices=('install', 'verify', *DISCOVERY_DIRECTORIES))
    args = parser.parse_args(argv)
    if args.action == 'verify':
        verify_provenance()
        return 0
    if args.action == 'install':
        command = ['-m', 'pip', 'install', '--disable-pip-version-check', *REQUIREMENTS]
    else:
        command = ['-m', 'unittest', 'discover', '-s', args.action, '-p', '*_test.py', '-v']
    os.execv(sys.executable, [sys.executable, *command])


if __name__ == '__main__':
    sys.exit(main())
