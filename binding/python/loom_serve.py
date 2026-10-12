"""Console entry point for the pinned HRX Loom Serve text server."""

from typing import NoReturn

from ._launcher import exec_loom_serve


def main() -> NoReturn:
    exec_loom_serve()
