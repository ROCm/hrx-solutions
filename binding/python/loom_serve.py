"""Console entry point for the HRX Loom Serve server."""

from typing import NoReturn

from ._launcher import exec_loom_serve


def main() -> NoReturn:
    exec_loom_serve()
