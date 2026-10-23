"""Shared argument spelling and validation for the four domain parsers."""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

from ..hem_common import DEFAULT_TIMEOUT, InvalidInput


def name(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise argparse.ArgumentTypeError("use a name containing letters, digits, '.', '_' or '-'")
    return value


def positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("expected a positive integer") from None
    if number <= 0:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return number


def timeout(value: str) -> float:
    try:
        number = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError("expected a finite positive timeout") from None
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("expected a finite positive timeout")
    return number


def output(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--json", action="store_true", default=argparse.SUPPRESS,
                        help="write one JSON action result to stdout; diagnostics and command "
                             "equivalents stay on stderr; help remains text")


def lifecycle(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--timeout", type=timeout, default=DEFAULT_TIMEOUT,
                        metavar="SECONDS")


def selector(parser: argparse.ArgumentParser) -> None:
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--server", type=name)
    group.add_argument("--url", help="native endpoint; does not grant process ownership")


def serving(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--name", type=name)
    parser.add_argument("--config", type=Path)
    for option in ("rows", "context-capacity", "pool-capacity", "prefill-capacity", "mtp-depth"):
        parser.add_argument("--" + option, type=positive_int, default=None)
    mtp = parser.add_mutually_exclusive_group()
    mtp.add_argument("--mtp", dest="mtp", action="store_true", default=None)
    mtp.add_argument("--no-mtp", dest="mtp", action="store_false")


def select_server(explicit: str | None, configured: list[str]) -> str:
    """Selection only; this helper does not load a registry or contact HTTP."""
    if explicit is not None:
        if explicit not in configured:
            raise InvalidInput(f"unknown server: {explicit}")
        return explicit
    if len(configured) != 1:
        raise InvalidInput("select --server NAME or --url URL; expected one configured server")
    return configured[0]
