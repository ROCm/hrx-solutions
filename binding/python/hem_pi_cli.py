"""The single Pi configuration command; operational wiring follows separately."""

from pathlib import Path

from . import hem_arguments as arguments
from .hem_common import unimplemented


def configure(parser):
    commands = parser.add_subparsers(dest="command", required=True)
    child = commands.add_parser("configure", help="configure or inspect Pi integration")
    arguments.output(child)
    arguments.lifecycle(child)
    mode = child.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="inspect without writing")
    mode.add_argument("--disable", action="store_true", help="remove unchanged owned files")
    child.add_argument("--pi-dir", type=Path, help="alternate Pi agent directory")


def run(args, deadline):
    # Parsing/help must never initialize state or imply configuration succeeded.
    deadline.remaining()
    return unimplemented("pi", args.command)
