"""Packaged pipeline inventory grammar."""

from . import hem_arguments as arguments
from .hem_common import unimplemented


def configure(parser):
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("list", "show"):
        child = commands.add_parser(command)
        arguments.output(child)
        if command == "show":
            child.add_argument("pipeline", type=arguments.name)


def run(args, deadline):
    return unimplemented("pipeline", args.command)
