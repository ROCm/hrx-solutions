"""Native registration and offline deployment grammar."""

from pathlib import Path

from . import hem_arguments as arguments
from .hem_common import unimplemented


def configure(parser):
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("start", "stop", "list", "show", "status", "logs", "export"):
        child = commands.add_parser(command)
        arguments.output(child)
        if command in ("start", "export"):
            child.add_argument("model", type=arguments.name)
            arguments.serving(child)
        elif command != "list":
            child.add_argument("instance", type=arguments.name)
        if command != "export":
            arguments.selector(child)
        if command in ("start", "stop"):
            arguments.lifecycle(child)
        if command == "stop":
            child.add_argument("--abort", action="store_true")
            child.add_argument("--no-wait", action="store_true")
        if command == "logs":
            child.add_argument("--follow", action="store_true")
        if command == "export":
            child.add_argument("--output", type=Path, required=True)


def run(args, deadline):
    return unimplemented("instance", args.command)
