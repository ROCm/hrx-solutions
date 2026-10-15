"""Saved server and owned process grammar."""

from . import hem_arguments as arguments
from .hem_common import InvalidInput, unimplemented


def configure(parser):
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("start", "list", "show", "status", "logs", "reconcile", "shutdown"):
        child = commands.add_parser(command)
        arguments.output(child)
        if command != "list":
            child.add_argument("server", nargs="?" if command == "start" else None,
                               type=arguments.name)
        if command == "start":
            child.add_argument("--name", type=arguments.name)
            child.add_argument("--device", default=None)
            child.add_argument("--listen", default=None)
        if command in ("start", "shutdown"):
            arguments.lifecycle(child)
        if command == "shutdown":
            child.add_argument("--abort", action="store_true")
        if command == "logs":
            child.add_argument("--follow", action="store_true")


def validate(args):
    if args.command == "start":
        if bool(args.server) == bool(args.name):
            raise InvalidInput("use server start NAME or server start --name NAME")
        if args.server and (args.device is not None or args.listen is not None):
            raise InvalidInput("saved server start reuses settings; use --name to create another definition")


def run(args, deadline):
    return unimplemented("server", args.command)
