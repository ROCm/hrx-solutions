"""Model acquisition and inventory grammar. Implementation follows separately."""

from pathlib import Path

from . import hem_arguments as arguments
from .hem_common import unimplemented


def configure(parser):
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("available", "download", "import", "validate", "list", "show"):
        child = commands.add_parser(command)
        arguments.output(child)
        if command == "download":
            child.add_argument("recipe")
            child.add_argument("--id", required=True, type=arguments.name)
            child.add_argument("--offline", action="store_true")
            child.add_argument("--dry-run", action="store_true")
        elif command in ("import", "validate", "show"):
            child.add_argument("model", type=arguments.name)
        if command in ("download", "import", "validate"):
            child.add_argument("--pipeline", type=arguments.name)
        if command == "import":
            child.add_argument("--gguf", nargs="+", required=True, type=Path)
            child.add_argument("--tokenizer", required=True, type=Path)
            child.add_argument("--alias", action="store_true")


def run(args, deadline):
    return unimplemented("model", args.command)
