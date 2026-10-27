"""Concrete executable inventory selected by safe declaration reference."""
from . import hem_arguments as arguments
from ..models import hem_config


def configure(parser):
    commands = parser.add_subparsers(dest="command", required=True)
    listing = commands.add_parser("list")
    arguments.output(listing)
    listing.add_argument("--modality", choices=("chat", "asr"))
    show = commands.add_parser("show")
    arguments.output(show)
    show.add_argument("executable", metavar="REF")


def run(args, deadline):
    if args.command == "list":
        return {"state": "available", "completed": True,
                "items": hem_config.executable_list(args.modality, deadline=deadline)}
    return {"state": "available", "completed": True,
            **hem_config.executable_info(args.executable, deadline=deadline)}
