"""Finite modality-contract discovery, distinct from executable selection."""

from . import hem_arguments as arguments
from ..models import hem_config


def configure(parser):
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("list", "show"):
        child = commands.add_parser(command)
        arguments.output(child)
        if command == "show":
            child.add_argument("modality", choices=("chat", "asr", "tts"))


def run(args, deadline):
    if args.command == "list":
        return {"state": "available", "completed": True,
                "items": hem_config.modality_list(deadline=deadline)}
    return {"state": "available", "completed": True,
            **hem_config.modality_info(args.modality, deadline=deadline)}
