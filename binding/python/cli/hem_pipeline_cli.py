"""Packaged pipeline inventory grammar."""

from . import hem_arguments as arguments
from ..models.hem_config import pipeline_info
from ..models.pipeline_validation import available_pipelines


def configure(parser):
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("list", "show"):
        child = commands.add_parser(command)
        arguments.output(child)
        if command == "show":
            child.add_argument("pipeline", type=arguments.name)


def run(args, deadline):
    if args.command == "list":
        return {"state": "available", "completed": True,
                "items": [pipeline_info(identifier) for identifier in available_pipelines()]}
    return {"state": "available", "completed": True, **pipeline_info(args.pipeline)}
