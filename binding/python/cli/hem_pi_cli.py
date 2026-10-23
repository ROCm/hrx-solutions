"""Pi command grammar and integration dispatch."""
from pathlib import Path

from . import hem_arguments as arguments
from ..hem_integrations import configure_integration


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
    mode = "disable" if args.disable else "check" if args.check else "configure"
    return configure_integration("pi", None, deadline, mode=mode, pi_dir=args.pi_dir)
