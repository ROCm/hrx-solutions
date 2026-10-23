"""OpenCode command grammar and fixed integration dispatch."""
from pathlib import Path

from . import hem_arguments as arguments
from ..hem_integrations import configure_integration


def configure(parser):
    commands = parser.add_subparsers(dest="command", required=True)
    child = commands.add_parser("configure", help="configure or inspect OpenCode 2 integration")
    arguments.output(child)
    arguments.lifecycle(child)
    mode = child.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="inspect without writes or client execution")
    mode.add_argument("--disable", action="store_true", help="remove unchanged owned files without client")
    child.add_argument("--opencode-dir", type=Path, help="alternate OpenCode configuration directory")


def run(args, deadline):
    mode = "disable" if args.disable else "check" if args.check else "configure"
    return configure_integration("opencode", None, deadline, mode=mode, opencode_dir=args.opencode_dir)
