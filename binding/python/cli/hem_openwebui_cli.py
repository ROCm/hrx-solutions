"""Open WebUI command grammar and fixed integration dispatch."""
import os

from . import hem_arguments as arguments
from ..integrations.hem_integrations import configure_integration


def configure(parser):
    commands = parser.add_subparsers(dest='command', required=True)
    child = commands.add_parser('configure', help='configure or inspect Open WebUI connections')
    arguments.output(child)
    arguments.lifecycle(child)
    mode = child.add_mutually_exclusive_group()
    mode.add_argument('--check', action='store_true')
    mode.add_argument('--disable', action='store_true')
    child.add_argument('--url', help='WebUI management origin; overrides OPENWEBUI_URL')
    child.add_argument('--stt-server', type=arguments.name, help='Saved server for STT selection; requires --stt-instance, may be stopped')
    child.add_argument('--stt-instance', type=arguments.name, help='Unprefixed STT instance name, including a future instance; requires --stt-server')


def safe_error(error):
    """Parser diagnostics may quote invalid arguments; never echo the admin token."""
    token = os.environ.get('OPENWEBUI_TOKEN')
    message = str(error)
    return message.replace(token, '<redacted>') if token else message


def run(args, deadline):
    mode = 'disable' if args.disable else 'check' if args.check else 'configure'
    return configure_integration('openwebui', None, deadline, mode=mode,
        url=args.url, stt_server=args.stt_server, stt_instance=args.stt_instance)
