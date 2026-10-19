"""Configure an existing local Open WebUI; Loom retains lifecycle control."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shlex

from . import hem_arguments as arguments, hem_display
from .hem_common import InvalidInput
from .hem_openwebui_http import canonical_target
from .hem_pi_files import private_path

GUIDANCE = ('Refresh model list in WebUI Admin Settings → Connections after model changes. '
            'Adding a saved server requires configure, then refresh. '
            'Persistent configuration must be enabled without startup resets, and global '
            'user-info forwarding must be disabled; these prerequisites are unverified. '
            'Disable requires reachable WebUI and an administrator credential. '
            'Export/backups are your responsibility.')


def configure(parser):
    commands = parser.add_subparsers(dest='command', required=True)
    child = commands.add_parser('configure', help='configure or inspect Open WebUI connections')
    arguments.output(child)
    arguments.lifecycle(child)
    mode = child.add_mutually_exclusive_group()
    mode.add_argument('--check', action='store_true')
    mode.add_argument('--disable', action='store_true')
    child.add_argument('--url', help='WebUI management origin; overrides OPENWEBUI_URL')


def resolve_target(override=None):
    value = override if override is not None else os.environ.get('OPENWEBUI_URL')
    if value is None:
        raise InvalidInput('Supply --url or OPENWEBUI_URL for the existing local WebUI management origin')
    return canonical_target(value)


def safe_error(error):
    """Parser diagnostics may quote invalid arguments; never echo the admin token."""
    token = os.environ.get('OPENWEBUI_TOKEN')
    message = str(error)
    return message.replace(token, '<redacted>') if token else message


def display_request(target):
    """Display actual routes without expanding credentials or foreign configuration."""
    def display(method, path, intent=None):
        # Routes are fixed adapter operations, never remote-controlled response fields.
        if (method, path) not in {('GET', '/api/version'), ('GET', '/openai/config'),
                                  ('POST', '/openai/config/update')}:
            raise InvalidInput('Unsupported WebUI management display operation')
        command = ('curl --fail --silent --show-error --request ' + method + ' '
                   + shlex.quote(target + path)
                   + ' --header "Authorization: Bearer ${OPENWEBUI_TOKEN}"')
        if method == 'POST':
            hem_display.emit('# WebUI API: REDACTED, NON-REPLAYABLE curl template')
            hem_display.emit(command + " --header 'Content-Type: application/json'"
                             + " --data-binary '<redacted whole configuration>'")
            hem_display.emit('# Read/merge/recheck before update; preserved provider keys omitted. '
                             'Do not submit the owned-only intent as a whole configuration.')
            hem_display.emit('Owned connection intent: ' + json.dumps(intent, ensure_ascii=True, sort_keys=True))
        else:
            hem_display.emit('# WebUI API request; credential supplied from environment')
            hem_display.emit(command)
    return display


def run(args, deadline):
    # Import lazily: help and input validation require no coordinator/runtime assets.
    target = resolve_target(args.url)
    token = os.environ.get('OPENWEBUI_TOKEN')
    if not token:
        raise InvalidInput('Supply an administrator bearer credential in OPENWEBUI_TOKEN; no token flag or automatic login')
    value = Path(os.environ.get('LOOM_SERVE_DIR', str(Path.home() / '.loom')))
    if not value.is_absolute():
        raise InvalidInput('LOOM_SERVE_DIR must be absolute')
    private_path(value)
    root = value.resolve()
    from .hem_openwebui import configure_openwebui
    response = configure_openwebui(root, target, token, deadline,
        check=args.check, disable=args.disable, display=display_request(target))
    response['guidance'] = GUIDANCE
    hem_display.emit(GUIDANCE)
    return response
