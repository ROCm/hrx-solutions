"""Help copy and presentation for the existing four-resource grammar.

This module only describes parsers. It never opens registries, inspects assets or
contacts services. Configuration numbers come from the validation authority.
"""
from __future__ import annotations

import argparse
import textwrap

from .hem_config import SETTINGS, EXPORT_RESPONSIBILITY
from .hem_common import DEFAULT_TIMEOUT

RESOURCES = {
    'model': 'Checkpoint files acquired or imported locally, with structural validation.',
    'pipeline': 'Packaged model implementation, compatibility rules and serving defaults.',
    'server': 'Saved device/listener settings and the managed native process.',
    'instance': 'A model loaded on a server, with its submitted serving configuration.',
}
ACTIONS = {
    'model': {
        'available': ('Discover downloadable recipes (not local inventory).', 'model available'),
        'download': ('Acquire external checkpoint files and validate them locally.', 'model download qwen35-08b-q5-k-xl --id qwen08'),
        'import': ('Copy existing GGUF fragments and tokenizer, or alias immutable files.', 'model import qwen08 --gguf weights.gguf --tokenizer tokenizer.json --alias'),
        'validate': ('Repeat structural compatibility validation; does not prove inference or memory fit.', 'model validate qwen08 --pipeline qwen-08b'),
        'list': ('List registered local models and asset accessibility.', 'model list'),
        'show': ('Inspect assets, validation and referencing instances.', 'model show qwen08'),
    },
    'pipeline': {
        'list': ('List packaged pipelines and their capabilities.', 'pipeline list'),
        'show': ('Inspect exact defaults, limits, capabilities and compatibility rules; no GPU or local model required.', 'pipeline show qwen-08b'),
    },
    'server': {
        'start': ('Start a new named server, or reuse saved settings. New processes start empty; prior instances are not reloaded.', 'server start --name local --device amdgpu --listen 127.0.0.1:8080\nloom-hem server start local'),
        'list': ('Observe saved servers, including stopped and unhealthy processes.', 'server list'),
        'show': ('Inspect saved device/listener settings; this is not a live health assertion.', 'server show local'),
        'status': ('Observe process ownership and HTTP health separately.', 'server status local'),
        'logs': ('Read managed process logs; following does not stop the service.', 'server logs local --follow'),
        'reconcile': ('Refresh observations and bookkeeping; does not restart the server.', 'server reconcile local'),
        'shutdown': ('Gracefully shut down an owned server, or abort it explicitly.', 'server shutdown local'),
    },
    'instance': {
        'start': ('Load a model and wait for readiness/admission on the selected server.', 'instance start qwen08 --name chat-small --server local'),
        'stop': ('Stop an instance gracefully and wait for retirement unless --no-wait is set.', 'instance stop chat-small --server local'),
        'list': ('Observe instances on the selected server, including external registrations.', 'instance list --server local'),
        'show': ('Inspect saved submission and native observation separately.', 'instance show chat-small --server local'),
        'status': ('Inspect readiness, admission and pending operations.', 'instance status chat-small --server local'),
        'logs': ('Read native instance logs; Ctrl-C stops following, not the service.', 'instance logs chat-small --server local --follow'),
        'export': ('Write a deployment document without launching a server.', 'instance export qwen08 --output registration.json'),
    },
}
ROOT_EXAMPLES = """Get started:
  loom-hem model available
  loom-hem model download qwen35-08b-q5-k-xl --id qwen08
  loom-hem server start --name local --device amdgpu --listen 127.0.0.1:8080
  loom-hem instance start qwen08 --name chat-small --server local
  loom-hem instance status chat-small --server local

This sequence requires an installed wheel, a supported AMD device and the
external download CLI required by the recipe. Downloads acquire external
checkpoint files. For existing assets: loom-hem model import --help
Chat requests use the server's HTTP endpoint.

Use loom-hem RESOURCE --help or loom-hem RESOURCE ACTION --help for details.
Bare root/resource commands show text guidance, even with --json.
"""


def configuration_reference():
    lines = ['Configuration:',
             '  --config accepts a flat JSON object. Flags override file values, which',
             '  override pipeline defaults. Accepted keys and numeric ranges:']
    for key, (default, maximum, _) in SETTINGS.items():
        suffix = 'when supported/enabled; otherwise not applicable' if key == 'mtp_depth' else ''
        lines.append(f'  {key}: integer 1..{maximum}; default {default} {suffix}'.rstrip())
    lines.extend([
        '  mtp: boolean; default is pipeline-dependent and requires MTP support.',
        '  Example file: {"rows": 4, "context_capacity": 8192}',
        '  Exact defaults/capabilities: loom-hem pipeline show PIPELINE',
        '  prefill_capacity must be <= context_capacity and >= mtp_depth + 1',
        '  (at least 1 when MTP is disabled). Explicit mtp_depth conflicts with',
        '  disabled MTP. Unsupported pipelines reject enabling MTP.',
        '  Duplicate/unknown keys, null, wrong types and nonfinite numbers fail.',
        '  Valid capacities do not guarantee device memory fit.',
    ])
    return '\n'.join(lines)


def describe(root, domains):
    """Attach contextual prose and examples to the configured parser tree."""
    root.description = 'Set up and control Loom Serve: loom-hem RESOURCE ACTION [options].'
    root.epilog = ROOT_EXAMPLES
    for resource, domain in domains.items():
        domain.description = RESOURCES[resource]
        examples = list(ACTIONS[resource].values())[:2]
        domain.epilog = ('Examples:\n  loom-hem ' + '\n  loom-hem '.join(value[1] for value in examples).replace('\nloom-hem', '\n  loom-hem')
                         + f'\n\nUse loom-hem {resource} ACTION --help for arguments and examples.')
        commands = next(a for a in domain._actions if isinstance(a, argparse._SubParsersAction))
        commands.required = False
        domain.set_defaults(_help_parser=domain)
        for command, child in commands.choices.items():
            description, example = ACTIONS[resource][command]
            child.description = textwrap.fill(description, width=78)
            child.epilog = 'Examples:\n  loom-hem ' + example.replace('\n', '\n  ')
            child.set_defaults(_help_parser=child)
            if resource == 'instance' and command in ('start', 'export'):
                child.epilog += '\n\n' + configuration_reference()
            if command == 'export':
                child.epilog += '\n\n' + textwrap.fill(EXPORT_RESPONSIBILITY, width=78)
            _arguments(child, resource, command)
        # argparse action summaries belong to the subparser choice entries.
        commands._choices_actions = [argparse._SubParsersAction._ChoicesPseudoAction(
            key, [], value[0]) for key, value in ACTIONS[resource].items()]
    root.set_defaults(_help_parser=root)


def _arguments(parser, resource, command):
    helps = {
        'json': 'Write one JSON action result to stdout; equivalents and diagnostics stay on stderr. Help remains text.',
        'model': 'Local model ID (letters/digits, dot, underscore or hyphen; starts with a letter/digit).',
        'instance': 'Instance name on the selected server, distinct from the local model ID.',
        'pipeline': ('Packaged pipeline ID to inspect (see pipeline list).' if resource == 'pipeline' else
                     "Packaged pipeline ID; omitted uses the recipe's declared pipeline (see model available)." if resource == 'model' and command == 'download' else
                     'Packaged pipeline ID; omitted selects the unique compatible pipeline (see pipeline list).'),
        'recipe': 'Download recipe ID from loom-hem model available.',
        'id': 'Required local model ID for this download; distinct from the recipe ID.',
        'offline': 'Use only files already present in the download CLI cache; missing cached assets fail.',
        'dry_run': 'Preview acquisition commands; acquire no files and create no model record.',
        'gguf': 'One or more matching GGUF fragment paths; copied by default.',
        'tokenizer': 'Matching tokenizer.json path; copied by default.',
        'alias': 'Reference immutable external files instead of copying; they must remain available.',
        'name': ('Required name for a new saved server; use positional SERVER for a saved start.' if resource == 'server' else
                 'Instance name; defaults to the local model ID.'),
        'device': 'Explicit native device specification for a new server; requires compatible installed runtime/hardware.',
        'listen': 'Required numeric loopback host and port for a new server, e.g. 127.0.0.1:8080; port 1..65535.',
        'server': ('Saved server name; start reuses saved settings.' if resource == 'server' else
                   'Configured server name; mutually exclusive with --url. Omission selects only when exactly one server is configured.'),
        'url': 'Native HTTP endpoint instead of --server; grants no ownership of the external process.',
        'timeout': f'Finite positive lifecycle budget in seconds (default: {DEFAULT_TIMEOUT:g}); expiry does not imply cancellation. Inspect status before retrying an accepted or unknown operation.',
        'abort': ('Abort owned server work instead of graceful shutdown.' if resource == 'server' else
                  'Abort instance work instead of graceful retirement.'),
        'no_wait': 'Return after stop acceptance; retirement is not yet confirmed. Inspect instance status.',
        'follow': 'Follow logs until Ctrl-C; ends observation without stopping the service.',
        'output': 'Required deployment JSON output path; does not start a server. Retain referenced dependencies through retirement.',
        'config': 'Flat JSON serving configuration path; flags override file, then pipeline defaults. See reference below.',
        'mtp': 'Enable multi-token prediction (propose tokens for verification), or disable it with --no-mtp; default depends on pipeline support.',
    }
    for key, (default, maximum, description) in SETTINGS.items():
        default_text = f'{default} when supported/enabled' if key == 'mtp_depth' else str(default)
        helps[key] = f'{description}; range 1..{maximum}, default {default_text}. JSON key: {key}.'
    groups = {name: parser.add_argument_group(name) for name in
              ('Targeting', 'Serving configuration', 'Lifecycle', 'Output')}
    for action in list(parser._actions):
        if action.dest == 'help':
            continue
        action.help = helps[action.dest]
        if not action.option_strings:
            continue
        group = ('Output' if action.dest in ('json', 'output', 'follow') else
                 'Lifecycle' if action.dest in ('timeout', 'abort', 'no_wait') else
                 'Serving configuration' if action.dest in (*SETTINGS, 'config', 'mtp', 'pipeline') else 'Targeting')
        parser._optionals._group_actions.remove(action)
        groups[group]._group_actions.append(action)
