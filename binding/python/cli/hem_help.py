"""Help copy and presentation for the resource grammar.

This module only describes parsers. It never opens registries, inspects assets or
contacts services. Configuration numbers come from the validation authority.
"""
from __future__ import annotations

import argparse
import textwrap

from ..models.hem_config import EXPORT_RESPONSIBILITY
from .hem_arguments import SERVING_OPTIONS
from ..hem_common import DEFAULT_TIMEOUT

RESOURCES = {
    'openwebui': 'Owned Open WebUI chat connections and selected finite dictation.',
    'omp': 'Owned Oh My Pi configuration for local managed servers.',
    'opencode': 'Owned OpenCode 2 configuration for local managed servers.',
    'tui': 'Interactive, read-only terminal dashboard for Loom Serve.',
    'pi': 'Owned Pi coding-agent configuration for local managed servers.',
    'model': 'Checkpoint files acquired or imported locally; prepared-asset checks do not prove native readiness.',
    'modality': 'CHAT/ASR modality contracts; capabilities belong to a selected executable.',
    'executable': 'Packaged executable declarations, program contracts and serving capabilities.',
    'server': 'Saved device/listener settings and the managed native process.',
    'instance': 'A model loaded on a server, with its submitted serving configuration.',
}
ACTIONS = {
    'openwebui': {'configure': ('Configure Open WebUI chat or selected STT, inspect them, or disconnect unchanged owned entries.', 'openwebui configure --url http://127.0.0.1:3000\nloom-control openwebui configure --url http://127.0.0.1:3000 --stt-server local --stt-instance dictation\nloom-control openwebui configure --url http://127.0.0.1:3000 --check')},
    'omp': {'configure': ('Configure Oh My Pi, inspect configuration, or remove unchanged owned entries.', 'omp configure\nloom-control omp configure --check')},
    'opencode': {'configure': ('Configure OpenCode 2, inspect configuration, or remove unchanged owned files.', 'opencode configure\nloom-control opencode configure --check')},
    'tui': {},
    'pi': {'configure': ('Configure Pi, inspect configuration, or remove unchanged owned files.', 'pi configure\nloom-control pi configure --check')},
    'model': {
        'available': ('Discover downloadable recipes (not local inventory).', 'model available'),
        'download': ('Acquire a recipe\'s pinned assets; its executable owns preparation and serving capabilities.', 'model download RECIPE --id local-model'),
        'import': ('Select --recipe explicitly; copy existing GGUF/tokenizer or prepared --assets-dir files, or --alias immutable references.', 'model import local-model --recipe RECIPE --gguf weights.gguf --tokenizer tokenizer.json --alias'),
        'validate': ('Repeat associated recipe, executable and asset checks; does not prove native readiness.', 'model validate local-model'),
        'list': ('List registered local models and asset accessibility.', 'model list'),
        'show': ('Inspect assets, validation and referencing instances.', 'model show local-model'),
    },
    'modality': {
        'list': ('List CHAT/ASR modality contracts; no modality chooses a model or executable.', 'modality list'),
        'show': ('Inspect modality semantics and per-executable capabilities; limits are not universal modality defaults.', 'modality show chat\nloom-control modality show asr'),
    },
    'executable': {
        'list': ('List executable declarations independently of available recipes; optionally filter by modality.', 'executable list\nloom-control executable list --modality chat'),
        'show': ('Inspect one direct executable reference, its serving limits, asset contract and programs.', 'executable show llm/example/executable.json'),
    },
    'server': {
        'start': ('Start a new named server, or reuse saved settings. New processes start empty; prior instances are not reloaded.', 'server start --name local --device amdgpu --listen 127.0.0.1:8080\nloom-control server start local'),
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
  loom-control model available
  loom-control model download RECIPE --id local-model
  loom-control server start --name local --device amdgpu --listen 127.0.0.1:8080
  loom-control instance start local-model --name chat-small --server local
  loom-control instance status chat-small --server local

This sequence requires an installed wheel, a supported AMD device and the
external download CLI required by the recipe. Downloads acquire external
checkpoint files. For existing assets: loom-control model import --help
Chat requests use the server's HTTP endpoint.
Inspect modality semantics: loom-control modality show asr
Inspect concrete capabilities: loom-control executable list
Choose RECIPE from model available. Preparation prerequisites belong to the
selected executable; complete prepared --assets-dir imports do not run preparation.

Use loom-control RESOURCE --help or loom-control RESOURCE ACTION --help for details.
Bare root/resource commands show text guidance, even with --json.
"""


def configuration_reference():
    return "\n".join([
        'Configuration applies through the selected recipe and modality policy.',
        '  --config accepts a flat JSON object. Explicit flags override file values,',
        '  then recipe configuration, then selected executable defaults.',
        '  Inspect exact types, fixed/unsupported values, defaults and limits:',
        '    loom-control executable show CATALOG/executable.json',
        '  Duplicate/unknown keys, null, wrong types and nonfinite values fail.',
        '  Each supplied layer is checked; a later override cannot repair bad input.',
        '  Cross-setting rules apply to the final effective configuration.',
        '  A later disabled MTP removes inherited mtp_depth; depth explicitly',
        '  supplied with disabled MTP is rejected. Valid capacities do not prove memory fit.',
    ])


def describe(root, domains):
    """Attach contextual prose and examples to the configured parser tree."""
    root.description = 'Set up and control Loom Serve: loom-control RESOURCE ACTION [options].'
    root.epilog = ROOT_EXAMPLES
    for resource, domain in domains.items():
        domain.description = RESOURCES[resource]
        examples = list(ACTIONS[resource].values())[:2]
        domain.epilog = ('Examples:\n  loom-control ' + '\n  loom-control '.join(value[1] for value in examples).replace('\nloom-control', '\n  loom-control')
                         + f'\n\nUse loom-control {resource} ACTION --help for arguments and examples.')
        commands = next(a for a in domain._actions if isinstance(a, argparse._SubParsersAction))
        commands.required = False
        commands.metavar = 'ACTION'
        domain._positionals.title = 'Actions'
        domain.set_defaults(_help_parser=domain)
        for command, child in commands.choices.items():
            description, example = ACTIONS[resource][command]
            child.description = textwrap.fill(description, width=78)
            child.epilog = 'Examples:\n  loom-control ' + example.replace('\n', '\n  ')
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
        'check': 'Inspect owned configuration and report Pi compatibility separately without writing.',
        'disable': 'Remove unchanged owned integration files without requiring Pi or Node; preserve edited files.',
        'pi_dir': 'Alternate Pi agent directory; default uses Pi’s configured agent directory.',
        'json': 'Write one JSON action result to stdout; equivalents and diagnostics stay on stderr. Help remains text.',
        'model': 'Local model ID (letters/digits, dot, underscore or hyphen; starts with a letter/digit).',
        'instance': 'Instance name on the selected server, distinct from the local model ID.',
        'modality': ('CHAT/ASR semantic contract; never selects one executable or model.' if resource == 'modality' else
                     'Optional declared modality assertion or executable-list filter; never selects a model.'),
        'executable': ('Direct safe catalog-relative executable.json reference to inspect.' if resource == 'executable' else
                       'Optional exact executable reference assertion on the selected recipe/model; no automatic recipe selection.'),
        'recipe': 'Recipe ID from model available; required to select an executable for local import.',
        'id': 'Required local model ID for this download; distinct from the recipe ID.',
        'offline': 'Use only files already present in the download CLI cache; missing cached assets fail.',
        'dry_run': 'Preview declared destinations and cache presence; acquire no files and create no model record.',
        'gguf': 'One or more matching GGUF fragment paths; copied by default.',
        'assets_dir': 'Complete caller-prepared files for explicit --recipe; copy by default or --alias. No preparation or fetch.',
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
        'config': 'Flat JSON serving configuration; flags override file, recipe values and selected executable defaults. See reference below.',
        'mtp': 'Enable multi-token prediction (propose tokens for verification), or disable with --no-mtp; support/default come from the selected executable.',
    }
    for key in SERVING_OPTIONS:
        helps[key] = f'Explicit {key.replace("_", " ")}; accepted values depend on the selected executable. JSON key: {key}.'
    if resource == 'omp':
        helps.update(check='Inspect activation and runtime compatibility separately; no writes or client execution.',
                     disable='Remove unchanged owned files and native entries without OMP or Bun; preserve edits.',
                     omp_dir='Explicit default-profile OMP agent directory; conflicts with named profile selectors. Use printed launch recipe.')
    if resource == 'opencode':
        helps.update(check='Inspect configuration and binary compatibility separately; no writes or client execution.',
                     disable='Remove unchanged owned files without requiring OpenCode; preserve edited files.',
                     opencode_dir='Explicit OpenCode configuration directory; overrides OPENCODE_CONFIG_DIR. Launch with the printed environment setting.')
    if resource == 'openwebui':
        helps.update(check='Inspect version and connection configuration without writes or probing model availability.',
                     tts_server='Managed Pocket speech server; requires --tts-instance and a cache-capable WebUI deployment.',
                     tts_instance='Exact native speech instance; requires --tts-server. Uses fixed Alba playback.',
                     disable_tts='Disconnect unchanged owned Pocket routing and return read-aloud to browser speech; preserve chat/STT.',
                     disable='Disconnect unchanged owned STT and remove owned model override/connections; preserve edits and foreign prerequisites. Requires administrator authentication.',
                     stt_server='Managed STT server; requires --stt-instance. Uses the existing host-network backend connection topology.',
                     stt_instance='Unprefixed served recognition instance; requires --stt-server. Configure validates readiness; --check remains unprobed.',
                     url='WebUI management HTTP origin with numeric loopback address and explicit port; overrides OPENWEBUI_URL. OPENWEBUI_TOKEN supplies the administrator credential.',
                     timeout=f'Finite positive total management budget in seconds (default: {DEFAULT_TIMEOUT:g}); inspect with --check before retrying an uncertain update.')
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
                 'Serving configuration' if action.dest in (*SERVING_OPTIONS, 'config', 'mtp', 'executable', 'modality') else 'Targeting')
        parser._optionals._group_actions.remove(action)
        groups[group]._group_actions.append(action)
