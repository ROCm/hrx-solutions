"""Human presentations of existing results; no model, registry or HTTP reads."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import shlex
import shutil
import textwrap

from ..control.hem_http import primary_parameters


class Literal(str):
    """A complete value or labeled record that prose wrapping must preserve."""


def text(value):
    if value is None:
        return 'Unknown'
    if value is True:
        return 'Yes'
    if value is False:
        return 'No'
    return str(value)


def details(values):
    values = [(label, value) for label, value in values if value is not None]
    if not values:
        return ''
    padding = max(len(label) for label, _ in values)
    return Literal('\n'.join(f'{label:<{padding}}  {text(value)}' for label, value in values))


def table(columns, rows, width):
    if not rows:
        return ''
    rows = [[text(value) for value in row] for row in rows]
    sizes = [max(len(column), *(len(row[i]) for row in rows)) for i, column in enumerate(columns)]
    if sum(sizes) + 2 * (len(columns) - 1) > width:
        return '\n\n'.join(details(list(zip(columns, row))) for row in rows)
    return '\n'.join('  '.join(value.ljust(size) for value, size in zip(row, sizes)).rstrip()
                     for row in [columns, *rows])


def observed(value):
    if value is None:
        return 'Unknown'
    try:
        return datetime.fromtimestamp(value, timezone.utc).isoformat(timespec='seconds')
    except (TypeError, ValueError, OverflowError):
        return text(value)


def issue(value):
    """Render errors/pending fields as labels, never as a dictionary dump."""
    if isinstance(value, dict):
        return '; '.join(f'{key.replace("_", " ")}: {issue(item)}' for key, item in value.items())
    if isinstance(value, (tuple, list)):
        return '; '.join(issue(item) for item in value)
    return text(value)


def warnings(data):
    lines = []
    if data.get('error') is not None:
        lines.append('Error: ' + issue(data['error']))
    validation = data.get('validation') or {}
    if validation.get('valid') is False:
        lines.append('Validation failed: ' + issue(validation.get('evidence')))
    pending = data.get('pending')
    if pending:
        lines.append('Pending: ' + issue(pending))
    native = data.get('native') or {}
    loom = native.get('loom') or {}
    if loom.get('error') is not None:
        lines.append('Native error: ' + issue(loom['error']))
    tracked = data.get('tracked_submission')
    if tracked is not None:
        lines.append('Warning: tracked/native incarnation mismatch; observed instance has no proven model association.')
        lines.append('Tracked incarnation: ' + text(tracked.get('incarnation')))
        if tracked.get('pending'):
            lines.append('Tracked pending: ' + issue(tracked['pending']))
    return '\n'.join(lines)


def settings(data, width):
    return table(['Setting', 'Submitted value'],
                 [(key, 'Not applicable' if key == 'mtp_depth' and data.get('mtp') is False else data[key])
                  for key in data
                  if key in data], width)


def target(args, action, name):
    argv = ['loom-control', 'instance', action, name]
    if getattr(args, 'server', None):
        argv += ['--server', args.server]
    elif getattr(args, 'url', None):
        argv += ['--url', args.url]
    return shlex.join(argv)


def model_state(data):
    state = data.get('state')
    if state == 'ready' and data.get('check_kind') == 'acquisition-presence':
        return 'Acquisition complete (native unprobed)'
    return 'Acquisition complete (native unprobed)' if state == 'ready' else text(state)


def model(args, data, width):
    command = args.command
    if command == 'available':
        rows = [(item.get('id'), item.get('name'), item.get('pipeline')) for item in data.get('items', [])]
        return ['Download recipes (external checkpoints; not local inventory).',
                table(['Recipe', 'Name', 'Pipeline'], rows, width) if rows else 'No download recipes are packaged.',
                'For existing files: loom-control model import --help']
    if command == 'list':
        items = data.get('items', [])
        return [table(['Model', 'Asset state', 'Pipeline'],
                      [(item.get('model', item.get('id')), model_state(item),
                        item.get('pipeline') or item.get('intended_pipeline')) for item in items], width)
                if items else 'No local models are registered. Discover recipes: loom-control model available',
                *[f'{item.get("model", item.get("id"))}: {warnings(item)}' for item in items if warnings(item)]]
    identifier = data.get('model', data.get('model_id', getattr(args, 'model', None)))
    if data.get('state') == 'preview':
        return [f'Preview for {identifier}: no files acquired and no model record created.',
                details([('Recipe', data.get('recipe_id')), ('Pipeline', data.get('pipeline')),
                         ('Offline cache only', data.get('offline'))]),
                *[details([('Artifact', item.get('id')), ('Destination', item.get('logical_path')),
                           ('Cached', item.get('cached')), ('Cache path', item.get('cache_path')),
                           ('Ownership', item.get('ownership'))]) for item in data.get('artifacts', [])]]
    provenance = data.get('provenance') or {}
    lines = [f'Model {identifier}: {model_state(data)}.', warnings(data),
             details([('Pipeline', data.get('intended_pipeline', data.get('pipeline'))),
                      ('Recipe', provenance.get('recipe_id')), ('Declaration SHA256', provenance.get('recipe_identity')),
                      ('Directory', data.get('directory')), ('Check', 'Acquisition completion and file presence'), ('Native readiness', 'Unprobed')])]
    for artifact in data.get('artifacts', []):
        lines.append(details([('Artifact', artifact.get('id')), ('Role', artifact.get('role')),
                              ('Format', artifact.get('format')), ('Destination', artifact.get('logical_path')),
                              ('Path', artifact.get('path')), ('Ownership', artifact.get('ownership'))]))
    lines.extend(Literal('Source: ' + text(asset.get('source_path'))) for asset in data.get('assets', []) if asset.get('source_path') is not None)
    lines.append('Prepared asset checks do not prove native inference readiness or device memory fit.')
    if 'instances' in data:
        lines.append('Referencing instances: ' + str(len(data['instances'])))
    return lines


def pipeline(args, data, width):
    if args.command == 'list':
        items = data.get('items', [])
        return [table(['Pipeline', 'Modality', 'Operation'],
                      [(item.get('pipeline'), item.get('modality'),
                        item.get('protocol', {}).get('operation')) for item in items], width)
                if items else 'No pipelines are packaged.',
                'Inspect parameters and limits: loom-control pipeline show PIPELINE']
    rows = []
    for name, parameter in data.get('parameters', {}).items():
        state = parameter['state']
        value = parameter.get('value') if state == 'fixed' else parameter.get('default')
        constraints = parameter.get('constraints', {})
        restriction = (parameter.get('reason', '') if state == 'unsupported' else
                       'Fixed assertion' if state == 'fixed' else
                       ', '.join(f'{key}={json.dumps(value)}' for key, value in constraints.items() if value is not None))
        if parameter.get('active_when'):
            restriction += ' active when ' + json.dumps(parameter['active_when'], sort_keys=True)
        rows.append((name, state, parameter['type'],
                     'Absent' if state == 'unsupported' else json.dumps(value, sort_keys=True), restriction))
    protocol = data.get('protocol', {})
    return [details([('Pipeline', data.get('pipeline')), ('Modality', data.get('modality')),
                     ('Operation', protocol.get('operation')), ('Endpoint', protocol.get('endpoint')),
                     ('Request encoding', protocol.get('request_encoding')),
                     ('Raw log events', data.get('telemetry', {}).get('raw_events')), ('Catalog', data.get('catalog'))]),
            table(['Parameter', 'State', 'Type', 'Default / fixed value', 'Constraints'], rows, width),
            'Packaged support is not acquired-model or native readiness.',
            'Valid capacities do not guarantee device memory fit.']


def server_observation(data):
    record = data.get('record') or {}
    native = data.get('native') or {}
    lines = [warnings(data), warnings(record), details([
        ('Server', data.get('server', record.get('id'))), ('State', data.get('state')),
        ('Endpoint', data.get('endpoint', record.get('endpoint'))),
        ('Process health', text(data.get('process_health'))), ('HTTP health', text(data.get('http_health'))),
        ('Observed at', observed(data.get('observed_at')))])]
    if 'data' in native:
        lines.append('Native registrations: ' + str(len(native['data'])))
        resources = native.get('server') or {}
        lines.append(details([('Workers', resources.get('worker_count')),
                              ('Model capacity', resources.get('model_capacity')),
                              ('Connection capacity', resources.get('connection_capacity')),
                              ('Held requests', resources.get('held_requests')),
                              ('Observers', resources.get('observers'))]))
    return lines


def server(args, data, width):
    name = data.get('server', getattr(args, 'server', None))
    if args.command == 'list':
        items = data.get('items', [])
        if not items:
            return ['No servers are configured.', 'Create a server: loom-control server start --help']
        rows = [(item.get('server'), (item.get('record') or {}).get('endpoint', item.get('endpoint')),
                 item.get('state'), item.get('process_health'), item.get('http_health')) for item in items]
        return [table(['Server', 'Endpoint', 'State', 'Process', 'HTTP'], rows, width),
                *[f'{item.get("server")}: {warning}' for item in items
                  for warning in (warnings(item), warnings(item.get('record') or {})) if warning]]
    if args.command == 'show':
        saved = data.get('saved_settings') or {}
        return [f'Saved settings for server {name} (not a live health observation).',
                details([('Device', text(saved.get('device'))), ('Listen', text(saved.get('listen')))])]
    if args.command in ('status', 'reconcile'):
        return server_observation(data)
    if args.command == 'start':
        record = data.get('record') or {}
        return [f'Server {name}: {text(data.get("state"))}.', warnings(record),
                details([('Endpoint', data.get('endpoint')), ('Device', (record.get('saved_settings') or {}).get('device'))]),
                'New processes start without loaded instances.',
                'Inspect: ' + shlex.join(['loom-control', 'server', 'status', name or 'SERVER']),
                'Logs: ' + shlex.join(['loom-control', 'server', 'logs', name or 'SERVER'])]
    mode = 'abort' if getattr(args, 'abort', False) else 'graceful shutdown'
    if data.get('state') == 'stopped' and data.get('completed'):
        return [f'Server {name} is stopped ({mode}).']
    return [f'Server {name}: {text(data.get("state"))}; shutdown is not confirmed ({mode}).',
            warnings(data), 'Inspect: ' + shlex.join(['loom-control', 'server', 'status', name or 'SERVER'])]


def instance_observation(data, detailed, width):
    native = data.get('native') or {}
    loom = native.get('loom') or {}
    status, parameters = loom.get('status') or {}, primary_parameters(loom)
    lines = [warnings(data), details([
        ('Instance', data.get('instance')), ('State', data.get('state')),
        ('Model', text(data.get('model'))), ('Server', data.get('server')),
        ('Endpoint', data.get('endpoint')), ('Server health', data.get('server_health')),
        ('Observed at', observed(data.get('observed_at'))), ('Incarnation', text(data.get('incarnation'))),
        ('Registration', text(loom.get('registration'))), ('Parameters active', text(parameters.get('active'))),
        ('Parameter operation', text(loom.get('parameter_operation'))), ('Admission', text(status.get('admission')))])]
    if detailed:
        if data.get('settings') is not None:
            lines.append('Saved submission settings:\n' + settings(data['settings'], width))
        lines.append(details([('Deployment document', data.get('document'))]))
        lines.extend(Literal('Dependency: ' + text(path)) for path in data.get('dependencies') or [])
    return lines


def instance_state(data):
    state = text(data.get('state'))
    loom = (data.get('native') or {}).get('loom') or {}
    if state != 'observed' or not loom:
        return state
    return (text(loom.get('registration')) + '; admission ' + text((loom.get('status') or {}).get('admission'))
            + '; parameters active: ' + text(primary_parameters(loom).get('active')))


def instance(args, data, width):
    command = args.command
    name = data.get('instance', getattr(args, 'instance', None))
    if command == 'list':
        context = details([('Server', data.get('server')), ('Endpoint', data.get('endpoint')),
                           ('Server health', text(data.get('server_health'))),
                           ('Observed at', observed(data.get('observed_at')))])
        items = data.get('items', [])
        if not items:
            if data.get('server_health') in ('unknown', 'unhealthy'):
                return [context, 'Instance inventory unavailable: server health is ' + data['server_health'] + '.', warnings(data)]
            return [context, 'No instances were observed.', 'Load a model: loom-control instance start --help']
        return [context, table(['Instance', 'Tracked model', 'Observed state'],
                [(item.get('instance'), item.get('model'), instance_state(item)) for item in items], width),
                *[f'{item.get("instance")}: {warnings(item)}' for item in items if warnings(item)]]
    if command in ('show', 'status'):
        return instance_observation(data, command == 'show', width)
    if command == 'start':
        ready = data.get('state') == 'running' and data.get('completed') is True
        return [f'Instance {name} is ready to accept requests.' if ready else
                f'Instance {name}: {text(data.get("state"))}; readiness is not confirmed.', warnings(data),
                details([('Model', data.get('model')), ('Pipeline', data.get('pipeline')),
                         ('Server', data.get('server')), ('Endpoint', data.get('endpoint', getattr(args, 'url', None)))]),
                settings(data.get('settings') or {}, width), 'Inspect: ' + target(args, 'status', name or 'INSTANCE')]
    if command == 'stop':
        mode = 'abort' if getattr(args, 'abort', False) else 'graceful stop'
        destination = data.get('server') or getattr(args, 'server', None) or getattr(args, 'url', None)
        context = f' on {destination}' if destination else ''
        if data.get('state') == 'stopped' and data.get('completed'):
            return [f'Instance {name}{context} is retired ({mode}).']
        return [f'Stop requested for {name}{context}; retirement is not yet confirmed ({mode}).', warnings(data),
                'Inspect: ' + target(args, 'status', name or 'INSTANCE')]
    # Export retains source/dependency lifetime responsibility in adapter stderr.
    return [f'Deployment exported for instance {name}.', details([
        ('Output', data.get('output')), ('Model', data.get('model')), ('Pipeline', data.get('pipeline'))]),
        settings(data.get('settings') or {}, width),
        'Preserve all referenced model assets and pipeline programs/policies through retirement.',
        'Registration curl is printed on stderr. Replace SERVER:PORT with a compatible server endpoint.']


def pi(args, data, width):
    return [f'Pi integration: {text(data.get("state"))}.',
            details([(key.replace('_', ' ').capitalize(), issue(value))
                     for key, value in data.items() if key != 'state'])]


def opencode(args, data, width):
    return [f'OpenCode integration: {text(data.get("state"))}.',
            details([(key.replace('_', ' ').capitalize(), issue(value))
                     for key, value in data.items() if key != 'state'])]


def omp(args, data, width):
    return ['OMP integration: ' + text(data.get('state')) + '.',
            details([(key.replace('_', ' ').capitalize(), issue(value)) for key,value in data.items() if key != 'state'])]


def openwebui(args, data, width):
    return ['Open WebUI integration: ' + text(data.get('state')) + '.',
            details([(key.replace('_', ' ').capitalize(), issue(value))
                     for key, value in data.items() if key not in ('state', 'guidance')]),
            data.get('guidance')]


RENDERERS = {'openwebui': openwebui, 'omp': omp, 'opencode': opencode, 'model': model, 'pipeline': pipeline, 'server': server, 'instance': instance, 'pi': pi}


def render(args, data, *, width=None):
    if args.command == 'logs':
        return '' if data.get('followed') else data.get('logs', '')
    width = width or shutil.get_terminal_size(fallback=(80, 24)).columns
    sections = RENDERERS[args.resource](args, data, width)
    return '\n\n'.join(textwrap.fill(section, width=width, break_long_words=False,
                                    break_on_hyphens=False)
                       if not isinstance(section, Literal) and '\n' not in section
                       and 'loom-control ' not in section and len(section) > width else section
                       for section in sections if section)
