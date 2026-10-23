"""Public instance lifecycle, native observations and offline deployment export."""
from contextlib import contextmanager
from pathlib import Path
import json
import sys
import time

from . import hem_arguments as arguments, hem_config, hem_display, hem_models
from .hem_common import Deadline, HemError, InvalidInput, result
from .hem_http import NativeClient, registration
from .hem_instances import InstanceManager
from .hem_process import ProcessManager
from .hem_server_cli import ObservationResult, resolve_server
from .hem_state import Registry


def configure(parser):
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("start", "stop", "list", "show", "status", "logs", "export"):
        child = commands.add_parser(command)
        arguments.output(child)
        if command in ("start", "export"):
            child.add_argument("model", type=arguments.name)
            arguments.serving(child)
            child.add_argument("--pipeline", type=arguments.name)
        elif command != "list":
            child.add_argument("instance", type=arguments.name)
        if command != "export":
            arguments.selector(child)
        if command in ("start", "stop"):
            arguments.lifecycle(child)
        if command == "stop":
            child.add_argument("--abort", action="store_true")
            child.add_argument("--no-wait", action="store_true")
        if command == "logs":
            child.add_argument("--follow", action="store_true")
        if command == "export":
            child.add_argument("--output", type=Path, required=True)


class InstanceObservation(ObservationResult):
    @property
    def exit_code(self):
        # A failed server observation remains nonzero even for an empty list.
        if self.get('server_health') == 'unknown':
            return 4
        return max(super().exit_code, 1 if self.get('server_health') == 'unhealthy' else 0)


def _overrides(args):
    return {key: getattr(args, key) for key in (*hem_config.DEFAULTS, 'mtp', 'mtp_depth')
            if getattr(args, key) is not None}


def _rows(registry, server):
    if server['external']:
        return []
    return [row for row in registry.list_instances(server_id=server['id'])
            if row['launch_token'] == server['launch_token']]


def _association(row):
    # Saved submissions remain distinct from native observations and today's
    # pipeline defaults. An external registration has no invented association.
    return dict(model=row['model_id'] if row else None,
                settings=row['settings'] if row else None,
                document=row['document'] if row else None,
                dependencies=row['dependencies'] if row else None,
                pending=row['pending'] if row else None)


def _item(server, name, row, native, observed_at, *, state=None):
    if state is None:
        state = 'unhealthy' if native and native['loom'].get('error') is not None else 'observed'
    return result(state, completed=True, server=server['id'], endpoint=server['endpoint'],
                  instance=name, incarnation=native['loom']['incarnation'] if native else
                  (row['incarnation'] if row else None), native=native,
                  observed_at=observed_at, **_association(row))


def _observe(registry, manager, server, args, deadline):
    rows = _rows(registry, server)
    observed_at = time.time()
    if server['external']:
        try:
            native = NativeClient(server['endpoint'], deadline).status()
            health = 'observed'
        except HemError as error:
            hem_display.emit(str(error))
            native, health = None, 'unknown'
    else:
        observed = manager.observe_and_reconcile(server['id'], deadline)
        observed_at = observed.observed_at
        native = observed.native
        health = observed.state
        if observed.error is not None:
            hem_display.emit(observed.error)
        # Reconciliation may clear exited launch rows. Reload current records;
        # observations never turn an old incarnation into a replacement mapping.
        server = dict(observed.record, external=False)
        rows = _rows(registry, server)
    by_name = {row['name']: row for row in rows}
    items = []
    if native is not None:
        for item in native['data']:
            row = by_name.get(item['id'])
            tracked = None
            if row and row['incarnation'] != item['loom']['incarnation']:
                tracked = dict(incarnation=row['incarnation'], **_association(row))
                row = None
            observed_item = _item(server, item['id'], row, item, observed_at)
            if tracked is not None:
                observed_item['tracked_submission'] = tracked
            items.append(observed_item)
    # Pending/uncertain current rows remain inspectable even if no native entry
    # is discoverable. No cleanup or replay follows a missing observation.
    visible = {item['instance'] for item in items}
    for row in rows:
        if row['name'] not in visible:
            items.append(_item(server, row['name'], row, None, observed_at,
                               state=health if native is None else 'absent'))
    if args.command == 'list':
        return InstanceObservation(result('inventory', completed=True,
                server=server['id'], endpoint=server['endpoint'], items=items,
                observed_at=observed_at, server_health=health))
    matches = [item for item in items if item['instance'] == args.instance]
    if matches:
        answer = matches[0]
        if native is None:
            answer['state'] = 'unknown' if health == 'unknown' else 'unhealthy'
        return InstanceObservation(answer)
    if native is None and health in ('unknown', 'unhealthy'):
        return InstanceObservation(_item(server, args.instance, None, None, observed_at, state=health))
    raise HemError('instance is not registered and has no current tracked submission')


@contextmanager
def _client(manager, server, deadline):
    if server['external']:
        yield NativeClient(server['endpoint'], deadline)
    else:
        with manager.pin(server) as pinned:
            yield manager.client(server, pinned, deadline)


def _logs(manager, server, args):
    cursor, snapshot = None, ''
    try:
        while True:
            # Each bounded snapshot has its own budget; follow has no lifecycle
            # deadline or lock. Ownership is re-proved for each connection.
            with _client(manager, server, Deadline.after()) as client:
                snapshot = client.logs(model=args.instance, cursor=cursor).body.decode('utf-8', errors='replace')
            if not args.follow:
                break
            sys.stderr.write(snapshot)
            sys.stderr.flush()
            for line in snapshot.splitlines():
                try:
                    event = json.loads(line)
                except ValueError:
                    raise HemError('native model log snapshot contains invalid NDJSON') from None
                if isinstance(event, dict) and event.get('event') == 'cursor':
                    cursor = event.get('value')
            if cursor is None:
                raise HemError('native model logs lack a continuation cursor')
            time.sleep(.1)
    except KeyboardInterrupt:
        pass
    return result('logs', completed=True, server=server['id'], instance=args.instance,
                  logs=snapshot, followed=args.follow)


def run(args, deadline):
    budget = deadline or Deadline.after()
    registry = Registry(deadline=budget)
    if args.command == 'export':
        # No server selection, launch, reservation or export history. Validation
        # and atomic output are owned by the shared configuration adapter.
        with registry.lock('model', args.model, deadline=budget):
            files = hem_models.model_files(registry, args.model, pipeline=args.pipeline)
            hem_display.emit('Local asset checks (Qwen structural, curated presence); no native equivalent')
            answer = hem_config.export_deployment(files, args.output, args.name or args.model,
                        overrides=_overrides(args), config_path=args.config, pipeline=args.pipeline, deadline=budget)
        hem_display.emit('# Registration example: replace SERVER:PORT with your compatible server endpoint.')
        hem_display.emit(hem_display.http('POST', 'http://SERVER:PORT/v1/models/register',
                            body_path=Path(answer['output']), body=Path(answer['output']).read_bytes()))
        return dict(answer, model=args.model)
    server = resolve_server(registry, args.server, args.url)
    manager = ProcessManager(registry)
    if args.command == 'start':
        hem_display.emit('Local asset checks and bookkeeping (Qwen structural, curated presence); no native equivalent')
        return InstanceManager(registry, processes=manager).start(server, args.model, budget,
                    name=args.name, overrides=_overrides(args), config_path=args.config, pipeline=args.pipeline)
    if args.command == 'stop':
        return InstanceManager(registry, processes=manager).stop(server, args.instance, budget,
                                                               abort=args.abort, no_wait=args.no_wait)
    if args.command in ('list', 'show', 'status'):
        return _observe(registry, manager, server, args, budget)
    if args.command == 'logs':
        return _logs(manager, server, args)
    raise InvalidInput('unknown instance command')
