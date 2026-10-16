"""Saved server definitions and bounded observations of owned native processes."""
from __future__ import annotations

import ipaddress
from pathlib import Path
import sys
import time

from . import hem_arguments as arguments, hem_display
from .hem_common import Deadline, HemError, InvalidInput, UncertainOutcome, result
from .hem_http import NativeClient
from .hem_process import ProcessManager, read_logs
from .hem_state import Registry
from .hem_shutdown import ShutdownManager


def configure(parser):
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("start", "list", "show", "status", "logs", "reconcile", "shutdown"):
        child = commands.add_parser(command)
        arguments.output(child)
        if command != "list":
            child.add_argument("server", nargs="?" if command == "start" else None,
                               type=arguments.name)
        if command == "start":
            child.add_argument("--name", type=arguments.name)
            child.add_argument("--device", default=None)
            child.add_argument("--listen", default=None)
        if command in ("start", "shutdown"):
            arguments.lifecycle(child)
        if command == "shutdown":
            child.add_argument("--abort", action="store_true")
        if command == "logs":
            child.add_argument("--follow", action="store_true")


def validate(args):
    if args.command == "start":
        if bool(args.server) == bool(args.name):
            raise InvalidInput("use server start NAME or server start --name NAME")
        if args.server and (args.device is not None or args.listen is not None):
            raise InvalidInput("saved server start reuses settings; use --name to create another definition")


def resolve_server(registry, explicit=None, url=None):
    """Resolve instance targeting once; an explicit URL never grants ownership."""
    if explicit is not None and url is not None:
        raise InvalidInput("select --server NAME or --url URL, not both")
    if url is not None:
        client = NativeClient(url)  # Validate without connecting or adopting.
        return dict(id=None, endpoint=client.endpoint, external=True)
    selected = arguments.select_server(explicit, [row['id'] for row in registry.list_servers()])
    return dict(registry.get_server(selected), external=False)


def _settings(args):
    # The native device and endpoint are explicit infrastructure choices. Do not
    # pick a GPU/backend or claim a port implicitly for a new saved definition.
    if not args.device or not args.device.strip() or not args.listen:
        raise InvalidInput("new server requires --device DEVICE and --listen LOOPBACK_HOST:PORT")
    try:
        host, port = args.listen.rsplit(':', 1)
        host = host.strip('[]')
        if not ipaddress.ip_address(host).is_loopback or not 0 < int(port) < 65536:
            raise ValueError()
    except (ValueError, AttributeError):
        raise InvalidInput("managed server listen must be a numeric loopback host and nonzero port") from None
    return dict(device=args.device, listen=args.listen)


def _log_paths(record):
    if not record.get('build'):
        return []
    directory = Path(record['build']['directory'])
    return [str(directory / name) for name in ('bootstrap-error.txt', 'process.log.1', 'process.log')]


class ObservationResult(dict):
    """An observation payload can remain useful when its health requires nonzero."""
    @property
    def exit_code(self):
        items = self.get('items', [self])
        if any(item['state'] == 'unknown' for item in items):
            return 4
        return 1 if any(item['state'] == 'unhealthy' for item in items) else 0


def _observed(state, **fields):
    return ObservationResult(result(state, **fields))


def _observation(registry, manager, server_id, deadline):
    """Reconcile local evidence, then report ownership and HTTP separately.

    No native mutation, cleanup or process signal occurs here. Unknown ownership
    retains evidence; a verified live process can have unhealthy/unreachable HTTP.
    """
    observed_at = time.time()
    record = registry.get_server(server_id)
    try:
        record = manager.reconcile(server_id, deadline)
    except HemError as error:
        # An inaccessible server lock or failed observation must not suppress
        # other servers in list. Retain its existing durable evidence.
        hem_display.emit(str(error))
        return _observed('unknown', completed=True, server=server_id, record=record,
                      observed_at=observed_at, process_health='unknown', http_health='unknown',
                      error=str(error), native=None)
    base = dict(server=server_id, record=record, observed_at=observed_at, native=None)
    if record['launch_token'] is None:
        return _observed('stopped', completed=True, process_health='exited', http_health='not-running', **base)
    try:
        with manager.pin(record) as pinned:
            try:
                native = manager.client(record, pinned, deadline).status()
            except HemError as error:
                # Socket/procfs association uncertainty is not mere ill health.
                if isinstance(error, UncertainOutcome):
                    raise
                pinned.verify(deadline)
                hem_display.emit(str(error))
                return _observed('unhealthy', completed=True, process_health='verified',
                              http_health='unhealthy', error=str(error), **base)
        base['native'] = native
        return _observed(record['phase'], completed=True, process_health='verified', http_health='healthy', **base)
    except HemError as error:
        hem_display.emit(str(error))
        return _observed('unknown', completed=True, process_health='unknown', http_health='unknown',
                      error=str(error), **base)


def _log_delta(snapshot, current):
    overlap = min(len(snapshot), len(current))
    while overlap and snapshot[-overlap:] != current[:overlap]:
        overlap -= 1
    return current[overlap:]


def _logs(registry, manager, record, follow, *, json_output=False):
    paths = _log_paths(record)
    if paths:
        hem_display.emit(hem_display.command(['tail', '-c', '65536', *(['-F'] if follow else []), *paths]))
    snapshot = read_logs(record)
    if follow:
        # Live diagnostics go to stderr, keeping --json stdout one final result.
        # No lifecycle lock spans this deliberate, caller-interruptible stream.
        if snapshot:
            sys.stderr.write(snapshot.decode('utf-8', errors='replace'))
            sys.stderr.flush()
        try:
            while record.get('process'):
                try:
                    with manager.pin(record) as pinned:
                        pinned.verify()
                except HemError:
                    break
                time.sleep(.1)
                current = read_logs(record)
                if current != snapshot:
                    # Preserve append/rotation continuity within the bounded
                    # combined tail. If no overlap survives, emit the new tail.
                    sys.stderr.write(_log_delta(snapshot, current).decode('utf-8', errors='replace'))
                    sys.stderr.flush()
                    snapshot = current
        except KeyboardInterrupt:
            pass
        final = read_logs(record)
        if not json_output:
            # Human mode has no duplicate final stdout snapshot: retain bytes
            # appended between the final observation and process exit/Ctrl-C.
            sys.stderr.write(_log_delta(snapshot, final).decode('utf-8', errors='replace'))
            sys.stderr.flush()
        snapshot = final
    return result('logs', completed=True, server=record['id'], paths=paths,
                  logs=snapshot.decode('utf-8', errors='replace'), followed=follow)


def run(args, deadline):
    settings = _settings(args) if args.command == 'start' and args.name else None
    budget = deadline or Deadline.after()
    registry = Registry(deadline=budget)
    manager = ProcessManager(registry)
    if args.command == 'shutdown':
        return ShutdownManager(registry, processes=manager).shutdown(args.server, budget, abort=args.abort)
    if args.command == 'start':
        server_id = args.name or args.server
        if args.name:
            with registry.lock('server', server_id, deadline=budget):
                if server_id in [row['id'] for row in registry.list_servers()]:
                    raise HemError('server name already exists; use server start NAME to reuse saved settings', accepted=False)
                registry.create_server(server_id, saved_settings=settings)
        else:
            settings = registry.get_server(server_id)['saved_settings']
        record = manager.start(server_id, settings, budget)
        return result('running', completed=True, server=server_id, accepted=True,
                      record=record, endpoint=record['endpoint'], process=record['process'],
                      log_paths=_log_paths(record))
    if args.command == 'show':
        return result('configured', completed=True, server=args.server,
                      saved_settings=registry.get_server(args.server)['saved_settings'])
    if args.command == 'list':
        # Give each independent observation a bounded budget; one unavailable
        # server must not consume or hide the remaining configured inventory.
        items = []
        for row in registry.list_servers():
            observation_budget = Deadline.after()
            scoped = Registry(registry.root, deadline=observation_budget)
            items.append(_observation(scoped, ProcessManager(scoped), row['id'], observation_budget))
        return ObservationResult(result('inventory', completed=True, items=items))
    if args.command in ('status', 'reconcile'):
        return _observation(registry, manager, args.server, budget)
    if args.command == 'logs':
        return _logs(registry, manager, registry.get_server(args.server), args.follow,
                     json_output=getattr(args, 'json', False))
    raise InvalidInput('unknown server command')
