"""Bounded eager instance operations; no background continuation or replay.

Managed callers provide a resolved server record. Native process ownership comes
from ProcessManager, and every phase consumes the original invocation deadline.
The model lock precedes the server lock and protects qualification through durable
reservation of the model dependencies. Current rows survive all unresolved work.
"""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
import json
from pathlib import Path
import uuid

from . import hem_process
from ..models import hem_config, hem_models, recipes
from ..hem_common import DeadlineExpired, HemError, InvalidInput, UncertainOutcome, result
from .hem_http import NativeClient, active_ready, primary_parameters, registration, retired
from .hem_state import _name


class InstanceManager:
    def __init__(self, registry, *, processes=None):
        self.registry = registry
        self.processes = processes or hem_process.ProcessManager(registry)

    @contextmanager
    def _budget(self, deadline):
        # Registry transactions must consume this invocation's remaining budget
        # too, including direct API callers outside the CLI dispatcher.
        prior = self.registry.deadline
        self.registry.deadline = deadline
        try:
            yield
        finally:
            self.registry.deadline = prior

    def start(self, server, model_id, deadline, *, name=None, overrides=None,
              config_path=None, pipeline=None, device=0):
        with self._budget(deadline):
            return self._start(server, model_id, deadline, name=name, overrides=overrides,
                               config_path=config_path, pipeline=pipeline, device=device)

    def stop(self, server, name, deadline, *, abort=False, no_wait=False):
        with self._budget(deadline):
            return self._stop(server, name, deadline, abort=abort, no_wait=no_wait)

    def _server(self, selected, *, starting=False):
        if selected.get('external'):
            if starting:
                raise InvalidInput('managed registration requires a tracked local server with a matching build; select --server')
            return selected
        current = self.registry.get_server(selected['id'])
        if current['launch_token'] != selected['launch_token']:
            raise UncertainOutcome('server launch changed; inspect before retrying')
        if current['phase'] != 'running' or current['pending']:
            raise HemError('server is not accepting instance operations; inspect/reconcile first')
        return current

    def _update(self, row, **changes):
        return self.registry.update_instance(row['server_id'], row['launch_token'], row['name'],
                                            changes, expected_revision=row['revision'])

    def _retire(self, row):
        self.registry.retire_instance(row['server_id'], row['launch_token'], row['name'],
                                      expected_revision=row['revision'])

    def _rows(self, server, name):
        return [row for row in self.registry.list_instances(server_id=server['id'])
                if row['launch_token'] == server['launch_token'] and row['name'] == name]

    def _document(self, server, name, action, document):
        # Unique inspectable request files avoid overwriting an earlier operation
        # or a document referenced by an unresolved native request.
        relative = Path('servers') / server['id'] / server['launch_token'] / 'instances' / name
        return self.registry.write_document(relative / f'{action}-{uuid.uuid4().hex}.json',
                                           json.dumps(document, allow_nan=False) + '\n')

    def _retained_catalog(self, server, resolved, catalog_root):
        """Map a qualified catalog into the matching immutable launch snapshot."""
        try:
            package = self.processes.package_root.resolve(strict=True)
            current_root = (package / 'loom-pipeline').resolve(strict=True)
            current_root.relative_to(package)
            current = resolved.catalog.resolve(strict=True)
            relative = current.relative_to(current_root)
            if relative != resolved.catalog_identity.relative_path:
                raise ValueError('checked catalog location differs')
            launch = Path(server['build']['directory']).resolve(strict=True)
            retained_root = catalog_root.resolve(strict=True)
            retained_root.relative_to(launch)
            retained = (retained_root / relative).resolve(strict=True)
            retained.relative_to(retained_root)
            if not current.is_dir() or not retained.is_dir():
                raise ValueError('catalog must be a directory')
        except (OSError, ValueError) as error:
            raise HemError('validated pipeline is outside or missing from the matching package catalog') from error
        return retained

    def _start(self, server, model_id, deadline, *, name=None, overrides=None,
              config_path=None, pipeline=None, device=0):
        """Return only after ready+active+idle and acknowledged open admission.

        A failed/uncertain start is never replayed. Inspect the pending current
        record and native status; explicitly stop it or coordinate native recovery.
        """
        name = _name(name or model_id)
        if server.get('external'):
            self._server(server, starting=True)
        with self.registry.lock('model', model_id, deadline=deadline):
            deadline.remaining()
            files = hem_models.model_files(self.registry, model_id, pipeline=pipeline)
            deadline.remaining()
            resolved = hem_config.resolve(recipes.load(files.recipe_id), files, overrides, config_path, deadline=deadline)
            deadline.remaining()
            with self.registry.lock('server', server['id'], deadline=deadline):
                server = self._server(server, starting=True)
                if self._rows(server, name):
                    raise HemError('instance name is already tracked; inspect or stop before retrying')
                with self.processes.pin(server) as pinned:
                    client = self.processes.client(server, pinned, deadline)
                    initial = client.status()
                    if registration(initial, name) is not None:
                        raise HemError('instance name is already registered; inspect or stop before retrying')
                    catalog_root = hem_process.require_build(server, self.processes.package_root, deadline)
                    catalog = self._retained_catalog(server, resolved, catalog_root)
                    document = hem_config.deployment_document(resolved, name, catalog=catalog, device=device,
                                                              deadline=deadline)
                    path = self._document(server, name, 'register', document)
                    deadline.remaining()
                    row = self.registry.create_instance(server['id'], server['launch_token'], name,
                            model_id=model_id, settings=resolved.settings_document(), document=str(path),
                            dependencies=[str(p) for p in (*resolved.dependencies, catalog)],
                            pending={'action': 'start', 'stage': 'register', 'accepted': None})
                    row = self._start_native(server, row, client, deadline)
                    return result('running', completed=True, server=server['id'], instance=name,
                                  model=model_id, pipeline=resolved.pipeline, settings=row['settings'],
                                  incarnation=row['incarnation'], accepted=True)

    def _post(self, row, client, route, path, *, deadline):
        try:
            client.post(route, path)
        except HemError as error:
            if error.accepted is not None:
                try:
                    deadline.remaining()
                    row = self._update(row, pending=dict(row['pending'], accepted=error.accepted))
                except HemError:
                    pass  # Preserve the original external outcome and unresolved reservation.
            raise
        return row

    def _acknowledged(self, row, **changes):
        try:
            return self._update(row, **changes)
        except HemError as error:
            error.accepted = True
            raise

    def _start_native(self, server, row, client, deadline):
        name = row['name']
        self._post(row, client, '/v1/models/register', row['document'], deadline=deadline)
        row = self._acknowledged(row, pending={'action': 'start', 'stage': 'registration', 'accepted': True})
        def observed_registration(document):
            nonlocal row
            item = registration(document, name, row['incarnation'])
            if item is None:
                if row['incarnation'] is not None:
                    raise HemError('registered incarnation disappeared; inspect before retrying', accepted=True)
                return False
            if row['incarnation'] is None:
                row = self._update(row, incarnation=item['loom']['incarnation'])
            if item['loom'].get('error') is not None:
                raise HemError(f"native preparation failed: {item['loom']['error']}", accepted=True)
            # Native register is metadata-only: no preparation or parameters
            # exist until explicit activation. Observe the exact drained
            # incarnation here, then let activation drive preparation.
            return item['loom'].get('status', {}).get('admission') == 'drained'
        client.wait(observed_registration)
        activate = self._document(server, name, 'activate', {'model': name})
        row = self._update(row, pending={'action': 'start', 'stage': 'activate', 'accepted': None})
        deadline.remaining()
        self._post(row, client, '/v1/models/activate', activate, deadline=deadline)
        row = self._acknowledged(row, pending={'action': 'start', 'stage': 'activation', 'accepted': True})
        client.wait(lambda document: active_ready(document, name, row['incarnation']))
        undrain = self._document(server, name, 'undrain', {'model': name})
        row = self._update(row, pending={'action': 'start', 'stage': 'undrain', 'accepted': None})
        deadline.remaining()
        self._post(row, client, '/v1/models/undrain', undrain, deadline=deadline)
        row = self._acknowledged(row, pending={'action': 'start', 'stage': 'admission', 'accepted': True})
        def opened(document):
            item = registration(document, name, row['incarnation'])
            if item is None:
                raise HemError('registered incarnation disappeared after undrain', accepted=True)
            loom = item['loom']
            if loom.get('error') is not None:
                raise HemError(f"native instance error after undrain: {loom['error']}", accepted=True)
            return (loom.get('registration') == 'ready' and loom.get('parameter_operation') == 'idle'
                    and primary_parameters(loom).get('active') is True
                    and loom.get('status', {}).get('admission') == 'open')
        client.wait(opened)
        return self._acknowledged(row, pending=None)

    def _stop(self, server, name, deadline, *, abort=False, no_wait=False):
        """Retire the original incarnation, including already-undiscoverable work.

        A pending stop with known acceptance only observes on a subsequent stop;
        uncertain submission is never replayed. A replacement name is untouched.
        External URL control confers no process or local-artifact cleanup rights.
        """
        name = _name(name)
        external = bool(server.get('external'))
        lock = nullcontext() if external else self.registry.lock('server', server['id'], deadline=deadline)
        with lock:
            server = self._server(server)
            pin = nullcontext(None) if external else self.processes.pin(server)
            with pin as pinned:
                client = (NativeClient(server['endpoint'], deadline) if external else
                          self.processes.client(server, pinned, deadline))
                rows = [] if external else self._rows(server, name)
                row = rows[0] if rows else None
                pending = (row['pending'] or {}) if row else {}
                try:
                    document = client.status()
                except DeadlineExpired as error:
                    if pending.get('action') == 'stop':
                        if pending.get('accepted') is None:
                            raise UncertainOutcome('unregister outcome remains unresolved at deadline; inspect before retrying') from error
                        error.accepted = pending['accepted']
                    raise
                except HemError as error:
                    if pending.get('action') == 'stop':
                        raise UncertainOutcome(f'native retirement observation failed: {error}',
                                               accepted=pending.get('accepted')) from error
                    raise
                item = registration(document, name)
                original = row['incarnation'] if row else (item['loom']['incarnation'] if item else None)
                if original is None and row:
                    pending = row['pending'] or {}
                    if item:
                        original = item['loom']['incarnation']
                        row = self._update(row, incarnation=original)
                    elif pending.get('stage') == 'register' and pending.get('accepted') is None:
                        raise UncertainOutcome('registration outcome/incarnation is unresolved; inspect before stopping')
                    elif pending.get('stage') == 'register' and pending.get('accepted') is False:
                        self._retire(row)
                        return result('stopped', completed=True, server=server.get('id'), instance=name)
                if original is None:
                    raise HemError('instance is not registered; no tracked incarnation to stop')
                if retired(document, name, original):
                    if row:
                        self._retire(row)
                    return result('stopped', completed=True, server=server.get('id'), instance=name,
                                  incarnation=original)
                pending = (row['pending'] or {}) if row else {}
                resubmit_abort = (abort and pending.get('action') == 'stop'
                                  and pending.get('accepted') is True and pending.get('mode') == 'drain')
                if pending.get('action') == 'stop' and not resubmit_abort:
                    if pending.get('accepted') is not True:
                        raise UncertainOutcome('unregister outcome is unresolved; inspect instead of replaying')
                    if pending.get('mode') != ('abort' if abort else 'drain'):
                        raise HemError('pending stop has a different mode; inspect instead of replacing it', accepted=True)
                else:
                    mode = 'abort' if abort else 'drain'
                    if not external:
                        if row is None:
                            row = self.registry.create_instance(server['id'], server['launch_token'], name,
                                        incarnation=original, pending=None)
                        row = self._update(row, pending={'action': 'stop', 'stage': 'unregister',
                                                        'mode': mode, 'accepted': None})
                        path = self._document(server, name, 'unregister', {'model': name, 'mode': mode})
                    else:
                        # External control still retains the printed request in
                        # private state, without claiming a server/process record.
                        path = self.registry.write_document(Path('external') / f'unregister-{uuid.uuid4().hex}.json',
                                                json.dumps({'model': name, 'mode': mode}) + '\n')
                    deadline.remaining()
                    if row:
                        self._post(row, client, '/v1/models/unregister', path, deadline=deadline)
                    else:
                        client.post('/v1/models/unregister', path)
                    if row:
                        row = self._acknowledged(row, pending={'action': 'stop', 'stage': 'retirement',
                                                        'mode': mode, 'accepted': True})
                if no_wait:
                    return result('stopping', completed=False, server=server.get('id'), instance=name,
                                  incarnation=original, accepted=True)
                client.wait(lambda document: retired(document, name, original))
                if row:
                    self._retire(row)
                return result('stopped', completed=True, server=server.get('id'), instance=name,
                              incarnation=original, accepted=True)
