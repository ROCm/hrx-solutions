"""Synchronous shared-server retirement, with only unresolved current markers.

One server lock spans enumeration through verified process exit. Native status
is authoritative; request acknowledgments never stand in for resource retirement.
"""
from __future__ import annotations

import json
from pathlib import Path
import signal
import time
import uuid

from .hem_common import DeadlineExpired, HemError, UncertainOutcome, result
from .hem_http import registration
from .hem_process import ProcessManager, _confirmed_exit


def _target_acceptance(targets):
    """Aggregate only unanimous acknowledgments of remaining native actions.

    A lost submission, an unsubmitted action, or mixed acknowledgments cannot
    establish acceptance of the remaining shutdown work as a whole.
    """
    stages = {item['stage'] for item in targets.values()}
    if stages == {'accepted'}:
        return True
    if stages == {'rejected'}:
        return False
    return None


class ShutdownManager:
    def __init__(self, registry, *, processes=None):
        self.registry = registry
        self.processes = processes or ProcessManager(registry)

    def _save(self, record, pending):
        return self.registry.update_server(record['id'],
                    {'phase': 'stopping', 'pending': pending},
                    expected_token=record['launch_token'], expected_revision=record['revision'])

    def _finish(self, record, deadline):
        # Keep the pre-clear launch snapshot so cleanup cannot accidentally
        # address another launch, and a failed cleanup remains explicitly retryable.
        while True:
            try:
                self.processes.cleanup(record, deadline)
                break
            except UncertainOutcome as error:
                # The log capture process can be between procfs teardown and
                # pidfd exit notification. Reobserve; never remove live or
                # unreadable artifacts, and preserve uncertainty at the bound.
                try:
                    time.sleep(min(.01, deadline.remaining()))
                except DeadlineExpired:
                    error.accepted = True
                    raise error
        self.registry.clear_server(record['id'], expected_token=record['launch_token'],
                                   expected_revision=record['revision'])
        self.processes.reap()
        return result('stopped', completed=True, server=record['id'], accepted=True)

    def shutdown(self, server_id, deadline, *, abort=False):
        previous = self.registry.deadline
        self.registry.deadline = deadline
        try:
            with self.registry.lock('server', server_id, deadline=deadline):
                return self._shutdown(server_id, deadline, abort)
        finally:
            self.registry.deadline = previous

    def _shutdown(self, server_id, deadline, abort):
        record = self.registry.get_server(server_id)
        if record['launch_token'] is None:
            return result('stopped', completed=True, server=server_id)
        if not record.get('process') or not record.get('build'):
            raise UncertainOutcome('shutdown requires established local process ownership; inspect/reconcile')
        if _confirmed_exit(record['process']):
            return self._finish(record, deadline)
        pending = record['pending'] or {}
        if pending and pending.get('action') != 'shutdown':
            raise UncertainOutcome('server launch is unresolved; inspect/reconcile before shutdown')
        with self.processes.pin(record) as pinned:
            # Verify before persisting shutdown or issuing any native action.
            pinned.verify(deadline)
            if pending.get('stage') == 'signal':
                if pending.get('accepted') is not True:
                    raise UncertainOutcome('termination submission is unresolved; inspect instead of replaying')
                return self._wait_exit(record, pinned, deadline)
            if not pending:
                pending = dict(action='shutdown', mode='abort' if abort else 'drain',
                               stage='retirement', targets={})
                record = self._save(record, pending)
            elif abort and pending['mode'] == 'drain':
                pending = dict(pending, mode='abort')
                record = self._save(record, pending)
            if pending['mode'] == 'abort':
                # Native process termination cancels registrations/admitted work.
                # This explicit escalation is never an implicit timeout fallback.
                return self._terminate(record, pinned, deadline)
            client = self.processes.client(record, pinned, deadline)
            while True:
                targets = dict(record['pending']['targets'])
                try:
                    document = client.status()
                except DeadlineExpired as error:
                    if any(item['stage'] == 'submitting' for item in targets.values()):
                        raise UncertainOutcome('unregister outcome remains unresolved at deadline') from error
                    error.accepted = _target_acceptance(targets)
                    raise
                except HemError as error:
                    raise UncertainOutcome(f'shutdown retirement observation failed: {error}',
                                           accepted=_target_acceptance(targets)) from error
                # Retired entries are deleted, rather than accumulating progress
                # history. A same-name replacement is a different native target.
                for name, item in list(targets.items()):
                    if registration(document, name, item['incarnation']) is None:
                        del targets[name]
                for item in document['data']:
                    name, incarnation = item['id'], item['loom']['incarnation']
                    if name not in targets:
                        stage = 'planned'
                        for row in self.registry.list_instances(server_id=server_id):
                            stopping = row['pending'] or {}
                            if (row['launch_token'] == record['launch_token'] and row['name'] == name
                                    and row['incarnation'] == incarnation and stopping.get('action') == 'stop'):
                                stage = {True: 'accepted', False: 'rejected', None: 'submitting'}[stopping.get('accepted')]
                        targets[name] = dict(incarnation=incarnation, stage=stage)
                pending = dict(record['pending'], targets=targets)
                record = self._save(record, pending)
                self._release_retired(record, document)
                if not targets:
                    return self._terminate(record, pinned, deadline)
                for name, item in list(targets.items()):
                    if item['stage'] == 'submitting':
                        raise UncertainOutcome('unregister outcome is unresolved; inspect instead of replaying')
                    if item['stage'] == 'accepted':
                        continue
                    path = self.registry.write_document(
                        Path('servers') / server_id / record['launch_token'] /
                        f'shutdown-{uuid.uuid4().hex}.json',
                        json.dumps({'model': name, 'mode': 'drain'}) + '\n')
                    targets[name] = dict(item, stage='submitting')
                    record = self._save(record, dict(record['pending'], targets=targets))
                    deadline.remaining()
                    try:
                        client.post('/v1/models/unregister', path)
                    except HemError as error:
                        if error.accepted is not None:
                            targets[name] = dict(item, stage='accepted' if error.accepted else 'rejected')
                            try:
                                record = self._save(record, dict(record['pending'], targets=targets))
                            except HemError:
                                pass
                        raise
                    targets[name] = dict(item, stage='accepted')
                    try:
                        record = self._save(record, dict(record['pending'], targets=targets))
                    except HemError as error:
                        error.accepted = True
                        raise
                try:
                    time.sleep(min(.01, deadline.remaining()))
                except DeadlineExpired as error:
                    error.accepted = True
                    raise

    def _release_retired(self, record, document):
        for row in self.registry.list_instances(server_id=record['id']):
            if (row['launch_token'] == record['launch_token'] and row['incarnation'] is not None
                    and registration(document, row['name'], row['incarnation']) is None):
                self.registry.retire_instance(row['server_id'], row['launch_token'], row['name'],
                                              expected_revision=row['revision'])

    def _terminate(self, record, pinned, deadline):
        deadline.remaining()
        record = self._save(record, dict(action='shutdown', mode=record['pending']['mode'],
                                       stage='signal', accepted=None))
        pinned.send(signal.SIGTERM, deadline)
        try:
            record = self._save(record, dict(record['pending'], accepted=True))
        except HemError as error:
            error.accepted = True
            raise
        return self._wait_exit(record, pinned, deadline)

    def _wait_exit(self, record, pinned, deadline):
        try:
            pinned.wait(deadline)
            return self._finish(record, deadline)
        except HemError as error:
            if error.accepted is None:
                error.accepted = True
            raise
