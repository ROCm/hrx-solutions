"""Eager lifecycle protocol and minimal inventory tests, using real loopback HTTP.

Synthetic payloads prove orchestration, never model/GPU compatibility. The small
process fixture additionally exercises real pin/accepted-socket ownership hooks.
"""
from dataclasses import replace
from contextlib import contextmanager, redirect_stderr
import io
import json
import os
from pathlib import Path
import shutil
import socket
import signal
import subprocess
import tempfile
import time
import unittest
from unittest.mock import patch

from binding.python import hem_instances, hem_process, hem_state
from binding.python.models import hem_config, hem_models, pipeline_validation
from binding.python.hem_common import Deadline, DeadlineExpired, HemError, InvalidInput, UncertainOutcome
from binding.python.hem_http import NativeClient
from tests.hem_lifecycle_fixture import fixture, inventory, model, reply
from tests.hem_lifecycle_fixture import INSTANCES_NATIVE as NATIVE, port, wait_exit


class Processes:
    def __init__(self, root):
        self.package_root = root
        self.deadlines = []
        self.pins = 0

    @contextmanager
    def pin(self, record):
        self.pins += 1
        try:
            yield self
        finally:
            self.pins -= 1

    def client(self, record, pinned, deadline):
        assert pinned is self and self.pins == 1
        self.deadlines.append(deadline)
        return NativeClient(record['endpoint'], deadline, emit=lambda _: None)


class LifecycleTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.registry = hem_state.Registry(self.root / 'state')
        self.processes = Processes(Path(hem_config.pipeline_info('qwen-08b')['catalog']).parents[3])
        self.manager = hem_instances.InstanceManager(self.registry, processes=self.processes)
        self.select = patch.object(hem_config, 'select_pipeline', return_value='qwen-08b')
        self.select.start()
        self.addCleanup(self.select.stop)
        output = redirect_stderr(io.StringIO())
        output.__enter__()
        self.addCleanup(output.__exit__, None, None, None)
        gguf, tokenizer = self.root / 'tiny.gguf', self.root / 'tokenizer.json'
        gguf.write_bytes(b'fixture')
        tokenizer.write_text('{}')
        hem_models.import_model(self.registry, 'tiny', ggufs=[gguf], tokenizer=tokenizer)
        self.build = patch.object(hem_process, 'require_build', side_effect=lambda server, package, deadline: Path(server['build']['directory']) / 'loom-pipeline')
        self.build.start()
        self.addCleanup(self.build.stop)
        self.items = []
        self.calls = []
        self.mode = None
        self.fail = None
        self.incarnation = 17

    def respond(self, handler, requests):
        method, path, body, _ = requests[-1]
        self.calls.append((method, path))
        if method == 'GET':
            if path != '/v1/status':
                raise AssertionError(path)
            reply(handler, inventory(*self.items))
            return
        if self.fail == path:
            handler.connection.shutdown(socket.SHUT_RDWR)
            handler.connection.close()
            return
        payload = json.loads(body)
        if path.endswith('/register'):
            self.assertEqual(payload['admission'], 'drained')
            self.items = [model(self.incarnation, registration='registered')]
            del self.items[0]['loom']['parameters']
        elif path.endswith('/activate'):
            self.assertEqual(self.items[0]['loom']['registration'], 'registered')
            self.assertNotIn('parameters', self.items[0]['loom'])
            self.items[0]['loom']['registration'] = 'ready'
            self.items[0]['loom']['parameters'] = {'primary': {'active': True}}
            if self.mode == 'activation-error':
                self.items[0]['loom']['error'] = 'fixture activation failed'
            elif self.mode == 'activation-pending':
                self.items[0]['loom']['parameter_operation'] = 'updating'
        elif path.endswith('/undrain'):
            self.assertTrue(self.items[0]['loom']['parameters']['primary']['active'])
            self.assertEqual(self.items[0]['loom']['parameter_operation'], 'idle')
            self.items[0]['loom']['status']['admission'] = 'open'
        elif path.endswith('/unregister'):
            self.assertEqual(payload['model'], 'chat')
            if self.mode != 'retirement-pending':
                self.items = []
        else:
            raise AssertionError(path)
        reply(handler, {'accepted': True}, 202)

    def server(self, endpoint):
        self.registry.create_server('server', saved_settings={})
        row = self.registry.reserve_launch('server', launch_token='launch', current_settings={}, endpoint=endpoint)
        directory = self.root / 'state/servers/server/launch'
        directory.mkdir(parents=True)
        shutil.copytree(self.processes.package_root / 'loom-pipeline', directory / 'loom-pipeline')
        return self.registry.update_server('server', {'phase': 'running', 'pending': None,
                    'build': {'directory': str(directory)}},
                    expected_token='launch', expected_revision=row['revision'])

    def start(self, server, deadline=None, **kwargs):
        return self.manager.start(server, 'tiny', deadline or Deadline.after(), name='chat', **kwargs)

    def row(self):
        return self.registry.get_instance('server', 'launch', 'chat')

    def mutations(self):
        return [path for method, path in self.calls if method == 'POST']

    def test_full_eager_order_retention_and_stop(self):
        with fixture(self.respond) as (endpoint, requests):
            server = self.server(endpoint)
            deadline = Deadline.after()
            answer = self.start(server, deadline)
            self.assertEqual(answer['state'], 'running')
            self.assertTrue(answer['completed'])
            self.assertEqual(self.mutations(), ['/v1/models/register', '/v1/models/activate', '/v1/models/undrain'])
            self.assertIs(self.processes.deadlines[0], deadline)
            row = self.row()
            self.assertIsNone(row['pending'])
            self.assertEqual(row['incarnation'], 17)
            document = json.loads(Path(row['document']).read_text())
            self.assertTrue(Path(document['base_directory']).is_relative_to(Path(server['build']['directory'])))
            self.assertTrue((Path(document['base_directory']) / hem_config.catalog_modules(Path(document['base_directory'])).control).is_file())
            self.assertEqual(self.registry.references('tiny'), [row])
            stopped = self.manager.stop(server, 'chat', Deadline.after())
            self.assertTrue(stopped['completed'])
            self.assertEqual(self.registry.list_instances(), [])
            self.assertTrue(Path(row['document']).is_file()) # server launch owns cleanup
            self.assertTrue(Path(self.registry.get_model('tiny')['directory']).is_dir())
        self.assertEqual(self.processes.pins, 0)

    def test_metadata_only_registration_activates_before_waiting_for_ready(self):
        with fixture(self.respond) as (endpoint, requests):
            answer = self.start(self.server(endpoint), Deadline.after(.25))
            self.assertTrue(answer['completed'])
            self.assertEqual(self.mutations(), ['/v1/models/register', '/v1/models/activate',
                                                '/v1/models/undrain'])
            self.assertEqual(self.row()['incarnation'], 17)
            self.assertIsNone(self.row()['pending'])

    def test_accepted_stop_retry_failed_initial_status_is_uncertain_without_replay(self):
        self.mode = 'retirement-pending'
        unavailable = False
        original = self.respond
        def respond(handler, requests):
            if unavailable and requests[-1][1] == '/v1/status':
                self.calls.append(('GET', '/v1/status'))
                reply(handler, {'error': 'unavailable'}, 503)
            else:
                original(handler, requests)
        with fixture(respond) as (endpoint, _):
            server = self.server(endpoint)
            self.start(server)
            self.manager.stop(server, 'chat', Deadline.after(), no_wait=True)
            before = list(self.mutations())
            saved = self.row()
            unavailable = True
            with self.assertRaises(UncertainOutcome) as error:
                self.manager.stop(server, 'chat', Deadline.after())
            self.assertIs(error.exception.accepted, True)
            self.assertEqual(error.exception.exit_code, 4)
            self.assertEqual(self.mutations(), before)
            self.assertEqual(self.row(), saved)

    def test_accepted_stop_retry_initial_deadline_retains_known_acceptance(self):
        self.mode = 'retirement-pending'
        with fixture(self.respond) as (endpoint, _):
            server = self.server(endpoint)
            self.start(server)
            self.manager.stop(server, 'chat', Deadline.after(), no_wait=True)
            before = list(self.mutations())
            with patch.object(NativeClient, 'status', side_effect=DeadlineExpired('fixture deadline')):
                with self.assertRaises(DeadlineExpired) as error:
                    self.manager.stop(server, 'chat', Deadline.after())
            self.assertEqual(error.exception.exit_code, 3)
            self.assertIs(error.exception.accepted, True)
            self.assertEqual(self.mutations(), before)
            self.assertEqual(self.row()['pending']['stage'], 'retirement')

    def test_unresolved_stop_retry_deadline_preserves_uncertainty(self):
        with fixture(self.respond) as (endpoint, _):
            server = self.server(endpoint)
            self.start(server)
            self.fail = '/v1/models/unregister'
            with self.assertRaises(UncertainOutcome):
                self.manager.stop(server, 'chat', Deadline.after())
            before = list(self.mutations())
            with patch.object(NativeClient, 'status', side_effect=DeadlineExpired('fixture deadline')):
                with self.assertRaises(UncertainOutcome) as error:
                    self.manager.stop(server, 'chat', Deadline.after())
            self.assertEqual(error.exception.exit_code, 4)
            self.assertIsNone(error.exception.accepted)
            self.assertEqual(self.mutations(), before)
            self.assertIsNone(self.row()['pending']['accepted'])

    def test_config_override_and_snapshot(self):
        config = self.root / 'config.json'
        config.write_text('{"rows":2,"context_capacity":4096}')
        with fixture(self.respond) as (endpoint, _):
            answer = self.start(self.server(endpoint), overrides={'rows': 3}, config_path=config)
            self.assertEqual(answer['settings']['rows'], 3)
            self.assertEqual(answer['settings']['context_capacity'], 4096)
            self.assertFalse(answer['settings']['mtp'])
            self.assertIsNone(answer['settings']['mtp_depth'])
            self.assertEqual(self.row()['settings'], answer['settings'])

    def test_duplicate_local_and_native_names_never_register(self):
        with fixture(self.respond) as (endpoint, _):
            server = self.server(endpoint)
            self.items = [model()]
            with self.assertRaisesRegex(HemError, 'already registered'):
                self.start(server)
            self.assertEqual(self.mutations(), [])
            self.items = []
            self.start(server)
            count = len(self.mutations())
            with self.assertRaisesRegex(HemError, 'already tracked'):
                self.start(server)
            self.assertEqual(len(self.mutations()), count)

    def test_shared_catalog_aliases_and_legacy_paths_survive_retirement(self):
        items = {}
        def respond(handler, requests):
            method, path, body, _ = requests[-1]
            if method == 'GET':
                reply(handler, inventory(*items.values()))
                return
            payload = json.loads(body)
            name = payload.get('name', payload.get('model'))
            if path.endswith('/register'):
                items[name] = dict(model(len(items) + 20), id=name)
            elif path.endswith('/activate'):
                items[name]['loom']['parameters']['active'] = True
            elif path.endswith('/undrain'):
                items[name]['loom']['status']['admission'] = 'open'
            elif path.endswith('/unregister'):
                del items[name]
            reply(handler, {'accepted': True}, 202)
        with fixture(respond) as (endpoint, _):
            server = self.server(endpoint)
            for name in ('one', 'two'):
                self.manager.start(server, 'tiny', Deadline.after(), name=name)
            rows = self.registry.list_instances()
            documents = [json.loads(Path(row['document']).read_text()) for row in rows]
            shared = Path(documents[0]['base_directory'])
            self.assertEqual(documents[1]['base_directory'], str(shared))
            self.assertTrue(all(str(shared) in row['dependencies'] for row in rows))
            self.assertFalse(list((Path(server['build']['directory']) / 'instances').glob('*/sources-*')))
            self.assertNotEqual(rows[0]['document'], rows[1]['document'])
            # Persisted legacy catalogs and role locators remain valid until
            # explicit retirement; there is no filesystem migration/sweep.
            legacy = Path(server['build']['directory']) / 'instances/legacy/sources-old'
            shutil.copytree(shared, legacy)
            document = dict(documents[0], name='legacy', base_directory=str(legacy))
            path = self.manager._document(server, 'legacy', 'register', document)
            self.registry.create_instance(server['id'], server['launch_token'], 'legacy',
                model_id='tiny', settings=rows[0]['settings'], document=str(path),
                dependencies=[str(legacy) if item == str(shared) else item for item in rows[0]['dependencies']],
                incarnation=33, pending=None)
            items['legacy'] = dict(model(33), id='legacy')
            self.manager.stop(server, 'legacy', Deadline.after())
            self.manager.stop(server, 'one', Deadline.after())
            self.assertTrue(legacy.is_dir())
            self.assertTrue(shared.is_dir())
            self.assertEqual(list(items), ['two'])
            self.assertEqual([row['name'] for row in self.registry.list_instances()], ['two'])
            self.manager.stop(server, 'two', Deadline.after())
            self.assertTrue(shared.is_dir())

    def test_catalog_mapping_rejects_missing_or_escaping_paths_before_registration(self):
        with fixture(self.respond) as (endpoint, _):
            server = self.server(endpoint)
            resolved = hem_config.resolve(hem_models.model_files(self.registry, 'tiny'))
            retained_root = Path(server['build']['directory']) / 'loom-pipeline'
            retained = retained_root / resolved.catalog.relative_to(self.processes.package_root / 'loom-pipeline')
            original = retained.with_name('saved-catalog')
            retained.rename(original)
            cases = [replace(resolved, catalog=self.root), resolved]
            for candidate in cases:
                with self.subTest(catalog=candidate.catalog), patch.object(hem_config, 'resolve', return_value=candidate):
                    with self.assertRaisesRegex(HemError, 'matching package catalog'):
                        self.start(server)
            retained.symlink_to(self.root)
            with self.assertRaisesRegex(HemError, 'matching package catalog'):
                self.start(server)
            escaped = self.root / 'catalog-link'
            escaped.symlink_to(self.root)
            with patch.object(hem_config, 'resolve', return_value=replace(resolved, catalog=escaped)):
                with self.assertRaisesRegex(HemError, 'matching package catalog'):
                    self.start(server)
            self.assertEqual(self.registry.list_instances(), [])
            self.assertEqual(self.mutations(), [])
            self.assertFalse(list(self.registry.root.rglob('register-*.json')))

    def test_checked_interpretation_is_selected_once_then_used_for_registration(self):
        with fixture(self.respond) as (endpoint, requests):
            server = self.server(endpoint)
            deadline = Deadline.after()
            checked = []
            original_resolve = hem_config.resolve
            original_emit = hem_config.deployment_document
            def resolve(*args, **kwargs):
                self.assertIs(kwargs['deadline'], deadline)
                answer = original_resolve(*args, **kwargs)
                checked.append(answer)
                return answer
            def emit(value, *args, **kwargs):
                self.assertIs(value, checked[0])
                self.assertIs(kwargs['deadline'], deadline)
                return original_emit(value, *args, **kwargs)
            with patch.object(hem_config, 'resolve', side_effect=resolve), \
                 patch.object(hem_config, 'deployment_document', side_effect=emit), \
                 patch.object(hem_config, 'catalog_modules', wraps=hem_config.catalog_modules) as modules:
                self.start(server, deadline, overrides={'rows': 2, 'context_capacity': 4096})
            self.assertEqual(len(checked), 1)
            modules.assert_called_once()
            row = self.row()
            sent = [json.loads(body) for method, path, body, _ in requests
                    if method == 'POST' and path.endswith('/register')][0]
            self.assertEqual(sent['definition'], checked[0].definition_document())
            self.assertEqual(row['settings'], checked[0].settings_document())
            self.assertEqual(row['dependencies'], [str(path) for path in (
                *checked[0].dependencies, Path(sent['base_directory']))])

    def test_retained_provider_drift_is_refused_before_reservation_or_registration(self):
        with fixture(self.respond) as (endpoint, _):
            server = self.server(endpoint)
            retained_root = Path(server['build']['directory']) / 'loom-pipeline'
            selected = pipeline_validation.resolve_catalog('qwen-08b')
            retained = retained_root / selected.relative_to(self.processes.package_root / 'loom-pipeline')
            modules = pipeline_validation.catalog_modules(retained)
            provider = retained / modules.providers[-1]
            provider.write_bytes(provider.read_bytes() + b'\n')
            with self.assertRaisesRegex(InvalidInput, 'checked serving interpretation'):
                self.start(server)
            self.assertEqual(self.registry.list_instances(), [])
            self.assertEqual(self.mutations(), [])
            self.assertFalse(list(self.registry.root.rglob('register-*.json')))

    def test_activation_failure_never_undrains_retains_dependencies(self):
        self.mode = 'activation-error'
        with fixture(self.respond) as (endpoint, _):
            with self.assertRaisesRegex(HemError, 'activation failed') as error:
                self.start(self.server(endpoint))
            self.assertIs(error.exception.accepted, True)
            self.assertEqual(self.mutations(), ['/v1/models/register', '/v1/models/activate'])
            self.assertEqual(self.row()['pending']['stage'], 'activation')
            self.assertTrue(self.row()['dependencies'])

    def test_activation_timeout_known_pending_no_late_undrain(self):
        self.mode = 'activation-pending'
        with fixture(self.respond) as (endpoint, _):
            with self.assertRaises(DeadlineExpired) as error:
                self.start(self.server(endpoint), Deadline.after(.2))
            self.assertIs(error.exception.accepted, True)
            before = list(self.mutations())
            time.sleep(.05)
            self.assertEqual(self.mutations(), before)
            self.assertNotIn('/v1/models/undrain', before)
            self.assertTrue(self.row()['dependencies'])

    def test_lost_register_response_uncertain_never_replayed(self):
        self.fail = '/v1/models/register'
        with fixture(self.respond) as (endpoint, _):
            server = self.server(endpoint)
            with self.assertRaises(UncertainOutcome) as error:
                self.start(server)
            self.assertIsNone(error.exception.accepted)
            self.assertEqual(self.row()['pending'], {'action':'start','stage':'register','accepted':None})
            with self.assertRaisesRegex(HemError, 'already tracked'):
                self.start(server)
            self.assertEqual(self.mutations(), ['/v1/models/register'])
            with self.assertRaises(UncertainOutcome):
                self.manager.stop(server, 'chat', Deadline.after())

    def test_lost_undrain_response_reports_uncertainty(self):
        self.fail = '/v1/models/undrain'
        with fixture(self.respond) as (endpoint, _):
            with self.assertRaises(UncertainOutcome):
                self.start(self.server(endpoint))
            self.assertEqual(self.row()['pending']['stage'], 'undrain')
            self.assertIsNone(self.row()['pending']['accepted'])
            self.assertEqual(self.mutations().count('/v1/models/undrain'), 1)

    def test_one_deadline_includes_validation_and_no_native_after_expiry(self):
        with fixture(self.respond) as (endpoint, _):
            server = self.server(endpoint)
            original = hem_models.model_files
            def slow(*args, **kwargs):
                result = original(*args, **kwargs)
                time.sleep(.06)
                return result
            with patch.object(hem_models, 'model_files', side_effect=slow):
                with self.assertRaises(DeadlineExpired):
                    self.start(server, Deadline.after(.03))
            self.assertEqual(self.calls, [])
            self.assertEqual(self.registry.list_instances(), [])

    def test_build_mismatch_never_registers(self):
        with fixture(self.respond) as (endpoint, _):
            server = self.server(endpoint)
            with patch.object(hem_process, 'require_build', side_effect=HemError('build mismatch')):
                with self.assertRaisesRegex(HemError, 'build mismatch'):
                    self.start(server)
            self.assertEqual(self.mutations(), [])
            self.assertEqual(self.registry.list_instances(), [])

    def test_stop_does_not_use_discovery_and_preserves_other_instance(self):
        with fixture(self.respond) as (endpoint, _):
            server = self.server(endpoint)
            self.start(server)
            self.items.append(dict(id='other', loom=model(22)['loom']))
            original_respond = self.respond
            def respond(handler, requests):
                if requests[-1][1] == '/v1/models/unregister':
                    self.calls.append(('POST', '/v1/models/unregister'))
                    self.items = [item for item in self.items if item['id'] != 'chat']
                    reply(handler, {}, 202)
                else:
                    original_respond(handler, requests)
            with fixture(respond) as (other_endpoint, _):
                current = self.registry.get_server('server')
                server = self.registry.update_server('server', {'endpoint': other_endpoint},
                              expected_token='launch', expected_revision=current['revision'])
                self.manager.stop(server, 'chat', Deadline.after())
            self.assertEqual([item['id'] for item in self.items], ['other'])
            self.assertNotIn(('GET', '/v1/models'), self.calls)

    def test_original_incarnation_retired_replacement_untouched(self):
        with fixture(self.respond) as (endpoint, _):
            server = self.server(endpoint)
            self.start(server)
            self.items = [model(18)]
            previous = self.mutations()
            answer = self.manager.stop(server, 'chat', Deadline.after())
            self.assertTrue(answer['completed'])
            self.assertEqual(answer['incarnation'], 17)
            self.assertEqual(self.mutations(), previous)
            self.assertEqual(self.registry.list_instances(), [])
            self.assertEqual(self.items[0]['loom']['incarnation'], 18)

    def test_stop_no_wait_native_continues_then_explicit_stop_observes_only(self):
        self.mode = 'retirement-pending'
        with fixture(self.respond) as (endpoint, _):
            server = self.server(endpoint)
            self.start(server)
            answer = self.manager.stop(server, 'chat', Deadline.after(), no_wait=True)
            self.assertFalse(answer['completed'])
            self.assertIs(answer['accepted'], True)
            self.assertEqual(self.row()['pending']['stage'], 'retirement')
            self.items = []
            answer = self.manager.stop(server, 'chat', Deadline.after())
            self.assertTrue(answer['completed'])
            self.assertEqual(self.mutations().count('/v1/models/unregister'), 1)
            self.assertEqual(self.registry.list_instances(), [])

    def test_acknowledged_retirement_deadline_is_pending(self):
        self.mode = 'retirement-pending'
        with fixture(self.respond) as (endpoint, _):
            server = self.server(endpoint)
            self.start(server)
            with self.assertRaises(DeadlineExpired) as error:
                self.manager.stop(server, 'chat', Deadline.after(.05))
            self.assertIs(error.exception.accepted, True)
            self.assertEqual(self.row()['incarnation'], 17)
            self.assertEqual(self.row()['pending']['stage'], 'retirement')

    def test_lost_unregister_is_uncertain_and_not_replayed(self):
        with fixture(self.respond) as (endpoint, _):
            server = self.server(endpoint)
            self.start(server)
            self.fail = '/v1/models/unregister'
            with self.assertRaises(UncertainOutcome):
                self.manager.stop(server, 'chat', Deadline.after())
            with self.assertRaisesRegex(UncertainOutcome, 'instead of replaying'):
                self.manager.stop(server, 'chat', Deadline.after())
            self.assertEqual(self.mutations().count('/v1/models/unregister'), 1)
            self.items = []
            self.assertTrue(self.manager.stop(server, 'chat', Deadline.after())['completed'])

    def test_unknown_external_build_refuses_start(self):
        with self.assertRaisesRegex(InvalidInput, 'tracked local server'):
            self.manager.start({'id':None, 'external':True, 'endpoint':'http://127.0.0.1:1'},
                               'tiny', Deadline.after())
        self.assertEqual(self.registry.list_instances(), [])

    def test_external_stop_does_not_create_server_ownership(self):
        self.items = [model()]
        with fixture(self.respond) as (endpoint, _):
            answer = self.manager.stop({'id':None, 'external':True, 'endpoint':endpoint},
                                       'chat', Deadline.after(), abort=True)
            self.assertTrue(answer['completed'])
        self.assertEqual(self.registry.list_servers(), [])
        self.assertEqual(self.registry.list_instances(), [])

    def test_server_launch_changed_refuses_native_actions(self):
        with fixture(self.respond) as (endpoint, _):
            server = self.server(endpoint)
            server = dict(server, launch_token='stale')
            with self.assertRaises(UncertainOutcome):
                self.start(server)
            self.assertEqual(self.calls, [])

    def test_start_does_not_reenter_model_or_server_lock(self):
        with fixture(self.respond) as (endpoint, _):
            server = self.server(endpoint)
            original = hem_models.model_files
            def check_lock(*args, **kwargs):
                with self.assertRaisesRegex(HemError, 'not reentrant'):
                    with self.registry.lock('model', 'tiny'):
                        pass
                return original(*args, **kwargs)
            with patch.object(hem_models, 'model_files', side_effect=check_lock):
                self.start(server)
            with self.registry.lock('server', 'server'):
                pass

    def test_register_rejected_records_known_nonacceptance_and_allows_explicit_cleanup(self):
        original = self.respond
        def reject(handler, requests):
            if requests[-1][1] == '/v1/models/register':
                self.calls.append(('POST', '/v1/models/register'))
                reply(handler, {'error': 'rejected fixture'}, 403)
            else:
                original(handler, requests)
        with fixture(reject) as (endpoint, _):
            server = self.server(endpoint)
            with self.assertRaises(HemError) as error:
                self.start(server)
            self.assertIs(error.exception.accepted, False)
            self.assertIs(self.row()['pending']['accepted'], False)
            answer = self.manager.stop(server, 'chat', Deadline.after())
            self.assertTrue(answer['completed'])
            self.assertEqual(self.registry.list_instances(), [])
            self.assertEqual(self.mutations(), ['/v1/models/register'])

    def test_lost_accepted_register_can_be_explicitly_stopped_after_observation(self):
        original = self.respond
        def lost(handler, requests):
            if requests[-1][1] == '/v1/models/register':
                self.calls.append(('POST', '/v1/models/register'))
                self.items = [model()]
                handler.connection.shutdown(socket.SHUT_RDWR)
                handler.connection.close()
            else:
                original(handler, requests)
        with fixture(lost) as (endpoint, _):
            server = self.server(endpoint)
            with self.assertRaises(UncertainOutcome):
                self.start(server)
            self.assertTrue(self.manager.stop(server, 'chat', Deadline.after())['completed'])
            self.assertEqual(self.registry.list_instances(), [])
            self.assertNotIn('/v1/models/activate', self.mutations())

    def test_explicit_abort_escalates_known_pending_drain(self):
        self.mode = 'retirement-pending'
        with fixture(self.respond) as (endpoint, requests):
            server = self.server(endpoint)
            self.start(server)
            self.manager.stop(server, 'chat', Deadline.after(), no_wait=True)
            self.mode = None
            answer = self.manager.stop(server, 'chat', Deadline.after(), abort=True)
            self.assertTrue(answer['completed'])
            modes = [json.loads(body)['mode'] for method, path, body, _ in requests
                     if path == '/v1/models/unregister']
            self.assertEqual(modes, ['drain', 'abort'])

    def test_failed_completion_observation_is_uncertain_even_after_acceptance(self):
        original = self.respond
        def broken(handler, requests):
            if requests[-1][1] == '/v1/status' and '/v1/models/activate' in self.mutations():
                reply(handler, {'bad': 'inventory'})
            else:
                original(handler, requests)
        with fixture(broken) as (endpoint, _):
            with self.assertRaises(UncertainOutcome) as error:
                self.start(self.server(endpoint))
            self.assertEqual(error.exception.exit_code, 4)
            self.assertIs(error.exception.accepted, True)
            self.assertNotIn('/v1/models/undrain', self.mutations())

    def test_sqlite_contention_consumes_original_deadline(self):
        import sqlite3
        with fixture(self.respond) as (endpoint, _):
            server = self.server(endpoint)
            connection = sqlite3.connect(self.registry.database)
            connection.execute('BEGIN IMMEDIATE')
            began = time.monotonic()
            try:
                with self.assertRaises(DeadlineExpired):
                    self.start(server, Deadline.after(.05))
            finally:
                connection.rollback()
                connection.close()
            self.assertLess(time.monotonic() - began, .5)
            self.assertEqual(self.calls, [])
            self.assertIsNone(self.registry.deadline)


class OwnedProcessLifecycleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compiled = tempfile.TemporaryDirectory()
        cls.binary = Path(cls.compiled.name) / 'loom-serve'
        source = Path(cls.compiled.name) / 'native.c'
        source.write_text(NATIVE)
        subprocess.run(['cc', '-O0', '-o', str(cls.binary), str(source)], check=True,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    @classmethod
    def tearDownClass(cls):
        cls.compiled.cleanup()

    def test_owned_process_registration_and_retirement_from_shared_snapshot(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            package = root / 'package'
            (package / '_native/bin').mkdir(parents=True)
            (package / '_native/lib').mkdir()
            shutil.copy2(self.binary, package / '_native/bin/loom-serve')
            (package / '_native/lib/libhsa-runtime64.so.1').write_bytes(b'CPU fixture only')
            original_info = hem_config.pipeline_info
            source_catalog = Path(original_info('qwen-08b')['catalog'])
            shutil.copytree(source_catalog.parents[2], package / 'loom-pipeline')
            registry = hem_state.Registry(root / 'state')
            processes = hem_process.ProcessManager(registry, package_root=package, emit=lambda _: None)
            manager = hem_instances.InstanceManager(registry, processes=processes)
            settings = {'device':'cpu-fixture','listen':f'127.0.0.1:{port()}'}
            registry.create_server('fixture', saved_settings=settings)
            server = None
            try:
                server = processes.start('fixture', settings, Deadline.after(5))
                weights, tokenizer = root / 'tiny.gguf', root / 'tokenizer.json'
                weights.write_bytes(b'CPU fixture weights')
                tokenizer.write_text('{}')
                with patch.object(hem_config, 'select_pipeline', return_value='qwen-08b'), \
                     patch.object(pipeline_validation, '__file__', str(package / 'models/pipeline_validation.py')):
                    hem_models.import_model(registry, 'tiny', ggufs=[weights], tokenizer=tokenizer)
                    answer = manager.start(server, 'tiny', Deadline.after(5), name='chat')
                self.assertTrue(answer['completed'])
                row = registry.get_instance('fixture', server['launch_token'], 'chat')
                deployment = json.loads(Path(row['document']).read_text())
                retained = Path(deployment['base_directory'])
                modules = hem_config.catalog_modules(retained)
                self.assertEqual((retained / modules.control).read_bytes(), (source_catalog / modules.control).read_bytes())
                self.assertEqual(retained, Path(server['build']['directory']) / 'loom-pipeline' / source_catalog.relative_to(source_catalog.parents[2]))
                self.assertFalse(list((Path(server['build']['directory']) / 'instances').glob('*/sources-*')))
                self.assertIn(str(retained), row['dependencies'])
                # Removing the package cannot invalidate the retained sources or
                # prevent safe stop of the already observed original incarnation.
                shutil.rmtree(package)
                self.assertTrue((retained / modules.prepare).is_file())
                with patch.object(hem_config, 'select_pipeline', return_value='qwen-08b'), \
                     patch.object(pipeline_validation, '__file__', str(package / 'models/pipeline_validation.py')):
                    with self.assertRaises(HemError):
                        manager.start(server, 'tiny', Deadline.after(3), name='new-alias')
                self.assertEqual(len(registry.list_instances()), 1)
                with self.assertRaisesRegex(HemError, 'live or unresolved'):
                    processes.cleanup(server, Deadline.after(2))
                stopped = manager.stop(server, 'chat', Deadline.after(3))
                self.assertTrue(stopped['completed'])
                self.assertEqual(registry.list_instances(), [])
                with processes.pin(server) as pinned:
                    self.assertEqual(processes.client(server, pinned, Deadline.after(2)).status()['data'], [])
                    pinned.send(signal.SIGTERM, Deadline.after(2))
                    pinned.wait(Deadline.after(2))
                processes.reconcile('fixture', Deadline.after(2))
                self.assertIsNone(registry.get_server('fixture')['launch_token'])
            finally:
                current = registry.get_server('fixture')
                owned = (server or current).get('process')
                if owned and not hem_process._confirmed_exit(owned):
                    with hem_process.PinnedProcess(owned, (server or current)['build']['executable'], emit=lambda _: None) as pinned:
                        pinned.send(signal.SIGKILL, Deadline.after(2))
                        pinned.wait(Deadline.after(2))
                if owned and owned.get('logger'):
                    wait_exit(owned['logger'])
                for child in processes.children.values():
                    child.wait(timeout=3)


if __name__ == '__main__':
    unittest.main()
