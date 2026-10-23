"""Composition faults with real atomic files/SQLite, never model acceptance."""
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from copy import deepcopy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib
import io
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch, Mock

PACKAGE = os.environ.get('HEM_TEST_PACKAGE', 'binding.python')
hem = importlib.import_module(PACKAGE + '.integrations.openwebui.hem_openwebui')
common = importlib.import_module(PACKAGE + '.hem_common')
files = importlib.import_module(PACKAGE + '.hem_pi_files')
state = importlib.import_module(PACKAGE + '.integrations.openwebui.hem_openwebui_state')
Registry = importlib.import_module(PACKAGE + '.hem_state').Registry

TARGET = 'http://127.0.0.1:3000'
SECRET = 'foreign-key-must-stay-in-memory'
TOKEN = 'admin-token-never-render'


def foreign():
    return dict(ENABLE_OPENAI_API=True,
        OPENAI_API_BASE_URLS=['https://foreign.example/v1'], OPENAI_API_KEYS=[SECRET],
        OPENAI_API_CONFIGS={'0': dict(prefix_id='foreign', extra={'opaque': SECRET})})


class Client:
    def __init__(self):
        self.remote = foreign()
        self.version_value = dict(version='0.12.3', deployment_id='')
        self.calls = []
        self.failure = None
        self.update_failure = None
        self.read_hook = None
        self.update_hook = None
        self.reads = 0
        self.factory_args = None

    def factory(self, target, token, deadline):
        self.factory_args = target, token, deadline
        return self

    def version(self):
        self.calls.append('version')
        if self.failure:
            raise self.failure
        return deepcopy(self.version_value)

    def config(self):
        self.calls.append('config')
        self.reads += 1
        if self.read_hook:
            self.read_hook(self)
        return deepcopy(self.remote)

    def update(self, config):
        self.calls.append('update')
        if self.update_hook:
            self.update_hook(self, config)
        else:
            self.remote = deepcopy(config)
        if self.update_failure:
            raise self.update_failure
        return deepcopy(self.remote)


class CoordinatorTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.root = self.base / 'loom'
        self.registry = Registry(self.root)
        self.registry.create_server('main', saved_settings=dict(device='task', listen='127.0.0.1:18080'))
        self.client = Client()
        self.path = state.ownership_path(self.root, TARGET)
        self.events = []

    def run_configure(self, **kwargs):
        kwargs.setdefault('client_factory', self.client.factory)
        kwargs.setdefault('display', lambda *args: self.events.append(deepcopy(args)))
        return hem.configure_openwebui(self.root, TARGET, TOKEN, common.Deadline.after(), **kwargs)

    def snapshot(self):
        return {str(p.relative_to(self.base)): (p.lstat().st_mode, p.lstat().st_mtime_ns,
            p.read_bytes() if p.is_file() and not p.is_symlink() else None)
            for p in self.base.rglob('*')}

    def owned(self):
        return state.decode_document(files.read_owned(self.path), self.root, TARGET)

    def assert_no_secrets(self, result=None):
        values = [repr(self.events), repr(result)]
        values += [p.read_text(errors='replace') for p in self.root.rglob('*.json')]
        for value in values:
            self.assertNotIn(SECRET, value)
            self.assertNotIn(TOKEN, value)

    def test_configure_preserves_foreign_and_noop_has_no_post_or_write(self):
        result = self.run_configure()
        self.assertEqual((result.exit_code, result['configuration'], result['completed']), (0, 'configured', True))
        self.assertTrue(result['accepted'])
        self.assertEqual(self.client.remote['OPENAI_API_KEYS'], [SECRET, ''])
        self.assertEqual(self.client.remote['OPENAI_API_CONFIGS']['0'], foreign()['OPENAI_API_CONFIGS']['0'])
        self.assertIsNone(self.owned()['pending'])
        self.assertEqual(self.client.calls, ['version', 'config', 'config', 'update', 'config'])
        # Registry startup is intentionally mutation-only; compare ownership mtime.
        stamp = self.path.stat().st_mtime_ns
        self.client.calls.clear()
        result = self.run_configure()
        self.assertEqual(self.client.calls, ['version', 'config'])
        self.assertFalse(result['changed'])
        self.assertEqual(self.path.stat().st_mtime_ns, stamp)
        self.assertEqual(result['model_availability'], 'unprobed')
        self.assert_no_secrets(result)

    def test_check_configured_is_fully_passive_and_pending_not_finalized(self):
        self.run_configure()
        before = self.snapshot()
        with patch.object(hem, 'Registry', side_effect=AssertionError('init')), \
             patch.object(hem, 'ensure_endpoints', side_effect=AssertionError('publish')), \
             patch.object(hem, 'integration_lock', side_effect=AssertionError('lock')), \
             patch.object(files, 'atomic_replace', side_effect=AssertionError('write')):
            self.assertEqual(self.run_configure(check=True).exit_code, 0)
        self.assertEqual(self.snapshot(), before)
        # A submitted-after-state intent remains untouched by passive check.
        doc = self.owned()
        row = doc['connections'][0]
        doc.update(connections=[], pending=dict(operation='configure', before=[], after=[row], deployment_id=''))
        self.path.write_bytes(files.encoded(doc))
        before = self.snapshot()
        result = self.run_configure(check=True)
        self.assertEqual((result.exit_code, result['configuration'], result['pending']), (1, 'pending', True))
        self.assertEqual(before, self.snapshot())

    def test_check_absent_root_never_creates_anything(self):
        absent = self.base / 'absent'
        before = self.snapshot()
        result = hem.configure_openwebui(absent, TARGET, TOKEN, common.Deadline.after(),
                                        check=True, client_factory=self.client.factory)
        self.assertEqual((result.exit_code, result['configuration']), (1, 'absent'))
        self.assertEqual(self.snapshot(), before)
        self.assertFalse(absent.exists())

    def test_direct_entry_resolves_environment_and_retains_safe_diagnostics(self):
        import inspect
        parameters = inspect.signature(hem.configure_openwebui).parameters
        self.assertNotIn('_chat_only', parameters)
        self.assertNotIn('_lock_held', parameters)
        absent = self.base / 'direct-absent'
        with patch.dict(os.environ, {'LOOM_SERVE_DIR': str(absent), 'OPENWEBUI_URL': TARGET,
                                     'OPENWEBUI_TOKEN': TOKEN}):
            result = hem.configure_openwebui(None, None, None, common.Deadline.after(),
                                            check=True, client_factory=self.client.factory)
        self.assertEqual(result['target_url'], TARGET)
        self.assertIn('operator-required/unverified', result['prerequisites']['persistence'])
        self.assertIn('Export/backups', result['guidance'])
        self.assertFalse(absent.exists())
        factory = Mock(side_effect=common.InvalidInput('rejected explicit-admin-secret'))
        with patch.dict(os.environ, {'OPENWEBUI_TOKEN': 'different-environment-token'}):
            with self.assertRaises(common.InvalidInput) as error:
                hem.configure_openwebui(absent, TARGET, 'explicit-admin-secret', common.Deadline.after(),
                                        client_factory=factory)
        self.assertNotIn('explicit-admin-secret', str(error.exception))
        self.assertIn('<redacted>', str(error.exception))
        self.assertFalse(absent.exists())

    def test_fixed_dispatch_preserves_webui_options_and_budget(self):
        dispatch = importlib.import_module(PACKAGE + '.hem_integrations').configure_integration
        deadline = common.Deadline.after()
        for mode in ('configure', 'check', 'disable'):
            with patch.object(hem, 'configure_openwebui', return_value={}) as entry:
                dispatch('openwebui', self.root, deadline, mode=mode, url=TARGET, token=TOKEN,
                         stt_server=None, stt_instance=None)
                entry.assert_called_once_with(self.root, TARGET, TOKEN, deadline,
                    check=mode == 'check', disable=mode == 'disable', stt_server=None, stt_instance=None)

    def test_detected_remote_race_aborts_before_post_pending_survives(self):
        def race(client):
            if client.reads == 2:
                client.remote['OPENAI_API_KEYS'][0] = 'new-foreign-key'
        self.client.read_hook = race
        with self.assertRaises(common.InvalidInput):
            self.run_configure()
        self.assertNotIn('update', self.client.calls)
        self.assertIsNotNone(self.owned()['pending'])
        self.client.read_hook = None
        result = self.run_configure()
        self.assertEqual(result.exit_code, 0)
        self.assertEqual(self.client.remote['OPENAI_API_KEYS'][0], 'new-foreign-key')
        self.assert_no_secrets(result)

    def test_lost_ack_or_5xx_readback_after_state_finalizes(self):
        for accepted in (None, True):
            with self.subTest(accepted=accepted):
                self.client.update_failure = common.UncertainOutcome('sanitized failure', accepted=accepted)
                result = self.run_configure()
                self.assertEqual(result.exit_code, 0)
                self.assertTrue(result['accepted'])
                self.assertIsNone(self.owned()['pending'])
                self.client.update_failure = None
                self.run_configure(disable=True)

    def test_unacknowledged_before_state_remains_uncertain_then_fresh_merge(self):
        self.client.update_hook = lambda client, config: None
        self.client.update_failure = common.UncertainOutcome('server error')
        with self.assertRaises(common.UncertainOutcome) as error:
            self.run_configure()
        self.assertIsNone(error.exception.accepted)
        self.assertIsNotNone(self.owned()['pending'])
        self.client.remote['OPENAI_API_KEYS'][0] = 'changed-between-attempts'
        self.client.update_hook = None
        self.client.update_failure = None
        self.assertEqual(self.run_configure().exit_code, 0)
        self.assertEqual(self.client.remote['OPENAI_API_KEYS'][0], 'changed-between-attempts')

    def test_post_readback_failure_retains_intent_for_later_recovery(self):
        def fail_read(client):
            if client.reads == 3:
                raise common.HemError('offline')
        self.client.read_hook = fail_read
        with self.assertRaises(common.UncertainOutcome) as error:
            self.run_configure()
        self.assertTrue(error.exception.accepted)
        self.assertIsNotNone(self.owned()['pending'])
        self.client.read_hook = None
        self.client.calls.clear()
        self.assertEqual(self.run_configure().exit_code, 0)
        self.assertNotIn('update', self.client.calls)
        self.assertIsNone(self.owned()['pending'])

    def test_readback_foreign_mismatch_is_uncertain_never_rollback(self):
        def foreign_edit(client):
            if client.reads == 3:
                client.remote['OPENAI_API_KEYS'][0] = 'late-browser-edit'
        self.client.read_hook = foreign_edit
        with self.assertRaises(common.UncertainOutcome) as error:
            self.run_configure()
        self.assertTrue(error.exception.accepted)
        self.assertEqual(self.client.remote['OPENAI_API_KEYS'][0], 'late-browser-edit')
        self.assertEqual(self.client.calls.count('update'), 1)
        self.assertIsNotNone(self.owned()['pending'])

    def test_local_finalize_failure_reports_accepted_then_recovers_without_post(self):
        original = files.atomic_replace
        def fail(path, data, **kwargs):
            if path == self.path and json.loads(data)['pending'] is None:
                raise OSError('disk full')
            return original(path, data, **kwargs)
        with patch.object(files, 'atomic_replace', side_effect=fail):
            with self.assertRaises(common.UncertainOutcome) as error:
                self.run_configure()
        self.assertTrue(error.exception.accepted)
        self.assertIsNotNone(self.owned()['pending'])
        self.client.calls.clear()
        result = self.run_configure()
        self.assertEqual(result.exit_code, 0)
        self.assertNotIn('update', self.client.calls)
        self.assertIsNone(self.owned()['pending'])

    def test_pending_third_state_conflict_is_retained_without_replay(self):
        def fail_read(client):
            if client.reads == 3:
                raise common.HemError('offline')
        self.client.read_hook = fail_read
        with self.assertRaises(common.UncertainOutcome):
            self.run_configure()
        evidence = self.path.read_bytes()
        self.client.read_hook = None
        self.client.remote['OPENAI_API_CONFIGS']['1']['model_ids'] = ['operator-change']
        self.client.calls.clear()
        with self.assertRaises(common.InvalidInput):
            self.run_configure()
        self.assertNotIn('update', self.client.calls)
        self.assertEqual(self.path.read_bytes(), evidence)
        before = self.snapshot()
        result = self.run_configure(check=True)
        self.assertEqual((result.exit_code, result['configuration'], result['pending']), (2, 'pending', True))
        self.assertEqual(self.snapshot(), before)

    def test_pending_stage_failure_prevents_post(self):
        original = files.atomic_replace
        def fail(path, data, **kwargs):
            if path == self.path:
                raise OSError('disk full')
            return original(path, data, **kwargs)
        with patch.object(files, 'atomic_replace', side_effect=fail):
            with self.assertRaises(common.HemError) as error:
                self.run_configure()
        self.assertEqual(error.exception.exit_code, 1)
        self.assertFalse(error.exception.accepted)
        self.assertNotIn('update', self.client.calls)
        self.assertFalse(self.path.exists())

    def test_atomic_pending_and_finalization_replace_crashes_resume_safely(self):
        for finalization in (False, True):
            with self.subTest(finalization=finalization):
                original = files.os.replace
                def crash(source, destination):
                    if Path(destination) == self.path:
                        pending = json.loads(Path(source).read_bytes())['pending']
                        if (pending is None) == finalization:
                            raise OSError('injected replacement crash')
                    return original(source, destination)
                with patch.object(files.os, 'replace', side_effect=crash):
                    with self.assertRaises(common.HemError):
                        self.run_configure()
                self.assertTrue(files.staged_path(self.path).exists())
                self.client.calls.clear()
                self.assertEqual(self.run_configure().exit_code, 0)
                self.assertEqual(self.client.calls.count('update'), 0 if finalization else 1)
                self.assertIsNone(self.owned()['pending'])
                self.assertFalse(files.staged_path(self.path).exists())
                self.run_configure(disable=True)

    def test_lock_order_publication_outside_http_inside_revalidation_lock(self):
        held = [False]
        publish = hem.ensure_endpoints
        lock = hem.integration_lock
        def checked_publish(registry, deadline):
            self.assertFalse(held[0])
            return publish(registry, deadline)
        @contextmanager
        def checked_lock(root, deadline):
            self.assertFalse(held[0])
            with lock(root, deadline):
                held[0] = True
                try:
                    yield
                finally:
                    held[0] = False
        def checked_display(*args):
            self.assertTrue(held[0])
        with patch.object(hem, 'ensure_endpoints', side_effect=checked_publish), \
             patch.object(hem, 'integration_lock', checked_lock):
            self.assertEqual(self.run_configure(display=checked_display).exit_code, 0)
        self.assertFalse(held[0])

    def test_public_cli_json_sanitizes_pre_submission_ownership_write_faults(self):
        cli = importlib.import_module(PACKAGE + '.hem_cli')
        coordinate = hem.configure_openwebui
        original_replace = files.atomic_replace
        for bookkeeping in (False, True):
            with self.subTest(bookkeeping=bookkeeping):
                if bookkeeping:
                    self.run_configure()
                    self.client.version_value['deployment_id'] = 'new-observed-identity'
                self.client.calls.clear()
                def disk_failure(path, data, **kwargs):
                    if path == self.path:
                        raise OSError('permission denied /private/' + TOKEN + '/' + SECRET)
                    return original_replace(path, data, **kwargs)
                def injected_client(*args, **kwargs):
                    return coordinate(*args, client_factory=self.client.factory, **kwargs)
                output, errors = io.StringIO(), io.StringIO()
                with patch.dict(os.environ, {'LOOM_SERVE_DIR': str(self.root),
                        'OPENWEBUI_URL': TARGET, 'OPENWEBUI_TOKEN': TOKEN}), \
                     patch.object(hem, 'configure_openwebui', side_effect=injected_client), \
                     patch.object(files, 'atomic_replace', side_effect=disk_failure), \
                     redirect_stdout(output), redirect_stderr(errors):
                    code = cli.main(['openwebui', 'configure', '--json'])
                self.assertEqual(code, 1)
                self.assertEqual(len(output.getvalue().splitlines()), 1)
                result = json.loads(output.getvalue())
                self.assertEqual(result['state'], 'failed')
                self.assertFalse(result['completed'])
                self.assertFalse(result['accepted'])
                self.assertIn('local ownership update failed', result['error'])
                self.assertNotIn('update', self.client.calls)
                combined = output.getvalue() + errors.getvalue()
                for value in (TOKEN, SECRET, '/private/', 'permission denied', 'Traceback'):
                    self.assertNotIn(value, combined)

    def test_partial_disable_removes_unchanged_reindexes_and_retains_edited(self):
        self.registry.create_server('other', saved_settings=dict(device='task', listen='127.0.0.1:18081'))
        self.run_configure()
        changed_provider = self.client.remote['OPENAI_API_CONFIGS']['1']['prefix_id']
        self.client.remote['OPENAI_API_KEYS'][1] = 'operator-added-credential'
        result = self.run_configure(disable=True)
        self.assertEqual((result.exit_code, result['configuration'], result['completed']), (1, 'partial', False))
        self.assertEqual(len(self.owned()['connections']), 1)
        self.assertEqual(self.owned()['connections'][0]['provider'], changed_provider)
        self.assertEqual(self.client.remote['OPENAI_API_KEYS'], [SECRET, 'operator-added-credential'])
        self.assertEqual(set(self.client.remote['OPENAI_API_CONFIGS']), {'0', '1'})
        self.assert_no_secrets(result)

    def test_offline_unauthorized_disable_retains_evidence(self):
        self.run_configure()
        old = self.path.read_bytes()
        for message, service in [('unreachable', 'offline'), ('WebUI HTTP 403: authorization rejected', 'unauthorized')]:
            self.client.failure = common.HemError(message)
            result = self.run_configure(disable=True)
            self.assertEqual((result.exit_code, result['service'], result['configuration']), (1, service, 'unknown'))
            self.assertFalse(result['completed'])
            self.assertEqual(old, self.path.read_bytes())
        self.client.failure = None
        self.assertEqual(self.run_configure(disable=True).exit_code, 0)
        self.assertEqual(self.client.remote, foreign())
        self.client.calls.clear()
        self.assertEqual(self.run_configure(disable=True).exit_code, 0)
        self.assertNotIn('update', self.client.calls)

    def test_unsupported_version_checks_only_mutation_refuses(self):
        self.client.version_value['version'] = '0.99.0'
        result = self.run_configure(check=True)
        self.assertEqual(result['compatibility'], 'unsupported')
        self.assertEqual(result.exit_code, 1)
        with self.assertRaises(common.InvalidInput):
            self.run_configure()
        self.assertNotIn('update', self.client.calls)
        self.assertFalse(self.path.exists())

    def test_owned_drift_global_disabled_and_identity_change_never_post(self):
        self.run_configure()
        self.client.calls.clear()
        self.client.remote['OPENAI_API_CONFIGS']['1']['headers'] = {'Authorization': SECRET}
        with self.assertRaises(common.InvalidInput):
            self.run_configure()
        self.assertNotIn('update', self.client.calls)
        self.assertEqual(self.run_configure(check=True).exit_code, 2)
        self.client.remote['OPENAI_API_CONFIGS']['1'].pop('headers')
        self.client.remote['ENABLE_OPENAI_API'] = False
        with self.assertRaises(common.InvalidInput):
            self.run_configure()
        self.assertFalse(self.client.remote['ENABLE_OPENAI_API'])
        self.client.remote['ENABLE_OPENAI_API'] = True
        self.client.version_value['deployment_id'] = 'observed'
        self.assertEqual(self.run_configure().exit_code, 0)
        self.assertEqual(self.owned()['deployment_id'], 'observed')
        self.client.version_value['deployment_id'] = ''
        with self.assertRaises(common.InvalidInput):
            self.run_configure(disable=True)

    def test_projection_changed_after_publication_refuses_without_http(self):
        original = hem.ensure_endpoints
        def publish_then_change(registry, deadline):
            original(registry, deadline)
            registry.create_server('later', saved_settings=dict(device='task', listen='127.0.0.1:18082'))
        with patch.object(hem, 'ensure_endpoints', side_effect=publish_then_change):
            with self.assertRaises(common.InvalidInput):
                self.run_configure()
        self.assertEqual(self.client.calls, [])

    def test_one_deadline_expiry_before_submission_is_three(self):
        clock = [0.0]
        deadline = common.Deadline.after(10, clock=lambda: clock[0])
        def expire(client):
            if client.reads == 2:
                clock[0] = 11.0
        self.client.read_hook = expire
        with self.assertRaises(common.DeadlineExpired) as error:
            hem.configure_openwebui(self.root, TARGET, TOKEN, deadline, client_factory=self.client.factory)
        self.assertEqual(error.exception.exit_code, 3)
        self.assertFalse(error.exception.accepted)
        self.assertIs(self.client.factory_args[2], deadline)
        self.assertNotIn('update', self.client.calls)
        self.assertIsNotNone(self.owned()['pending'])

    def test_real_management_http_composes_with_real_owned_io(self):
        fixture = self.client
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass
            def send_json(self, data):
                body = json.dumps(data).encode()
                self.send_response(200)
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            def do_GET(self):
                self.assert_auth()
                if self.path == '/api/version':
                    self.send_json(fixture.version())
                elif self.path == '/openai/config':
                    self.send_json(fixture.config())
                else:
                    self.send_error(404)
            def assert_auth(self):
                if self.headers.get('Authorization') != 'Bearer ' + TOKEN:
                    raise AssertionError('credential not sent to management')
            def do_POST(self):
                self.assert_auth()
                if self.path != '/openai/config/update':
                    self.send_error(404)
                    return
                self.send_json(fixture.update(json.loads(self.rfile.read(int(self.headers['Content-Length'])))))
        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(thread.join, 3)
        self.addCleanup(server.shutdown)
        target = 'http://127.0.0.1:' + str(server.server_port)
        result = hem.configure_openwebui(self.root, target, TOKEN, common.Deadline.after())
        self.assertEqual(result.exit_code, 0)
        self.assertEqual(fixture.remote['OPENAI_API_KEYS'], [SECRET, ''])
        result = hem.configure_openwebui(self.root, target, TOKEN, common.Deadline.after(), check=True)
        self.assertEqual(result.exit_code, 0)
        result = hem.configure_openwebui(self.root, target, TOKEN, common.Deadline.after(), disable=True)
        self.assertEqual(result.exit_code, 0)
        self.assertEqual(fixture.remote, foreign())


if __name__ == '__main__':
    unittest.main()
