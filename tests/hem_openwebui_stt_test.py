"""STT ownership boundaries with real private files and existing chat coordinator."""
from copy import deepcopy
from contextlib import nullcontext, contextmanager
from contextlib import redirect_stdout, redirect_stderr
import importlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, Mock

PACKAGE = os.environ.get('HEM_TEST_PACKAGE', 'binding.python')
hem = importlib.import_module(PACKAGE + '.hem_openwebui')
stt = importlib.import_module(PACKAGE + '.hem_openwebui_stt')
state = importlib.import_module(PACKAGE + '.hem_openwebui_state')
files = importlib.import_module(PACKAGE + '.hem_pi_files')
common = importlib.import_module(PACKAGE + '.hem_common')
Registry = importlib.import_module(PACKAGE + '.hem_state').Registry
contracts = importlib.import_module(PACKAGE + '.hem_pi_contracts')
core = importlib.import_module(PACKAGE + '.hem_operation_targets')
process = importlib.import_module(PACKAGE + '.hem_process')
TARGET = 'http://127.0.0.1:3000'
SECRET = 'foreign-audio-secret-do-not-persist'


def audio():
    tts = dict.fromkeys(stt.TTS_STRINGS, '')
    tts.update(OPENAI_API_KEY=SECRET, OPENAI_PARAMS={'secret': SECRET}, REALTIME_TTS_PROMPT_TEMPLATE=None)
    row = dict.fromkeys(stt.STT_STRINGS, '')
    row.update(OPENAI_API_BASE_URL='https://api.openai.com/v1', OPENAI_API_REQUEST_FORMAT='multipart',
        WHISPER_MODEL='base', MISTRAL_API_BASE_URL='https://api.mistral.ai/v1', MISTRAL_USE_CHAT_COMPLETIONS=False,
        SUPPORTED_CONTENT_TYPES=[], ALLOWED_EXTENSIONS=['wav', 'mp3', 'flac'])
    return dict(tts=tts, stt=row, realtime={'OPENAI_API_KEY': SECRET, 'ENABLED': True})


class Client:
    def __init__(self):
        self.remote = dict(ENABLE_OPENAI_API=True, OPENAI_API_BASE_URLS=['https://foreign.test/v1'],
            OPENAI_API_KEYS=[SECRET], OPENAI_API_CONFIGS={'0': {'prefix_id': 'foreign'}})
        self.audio = audio(); self.rows = {}; self.calls = []; self.fail_chat = False
        self.audio_error = None; self.audio_commit = True; self.fail_delete = False
        self.audio_reads = 0; self.on_audio_read = None

    def factory(self, *args): return self
    def version(self): return dict(version='0.12.3', deployment_id='test-deployment')
    def config(self): return deepcopy(self.remote)
    def update(self, body):
        self.calls.append('chat_update')
        if self.fail_chat: raise common.HemError('deliberate chat failure', accepted=False)
        self.remote = deepcopy(body); return deepcopy(body)
    def audio_config(self):
        self.audio_reads += 1
        if self.on_audio_read: self.on_audio_read(self)
        self.calls.append('audio_get'); return deepcopy(self.audio)
    def update_audio(self, body):
        self.calls.append('audio_update')
        assert 'realtime' not in body
        if self.audio_commit:
            self.audio.update(deepcopy(body))
        if self.audio_error: raise self.audio_error
        return deepcopy(self.audio)
    def model(self, name):
        self.calls.append('model_get'); return deepcopy(self.rows.get(name))
    def create_model(self, body):
        self.calls.append('model_create')
        assert body['id'] not in self.rows
        self.rows[body['id']] = dict(deepcopy(body), user_id='admin', version_id='version1',
            created_at=100, updated_at=100, write_access=True, user={'name': 'admin'})
        self.rows[body['id']]['meta'] = dict.fromkeys(('profile_image_url', 'background_image_url', 'description', 'i18n', 'capabilities', 'knowledge', 'voice', 'voice_avatar'), None) | body['meta']
        return deepcopy(self.rows[body['id']])
    def delete_model(self, name):
        self.calls.append('model_delete')
        if self.fail_delete: raise common.UncertainOutcome('deliberate lost delete')
        del self.rows[name]; return True


class STTTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / 'state'
        Registry(self.root).create_server('local', saved_settings=dict(device='task', listen='127.0.0.1:18080'))
        self.client = Client(); self.events = []
        self.selected = dict(server='local', instance='dictation', base_url='http://127.0.0.1:18080/v1',
            prefixed_id=contracts.provider_id('local') + '.dictation')
        self.registry = Registry(self.root)
        server = self.registry.reserve_launch('local', launch_token='launch', current_settings={},
                                               endpoint='http://127.0.0.1:18080')
        self.registry.update_server('local', dict(phase='running', pending=None,
            process=dict(pid=123, boot_id='boot', start_ticks=42),
            build=dict(directory='/retained', native_sha256='a'*64, payload_sha256='b'*64,
                       catalog_sha256='c'*64, executable=dict(device=1, inode=2))),
            expected_token='launch', expected_revision=server['revision'])
        self.registry.create_instance('local', 'launch', 'dictation', incarnation=7)
        self.native_row = dict(id='dictation', input_modalities=['audio'], output_modalities=['text'],
            loom=dict(incarnation=7, kind='recognition', endpoint='/v1/audio/transcriptions',
                      registration='ready', parameter_operation='idle', error=None,
                      parameters={'primary': {'active': True}}, status={'admission': 'open'}))
        self.native_reads = 0; self.on_native_read = None
        def status(**kwargs):
            self.native_reads += 1
            if self.on_native_read:self.on_native_read(self)
            return dict(object='list', server={}, data=[deepcopy(self.native_row)])
        self.processes = Mock(package_root=Path('/package'))
        self.processes.pin.side_effect = lambda _: nullcontext(object())
        self.processes.client.return_value.status.side_effect = status
        for owner in (process, core):
            patch.object(owner, 'ProcessManager', return_value=self.processes).start()
            patch.object(owner, 'require_build', return_value=Path('/retained/loom-pipeline')).start()
        self.native_patch = patch.object(core, 'resolve_operation_target', wraps=core.resolve_operation_target)
        self.native = self.native_patch.start()
        self.revalidate = patch.object(core, 'revalidate_operation_target', wraps=core.revalidate_operation_target).start()
        self.addCleanup(patch.stopall)
        self.path = state.ownership_path(self.root, TARGET)

    def run_configure(self, **kwargs):
        return hem.configure_openwebui(self.root, TARGET, 'admin-token', common.Deadline.after(),
            client_factory=self.client.factory, display=lambda *args: self.events.append(deepcopy(args)), **kwargs)

    def configure(self): return self.run_configure(stt_server='local', stt_instance='dictation')
    def record(self): return state.decode_document(files.read_owned(self.path), self.root, TARGET).get('stt')
    def assert_no_foreign_secret(self, result=None):
        self.assertNotIn(SECRET, repr(self.events) + repr(result))
        for path in self.root.rglob('*.json'): self.assertNotIn(SECRET, path.read_text())

    def test_missing_and_stopped_names_save_then_only_explicit_configure_applies(self):
        self.registry.create_server('offline', saved_settings=dict(device='task', listen='127.0.0.1:18081'))
        for server, instance, reason in (('offline', 'future', 'unavailable'),
                                         ('local', 'future-dictation', 'missing_instance')):
            with self.subTest(server=server):
                result = self.run_configure(stt_server=server, stt_instance=instance)
                self.assertEqual((result.exit_code, result['completed'], result['pending']), (0, True, False))
                self.assertEqual(result['stt'], dict(selection=dict(server=server, instance=instance),
                    configuration='deferred', reason=reason))
                self.assertIsNone(self.record())
                self.assertFalse(any(call.startswith(('audio_', 'model_')) for call in self.client.calls))
        before = self.path.read_bytes(); stamp = self.path.stat().st_mtime_ns
        self.native.reset_mock()
        self.assertEqual(self.run_configure(check=True).exit_code, 1)
        self.native.assert_not_called()
        self.assertEqual((self.path.read_bytes(), self.path.stat().st_mtime_ns), (before, stamp))
        self.assertEqual(self.run_configure()['stt']['configuration'], 'deferred')
        self.assertEqual(self.path.read_bytes(), before)
        result = self.configure()
        self.assertEqual(result['stt']['configuration'], 'configured')
        self.assertEqual(result['stt']['selection'], dict(server='local', instance='dictation'))
        self.assertEqual(self.client.calls.count('model_create'), 1)
        self.assertEqual(self.client.calls.count('audio_update'), 1)
        self.native.reset_mock(); self.client.calls.clear()
        self.assertEqual(self.run_configure()['stt']['configuration'], 'configured')
        self.native.assert_called_once()
        self.assertFalse(any(call.endswith(('update', 'create', 'delete')) for call in self.client.calls))

    def test_new_missing_server_rejects_but_previously_selected_removed_server_is_retained(self):
        with self.assertRaises(common.InvalidInput):
            self.run_configure(stt_server='typo', stt_instance='future')
        self.assertIsNone(files.read_owned(self.path)); self.assertFalse(self.client.calls)
        self.run_configure(stt_server='local', stt_instance='future')
        with self.registry._transaction() as connection:
            connection.execute('DELETE FROM instances'); connection.execute('DELETE FROM servers')
        self.client.calls.clear()
        result = self.run_configure()
        self.assertEqual(result['stt']['reason'], 'missing_saved_server')
        self.assertEqual(state.stt_selection(state.decode_document(files.read_owned(self.path), self.root, TARGET)),
                         dict(server='local', instance='future'))
        before = self.path.read_bytes()
        with self.assertRaises(common.InvalidInput):
            self.run_configure(stt_server='local', stt_instance='another')
        self.assertEqual(self.path.read_bytes(), before)
        self.assertFalse(any(call.startswith(('audio_', 'model_')) for call in self.client.calls))

    def test_unknown_and_incompatible_keep_exact_failure_after_saving_names_without_effects(self):
        for error in (common.UncertainOutcome('unresolved native identity', status=503, reason='unknown'),
                      common.HemError('incompatible operation', status=409, reason='incompatible')):
            with self.subTest(reason=error.reason), patch.object(core, 'resolve_operation_target', side_effect=error):
                self.client.calls.clear()
                with self.assertRaises(type(error)) as caught: self.configure()
                self.assertIs(caught.exception, error)
                self.assertEqual((caught.exception.status, caught.exception.reason, caught.exception.accepted),
                                 (error.status, error.reason, None))
                value = state.decode_document(files.read_owned(self.path), self.root, TARGET)
                self.assertEqual(value['stt_selection'], dict(server='local', instance='dictation'))
                self.assertIsNone(value['stt']); self.assertIsNone(value['pending'])
                self.assertFalse(any(call.endswith(('update', 'create', 'delete')) for call in self.client.calls))
                self.assertFalse(any(call.startswith(('audio_', 'model_')) for call in self.client.calls))

    def test_selection_preflight_conflicts_and_failed_selection_write_stop_new_effects(self):
        self.client.remote['ENABLE_OPENAI_API'] = False
        with self.assertRaises(common.InvalidInput): self.configure()
        self.assertIsNone(files.read_owned(self.path))
        self.client.remote['ENABLE_OPENAI_API'] = True
        with patch.object(hem, '_replace_before_submission', side_effect=common.HemError('selection disk failure', accepted=False)):
            with self.assertRaises(common.HemError): self.configure()
        self.assertIsNone(files.read_owned(self.path))
        self.assertFalse(any(call.endswith(('update', 'create', 'delete')) for call in self.client.calls))

    def test_unavailable_old_route_is_visible_and_retarget_refuses_without_erasing_it(self):
        self.configure(); old = deepcopy(self.record()); self.client.calls.clear()
        self.native_row['loom']['registration'] = 'loading'
        result = self.run_configure()
        self.assertEqual((result.exit_code, result['completed'], result['pending']), (0, True, False))
        self.assertEqual(result['stt']['configuration'], 'deferred')
        self.assertEqual(result['stt']['target'], old['target'])
        self.assertEqual(self.record(), old)
        before = self.path.read_bytes()
        with self.assertRaisesRegex(common.InvalidInput, 'Disable'):
            self.run_configure(stt_server='local', stt_instance='future')
        self.assertEqual(self.path.read_bytes(), before)
        self.assertFalse(any(call.endswith(('update', 'create', 'delete')) for call in self.client.calls))

    def test_saved_selection_survives_later_chat_failure_without_claiming_deferred_success(self):
        self.client.fail_chat = True
        with self.assertRaises(common.HemError) as error:
            self.run_configure(stt_server='local', stt_instance='future')
        self.assertEqual(error.exception.exit_code, 1)
        value = state.decode_document(files.read_owned(self.path), self.root, TARGET)
        self.assertEqual(value['stt_selection'], dict(server='local', instance='future'))
        self.assertIsNotNone(value['pending']); self.assertIsNone(value['stt'])
        self.assertFalse(any(call.startswith(('audio_', 'model_')) for call in self.client.calls))
        before = self.path.read_bytes(); writes = self.client.calls.count('chat_update')
        self.client.fail_chat = False
        with self.assertRaises(common.UncertainOutcome):
            self.run_configure(stt_server='local', stt_instance='another-future')
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(self.client.calls.count('chat_update'), writes)

    def cli_configure(self, *flags):
        cli = importlib.import_module(PACKAGE + '.hem_cli')
        entry = hem.configure_openwebui
        def configure(*args, **kwargs):
            return entry(*args, **kwargs, client_factory=self.client.factory)
        output, errors = io.StringIO(), io.StringIO()
        with patch.dict(os.environ, dict(LOOM_SERVE_DIR=str(self.root), OPENWEBUI_URL=TARGET, OPENWEBUI_TOKEN='admin-token')), \
                patch.object(hem, 'configure_openwebui', side_effect=configure), \
                redirect_stdout(output), redirect_stderr(errors):
            code = cli.main(['openwebui', 'configure', *flags, '--json'])
        value = json.loads(output.getvalue())
        self.assertNotIn('admin-token', output.getvalue()+errors.getvalue())
        return code, value

    def test_real_cli_reports_deferred_selection_separately_from_remote_pending(self):
        code, value = self.cli_configure('--stt-server', 'local', '--stt-instance', 'future')
        self.assertEqual((code, value['completed'], value['pending']), (0, True, False))
        self.assertEqual(value['stt'], dict(selection=dict(server='local', instance='future'),
                         configuration='deferred', reason='missing_instance'))

    def assert_confirmed_history_survives_later_failure(self, history, phase):
        if history == 'audio':
            self.pending_audio()
            self.client.audio['stt'].update(stt.desired(self.selected))
            self.native_row['loom']['registration'] = 'loading'
        else:
            self.pending_chat(completed=True)
        self.registry.create_server('later', saved_settings=dict(device='task', listen='127.0.0.1:18082'))
        error = (common.InvalidInput('WebUI owned STT routing/model row drift; preserve evidence')
                 if phase == 'ownership' else common.HemError('later chat step failed',
                     accepted=False, status=503, reason='transport_error'))
        reads = [0]
        config = self.client.config
        def read():
            reads[0] += 1
            if phase == 'preflight' and reads[0] == (1 if history == 'audio' else 2):
                raise error
            return config()
        def update(body):
            self.client.calls.append('chat_update')
            raise error
        if phase == 'ownership':
            self.client.rows[self.selected['prefixed_id']]['name'] = 'administrator edit'
        with patch.object(self.client, 'config', side_effect=read), \
                patch.object(self.client, 'update', side_effect=update):
            with self.assertRaises(type(error)) as caught:
                self.configure()
        if phase != 'ownership':
            self.assertIs(caught.exception, error)
            self.assertEqual((caught.exception.reason, caught.exception.status, caught.exception.exit_code),
                             ('transport_error', 503, 1))
        else:
            self.assertEqual(caught.exception.exit_code, 2)
        self.assertIs(caught.exception.accepted, True)
        self.assertIn('prior '+('audio routing update' if history == 'audio' else 'chat update'), str(caught.exception))
        self.assertIn('not submitted', str(caught.exception))
        current = state.decode_document(files.read_owned(self.path), self.root, TARGET)
        self.assertIsNone(current['stt']['pending'])
        if history == 'audio':
            self.assertEqual(current['stt']['routing'], stt.desired(self.selected))
        if phase == 'chat':
            self.assertEqual(current['pending']['operation'], 'configure')
        else:
            self.assertIsNone(current['pending'])
        self.assertEqual(current['stt_selection'], dict(server='local', instance='dictation'))
        self.assertNotIn('audio_update', self.client.calls)
        self.assertNotIn('model_create', self.client.calls)

    def test_confirmed_audio_history_survives_deferred_chat_throw(self):
        self.assert_confirmed_history_survives_later_failure('audio', 'chat')

    def test_confirmed_chat_history_survives_deferred_chat_throw(self):
        self.assert_confirmed_history_survives_later_failure('chat', 'chat')

    def test_confirmed_audio_history_survives_later_preflight_throw(self):
        self.assert_confirmed_history_survives_later_failure('audio', 'preflight')

    def test_confirmed_chat_history_survives_later_preflight_throw(self):
        self.assert_confirmed_history_survives_later_failure('chat', 'preflight')

    def test_confirmed_audio_history_survives_owned_preflight_conflict(self):
        self.assert_confirmed_history_survives_later_failure('audio', 'ownership')

    def test_partial_disable_cli_reports_retained_selection_and_success_clears_it(self):
        self.configure()
        self.client.remote['OPENAI_API_KEYS'][-1] = 'administrator edit'
        code, value = self.cli_configure('--disable')
        self.assertEqual((code, value['completed'], value['configuration']), (1, False, 'partial'))
        current = state.decode_document(files.read_owned(self.path), self.root, TARGET)
        self.assertIsNone(current['stt'])
        selected = dict(server='local', instance='dictation')
        self.assertEqual(current['stt_selection'], selected)
        self.assertEqual(value['stt']['selection'], selected)
        self.assertEqual(value['stt']['configuration'], 'absent')
        self.assertIsNone(value['stt']['target'])
        self.client.remote['OPENAI_API_KEYS'][-1] = ''
        self.configure()
        code, value = self.cli_configure('--disable')
        self.assertEqual((code, value['completed']), (0, True))
        self.assertIsNone(value['stt']['selection'])
        self.assertIsNone(state.decode_document(files.read_owned(self.path), self.root, TARGET)['stt_selection'])

    def test_unavailable_before_new_intent_can_defer_but_staged_intent_is_not_harmless(self):
        guard = core.revalidate_operation_target
        def unavailable(*args):
            guard(*args)
            raise common.HemError('known not ready', reason='unavailable')
        with patch.object(core, 'revalidate_operation_target', side_effect=unavailable):
            result = self.configure()
        self.assertEqual(result['stt']['configuration'], 'deferred')
        self.assertIsNone(self.record())
        self.assertNotIn('model_create', self.client.calls)
        self.client.calls.clear()
        def after_intent(*args):
            guard(*args)
            if self.record() is not None:
                raise common.HemError('known not ready', reason='unavailable')
        with patch.object(core, 'revalidate_operation_target', side_effect=after_intent):
            with self.assertRaises(common.HemError): self.configure()
        self.assertEqual(self.record()['pending']['operation'], 'create_overlay')
        self.assertNotIn('model_create', self.client.calls)

    def test_configure_check_disable_reenable_preserves_secrets_and_dynamic_chat(self):
        original = deepcopy(self.client.audio)
        result = self.configure(); self.assertEqual(result.exit_code, 0)
        self.assertEqual(result['stt']['configuration'], 'configured')
        self.assertEqual(stt.projection(self.client.audio), stt.desired(self.selected))
        self.assertEqual(self.client.audio['tts'], original['tts'])
        self.assertEqual(self.client.audio['realtime'], original['realtime'])
        for key in original['stt'].keys() - stt.ROUTING:
            self.assertEqual(self.client.audio['stt'][key], original['stt'][key])
        owned_connection = list(self.client.remote['OPENAI_API_CONFIGS'].values())[-1]
        self.assertEqual(owned_connection['model_ids'], [])
        self.assertLess(self.client.calls.index('model_create'), self.client.calls.index('chat_update'))
        self.assertLess(self.client.calls.index('chat_update'), self.client.calls.index('audio_update'))
        before = self.path.read_bytes(); calls = list(self.client.calls); self.native.reset_mock()
        check = self.run_configure(check=True)
        self.assertEqual(check.exit_code, 0); self.assertEqual(check['model_availability'], 'unprobed')
        self.native.assert_not_called(); self.assertEqual(self.path.read_bytes(), before)
        self.assertFalse(any(x.endswith(('update', 'create', 'delete')) for x in self.client.calls[len(calls):]))
        result = self.run_configure(disable=True); self.assertEqual(result.exit_code, 0)
        self.assertEqual(stt.projection(self.client.audio), stt.CLEAR)
        self.assertEqual(self.client.audio['tts'], original['tts']); self.assertFalse(self.client.rows)
        self.assertIsNone(self.record()); self.assert_no_foreign_secret(result)
        self.configure(); self.assertEqual(self.record()['routing'], stt.desired(self.selected))

    def test_chat_only_workflow_does_not_read_audio_or_models(self):
        self.run_configure(); self.run_configure(check=True); self.run_configure(disable=True)
        self.assertFalse(any(call.startswith(('audio_', 'model_')) for call in self.client.calls))
        self.assertEqual(state.decode_document(files.read_owned(self.path), self.root, TARGET)['schema_version'], 1)

    def test_legacy_stt_check_is_write_free_and_explicit_mutation_upgrades_only_locally(self):
        self.configure()
        legacy = state.decode_document(files.read_owned(self.path), self.root, TARGET)
        legacy['schema_version'] = 2; del legacy['stt_selection']
        self.path.write_bytes(files.encoded(legacy))
        before = self.path.read_bytes(); stamp = self.path.stat().st_mtime_ns
        self.native.reset_mock(); self.revalidate.reset_mock(); self.client.calls.clear()
        self.assertEqual(self.run_configure(check=True).exit_code, 0)
        self.assertEqual((self.path.read_bytes(), self.path.stat().st_mtime_ns), (before, stamp))
        self.native.assert_not_called(); self.revalidate.assert_not_called()
        self.client.calls.clear()
        result = self.configure()
        current = state.decode_document(files.read_owned(self.path), self.root, TARGET)
        self.assertTrue(result['changed'])
        self.assertEqual(current['schema_version'], 3)
        self.assertEqual(current['stt_selection'], dict(server='local', instance='dictation'))
        self.assertEqual(current['stt'], legacy['stt'])
        self.assertFalse(any(call.endswith(('update', 'create', 'delete')) for call in self.client.calls))

    def test_selection_only_chat_writes_preserve_names_and_disable_failure_retains_intent(self):
        selected = dict(server='local', instance='not-yet-served')
        value = state.with_stt_selection(state.decode_document(None, self.root, TARGET), selected)
        files.atomic_replace(self.path, files.encoded(value), expected=None, deadline=common.Deadline.after())
        self.native.reset_mock(); self.revalidate.reset_mock()
        self.run_configure()
        self.assertEqual(state.decode_document(files.read_owned(self.path), self.root, TARGET)['stt_selection'], selected)
        before = self.path.read_bytes(); stamp = self.path.stat().st_mtime_ns
        self.assertEqual(self.run_configure(check=True).exit_code, 1)
        self.assertEqual((self.path.read_bytes(), self.path.stat().st_mtime_ns), (before, stamp))
        self.client.fail_chat = True
        with self.assertRaises(common.HemError): self.run_configure(disable=True)
        pending = state.decode_document(files.read_owned(self.path), self.root, TARGET)
        self.assertEqual(pending['stt_selection'], selected)
        self.assertEqual(pending['pending']['operation'], 'disable')
        self.assertIsNone(pending['stt'])
        self.client.fail_chat = False
        self.assertEqual(self.run_configure(disable=True).exit_code, 0)
        current = state.decode_document(files.read_owned(self.path), self.root, TARGET)
        self.assertEqual(current['schema_version'], 3)
        self.assertIsNone(current['stt_selection']); self.assertIsNone(current['stt'])
        self.assertEqual(self.native.call_count, 1); self.revalidate.assert_not_called()
        self.assertFalse(any(call.startswith(('audio_', 'model_')) for call in self.client.calls))

    def test_legacy_pending_finalization_keeps_derived_selection_and_original_native_failure(self):
        self.configure()
        legacy = state.decode_document(files.read_owned(self.path), self.root, TARGET)
        legacy['schema_version'] = 2; del legacy['stt_selection']
        legacy['stt']['pending'] = dict(operation='configure_audio', intent=deepcopy(legacy['stt']['routing']),
                                        outcome='unconfirmed')
        self.path.write_bytes(files.encoded(legacy))
        self.native_row['loom']['kind'] = 'text'
        self.client.calls.clear()
        with self.assertRaises(common.HemError) as error: self.configure()
        self.assertEqual(error.exception.reason, 'incompatible')
        self.assertIs(error.exception.accepted, True)
        current = state.decode_document(files.read_owned(self.path), self.root, TARGET)
        self.assertEqual(current['schema_version'], 3)
        self.assertEqual(current['stt_selection'], dict(server='local', instance='dictation'))
        self.assertIsNone(current['stt']['pending'])
        self.assertEqual(current['stt']['routing'], legacy['stt']['routing'])
        self.assertFalse(any(call.endswith(('update', 'create', 'delete')) for call in self.client.calls))

    def test_fixed_entry_one_client_one_mutation_lock_and_no_recursion(self):
        dispatch = importlib.import_module(PACKAGE + '.hem_integrations').configure_integration
        deadline = common.Deadline.after()
        original, publish, lock = hem.configure_openwebui, hem.ensure_endpoints, hem.integration_lock
        held = [False]
        def checked_publish(*args):
            self.assertFalse(held[0])
            return publish(*args)
        @contextmanager
        def checked_lock(*args):
            self.assertFalse(held[0])
            with lock(*args):
                held[0] = True
                try:
                    yield
                finally:
                    held[0] = False
        def display(*args):
            self.assertTrue(held[0])
        factory = Mock(side_effect=self.client.factory)
        def complete(*args, **kwargs):
            return original(*args, client_factory=factory, display=display, **kwargs)
        with patch.dict(os.environ, {'LOOM_SERVE_DIR': str(self.root), 'OPENWEBUI_URL': 'invalid',
                                     'OPENWEBUI_TOKEN': 'wrong-environment-token'}), patch.object(
                hem, 'configure_openwebui', side_effect=complete) as entry, patch.object(
                hem, 'ensure_endpoints', side_effect=checked_publish) as publisher, patch.object(
                hem, 'integration_lock', side_effect=checked_lock) as mutation_lock, patch.object(
                hem, 'passive_snapshot', wraps=hem.passive_snapshot) as snapshot:
            result = dispatch('openwebui', None, deadline, url=TARGET, token='admin-token',
                              stt_server='local', stt_instance='dictation')
        self.assertEqual(result.exit_code, 0)
        self.assertEqual(entry.call_count, 1)
        self.assertEqual(mutation_lock.call_count, 1)
        self.assertEqual(publisher.call_count, 1)
        self.assertEqual(snapshot.call_count, 1)
        factory.assert_called_once_with(TARGET, 'admin-token', deadline)
        for call in ('model_create', 'chat_update', 'audio_update'):
            self.assertIn(call, self.client.calls)
        self.assertFalse(held[0])
        self.assert_no_foreign_secret(result)

    def test_owned_check_reuses_version_without_publication_lock_or_writes(self):
        self.configure()
        before = self.path.read_bytes()
        with patch.object(hem, 'Registry', side_effect=AssertionError('check initialized state')), patch.object(
                hem, 'ensure_endpoints', side_effect=AssertionError('check published')), patch.object(
                hem, 'integration_lock', side_effect=AssertionError('check locked')), patch.object(
                files, 'atomic_replace', side_effect=AssertionError('check wrote')), patch.object(
                self.client, 'version', wraps=self.client.version) as version, patch.object(
                self.client, 'factory', wraps=self.client.factory) as factory:
            result = self.run_configure(check=True)
        self.assertEqual(result.exit_code, 0)
        self.assertEqual(before, self.path.read_bytes())
        self.assertEqual(version.call_count, 1)
        self.assertEqual(factory.call_count, 1)

    def test_audio_mutation_rechecks_deployment_before_chat_effects(self):
        version = self.client.version
        def observed():
            value = version()
            if self.client.rows:
                value['deployment_id'] = 'restarted-during-overlay-create'
            return value
        with patch.object(self.client, 'version', side_effect=observed):
            with self.assertRaises(common.InvalidInput):
                self.configure()
        self.assertIn('model_create', self.client.calls)
        self.assertNotIn('chat_update', self.client.calls)
        self.assertNotIn('audio_update', self.client.calls)
        self.assertEqual(self.record()['overlay']['ownership'], 'created')

    def test_foreign_stt_refuses_before_any_write_even_matching_our_url(self):
        for change in ({'ENGINE': 'openai', 'MODEL': 'foreign'}, {'OPENAI_API_KEY': SECRET},
                       {'WHISPER_MODEL': 'large'}, {'OPENAI_API_BASE_URL': 'https://foreign/v1'},
                       stt.desired(self.selected)):
            with self.subTest(change=change):
                self.client.audio = audio(); self.client.audio['stt'].update(change)
                with self.assertRaises(common.InvalidInput): self.configure()
                self.assertFalse(self.client.rows); self.assertNotIn('audio_update', self.client.calls)
                self.assertNotIn('chat_update', self.client.calls); self.assert_no_foreign_secret()

    def test_foreign_inactive_prerequisite_is_not_adopted_or_deleted(self):
        row = dict(id=self.selected['prefixed_id'], base_model_id=None, is_active=False,
                   write_access=True, meta={'foreign-secret': SECRET})
        self.client.rows[row['id']] = deepcopy(row)
        self.configure(); self.assertEqual(self.record()['overlay']['ownership'], 'external')
        self.run_configure(disable=True)
        self.assertEqual(self.client.rows[row['id']], row); self.assertNotIn('model_delete', self.client.calls)
        self.assert_no_foreign_secret()

    def test_active_custom_or_inaccessible_model_conflicts_before_any_write(self):
        for changes in ({'is_active': True}, {'base_model_id': 'custom'}, {'write_access': False}):
            with self.subTest(changes=changes):
                row = dict(id=self.selected['prefixed_id'], base_model_id=None,is_active=False,write_access=True)
                row.update(changes); self.client.rows[row['id']] = row
                with self.assertRaises(common.InvalidInput): self.configure()
                self.assertNotIn('chat_update', self.client.calls); self.assertNotIn('audio_update', self.client.calls)

    def test_partial_overlay_creation_does_not_own_or_clear_audio(self):
        self.client.fail_chat = True
        with self.assertRaises(common.HemError): self.configure()
        self.assertIsNone(self.record()['routing']); self.assertEqual(self.record()['overlay']['ownership'], 'created')
        self.client.fail_chat = False; foreign = deepcopy(self.client.audio)
        foreign['stt'].update(ENGINE='foreign',MODEL='administrator');self.client.audio=foreign
        self.run_configure(disable=True)
        self.assertEqual(self.client.audio,foreign);self.assertNotIn('audio_update',self.client.calls)

    def test_owned_drift_in_audio_or_complete_model_row_refuses_disable(self):
        self.configure(); before=self.path.read_bytes()
        self.client.audio['stt']['MODEL']='edited'
        with self.assertRaises(common.InvalidInput):self.run_configure(disable=True)
        self.assertEqual(self.path.read_bytes(),before)
        self.client.audio['stt']['MODEL']='dictation'
        self.client.rows[self.selected['prefixed_id']]['name']='edited'
        with self.assertRaises(common.InvalidInput):self.run_configure(disable=True)
        self.assertNotIn('model_delete',self.client.calls)

    def test_committed_audio_error_resolved_by_readback_no_replay(self):
        self.client.audio_error=common.UncertainOutcome('late server event failure')
        self.configure(); self.assertIsNone(self.record()['pending'])
        self.assertEqual(self.client.calls.count('audio_update'),1)

    def test_unconfirmed_audio_is_retained_and_not_replayed_by_check_or_configure(self):
        self.client.audio_commit=False;self.client.audio_error=common.UncertainOutcome('lost ack')
        with self.assertRaises(common.UncertainOutcome):self.configure()
        self.assertEqual(self.record()['pending']['operation'],'configure_audio')
        before=self.path.read_bytes();self.assertEqual(self.run_configure(check=True).exit_code,1)
        self.assertEqual(self.path.read_bytes(),before)
        with self.assertRaises(common.UncertainOutcome):self.configure()
        self.assertEqual(self.client.calls.count('audio_update'),1)
        self.client.audio['stt'].update(stt.desired(self.selected));self.client.audio_error=None
        self.configure();self.assertEqual(self.client.calls.count('audio_update'),1)

    def test_clear_then_uncertain_delete_never_clears_new_foreign_audio(self):
        self.configure();self.client.fail_delete=True
        with self.assertRaises(common.UncertainOutcome):self.run_configure(disable=True)
        self.assertIsNone(self.record()['routing']);self.assertEqual(self.record()['pending']['operation'],'delete_overlay')
        self.assertEqual(state.stt_selection(state.decode_document(files.read_owned(self.path), self.root, TARGET)),
                         dict(server='local', instance='dictation'))
        writes=self.client.calls.count('audio_update');self.client.audio['stt'].update(ENGINE='foreign',MODEL='admin')
        with self.assertRaises(common.UncertainOutcome):self.run_configure(disable=True)
        self.assertEqual(self.client.calls.count('audio_update'),writes)
        del self.client.rows[self.selected['prefixed_id']]
        self.run_configure(disable=True);self.assertEqual(self.client.audio['stt']['ENGINE'],'foreign')
        self.assertIsNone(state.stt_selection(state.decode_document(files.read_owned(self.path), self.root, TARGET)))

    def test_audio_recheck_race_keeps_intent_and_does_not_post(self):
        def race(client):
            if client.audio_reads==2:client.audio['tts']['VOICE']='administrator edit'
        self.client.on_audio_read=race
        with self.assertRaises(common.InvalidInput):self.configure()
        self.assertNotIn('audio_update',self.client.calls);self.assertIsNotNone(self.record()['pending'])

    def test_remote_create_then_local_finalize_failure_retains_intent_and_no_replay(self):
        replace = hem._replace_before_submission
        count = 0
        def fault(*args, **kwargs):
            nonlocal count
            count += 1
            if count == 3:raise common.HemError('disk failure', accepted=False)
            return replace(*args, **kwargs)
        with patch.object(hem, '_replace_before_submission', side_effect=fault):
            with self.assertRaises(common.UncertainOutcome) as error:self.configure()
        self.assertIs(error.exception.accepted, True)
        self.assertEqual(self.record()['pending']['operation'], 'create_overlay')
        self.configure();self.assertEqual(self.client.calls.count('model_create'), 1)

    def test_idempotent_stt_configure_has_no_write(self):
        self.configure();before=self.path.read_bytes();self.client.calls.clear()
        result=self.configure()
        self.assertFalse(result['changed']);self.assertEqual(self.path.read_bytes(),before)
        self.assertFalse(any(call.endswith(('update','create','delete')) for call in self.client.calls))

    def test_custom_upload_restriction_is_not_broadened(self):
        self.client.audio['stt']['ALLOWED_EXTENSIONS']=['wav']
        with self.assertRaises(common.InvalidInput):self.configure()
        self.assertFalse(self.client.rows);self.assertNotIn('audio_update',self.client.calls)

    def test_optional_audio_types_and_unknown_fields_refuse_lossy_merge(self):
        for changes in ({'ALLOWED_EXTENSIONS':None},{'unknown':SECRET},{'MISTRAL_USE_CHAT_COMPLETIONS':1}):
            value=audio();value['stt'].update(changes)
            with self.assertRaises(common.InvalidInput):stt.validate_audio(value)
        for value in ({'OPENAI_PARAMS':None},{'OPENAI_PARAMS':{}},{'REALTIME_TTS_PROMPT_TEMPLATE':''}):
            row=audio();row['tts'].update(value);self.assertIs(stt.validate_audio(row),row)

    def test_core_native_guards_are_preserved_through_public_configure(self):
        self.configure()
        for changes in ({'incarnation': 8}, {'kind': 'text'}, {'registration': 'loading'},
                        {'parameters': {'primary': {'active': False}, 'image': {'active': True}}},
                        {'status': {'admission': 'drained'}}):
            with self.subTest(changes=changes):
                original = deepcopy(self.native_row['loom'])
                self.native_row['loom'].update(changes)
                if 'incarnation' in changes or 'kind' in changes:
                    with self.assertRaises(common.HemError):self.configure()
                else:
                    result = self.configure()
                    self.assertEqual(result['stt']['configuration'], 'deferred')
                    self.assertEqual(result['stt']['target'], self.selected)
                self.native_row['loom'] = original

    def test_original_target_is_resolved_outside_integration_and_revalidated_inside(self):
        original_lock = hem.integration_lock
        held = [False]
        @contextmanager
        def lock(*args):
            with original_lock(*args):
                held[0] = True
                try:yield
                finally:held[0] = False
        resolve, revalidate = core.resolve_operation_target, core.revalidate_operation_target
        selected = []
        def initial(registry, server, instance, operation, deadline):
            self.assertFalse(held[0]); self.assertEqual(operation, 'transcription')
            value = resolve(registry, server, instance, operation, deadline)
            selected.append((registry, value, deadline)); return value
        def guard(registry, target, deadline):
            self.assertTrue(held[0])
            self.assertIs(registry, selected[0][0]); self.assertIs(target, selected[0][1])
            self.assertIs(deadline, selected[0][2])
            return revalidate(registry, target, deadline)
        with patch.object(hem, 'integration_lock', side_effect=lock), \
                patch.object(core, 'resolve_operation_target', side_effect=initial) as first, \
                patch.object(core, 'revalidate_operation_target', side_effect=guard) as guards:
            self.configure()
        self.assertEqual(first.call_count, 1); self.assertEqual(guards.call_count, 4)

    def test_tracked_replacement_during_integration_wait_refuses_before_binding(self):
        original = hem.integration_lock
        @contextmanager
        def wait(*args):
            row = self.registry.get_instance('local', 'launch', 'dictation')
            self.registry.update_instance('local', 'launch', 'dictation', {'incarnation': 8}, expected_revision=row['revision'])
            self.native_row['loom']['incarnation'] = 8
            with original(*args):yield
        with patch.object(hem, 'integration_lock', side_effect=wait):
            with self.assertRaises(common.UncertainOutcome):self.configure()
        self.assertFalse(any(call.endswith(('create', 'update')) for call in self.client.calls))

    def test_native_replacement_during_immediate_picker_read_refuses_create(self):
        model = self.client.model
        reads = [0]
        def picker(name):
            reads[0] += 1
            if reads[0] == 2:self.native_row['loom']['incarnation'] = 8
            return model(name)
        with patch.object(self.client, 'model', side_effect=picker):
            with self.assertRaises(common.UncertainOutcome):self.configure()
        self.assertNotIn('model_create', self.client.calls)
        self.assertEqual(self.record()['pending']['operation'], 'create_overlay')

    def test_native_replacement_during_immediate_audio_read_preserves_confirmed_effects(self):
        def race(client):
            if client.audio_reads == 2:self.native_row['loom']['incarnation'] = 8
        self.client.on_audio_read = race
        with self.assertRaises(common.UncertainOutcome) as error:self.configure()
        self.assertIs(error.exception.accepted, True)
        self.assertIn('overlay', str(error.exception)); self.assertIn('chat', str(error.exception))
        self.assertIn('not submitted', str(error.exception))
        self.assertNotIn('audio_update', self.client.calls)
        self.assertEqual(self.record()['pending']['operation'], 'configure_audio')
        self.assertEqual(self.record()['overlay']['ownership'], 'created')

    def test_status_reread_refuses_tracked_replacement_through_public_entry(self):
        def race(fixture):
            row = self.registry.get_instance('local', 'launch', 'dictation')
            self.registry.update_instance('local', 'launch', 'dictation', {'incarnation': 8}, expected_revision=row['revision'])
        self.on_native_read = race
        with self.assertRaises(common.UncertainOutcome):self.configure()
        self.assertFalse(any(call.endswith(('create', 'update')) for call in self.client.calls))

    def external_row(self):
        self.client.rows[self.selected['prefixed_id']] = dict(id=self.selected['prefixed_id'],
            base_model_id=None, is_active=False, write_access=True)

    def test_confirmed_chat_without_created_overlay_keeps_accepted_history(self):
        self.external_row()
        update = self.client.update
        def changed(body):
            value = update(body)
            self.native_row['loom']['incarnation'] = 8
            return value
        with patch.object(self.client, 'update', side_effect=changed):
            with self.assertRaises(common.UncertainOutcome) as error:self.configure()
        self.assertIs(error.exception.accepted, True)
        self.assertIn('chat', str(error.exception)); self.assertNotIn('overlay creation', str(error.exception))
        self.assertIn('not submitted', str(error.exception)); self.assertNotIn('audio_update', self.client.calls)
        self.assertEqual(self.record()['overlay']['ownership'], 'external')
        owned = state.decode_document(files.read_owned(self.path), self.root, TARGET)
        self.assertTrue(owned['connections'])

    def test_local_external_binding_alone_does_not_claim_remote_acceptance(self):
        self.run_configure()  # Existing chat, so the next chat participant is a no-op.
        self.external_row()
        config = self.client.config
        def changed():
            if self.record() is not None:self.native_row['loom']['incarnation'] = 8
            return config()
        with patch.object(self.client, 'config', side_effect=changed):
            with self.assertRaises(common.UncertainOutcome) as error:self.configure()
        self.assertIsNone(error.exception.accepted)
        self.assertNotIn('audio_update', self.client.calls)
        self.assertEqual(self.record()['overlay']['ownership'], 'external')

    def pending_audio(self):
        self.client.audio_commit = False
        self.client.audio_error = common.UncertainOutcome('lost audio response')
        with self.assertRaises(common.UncertainOutcome):self.configure()
        self.client.audio_error = None
        self.client.calls.clear()

    def test_initial_native_failure_after_pending_audio_finalization_still_surfaces(self):
        self.pending_audio()
        self.client.audio['stt'].update(stt.desired(self.selected))
        self.native_row['loom']['kind'] = 'text'
        with self.assertRaises(common.HemError) as error:self.configure()
        self.assertEqual(error.exception.reason, 'incompatible'); self.assertIs(error.exception.accepted, True)
        self.assertIsNone(self.record()['pending']); self.assertEqual(self.record()['routing'], stt.desired(self.selected))
        self.assertNotIn('audio_update', self.client.calls)

    def test_initial_native_failure_does_not_override_unresolved_pending(self):
        self.pending_audio(); self.native_row['loom']['registration'] = 'loading'
        before = self.path.read_bytes()
        with self.assertRaises(common.UncertainOutcome):self.configure()
        self.assertEqual(self.path.read_bytes(), before); self.assertNotIn('audio_update', self.client.calls)

    def test_initial_native_failure_idempotent_and_no_receipt_never_succeeds(self):
        self.native_row['loom']['kind'] = 'text'
        with self.assertRaises(common.HemError):self.configure()
        self.assertFalse(any(call.endswith(('create', 'update')) for call in self.client.calls))
        self.native_row['loom']['kind'] = 'recognition'; self.configure()
        before = self.path.read_bytes(); self.client.calls.clear()
        self.native_row['loom']['kind'] = 'text'
        with self.assertRaises(common.HemError):self.configure()
        self.assertEqual(self.path.read_bytes(), before)
        self.assertFalse(any(call.endswith(('create', 'update')) for call in self.client.calls))

    def test_external_binding_guard_runs_before_first_local_attribution(self):
        self.external_row()
        model = self.client.model
        def changed(name):
            self.native_row['loom']['incarnation'] = 8
            return model(name)
        with patch.object(self.client, 'model', side_effect=changed):
            with self.assertRaises(common.UncertainOutcome):self.configure()
        self.assertIsNone(self.record())
        self.assertNotIn('chat_update', self.client.calls); self.assertNotIn('audio_update', self.client.calls)

    def test_confirmed_overlay_without_new_chat_keeps_exact_guard_metadata(self):
        self.run_configure(); self.client.calls.clear()
        revalidate = core.revalidate_operation_target
        def guard(*args):
            revalidate(*args)
            if 'model_create' in self.client.calls:
                raise common.HemError('native operation incompatible', status=409, reason='incompatible')
        with patch.object(core, 'revalidate_operation_target', side_effect=guard):
            with self.assertRaises(common.HemError) as error:self.configure()
        self.assertIs(type(error.exception), common.HemError)
        self.assertEqual((error.exception.reason, error.exception.status, error.exception.exit_code), ('incompatible', 409, 1))
        self.assertIs(error.exception.accepted, True)
        self.assertIn('overlay creation', str(error.exception)); self.assertNotIn('chat update', str(error.exception))
        self.assertIn('not submitted', str(error.exception))
        self.assertNotIn('audio_update', self.client.calls)
        self.assertEqual(self.record()['overlay']['ownership'], 'created')

    def test_pending_created_overlay_can_finalize_before_initial_failure_surfaces(self):
        replace = hem._replace_before_submission
        count = [0]
        def fault(*args, **kwargs):
            count[0] += 1
            if count[0] == 3:raise common.HemError('disk failure')
            return replace(*args, **kwargs)
        with patch.object(hem, '_replace_before_submission', side_effect=fault):
            with self.assertRaises(common.UncertainOutcome):self.configure()
        self.assertEqual(self.record()['pending']['operation'], 'create_overlay')
        self.native_row['loom']['kind'] = 'text'; self.client.calls.clear()
        with self.assertRaises(common.HemError) as error:self.configure()
        self.assertEqual(error.exception.reason, 'incompatible'); self.assertIs(error.exception.accepted, True)
        self.assertIn('prior overlay creation', str(error.exception))
        self.assertIsNone(self.record()['pending']); self.assertEqual(self.record()['overlay']['ownership'], 'created')
        self.assertNotIn('model_create', self.client.calls); self.assertNotIn('audio_update', self.client.calls)

    def test_initial_deadline_expiry_retains_unresolved_pending_without_restart(self):
        self.pending_audio(); before = self.path.read_bytes()
        clock = [0]
        deadline = common.Deadline(10, clock=lambda: clock[0])
        self.on_native_read = lambda fixture: clock.__setitem__(0, 11)
        with self.assertRaises(common.UncertainOutcome):
            hem.configure_openwebui(self.root, TARGET, 'admin-token', deadline,
                client_factory=self.client.factory, stt_server='local', stt_instance='dictation', display=None)
        self.assertEqual(self.path.read_bytes(), before)
        self.assertIs(self.processes.client.call_args.args[2], deadline)
        self.assertNotIn('audio_update', self.client.calls)

    def assert_recovery_deadline(self, *, unavailable):
        self.pending_audio(); before = self.path.read_bytes()
        if unavailable:self.native_row['loom']['registration'] = 'loading'
        clock = [0]; deadline = common.Deadline(10, clock=lambda: clock[0])
        version = self.client.version
        def expire():
            value = version(); clock[0] = 11; return value
        with patch.object(self.client, 'version', side_effect=expire):
            with self.assertRaises(common.UncertainOutcome):
                hem.configure_openwebui(self.root, TARGET, 'admin-token', deadline,
                    client_factory=self.client.factory, stt_server='local', stt_instance='dictation', display=None)
        self.assertEqual(self.path.read_bytes(), before); self.assertNotIn('audio_update', self.client.calls)

    def test_recovery_deadline_expiry_preserves_pending_when_native_is_unavailable(self):
        self.assert_recovery_deadline(unavailable=True)

    def test_recovery_deadline_expiry_preserves_pending_with_healthy_initial_target(self):
        self.assert_recovery_deadline(unavailable=False)

    def test_native_loss_after_confirmed_audio_does_not_veto_finalization(self):
        update = self.client.update_audio
        def changed(body):
            value = update(body); self.native_row['loom']['incarnation'] = 8; return value
        with patch.object(self.client, 'update_audio', side_effect=changed):
            result = self.configure()
        self.assertEqual(result.exit_code, 0)
        self.assertIsNone(self.record()['pending']); self.assertEqual(self.record()['routing'], stt.desired(self.selected))

    def test_passive_and_disable_stay_native_free_but_saved_selection_configure_retries(self):
        self.configure(); self.native.reset_mock(); self.revalidate.reset_mock()
        self.native_row['loom']['registration'] = 'loading'
        self.run_configure(check=True, stt_server='local', stt_instance='dictation')
        self.run_configure(disable=True)
        self.native.assert_not_called(); self.revalidate.assert_not_called()

    def pending_chat(self, *, completed=False):
        submitted = [False]
        config = self.client.config
        def lost_update(body):
            self.client.calls.append('chat_update'); submitted[0] = True
            if completed:self.client.remote = deepcopy(body)
            raise common.UncertainOutcome('lost chat response')
        def lost_readback():
            if submitted[0]:raise common.HemError('chat readback unavailable')
            return config()
        with patch.object(self.client, 'update', side_effect=lost_update), \
                patch.object(self.client, 'config', side_effect=lost_readback):
            with self.assertRaises(common.UncertainOutcome):self.configure()
        owned = state.decode_document(files.read_owned(self.path), self.root, TARGET)
        self.assertIsNotNone(owned['pending']); self.assertIsNone(owned['stt']['pending'])
        self.native_row['loom']['registration'] = 'loading'; self.client.calls.clear()
        return owned

    def test_initial_failure_observes_unresolved_chat_pending_without_replay(self):
        self.pending_chat(); before = self.path.read_bytes()
        with patch.object(self.client, 'version', wraps=self.client.version) as version, \
                patch.object(self.client, 'config', wraps=self.client.config) as config:
            with self.assertRaises(common.UncertainOutcome) as error:self.configure()
        self.assertIsNone(error.exception.accepted)
        self.assertEqual(version.call_count, 1); self.assertEqual(config.call_count, 1)
        self.assertEqual(self.path.read_bytes(), before)
        self.assertFalse(any(call.endswith(('create', 'update', 'delete')) for call in self.client.calls))

    def test_initial_failure_finalizes_only_observed_chat_not_new_desired_work(self):
        previous = self.pending_chat(completed=True)
        self.native_row['loom']['kind'] = 'text'
        self.registry.create_server('later', saved_settings=dict(device='task', listen='127.0.0.1:18082'))
        with self.assertRaises(common.HemError) as error:self.configure()
        self.assertIs(type(error.exception), common.HemError)
        self.assertEqual(error.exception.reason, 'incompatible'); self.assertIs(error.exception.accepted, True)
        self.assertIn('prior chat update', str(error.exception)); self.assertIn('not submitted', str(error.exception))
        owned = state.decode_document(files.read_owned(self.path), self.root, TARGET)
        self.assertIsNone(owned['pending'])
        self.assertEqual(owned['connections'], previous['pending']['after'])
        self.assertEqual(owned['stt'], previous['stt'])
        self.assertFalse(any(call.endswith(('create', 'update', 'delete')) for call in self.client.calls))
        # Existing applied receipt still receives its required read-only drift
        # checks before names can be retained with the incompatible error.

    def test_initial_failure_chat_pending_deadline_preserves_uncertainty(self):
        self.pending_chat(); before = self.path.read_bytes()
        clock = [0]; deadline = common.Deadline(10, clock=lambda: clock[0])
        version = self.client.version
        def expired_version():
            value = version(); clock[0] = 11; return value
        with patch.object(self.client, 'version', side_effect=expired_version) as version_read:
            with self.assertRaises(common.UncertainOutcome):
                hem.configure_openwebui(self.root, TARGET, 'admin-token', deadline,
                    client_factory=self.client.factory, stt_server='local', stt_instance='dictation', display=None)
        self.assertEqual(version_read.call_count, 1); self.assertIs(self.processes.client.call_args.args[2], deadline)
        self.assertEqual(self.path.read_bytes(), before)
        self.assertFalse(any(call.endswith(('create', 'update', 'delete')) for call in self.client.calls))

    def assert_deferred_stt_inspection_uncertain(self, method):
        self.pending_audio(); before = self.path.read_bytes()
        self.native_row['loom']['registration'] = 'loading'
        failure = common.HemError('management offline', status=503)
        with patch.object(self.client, method, side_effect=failure):
            with self.assertRaises(common.UncertainOutcome) as error:self.configure()
        self.assertIsNone(error.exception.accepted)
        self.assertEqual(self.path.read_bytes(), before)
        self.assertFalse(any(call.endswith(('create', 'update', 'delete')) for call in self.client.calls))
        # Qualified and selectorless/passive/disable paths keep their existing
        # management category even with the same retained STT intent.
        self.native_row['loom']['registration'] = 'ready'
        for kwargs in ({'stt_server': 'local', 'stt_instance': 'dictation'}, {}, {'check': True}, {'disable': True}):
            with patch.object(self.client, method, side_effect=failure):
                with self.assertRaises(common.HemError) as ordinary:self.run_configure(**kwargs)
            self.assertIs(ordinary.exception, failure)
            self.assertEqual((ordinary.exception.exit_code, ordinary.exception.status), (1, 503))
            self.assertEqual(self.path.read_bytes(), before)

    def test_deferred_stt_pending_version_failure_is_uncertain(self):
        self.assert_deferred_stt_inspection_uncertain('version')

    def test_deferred_stt_pending_audio_failure_is_uncertain(self):
        self.assert_deferred_stt_inspection_uncertain('audio_config')

    def test_deferred_stt_pending_picker_failure_is_uncertain(self):
        self.assert_deferred_stt_inspection_uncertain('model')

    def test_absent_selection_check_still_observes_webui_version_without_native_io(self):
        with patch.object(self.client, 'version', wraps=self.client.version) as version:
            result = self.run_configure(check=True, stt_server='local', stt_instance='dictation')
        self.assertEqual(result.exit_code, 1); self.assertEqual(version.call_count, 1)
        self.native.assert_not_called(); self.revalidate.assert_not_called()
        self.assertFalse(any(call.startswith(('audio_', 'model_')) for call in self.client.calls))

    def test_pair_and_disable_selectors_reject_before_http(self):
        for kwargs in ({'stt_server':'local'},{'stt_instance':'dictation'},
                       {'stt_server':'local','stt_instance':'dictation','disable':True}):
            with self.assertRaises(common.InvalidInput):self.run_configure(**kwargs)
            self.assertFalse(self.client.calls)


if __name__=='__main__':unittest.main()
