"""STT ownership boundaries with real private files and existing chat coordinator."""
from copy import deepcopy
from contextlib import nullcontext
import importlib
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
        self.native_patch = patch.object(stt, 'native_target', return_value=self.selected)
        self.native = self.native_patch.start()
        self.addCleanup(patch.stopall)
        self.path = state.ownership_path(self.root, TARGET)

    def run_configure(self, **kwargs):
        return hem.configure_openwebui(self.root, TARGET, 'admin-token', common.Deadline.after(),
            client_factory=self.client.factory, display=lambda *args: self.events.append(deepcopy(args)), **kwargs)

    def configure(self): return self.run_configure(stt_server='local', stt_instance='dictation')
    def record(self): return state.decode_document(files.read_owned(self.path), self.root, TARGET)['stt']
    def assert_no_foreign_secret(self, result=None):
        self.assertNotIn(SECRET, repr(self.events) + repr(result))
        for path in self.root.rglob('*.json'): self.assertNotIn(SECRET, path.read_text())

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
        writes=self.client.calls.count('audio_update');self.client.audio['stt'].update(ENGINE='foreign',MODEL='admin')
        with self.assertRaises(common.UncertainOutcome):self.run_configure(disable=True)
        self.assertEqual(self.client.calls.count('audio_update'),writes)
        del self.client.rows[self.selected['prefixed_id']]
        self.run_configure(disable=True);self.assertEqual(self.client.audio['stt']['ENGINE'],'foreign')

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
            if count == 2:raise common.HemError('disk failure', accepted=False)
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

    def test_native_target_requires_matching_build_incarnation_primary_and_recognition(self):
        self.native_patch.stop()
        process = importlib.import_module(PACKAGE + '.hem_process')
        registry = Mock()
        registry.get_server.return_value = dict(id='local', phase='running', pending=None, launch_token='launch', endpoint='http://127.0.0.1:18080')
        registry.list_instances.return_value = [dict(name='dictation', launch_token='launch', incarnation=7, pending=None)]
        row = dict(id='dictation', input_modalities=['audio'], output_modalities=['text'],
                   loom=dict(incarnation=7, kind='recognition', endpoint='/v1/audio/transcriptions',
                             registration='ready', parameter_operation='idle', parameters={'primary': {'active': True}},
                             status={'admission': 'open'}))
        manager = Mock(); manager.pin.side_effect = lambda _: nullcontext(object())
        manager.client.return_value.status.return_value = {'object':'list','server':{},'data': [row]}
        with patch.object(stt, 'Registry', return_value=registry), patch.object(process, 'ProcessManager', return_value=manager), \
                patch.object(process, 'require_build') as build:
            self.assertEqual(stt.native_target(self.root, 'local', 'dictation', common.Deadline.after()), self.selected)
            build.assert_called_once()
            for changes in ({'incarnation': 8}, {'kind': 'text'}, {'registration': 'loading'},
                            {'parameters': {'primary': {'active': False}, 'image': {'active': True}}},
                            {'status': {'admission': 'drained'}}):
                manager.client.return_value.status.return_value = {'object':'list','server':{},'data': [dict(row, loom=dict(row['loom'], **changes))]}
                with self.assertRaises(common.InvalidInput):stt.native_target(self.root, 'local', 'dictation', common.Deadline.after())

    def test_pair_and_disable_selectors_reject_before_http(self):
        for kwargs in ({'stt_server':'local'},{'stt_instance':'dictation'},
                       {'stt_server':'local','stt_instance':'dictation','disable':True}):
            with self.assertRaises(common.InvalidInput):self.run_configure(**kwargs)
            self.assertFalse(self.client.calls)


if __name__=='__main__':unittest.main()
