"""Pocket uses the existing audio owner while preserving foreign configuration."""
from copy import deepcopy
import unittest
from unittest.mock import patch
from tests import hem_openwebui_stt_test as existing
from binding.python.integrations.openwebui import hem_openwebui_tts as tts
from binding.python.hem_common import InvalidInput, UncertainOutcome

class TTSTest(unittest.TestCase):
    run_configure = existing.STTTest.run_configure
    assert_no_foreign_secret = existing.STTTest.assert_no_foreign_secret
    def setUp(self):
        existing.STTTest.setUp(self)
        self.native_row.update(input_modalities=['text'],output_modalities=['audio'])
        self.native_row['loom'].update(kind='speech',endpoint='/v1/audio/speech')
        self.client.audio['tts'].update(ENGINE='',OPENAI_API_BASE_URL='https://api.openai.com/v1',
            OPENAI_API_KEY='',MODEL='tts-1',VOICE='alloy',OPENAI_PARAMS={},CACHE_ENABLED=True)
        self.client.audio['stt'].update(ENGINE='openai',OPENAI_API_BASE_URL='https://foreign.test/v1',
            OPENAI_API_KEY=existing.SECRET,MODEL='foreign-transcription')
    def configure(self):
        return self.run_configure(tts_server='local',tts_instance='dictation')
    def test_configure_check_selected_disconnect_preserve_other_audio_chat(self):
        before=deepcopy(self.client.audio)
        result=self.configure();self.assertEqual(result.exit_code,0)
        self.assertEqual(result['tts']['configuration'],'configured')
        self.assertEqual(self.client.audio['stt'],before['stt'])
        self.assertEqual(self.client.audio['realtime'],before['realtime'])
        self.assertEqual(tts.projection(self.client.audio),tts.desired(self.selected))
        calls=list(self.client.calls)
        self.assertEqual(self.run_configure(check=True)['tts']['configuration'],'configured')
        self.assertNotIn('audio_update',self.client.calls[len(calls):])
        connected=deepcopy(self.client.remote)
        result=self.run_configure(disable_tts=True)
        self.assertEqual(result['tts']['disconnect'],'browser-speech')
        self.assertEqual(self.client.remote,connected)
        self.assertEqual(self.client.audio['stt'],before['stt'])
        self.assertEqual(tts.projection(self.client.audio),tts.CLEAR)
        self.assert_no_foreign_secret(result)
        self.assertEqual(self.configure()['tts']['configuration'],'configured')
    def test_unsupported_cache_and_foreign_route_refuse_without_effect(self):
        del self.client.audio['tts']['CACHE_ENABLED']
        with self.assertRaisesRegex(InvalidInput,'CACHE_ENABLED'):self.configure()
        self.assertNotIn('audio_update',self.client.calls)
        self.assertNotIn('model_create',self.client.calls)
        self.client.audio['tts']['CACHE_ENABLED']=True
        self.client.audio['tts']['ENGINE']='elevenlabs'
        with self.assertRaisesRegex(InvalidInput,'Foreign'):self.configure()
    def test_drift_and_uncertain_update_retained_without_replay(self):
        self.client.audio_commit=False
        with self.assertRaises(UncertainOutcome):self.configure()
        updates=self.client.calls.count('audio_update')
        with self.assertRaises(UncertainOutcome):self.configure()
        self.assertEqual(updates,self.client.calls.count('audio_update'))
    def test_owned_tts_drift_refuses_disconnect(self):
        self.configure();self.client.audio['tts']['OPENAI_PARAMS']['speed']=2
        with self.assertRaisesRegex(InvalidInput,'drift'):self.run_configure(disable_tts=True)
    def test_pending_tts_deadline_preserves_uncertain_outcome(self):
        self.client.audio_commit=False
        with self.assertRaises(UncertainOutcome):self.configure()
        before=self.path.read_bytes()
        with patch.object(existing.hem,'passive_snapshot',side_effect=existing.common.DeadlineExpired('expired')):
            with self.assertRaises(UncertainOutcome):self.configure()
        self.assertEqual(self.path.read_bytes(),before)
    def test_selector_pair_and_conflicting_disconnect_rejected(self):
        for kwargs in (dict(tts_server='local'),dict(tts_instance='dictation'),
            dict(tts_server='local',tts_instance='dictation',disable_tts=True),dict(disable=True,disable_tts=True)):
            with self.subTest(kwargs=kwargs),self.assertRaises(InvalidInput):self.run_configure(**kwargs)
