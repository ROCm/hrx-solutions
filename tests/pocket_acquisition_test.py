"""Pocket acquisition uses the common stage and retries, with no runtime WAV."""
import json
from unittest.mock import patch
from tests.recipe_rewrite_acquisition_test import AcquisitionDraftTest
from binding.python.hem_common import HemError
from binding.python.models import hem_models as models, model_download
from binding.python.models.preparation import pocket

class PocketAcquisitionTest(AcquisitionDraftTest):
    def test_pocket_prepare_retry_and_preparation_only_input(self):
        doc=self.documents['pocket-tts-alba']
        self.cache(doc)
        wav=model_download._cache_path(doc['preparation_inputs']['alba.wav'])
        wav.parent.mkdir(parents=True); wav.write_bytes(b'voice')
        calls=[]
        def prepare(inputs, attempt, deadline):
            self.assertEqual(set(inputs), {'model.safetensors','tokenizer.json','alba.wav'})
            self.assertEqual(inputs['alba.wav'],wav)
            calls.append(attempt)
            if len(calls)==1:
                (attempt/'partial').write_bytes(b'incomplete')
                raise HemError('interrupted preparation')
            out=attempt/'output';out.mkdir()
            (out/'rotary.safetensors').write_bytes(b'rotary')
            (out/'voices.safetensors').write_bytes(b'voices')
            (out/'model.json').write_text(json.dumps({'definition':{'options':dict(context_capacity=512,rotary_capacity=4096,chunk_tokens=50,voices=[dict(id='alba',frames=126)])}}))
            return {n:out/n for n in ('rotary.safetensors','voices.safetensors','model.json')}
        with patch.object(pocket,'preflight'), patch.object(pocket,'prepare',side_effect=prepare):
            with self.assertRaisesRegex(HemError,'interrupted'):
                self.download(doc,'pocket')
            self.assertIsNotNone(self.registry.get_model('pocket')['pending'])
            self.assertEqual(self.download(doc,'pocket')['state'],'ready')
            self.assertNotEqual(calls[0],calls[1])
            record=self.registry.get_model('pocket')
            self.assertNotIn('alba.wav',[a['logical_path'] for a in record['assets']])
            wav.unlink()
            with patch.object(pocket,'preflight',side_effect=AssertionError('no repeated prep')):
                self.assertEqual(self.download(doc,'pocket')['state'],'ready')
    def test_pocket_offline_missing_wav(self):
        doc=self.documents['pocket-tts-alba'];self.cache(doc)
        with patch.object(pocket,'preflight'),patch.object(pocket,'prepare',side_effect=AssertionError('no helper')):
            with self.assertRaisesRegex(HemError,'Offline cache is missing kyutai/tts-voices'):
                self.download(doc,'pocket')
