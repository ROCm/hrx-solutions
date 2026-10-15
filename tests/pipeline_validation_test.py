from contextlib import redirect_stdout, redirect_stderr
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from binding.python import _launcher, pipeline_validation
from binding.python.model_store import ModelFiles
from binding.python.pipeline_validation import validate_model
from tests.gguf_fixtures import profile, write_matching_gguf


class PipelineValidationTest(unittest.TestCase):
    def select_08b_contract(self):
        digest = self.contract['tokenizer_canonical_sha256']
        self.contract.update(profile('qwen_08b'))
        self.contract['tokenizer_canonical_sha256'] = digest

    def setUp(self):
        temporary=tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root=Path(temporary.name)
        self.directory=self.root/'test-model'
        self.directory.mkdir()
        self.weights=self.directory/'model.gguf'
        self.tokenizer=self.directory/'tokenizer.json'
        self.contract=profile()
        data={'model':{'type':'BPE','vocab':{'a':0}}}
        self.contract['tokenizer_canonical_sha256']=hashlib.sha256(json.dumps(data,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
        self.tokenizer.write_text(json.dumps(data))
        (self.directory/'manifest.json').write_text(json.dumps({'model_id':'test-model'}))
        self.profile_patch=patch('binding.python.pipeline_validation.load_profile',return_value=self.contract)
        self.profile_patch.start()
        self.addCleanup(self.profile_patch.stop)



    def test_errors_and_unknown_pipeline(self):
        files=ModelFiles(self.weights,self.tokenizer)
        self.assertIn('Unknown pipeline',validate_model(files,'unknown').message)
        self.assertIn('missing:',validate_model(files,'qwen').message)
        self.weights.write_bytes(b'GGUF')
        self.assertIn('Truncated',validate_model(files,'qwen').message)
        write_matching_gguf(self.weights)
        self.tokenizer.write_text('invalid')
        self.assertIn('not valid UTF-8 JSON',validate_model(files,'qwen').message)

    def test_symlink_without_extension_on_target_and_missing_mtp(self):
        blob=self.root/'extensionless'
        write_matching_gguf(blob,missing=['blk.64.nextn.eh_proj.weight'])
        self.weights.symlink_to(blob)
        result=validate_model(ModelFiles(self.weights,self.tokenizer),'qwen')
        self.assertFalse(result.compatible)
        self.assertIn('MTP: missing required tensor',result.message)

    def test_catalog_customization(self):
        source=Path(__file__).resolve().parents[1]/'loom-pipeline/llm/qwen/qwen_27b'
        with patch.object(Path,'home',return_value=self.root):
            configured=pipeline_validation.catalog_for_model('qwen',source,'test-model')
            self.assertIn('746573742d6d6f64656c',(configured/'control.loom').read_text())
            self.assertTrue((configured/'compatibility.json').is_file())


    def test_08b_catalog_cache_preserves_other_architecture_and_dependencies(self):
        source = pipeline_validation.resolve_catalog('qwen-08b')
        self.assertEqual(source.name, 'qwen_08b')
        self.assertEqual(pipeline_validation.resolve_catalog('qwen').name, 'qwen_27b')
        original = (source/'control.loom').read_bytes()
        with patch.object(Path, 'home', return_value=self.root):
            configured = pipeline_validation.catalog_for_model('qwen-08b', source, 'small-qwen')
            self.assertIn('736d616c6c2d7177656e', (configured/'control.loom').read_text())
            self.assertEqual((configured.parent/'qwen_27b/control.loom').read_bytes(), original)
            for entry in (configured/'sources.txt').read_text().splitlines():
                if entry:
                    self.assertTrue((configured/entry).is_file(), entry)
            self.assertEqual(pipeline_validation.catalog_for_model('qwen-08b', source, 'small-qwen'), configured)
        self.assertEqual((source/'control.loom').read_bytes(), original)


    def test_deployment_capacity_and_native_document_passthrough(self):
        self.select_08b_contract()
        selected = pipeline_validation.resolve_catalog('qwen-08b')
        files = pipeline_validation.ModelFiles(self.root/'model.gguf', self.root/'tokenizer.json')
        with self.assertRaisesRegex(ValueError, 'positive --pool-capacity'):
            pipeline_validation.deployment_for_model(selected, files, 'small', pool_capacity=0)
        with self.assertRaisesRegex(ValueError, 'does not support MTP'):
            pipeline_validation.deployment_for_model(selected, files, 'small', draft_depth=1)
        package = self.root/'native-package'
        binary = package/'_native/bin/loom-serve'
        binary.parent.mkdir(parents=True)
        binary.write_text('test executable')
        native_args = ['--device=amdgpu', '--model=one.json', '--model=two.json', '--listen=127.0.0.1:19000']
        with patch.object(_launcher, '__file__', str(package/'_launcher.py')), patch('sys.argv', ['loom-serve', *native_args]), patch.object(_launcher, 'native_environment', return_value={}), patch.object(_launcher.os, 'execve', side_effect=SystemExit(0)) as execute:
            with self.assertRaises(SystemExit):
                _launcher.exec_loom_serve()
            self.assertEqual(execute.call_args.args[1], [str(binary), *native_args])

    def test_installed_catalog_selection_and_low_level_mtp_rejection(self):
        package = self.root/'installed/hrx_solutions'
        for architecture in ('qwen_27b', 'qwen_08b'):
            catalog = package/'loom-pipeline/llm/qwen'/architecture
            catalog.mkdir(parents=True)
            (catalog/'compatibility.json').write_text(json.dumps(profile(architecture)))
        self.select_08b_contract()
        with patch.object(pipeline_validation, '__file__', str(package/'pipeline_validation.py')):
            selected = pipeline_validation.resolve_catalog('qwen-08b')
            self.assertEqual(selected, package/'loom-pipeline/llm/qwen/qwen_08b')
            self.assertEqual(pipeline_validation.resolve_catalog('qwen').name, 'qwen_27b')
        with patch('sys.argv', ['loom-serve', f'--model={selected}', '--mtp']), patch.object(_launcher.os, 'execve') as execute:
            with self.assertRaisesRegex(ValueError, 'does not support MTP'):
                _launcher.exec_loom_serve()
            execute.assert_not_called()


if __name__ == '__main__':
    unittest.main()
