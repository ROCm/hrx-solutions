from contextlib import redirect_stdout, redirect_stderr
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from binding.python import model_store, pipeline_validation, setup_cli
from binding.python.pipeline_validation import ModelFiles, validate_model
from tests.gguf_fixtures import profile, write_matching_gguf


class PipelineValidationTest(unittest.TestCase):
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

    def test_cli_success_cpu_only_and_success_manifest_record(self):
        write_matching_gguf(self.weights)
        with patch('subprocess.run',side_effect=AssertionError('validation must not execute tools')), patch('binding.python._launcher.native_environment',side_effect=AssertionError('validation must not initialize native environment')):
            with redirect_stdout(io.StringIO()) as output:
                code=setup_cli.main(['models','validate','test-model','--pipeline','qwen','--model-root',str(self.root)])
        self.assertEqual(code,0)
        self.assertIn('64 base layers and MTP',output.getvalue())
        self.assertEqual(model_store.read_manifest('test-model',self.root)['pipeline'],'qwen')

    def test_failure_preserves_manifest(self):
        write_matching_gguf(self.weights,changes={'blk.3.attn_v.weight':{'type':'Q6_K'}})
        manifest=self.directory/'manifest.json'
        original=manifest.read_bytes()
        with redirect_stdout(io.StringIO()) as output:
            code=setup_cli.main(['models','validate','test-model','--pipeline','qwen','--model-root',str(self.root)])
        self.assertEqual(code,1)
        self.assertIn('expected Q8_0, found Q6_K',output.getvalue())
        self.assertEqual(manifest.read_bytes(),original)

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

    def test_removed_checker_flags_and_catalog_customization(self):
        for flag,value in [('--checker','binary'),('--timeout','1')]:
            with self.subTest(flag=flag),redirect_stderr(io.StringIO()),self.assertRaises(SystemExit) as caught:
                setup_cli.build_parser().parse_args(['models','validate','test-model','--pipeline','qwen',flag,value])
            self.assertEqual(caught.exception.code,2)
        source=Path(__file__).resolve().parents[1]/'loom-pipeline/llm/qwen'
        with patch.object(Path,'home',return_value=self.root):
            configured=pipeline_validation.catalog_for_model('qwen',source,'test-model')
            self.assertIn('746573742d6d6f64656c',(configured/'control.loom').read_text())
            self.assertTrue((configured/'compatibility.json').is_file())


if __name__ == '__main__':
    unittest.main()
