from contextlib import redirect_stdout, redirect_stderr
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from binding.python import _launcher, model_store, pipeline_validation, setup_cli
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
        source=Path(__file__).resolve().parents[1]/'loom-pipeline/llm/qwen/qwen_27b'
        with patch.object(Path,'home',return_value=self.root):
            configured=pipeline_validation.catalog_for_model('qwen',source,'test-model')
            self.assertIn('746573742d6d6f64656c',(configured/'control.loom').read_text())
            self.assertTrue((configured/'compatibility.json').is_file())

    def test_08b_validation_records_explicit_pipeline(self):
        self.select_08b_contract()
        write_matching_gguf(self.weights, contract=self.contract)
        with redirect_stdout(io.StringIO()) as output:
            code = setup_cli.main(['models', 'validate', 'test-model', '--pipeline', 'qwen-08b', '--model-root', str(self.root)])
        self.assertEqual(code, 0)
        self.assertIn('24 base layers without MTP', output.getvalue())
        self.assertEqual(model_store.read_manifest('test-model', self.root)['pipeline'], 'qwen-08b')
        self.assertEqual(set(pipeline_validation.available_pipelines()), {'qwen', 'qwen-08b'})

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

    def test_serving_capability_defaults_and_explicit_requests(self):
        package = self.root/'package'
        binary = package/'_native/bin/loom-serve'
        binary.parent.mkdir(parents=True)
        binary.write_text('test executable')
        args = ['serve', 'test-model', '--model-root', str(self.root), '--foreground']
        for pipeline, automatic in [('qwen', True), ('qwen-08b', False)]:
            if pipeline == 'qwen-08b':
                self.select_08b_contract()
            write_matching_gguf(self.weights, contract=self.contract)
            model_store.record_pipeline('test-model', pipeline, self.root)
            selected = pipeline_validation.resolve_catalog(pipeline)
            choices = [([], automatic), (['--no-mtp'], False)]
            if automatic:
                choices.append((['--mtp', '--mtp-depth', '2'], True))
            for flags, enabled in choices:
                with self.subTest(pipeline=pipeline, flags=flags), patch.object(setup_cli, '__file__', str(package/'setup_cli.py')), patch.object(pipeline_validation, 'catalog_for_model', return_value=selected), patch.object(setup_cli, 'native_environment', return_value={}), patch.object(setup_cli.os, 'execve', side_effect=SystemExit(0)) as execute:
                    with self.assertRaises(SystemExit) as stopped:
                        setup_cli.main(args+flags)
                    self.assertEqual(stopped.exception.code, 0)
                    argv = execute.call_args.args[1]
                    document_path = Path(next(value.partition('=')[2] for value in argv if value.startswith('--model=')))
                    document = json.loads(document_path.read_text())
                    self.assertEqual(document['definition']['draft_depth'] > 0, enabled)
                    self.assertEqual(document['base_directory'], str(selected))
                    self.assertIn('--device=amdgpu', argv)
                    self.assertIn('--listen=127.0.0.1:8080', argv)
                    self.assertNotIn('prepare.loom', document['definition']['sources'])
                    self.assertIn('../programs/qwen38/ffn.loom', document['definition']['sources'])
            if pipeline == 'qwen-08b':
                for flags in (['--mtp'], ['--mtp-depth', '2']):
                    with patch.object(setup_cli.os, 'execve') as execute, redirect_stderr(io.StringIO()) as error:
                        self.assertEqual(setup_cli.main(args+flags), 1)
                        self.assertIn('does not support MTP', error.getvalue())
                        execute.assert_not_called()

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
