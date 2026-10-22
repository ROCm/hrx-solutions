"""CPU-only contracts; synthetic sparse GGUFs contain no checkpoint payload."""
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from binding.python import hem_config as config, hem_cli
from binding.python.hem_common import InvalidInput
from binding.python.model_store import ModelFiles
from tests.gguf_fixtures import expected_tensors, lifecycle_profile, profile, write_matching_gguf


class HemConfigTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.models = {}
        cls.tokenizer = cls.root / 'tokenizer.json'
        cls.tokenizer.write_text('{"model":{"fixture":true}}')
        cls.digest = hashlib.sha256(json.dumps(json.loads(cls.tokenizer.read_text()), sort_keys=True,
                                               ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()
        for identifier, catalog in (('qwen', 'qwen_27b'), ('qwen-08b', 'qwen_08b')):
            weights = cls.root / f'{identifier}.gguf'
            contract = lifecycle_profile() if identifier == 'qwen-08b' else profile(catalog)
            write_matching_gguf(weights, contract=contract)
            cls.models[identifier] = ModelFiles(weights, cls.tokenizer)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def setUp(self):
        self.load_profile = config.load_profile
        def fixture_profile(catalog):
            contract = lifecycle_profile() if Path(catalog).name == 'qwen_08b' else self.load_profile(catalog)
            contract['tokenizer_canonical_sha256'] = self.digest
            return contract
        self.profile_patch = patch.object(config, 'load_profile', side_effect=fixture_profile)
        self.profile_patch.start()
        self.addCleanup(self.profile_patch.stop)

    def resolve(self, pipeline='qwen-08b', flags=None, file=None):
        return config.resolve(self.models[pipeline], flags, file, pipeline=pipeline)

    def file(self, contents):
        path = self.root / 'settings.json'
        path.write_text(contents)
        return path

    def test_unique_selection_uses_descriptors_for_both_profiles(self):
        for pipeline, model in self.models.items():
            self.assertEqual(config.select_pipeline(model), pipeline)

    def test_explicit_pipeline_constrains_without_bypassing_qualification(self):
        with self.assertRaisesRegex(InvalidInput, 'incompatible with qwen:'):
            config.select_pipeline(self.models['qwen-08b'], 'qwen')
        with self.assertRaisesRegex(InvalidInput, 'Unknown pipeline'):
            config.select_pipeline(self.models['qwen'], 'unknown')

    def test_ambiguous_selection_and_explicit_constraint(self):
        with patch.object(config, 'available_pipelines', return_value=('qwen-08b', 'duplicate')), \
             patch.object(config, 'resolve_catalog', return_value=config.resolve_catalog('qwen-08b')):
            with self.assertRaisesRegex(InvalidInput, 'Ambiguous.*--pipeline'):
                config.select_pipeline(self.models['qwen-08b'])
            self.assertEqual(config.select_pipeline(self.models['qwen-08b'], 'duplicate'), 'duplicate')

    def test_tokenizer_and_tensor_mismatch_fail(self):
        changed = self.root / 'changed-tokenizer.json'
        changed.write_text('{"model":{"fixture":false}}')
        with self.assertRaisesRegex(InvalidInput, 'Tokenizer configuration differs'):
            config.select_pipeline(replace(self.models['qwen-08b'], tokenizer=changed))
        weights = self.root / 'wrong-tensor.gguf'
        write_matching_gguf(weights, contract=lifecycle_profile(), missing=('blk.0.ssm_a',))
        with self.assertRaisesRegex(InvalidInput, 'missing required tensor'):
            config.select_pipeline(ModelFiles(weights, self.tokenizer), 'qwen-08b')

    def test_strict_configuration_json(self):
        for contents in ('[]', 'null', '{"rows":1,"rows":2}', '{"rows":true}', '{"rows":1.0}',
                         '{"rows":0}', '{"rows":17}', '{"mtp":null}', '{"unknown":1}',
                         '{"rows":NaN}', '{"mtp_depth":0}', '{"rows":{"x":1}}', '{'):
            with self.subTest(contents=contents), self.assertRaises(InvalidInput):
                config.load_config(self.file(contents))
        self.assertEqual(config.load_config(None), {})
        self.assertEqual(config.load_config(self.file('{"rows":8,"mtp":false}')), {'rows':8,'mtp':False})

    def test_precedence_and_provenance_snapshot(self):
        path = self.file('{"rows":8,"context_capacity":16384}')
        result = self.resolve(flags={'context_capacity':32768}, file=path)
        self.assertEqual(result.settings, {**config.DEFAULTS, 'rows':8, 'context_capacity':32768,
                                          'mtp':False, 'mtp_depth':None})
        self.assertEqual(result.sources['rows'], 'file')
        self.assertEqual(result.sources['context_capacity'], 'flag')
        self.assertEqual(result.sources['pool_capacity'], 'default')
        path.write_text('{"rows":1}')
        self.assertEqual(result.settings['rows'],8)

    def test_mtp_defaults_disable_and_enabled_depth(self):
        self.assertEqual(self.resolve('qwen').settings['mtp_depth'],3)
        self.assertIsNone(self.resolve('qwen', {'mtp':False}).settings['mtp_depth'])
        self.assertIsNone(self.resolve().settings['mtp_depth'])
        result = self.resolve('qwen', {'mtp':True,'mtp_depth':2}, self.file('{"mtp":false}'))
        self.assertEqual(result.settings['mtp_depth'],2)

    def test_explicit_depth_disable_conflicts_across_sources(self):
        for flags, contents in (({'mtp_depth':3},'{"mtp":false}'),
                                ({'mtp':False},'{"mtp":true,"mtp_depth":3}'),
                                ({'mtp':False,'mtp_depth':3},'{}'),
                                ({},'{"mtp":false,"mtp_depth":3}')):
            with self.subTest(flags=flags, contents=contents), self.assertRaisesRegex(InvalidInput, 'remove mtp_depth or enable'):
                self.resolve('qwen', flags, self.file(contents))
        for flags in ({'mtp':True}, {'mtp':True,'mtp_depth':1}, {'mtp_depth':1}):
            with self.subTest(flags=flags), self.assertRaises(InvalidInput):
                self.resolve(flags=flags)

    def test_capacity_limits_and_relationships(self):
        for flags in ({'rows':17}, {'context_capacity':262145}, {'pool_capacity':4194305},
                      {'prefill_capacity':262145}, {'prefill_capacity':129,'context_capacity':128},
                      {'mtp_depth':4}, {'prefill_capacity':3}, {'rows':True}):
            with self.subTest(flags=flags), self.assertRaises(InvalidInput):
                self.resolve('qwen', flags)
        self.assertEqual(self.resolve('qwen', {'prefill_capacity':4}).settings['prefill_capacity'],4)

    def test_native_document_exact_drained_settings_and_local_paths(self):
        resolved = self.resolve()
        body = config.deployment_document(resolved, 'chat-small', catalog=resolved.catalog)
        self.assertEqual(body['admission'],'drained')
        self.assertEqual(body['device'],0)
        self.assertEqual(body['definition']['draft_depth'],0)
        self.assertEqual(body['definition']['weights'],str(self.models['qwen-08b'].weights))
        self.assertEqual(body['definition']['tokenizer'],str(self.tokenizer))
        for key, value in config.DEFAULTS.items():
            self.assertEqual(body['definition'][key],value)
        self.assertNotIn('weights.loom',body['definition']['sources'])
        self.assertTrue(all((Path(body['base_directory']) / item).is_file() for item in body['definition']['sources']))
        self.assertIn(self.tokenizer,resolved.dependencies)

    def test_fragmented_index_is_validated_and_retained(self):
        # Two actual sparse GGUF members, with disjoint tensor descriptors.
        contract = lifecycle_profile()
        names = tuple(expected_tensors(contract))
        first, second = self.root / 'part-one.gguf', self.root / 'part-two.gguf'
        write_matching_gguf(first, contract=contract, missing=names[::2])
        write_matching_gguf(second, contract=contract, missing=names[1::2])
        model = ModelFiles(first, self.tokenizer, (first, second))
        inspection = config.inspect_gguf_set(model.gguf_paths)
        index = self.root / 'parts.safetensors.index.json'
        index.write_text(json.dumps({'weight_map':{name:path.name for name,path in inspection.tensor_files.items()}}))
        model = replace(model, weights=index, fragments=model.gguf_paths)
        resolved = config.resolve(model)
        self.assertIn(index,resolved.dependencies)
        self.assertEqual(config.deployment_document(resolved,'parts',catalog=resolved.catalog)['definition']['weights'],str(index))
        index.write_text('{"weight_map":{}}')
        with self.assertRaisesRegex(InvalidInput,'index does not match'):
            config.resolve(model)

    def test_export_revalidates_and_has_no_registry_reservation(self):
        output = self.root / 'export.json'
        stderr = io.StringIO()
        resolved = self.resolve()
        with patch.object(config,'deployment_document',wraps=config.deployment_document) as document, \
             patch.object(config,'select_pipeline',wraps=config.select_pipeline) as validation, \
             redirect_stderr(stderr):
            result = config.export_deployment(self.models['qwen-08b'], output,'chat-small')
        self.assertTrue(result['completed'])
        validation.assert_called_once()
        document.assert_called_once()
        self.assertEqual(json.loads(output.read_text()), config.deployment_document(resolved,'chat-small',catalog=resolved.catalog))
        self.assertIn('does not reserve',stderr.getvalue())
        self.assertIn('compatible server',stderr.getvalue())
        self.assertIn('entire raw-curl',stderr.getvalue())
        self.assertFalse(list(self.root.glob('*.sqlite3')))
        original = output.read_bytes()
        with patch.object(config.tempfile, 'NamedTemporaryFile') as staging, self.assertRaises(InvalidInput):
            config.export_deployment(self.models['qwen-08b'],output,'../bad')
        staging.assert_not_called()
        self.assertEqual(output.read_bytes(), original)

    def test_pipeline_cli_exposes_profiles_capabilities_and_defaults(self):
        for argv in (['pipeline','list','--json'],['pipeline','show','qwen-08b','--json']):
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                code = hem_cli.main(argv)
            self.assertEqual(code,0)
            body=json.loads(stdout.getvalue())
            entries=body.get('items',[body])
            self.assertTrue(entries)
            if 'items' in body:
                self.assertEqual({entry['pipeline'] for entry in entries},
                                 {'qwen','qwen-08b','moonshine-streaming-tiny'})
            for entry in entries:
                self.assertIn('defaults',entry)
                self.assertIn('limits',entry)
                if entry['pipeline'] in ('qwen','qwen-08b'):
                    self.assertIn('compatibility',entry)
                    self.assertIn('mtp',entry['capabilities'])
                else:
                    self.assertEqual(entry['pipeline'],'moonshine-streaming-tiny')
                    self.assertEqual(entry['kind'],'recognition')
                    self.assertEqual(entry['check_kind'],'acquisition-presence')
                    self.assertEqual(entry['defaults'],{'pending_requests':4,'options':{}})
                    self.assertEqual({asset['path'] for asset in entry['assets']},
                                     {'model.safetensors','tokenizer.json','config.json','rotary.safetensors'})
                    self.assertNotIn('compatibility',entry)
