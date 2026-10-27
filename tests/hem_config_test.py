"""Shared configuration semantics over real authored recipes and direct modality policy."""
from contextlib import redirect_stderr
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from binding.python.models import hem_config as config
from binding.python.hem_common import Deadline, DeadlineExpired, InvalidInput
from tests.recipe_config_fixture import ConfigurationFixture, recipe_for, record_for


class HemConfigTest(ConfigurationFixture, unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.configure()

    def resolve(self, overrides=None, config_path=None, **kwargs):
        return config.resolve(self.record, self.paths, self.dependencies, self.evidence, overrides, config_path, **kwargs)

    def test_strict_json_does_not_apply_an_unselected_parameter_schema(self):
        for value in ('[]', 'null', '{"rows":1,"rows":2}', '{"rows":NaN}', '{"options":Infinity}', '{'):
            with self.subTest(value=value), self.assertRaises(InvalidInput):
                config.load_config(self.config_file(value))
        self.assertEqual(config.load_config(None), {})
        self.assertEqual(config.load_config(self.config_file('{"pending_requests":4}')), {'pending_requests': 4})

    def test_all_four_layers_and_immutable_source_snapshot(self):
        self.replace_recipe(configuration={**self.recipe['configuration'], 'rows': 5})
        path = self.config_file('{"rows":8,"context_capacity":16384}')
        result = self.resolve({'context_capacity': 32768}, path)
        self.assertEqual(result.settings['rows'], 8)
        self.assertEqual(result.settings['context_capacity'], 32768)
        self.assertEqual(result.sources['rows'], 'file')
        self.assertEqual(result.sources['context_capacity'], 'override')
        self.assertEqual(result.sources['pool_capacity'], 'recipe')
        self.assertNotIn('mtp_depth', result.settings)
        with self.assertRaises(TypeError): result.settings['rows'] = 1
        path.write_text('{"rows":1}')
        self.assertEqual(result.settings['rows'], 8)

    def test_final_cross_rules_allow_later_layer_to_complete_partial_file(self):
        result = self.resolve({'prefill_capacity': 32}, self.config_file('{"context_capacity":64}'))
        self.assertEqual((result.settings['context_capacity'], result.settings['prefill_capacity']), (64, 32))
        with self.assertRaisesRegex(InvalidInput, 'prefill_capacity'):
            self.resolve(config_path=self.config_file('{"context_capacity":64}'))

    def test_bad_layer_cannot_be_repaired_by_later_override(self):
        for value in ('"2"', 'true', '1.0', 'null', '0', '17'):
            with self.subTest(value=value), self.assertRaises(InvalidInput):
                self.resolve({'rows': 2}, self.config_file('{"rows":' + value + '}'))
        for layer in ({'unknown': 1}, {'mtp': True}, {'mtp_depth': None}, {'mtp_depth': False}, {'mtp_depth': 1}):
            with self.subTest(layer=layer), self.assertRaises(InvalidInput): self.resolve(layer)
        for value in ([], None, {'x': set()}):
            if value is None: continue
            with self.subTest(value=value), self.assertRaises(InvalidInput): self.resolve(value)

    def test_conditional_depth_uses_final_winning_layer(self):
        recipe = recipe_for('qwen38-27b-q5-k-xl'); modality = config.modalities.get('chat')
        executable, _, _ = config.pipelines.load(recipe['executable'])
        serving = executable['serving']
        settings, sources = config.effective_settings(recipe, serving, modality, overrides={'mtp': False})
        self.assertNotIn('mtp_depth', settings)
        self.assertEqual(sources['mtp_depth'], {'state': 'inactive', 'by': 'override'})
        for file, override in (({}, {'mtp': False, 'mtp_depth': 2}), ({'mtp': False}, {'mtp_depth': 2})):
            with self.subTest(file=file, override=override), self.assertRaises(InvalidInput):
                config.effective_settings(recipe, serving, modality, file, override)
        settings, _ = config.effective_settings(recipe, serving, modality, {'mtp': False}, {'mtp': True})
        self.assertEqual(settings['mtp_depth'], 3)

    def test_asr_uses_same_merge_and_fixed_assertions(self):
        recipe = recipe_for('moonshine-streaming-tiny'); modality = config.modalities.get('asr')
        executable, _, _ = config.pipelines.load(recipe['executable'])
        serving = executable['serving']
        settings, _ = config.effective_settings(recipe, serving, modality, {'pending_requests': 4, 'options': {}})
        self.assertEqual(settings, {'pending_requests': 4, 'options': {}})
        for layer in ({'rows': 2}, {'pending_requests': 8}, {'options': {'x': 1}}, {'pending_requests': True}, {'options': None}):
            with self.subTest(layer=layer), self.assertRaises(InvalidInput):
                config.effective_settings(recipe, serving, modality, overrides=layer)

    def test_record_identity_inventory_and_files_are_checked(self):
        for key in ('recipe_id', 'recipe_identity', 'pipeline'):
            record = deepcopy(self.record)
            record['provenance'][key] = 'different'
            with self.subTest(key=key), self.assertRaises(config.HemError):
                config.resolve(record, self.paths, self.dependencies, self.evidence)
        for assets in (self.record['assets'][:-1], self.record['assets'] + self.record['assets'][:1]):
            with self.subTest(assets=assets), self.assertRaises(InvalidInput):
                config.resolve(dict(self.record, assets=assets), self.paths, self.dependencies, self.evidence)
        self.paths['weights'].unlink()
        with self.assertRaisesRegex(InvalidInput, 'unreadable'): self.resolve()

    def test_selected_resolve_does_not_enumerate_recipes_or_initialize_state(self):
        with patch.object(config.recipes, 'list', side_effect=AssertionError('global recipes')), \
             patch('binding.python.control.hem_state.Registry', side_effect=AssertionError('state')):
            result = self.resolve()
        self.assertEqual(result.pipeline, self.recipe['executable'])
        self.assertEqual(result.recipe_identity, self.recipe_identity)
        self.assertEqual(result.modality, 'chat')

    def test_export_is_atomic_and_uses_the_checked_value_without_reselection(self):
        output = self.root / 'export.json'; budget = Deadline.after()
        checked = self.resolve(deadline=budget)
        with patch.object(config, 'resolve', return_value=checked) as resolution, \
             patch.object(config.pipelines, 'load', side_effect=AssertionError('reselected')), redirect_stderr(io.StringIO()):
            answer = config.export_deployment(self.record, self.paths, self.dependencies, self.evidence, output, 'fixture', deadline=budget)
        self.assertTrue(answer['completed'])
        self.assertIs(resolution.call_args.kwargs['deadline'], budget)
        self.assertEqual(json.loads(output.read_text())['admission'], 'drained')
        original = output.read_bytes()
        with self.assertRaises(InvalidInput), redirect_stderr(io.StringIO()):
            config.export_deployment(self.record, self.paths, self.dependencies, self.evidence, output, '../bad')
        self.assertEqual(output.read_bytes(), original)

    def test_export_expiry_after_serialization_preserves_output(self):
        output = self.root / 'old.json'; output.write_text('old')
        clock = [0.0]; budget = Deadline(1.0, clock=lambda: clock[0]); dump = json.dump
        def expire(*args, **kwargs):
            result = dump(*args, **kwargs); clock[0] = 2.0; return result
        with patch.object(config.json, 'dump', side_effect=expire), self.assertRaises(DeadlineExpired):
            config.export_deployment(self.record, self.paths, self.dependencies, self.evidence, output, 'fixture', deadline=budget)
        self.assertEqual(output.read_text(), 'old')
        self.assertFalse(list(self.root.glob('.old.json.*')))


class RealRecipeResolutionTest(unittest.TestCase):
    def test_three_authored_recipes_drive_actual_modality_native_definitions(self):
        from binding.python.models import recipes
        for identifier in ('qwen35-08b-q5-k-xl', 'qwen38-27b-q5-k-xl', 'moonshine-streaming-tiny'):
            with self.subTest(identifier=identifier), tempfile.TemporaryDirectory() as directory:
                recipe, identity = recipes.load(identifier)
                record, paths, dependencies, evidence = record_for(Path(directory), recipe, identity)
                with patch.object(recipes, 'list', side_effect=AssertionError('unrelated recipe discovery')), \
                     patch('subprocess.Popen', side_effect=AssertionError('helper/native launch')), \
                     patch('binding.python.control.hem_state.Registry', side_effect=AssertionError('state')):
                    checked = config.resolve(record, paths, dependencies, evidence, {'rows': 2} if identifier != 'moonshine-streaming-tiny' else {'pending_requests': 4})
                definition = checked.definition_document()
                if checked.modality == 'chat':
                    self.assertEqual(definition['kind'], 'text')
                    self.assertEqual(definition['rows'], 2)
                    self.assertEqual(definition['weights'], str(paths['weights']))
                    self.assertEqual(definition['draft_depth'], 0 if identifier == 'qwen35-08b-q5-k-xl' else 3)
                else:
                    self.assertEqual(definition['kind'], 'recognition')
                    self.assertEqual(definition['assets'], record['directory'])
                    self.assertEqual(definition['pending_requests'], 4)
                    self.assertEqual(definition['options'], {})
                self.assertEqual(checked.recipe_identity, identity)
                self.assertEqual(set(checked.dependencies), set(dependencies))

    def test_caller_imports_and_explicit_shard_index_survive_without_shape_matching(self):
        from binding.python.models import recipes, hem_models
        from binding.python.control.hem_state import Registry
        from tests.gguf_fixtures import write_gguf
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry = Registry(root / 'state')
            tokenizer = root / 'tokenizer.json'; tokenizer.write_text('{}')
            first, second = root / 'first.gguf', root / 'second.gguf'
            write_gguf(first, tensors=[('arbitrary.first', [1], 0, 0)])
            write_gguf(second, tensors=[('arbitrary.second', [1], 0, 0)])
            hem_models.import_model(registry, 'single', ggufs=[first], tokenizer=tokenizer,
                                    alias=True, recipe_id='qwen35-08b-q5-k-xl')
            with registry.lock('model', 'single'):
                single = hem_models.qualify(registry, 'single')
                self.assertEqual(config.resolve(*single).definition['kind'], 'text')
            hem_models.import_model(registry, 'split', ggufs=[first, second], tokenizer=tokenizer,
                                    alias=True, recipe_id='qwen35-08b-q5-k-xl')
            with registry.lock('model', 'split'):
                split = hem_models.qualify(registry, 'split')
                checked = config.resolve(*split)
            self.assertTrue(checked.definition['weights'].endswith('.safetensors.index.json'))
            for path in split[2]:
                self.assertIn(path, checked.dependencies)


class PreparationAssociationTest(ConfigurationFixture, unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.configure('moonshine-streaming-tiny')
        from binding.python.models.preparation import moonshine_contract as operation
        self.operation = operation
        self.operation_path = self.root / 'operation.json'
        self.operation_path.write_text(json.dumps(operation.load_operation()))
        patcher = patch.object(operation, 'operation_path', return_value=self.operation_path)
        patcher.start(); self.addCleanup(patcher.stop)

    def resolve(self):
        return config.resolve(self.record, self.paths, self.dependencies, self.evidence)

    def change_operation(self):
        document = json.loads(self.operation_path.read_text())
        document['tool_revision'] = '0' * 40
        self.operation_path.write_text(json.dumps(document))

    def test_stale_preparation_cannot_cross_the_checked_configuration_boundary(self):
        with patch('subprocess.Popen', side_effect=AssertionError('helper/native process')), \
             patch('binding.python.control.hem_state.Registry', side_effect=AssertionError('state')):
            self.assertEqual(self.resolve().definition['kind'], 'recognition')
            self.change_operation()
            with self.assertRaisesRegex(InvalidInput, 'preparation association'):
                self.resolve()

    def test_operation_drift_outside_catalog_during_definition_is_rejected(self):
        original = self.modality.definition
        def mutate(*args):
            result = original(*args)
            self.change_operation()
            return result
        with patch.object(self.modality, 'definition', side_effect=mutate):
            with self.assertRaisesRegex(InvalidInput, 'Preparation contract changed'):
                self.resolve()


class DataOnlyConfigurationTest(ConfigurationFixture, unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.configure()

    def add_declaration(self, recipe_id, identifier, configure):
        recipe, _ = config.recipes.load(recipe_id)
        recipe = deepcopy(recipe)
        executable, _, _ = config.pipelines.load(recipe['executable'])
        executable = deepcopy(executable)
        executable['id'] = identifier
        configure(executable, recipe)
        reference = executable['programs']['root'] + '/examples/' + identifier + '/executable.json'
        manifest = self.catalog_root / reference
        manifest.parent.mkdir(parents=True)
        manifest.write_text(json.dumps(executable))
        recipe.update(id=identifier, name='Data-only ' + identifier, executable=reference, configuration={})
        path = manifest.parent / 'recipes' / (identifier + '.json')
        path.parent.mkdir(); path.write_text(json.dumps(recipe))
        return identifier

    def resolve_new(self, identifier):
        recipe, identity = config.recipes.load(identifier)
        directory = self.root / identifier; directory.mkdir()
        record, paths, dependencies, evidence = record_for(directory, recipe, identity)
        return config.resolve(record, paths, dependencies, evidence)

    def test_additional_chat_declaration_changes_caps_and_defaults_without_python_policy(self):
        def configure(executable, recipe):
            executable['serving']['bounds']['rows'] = [1, 8]
            executable['serving']['defaults']['rows'] = 2
        identifier = self.add_declaration('qwen35-08b-q5-k-xl', 'another-text-native', configure)
        checked = self.resolve_new(identifier)
        self.assertEqual(checked.executable_id, identifier)
        self.assertEqual(checked.modality, 'chat')
        self.assertEqual(checked.settings['rows'], 2)
        self.assertEqual(checked.definition['rows'], 2)
        self.assertEqual(checked.definition['draft_depth'], 0)
        view = config.modality_info('chat')
        self.assertNotIn('parameters', view)
        self.assertNotIn('defaults', view)
        selected = next(row for row in view['executables'] if row['executable_id'] == identifier)
        self.assertEqual(selected['parameters']['rows']['constraints']['maximum'], 8)
        self.assertEqual(selected['parameters']['mtp_depth']['state'], 'unsupported')

    def test_additional_asr_pending_range_and_nonpreparing_assets_are_data(self):
        def configure(executable, recipe):
            executable['serving']['pending_requests'] = [1, 8]
            executable['serving']['defaults']['pending_requests'] = 2
            executable['preparation'] = None
            recipe['files']['rotary']['source'] = dict(kind='huggingface', repo_id='example/recognizer',
                                                      revision='a'*40, filename='rotary.safetensors')
        identifier = self.add_declaration('moonshine-streaming-tiny', 'another-recognition-native', configure)
        checked = self.resolve_new(identifier)
        self.assertEqual(checked.modality, 'asr')
        self.assertEqual(checked.settings, {'pending_requests':2, 'options':{}})
        self.assertEqual(checked.definition['pending_requests'], 2)
        view = config.modality_info('asr')
        self.assertNotIn('parameters', view)
        current = {row['executable_id']:row for row in view['executables']}
        self.assertEqual(current[identifier]['parameters']['pending_requests']['constraints']['maximum'], 8)
        fixed = next(row for row in view['executables'] if row['executable_id'] != identifier)
        self.assertEqual(fixed['parameters']['pending_requests']['value'], 4)
