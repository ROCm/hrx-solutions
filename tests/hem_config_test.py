"""Shared configuration semantics over real authored recipes and typed doubles."""
from contextlib import redirect_stderr
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from binding.python.models import hem_config as config
from binding.python.hem_common import Deadline, DeadlineExpired, InvalidInput
from tests.recipe_config_fixture import ConfigurationFixture, recipe_for, descriptor_for


class HemConfigTest(ConfigurationFixture, unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.configure()

    def resolve(self, overrides=None, config_path=None, **kwargs):
        return config.resolve(self.recipe, self.model, overrides, config_path, **kwargs)

    def test_strict_json_does_not_apply_an_unselected_parameter_schema(self):
        for value in ('[]', 'null', '{"rows":1,"rows":2}', '{"rows":NaN}', '{"options":Infinity}', '{'):
            with self.subTest(value=value), self.assertRaises(InvalidInput):
                config.load_config(self.config_file(value))
        self.assertEqual(config.load_config(None), {})
        self.assertEqual(config.load_config(self.config_file('{"pending_requests":4}')), {'pending_requests': 4})

    def test_all_four_layers_and_immutable_source_snapshot(self):
        recipe = replace(self.recipe, configuration={**self.recipe.configuration, 'rows': 5})
        path = self.config_file('{"rows":8,"context_capacity":16384}')
        result = config.resolve(recipe, replace(self.model, recipe_identity=recipe.identity),
                                {'context_capacity': 32768}, path)
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
        recipe = recipe_for('qwen'); descriptor = descriptor_for(recipe)
        settings, sources = config.effective_settings(recipe, descriptor, overrides={'mtp': False})
        self.assertNotIn('mtp_depth', settings)
        self.assertEqual(sources['mtp_depth'], {'state': 'inactive', 'by': 'override'})
        for file, override in (({}, {'mtp': False, 'mtp_depth': 2}), ({'mtp': False}, {'mtp_depth': 2})):
            with self.subTest(file=file, override=override), self.assertRaises(InvalidInput):
                config.effective_settings(recipe, descriptor, file, override)
        settings, _ = config.effective_settings(recipe, descriptor, {'mtp': False}, {'mtp': True})
        self.assertEqual(settings['mtp_depth'], 3)

    def test_asr_uses_same_merge_and_fixed_assertions(self):
        recipe = recipe_for('moonshine-streaming-tiny'); descriptor = descriptor_for(recipe)
        settings, _ = config.effective_settings(recipe, descriptor, {'pending_requests': 4, 'options': {}})
        self.assertEqual(settings, {'pending_requests': 4, 'options': {}})
        for layer in ({'rows': 2}, {'pending_requests': 8}, {'options': {'x': 1}}, {'pending_requests': True}, {'options': None}):
            with self.subTest(layer=layer), self.assertRaises(InvalidInput):
                config.effective_settings(recipe, descriptor, overrides=layer)

    def test_recipe_and_prepared_identity_inventory_and_files_are_checked(self):
        for model in (replace(self.model, recipe_id='different'), replace(self.model, recipe_identity='different'),
                      replace(self.model, pipeline='different'), replace(self.model, files=self.model.files[:-1]),
                      replace(self.model, files=self.model.files + self.model.files[:1])):
            with self.subTest(model=model), self.assertRaises(InvalidInput):
                config.resolve(self.recipe, model)
        self.model.files[0].path.unlink()
        with self.assertRaisesRegex(InvalidInput, 'unreadable'): self.resolve()

    def test_selected_resolve_does_not_enumerate_recipes_or_initialize_state(self):
        with patch.object(config.recipes, 'list', side_effect=AssertionError('global recipes')), \
             patch('binding.python.control.hem_state.Registry', side_effect=AssertionError('state')):
            result = self.resolve()
        self.assertEqual(result.pipeline, self.recipe.pipeline)
        self.assertEqual(result.recipe_identity, self.recipe.identity)
        self.assertEqual(result.modality, 'chat')

    def test_export_is_atomic_and_uses_the_checked_value_without_reselection(self):
        output = self.root / 'export.json'; budget = Deadline.after()
        checked = self.resolve(deadline=budget)
        with patch.object(config, 'resolve', return_value=checked) as resolution, \
             patch.object(config.pipelines, 'get', side_effect=AssertionError('reselected')), redirect_stderr(io.StringIO()):
            answer = config.export_deployment(self.recipe, self.model, output, 'fixture', deadline=budget)
        self.assertTrue(answer['completed'])
        self.assertIs(resolution.call_args.kwargs['deadline'], budget)
        self.assertEqual(json.loads(output.read_text())['admission'], 'drained')
        original = output.read_bytes()
        with self.assertRaises(InvalidInput), redirect_stderr(io.StringIO()):
            config.export_deployment(self.recipe, self.model, output, '../bad')
        self.assertEqual(output.read_bytes(), original)

    def test_export_expiry_after_serialization_preserves_output(self):
        output = self.root / 'old.json'; output.write_text('old')
        clock = [0.0]; budget = Deadline(1.0, clock=lambda: clock[0]); dump = json.dump
        def expire(*args, **kwargs):
            result = dump(*args, **kwargs); clock[0] = 2.0; return result
        with patch.object(config.json, 'dump', side_effect=expire), self.assertRaises(DeadlineExpired):
            config.export_deployment(self.recipe, self.model, output, 'fixture', deadline=budget)
        self.assertEqual(output.read_text(), 'old')
        self.assertFalse(list(self.root.glob('.old.json.*')))


class RealRecipeResolutionTest(unittest.TestCase):
    def test_three_authored_recipes_drive_actual_modality_native_definitions(self):
        from binding.python.models import recipes
        from tests.recipe_config_fixture import prepared_for
        for identifier in ('qwen35-08b-q5-k-xl', 'qwen38-27b-q5-k-xl', 'moonshine-streaming-tiny'):
            with self.subTest(identifier=identifier), tempfile.TemporaryDirectory() as directory:
                recipe = recipes.load(identifier)
                prepared = prepared_for(Path(directory), recipe)
                with patch.object(recipes, 'list', side_effect=AssertionError('unrelated recipe discovery')), \
                     patch('subprocess.Popen', side_effect=AssertionError('helper/native launch')), \
                     patch('binding.python.control.hem_state.Registry', side_effect=AssertionError('state')):
                    checked = config.resolve(recipe, prepared, {'rows': 2} if recipe.modality == 'chat' else {'pending_requests': 4})
                definition = checked.definition_document()
                if recipe.modality == 'chat':
                    self.assertEqual(definition['kind'], 'text')
                    self.assertEqual(definition['rows'], 2)
                    self.assertEqual(definition['weights'], str(prepared.directory / 'model.gguf'))
                    self.assertEqual(definition['draft_depth'], 0 if recipe.pipeline == 'qwen-08b' else 3)
                else:
                    self.assertEqual(definition['kind'], 'recognition')
                    self.assertEqual(definition['assets'], str(prepared.directory))
                    self.assertEqual(definition['pending_requests'], 4)
                    self.assertEqual(definition['options'], {})
                self.assertEqual(checked.recipe_identity, recipe.identity)
                self.assertEqual(set(checked.dependencies), {item.path for item in prepared.files})

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
                                    alias=True, pipeline='qwen-08b')
            single = hem_models.model_files(registry, 'single')
            self.assertEqual(config.resolve(recipes.load(single.recipe_id), single).definition['kind'], 'text')
            hem_models.import_model(registry, 'split', ggufs=[first, second], tokenizer=tokenizer,
                                    alias=True, pipeline='qwen-08b')
            split = hem_models.model_files(registry, 'split')
            checked = config.resolve(recipes.load(split.recipe_id), split)
            self.assertTrue(checked.definition['weights'].endswith('.safetensors.index.json'))
            for item in split.files:
                self.assertIn(item.path, checked.dependencies)
