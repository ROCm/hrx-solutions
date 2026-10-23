"""CHAT policy and concrete Qwen contracts without checkpoints or services."""
from contextlib import ExitStack
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from binding.python.hem_common import HemError, InvalidInput
from binding.python.models import modalities, pipelines, pipelines_qwen, recipes
from binding.python.models.modalities import chat
from binding.python.models.recipe_types import (
    AcquisitionPlan, Artifact, Fixed, GeneratedSource, HFSource, PreparedArtifact,
    PreparedModel, TextPrograms, Unsupported,
)

ROOT = Path(__file__).resolve().parents[1]
QWEN08 = ROOT / 'loom-pipeline/llm/qwen/qwen_08b/recipes/qwen35-08b-q5-k-xl.json'
QWEN27 = ROOT / 'loom-pipeline/llm/qwen/qwen_27b/recipes/qwen38-27b-q5-k-xl.json'


def effective(recipe):
    # Actual endorsed recipes specify the full active settings; merge orchestration
    # is intentionally owned/tested by the shared caller lane, not duplicated here.
    return dict(recipe.configuration)


class ChatPolicyTest(unittest.TestCase):
    def setUp(self):
        self.small = pipelines.get('qwen-08b')
        self.large = pipelines.get('qwen')
        self.recipe = recipes.load('qwen35-08b-q5-k-xl')
        self.large_recipe = recipes.load('qwen38-27b-q5-k-xl')

    def test_real_recipes_same_vocabulary_and_selected_states(self):
        a, b = self.small.describe(), self.large.describe()
        self.assertEqual(tuple(a.parameters), tuple(b.parameters))
        self.assertEqual(set(a.parameters), {'rows', 'context_capacity', 'pool_capacity',
                                              'prefill_capacity', 'mtp', 'mtp_depth'})
        self.assertIsInstance(a.parameters['mtp'], Fixed)
        self.assertIs(a.parameters['mtp'].value, False)
        self.assertIsInstance(a.parameters['mtp_depth'], Unsupported)
        self.assertEqual(b.parameters['mtp'].state, 'configurable')
        self.assertEqual(b.parameters['mtp_depth'].active_when, {'parameter': 'mtp', 'equals': True})
        self.assertEqual(a.protocol, chat.PROTOCOL)
        self.assertEqual(a.protocol.native_kind, 'text')
        for binding, recipe in ((self.small, self.recipe), (self.large, self.large_recipe)):
            chat.validate(effective(recipe), binding.describe())
            self.assertIsNone(binding.describe().preparation)
            self.assertEqual(binding.describe().telemetry.metrics, ())

    def test_modality_name_is_policy_not_recipe_or_local_identity(self):
        changed = replace(self.recipe, id='another-name', name='Different display')
        self.small.validate_artifacts(changed.artifacts)
        chat.validate(effective(changed), self.small.describe())
        self.assertIs(modalities.get(changed.modality), chat)
        for value in ('qwen', 'moonshine', 'transcription', None):
            with self.assertRaises(InvalidInput):
                modalities.get(value)

    def test_exact_types_ranges_fixed_unsupported_and_missing(self):
        settings = effective(self.recipe)
        for key, value in (('rows', True), ('rows', 1.0), ('rows', 0), ('rows', 17),
                           ('context_capacity', None), ('mtp', True), ('mtp_depth', 1),
                           ('unknown', 1)):
            with self.subTest(key=key, value=value), self.assertRaises(InvalidInput):
                chat.validate({**settings, key: value}, self.small.describe())
        for key in settings:
            bad = dict(settings)
            del bad[key]
            with self.subTest(missing=key), self.assertRaises(InvalidInput):
                chat.validate(bad, self.small.describe())

    def test_final_capacity_and_mtp_cross_rules(self):
        large = effective(self.large_recipe)
        for values in ({'context_capacity': 64, 'prefill_capacity': 128},
                       {'prefill_capacity': 3}, {'mtp': False}, {'mtp_depth': 4}):
            with self.subTest(values=values), self.assertRaises(InvalidInput):
                chat.validate({**large, **values}, self.large.describe())
        disabled = {key: value for key, value in large.items() if key != 'mtp_depth'}
        disabled['mtp'] = False
        chat.validate(disabled, self.large.describe())
        with self.assertRaisesRegex(InvalidInput, 'Missing CHAT setting mtp_depth'):
            chat.validate({key: value for key, value in large.items() if key != 'mtp_depth'}, self.large.describe())
        chat.validate({**effective(self.recipe), 'context_capacity': 64, 'prefill_capacity': 32},
                      self.small.describe())

    def test_native_definition_uses_explicit_prepared_paths_and_mtp(self):
        for binding, recipe in ((self.small, self.recipe), (self.large, self.large_recipe)):
            prepared = PreparedModel('local-name', recipe.id, recipe.identity, binding.id, Path('/private/model'),
                (PreparedArtifact('weights', 'weights', 'gguf', 'custom.gguf', Path('/hf/cache/custom.gguf'), 'hf-reference'),
                 PreparedArtifact('tokenizer', 'tokenizer', 'json', 'tokenizer.json', Path('/hf/cache/custom.json'), 'hf-reference')))
            settings = effective(recipe)
            chat.validate(settings, binding.describe())
            result = chat.service_definition(settings, prepared, TextPrograms('p.loom', 'c.loom', ('first.loom', 'next.loom')))
            self.assertEqual(result, {'kind': 'text', 'prepare': 'p.loom', 'control': 'c.loom',
                'sources': ['first.loom', 'next.loom'], 'weights': '/hf/cache/custom.gguf',
                'tokenizer': '/hf/cache/custom.json', 'rows': 4, 'context_capacity': 8192,
                'pool_capacity': 32768, 'prefill_capacity': 128,
                'draft_depth': 3 if binding.id == 'qwen' else 0, 'options': {}})
            self.assertEqual(prepared.local_id, 'local-name')

    def test_policy_and_raw_fallback_have_no_io(self):
        descriptor = self.small.describe()
        prepared = PreparedModel('local', 'recipe', 'identity', 'qwen-08b', Path('/model'),
            (PreparedArtifact('w', 'weights', 'gguf', 'w.gguf', Path('/model/w.gguf'), 'hf-reference'),
             PreparedArtifact('t', 'tokenizer', 'json', 't.json', Path('/model/t.json'), 'hf-reference')))
        with ExitStack() as stack:
            for target in ('builtins.open', 'pathlib.Path.open', 'subprocess.run', 'socket.create_connection'):
                stack.enter_context(patch(target, side_effect=AssertionError('modality performed I/O')))
            chat.describe(self.small)
            chat.validate(effective(self.recipe), descriptor)
            chat.service_definition(effective(self.recipe), prepared, descriptor.programs)
            view = chat.interpret({'event': 'unknown', 'value': 'line\nbreak'}, descriptor)
        self.assertEqual(json.loads(view.text), {'event': 'unknown', 'value': 'line\nbreak'})
        self.assertEqual(view.metrics, ())
        self.assertFalse(view.recognized)

    def test_chat_selection_does_not_import_asr_or_shape_checker(self):
        code = ('import sys; from binding.python.models import pipelines, modalities; '
                'modalities.get("chat"); pipelines.get("qwen-08b").describe(); '
                'assert "binding.python.models.modalities.asr" not in sys.modules; '
                'assert "binding.python.models.qwen_validation" not in sys.modules')
        subprocess.run([sys.executable, '-c', code], cwd=ROOT, check=True, capture_output=True, text=True)


class QwenProgramsTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name) / 'loom-pipeline'
        self.catalog = self.root / 'llm/qwen/qwen_08b'
        self.catalog.mkdir(parents=True)
        (self.root / 'llm/LICENSE-APACHE-2.0').write_text('test license')
        self.roots = patch.object(pipelines, 'catalog_roots', return_value=(self.root,))
        self.roots.start()
        self.addCleanup(self.roots.stop)

    def source(self):
        entries = ('config.loom', 'prepare.loom', 'control.loom', 'weights.loom', 'provider.loom')
        for name in entries:
            (self.catalog / name).write_text('fixture program')
        (self.catalog / 'sources.txt').write_text('\n'.join(entries) + '\n')
        return entries

    def bytecode(self):
        content = b'LOOMtest bytecode'
        (self.catalog / 'pipeline.loombc').write_bytes(content)
        descriptor = {'format': 1, 'variant': 'qwen35-08b-ud-q5-k-xl-v1',
                      'module': 'pipeline.loombc', 'sha256': hashlib.sha256(content).hexdigest()}
        (self.catalog / 'pipeline.json').write_text(json.dumps(descriptor))
        return descriptor

    def test_source_roles_and_package_relative_shipping_manifest(self):
        self.source()
        descriptor = pipelines_qwen.get('qwen-08b').describe()
        self.assertEqual(descriptor.programs, TextPrograms('prepare.loom', 'control.loom', ('config.loom', 'provider.loom')))
        self.assertEqual(descriptor.bundle.catalog_relative_path, 'llm/qwen/qwen_08b')
        paths = {entry.path for entry in descriptor.bundle.files}
        self.assertIn('loom-pipeline/llm/qwen/qwen_08b/pipeline.loombc', paths)
        self.assertIn('loom-pipeline/llm/LICENSE-APACHE-2.0', paths)
        self.assertNotIn('compatibility.json', ' '.join(paths))
        self.assertFalse(any(path.endswith('.loom') for path in paths))

    def test_bytecode_roles_and_hash_variant_format_rejection(self):
        metadata = self.bytecode()
        self.assertEqual(pipelines_qwen.get('qwen-08b').programs,
                         TextPrograms('pipeline.loombc', 'pipeline.loombc', ('pipeline.loombc',)))
        for field, value in (('format', True), ('variant', 'other'), ('module', '../outside.loombc'), ('sha256', '0' * 64)):
            with self.subTest(field=field):
                (self.catalog / 'pipeline.json').write_text(json.dumps({**metadata, field: value}))
                with self.assertRaises(HemError):
                    pipelines_qwen.get('qwen-08b')
        (self.catalog / 'pipeline.json').write_text(json.dumps(metadata))
        (self.catalog / 'pipeline.loombc').write_bytes(b'not bytecode')
        with self.assertRaises(HemError):
            pipelines_qwen.get('qwen-08b')

    def test_singleton_rejects_duplicate_descriptor_and_extra_program(self):
        metadata = self.bytecode()
        text = json.dumps(metadata)
        (self.catalog / 'pipeline.json').write_text(text[:-1] + ', "format": 1}')
        with self.assertRaisesRegex(HemError, 'Duplicate'):
            pipelines_qwen.get('qwen-08b')
        (self.catalog / 'pipeline.json').write_text(text)
        (self.catalog / 'extra.loom').write_text('undeclared')
        with self.assertRaisesRegex(HemError, 'unexpected executable'):
            pipelines_qwen.get('qwen-08b')

    def test_compiled_variant_identities_do_not_swap(self):
        metadata = self.bytecode()
        other = self.catalog.parent / 'qwen_27b'
        other.mkdir()
        (other / 'pipeline.loombc').write_bytes((self.catalog / 'pipeline.loombc').read_bytes())
        (other / 'pipeline.json').write_text(json.dumps({**metadata, 'variant': 'qwen38-27b-ud-q5-k-xl-v1'}))
        self.assertTrue(pipelines_qwen.get('qwen').describe().parameters['mtp'].default)
        (other / 'pipeline.json').write_text(json.dumps(metadata))
        with self.assertRaisesRegex(HemError, 'variant'):
            pipelines_qwen.get('qwen')

    def test_source_manifest_requires_roles_unique_identity_and_containment(self):
        entries = self.source()
        for values in (entries[:-2] + ('provider.loom',), entries + ('prepare.loom',),
                       entries + ('../qwen_08b/prepare.loom',), entries + ('/absolute.loom',)):
            with self.subTest(values=values):
                (self.catalog / 'sources.txt').write_text('\n'.join(values))
                with self.assertRaises(HemError):
                    pipelines_qwen.get('qwen-08b')
        (self.catalog / 'sources.txt').write_text('\n'.join(entries))
        outside = self.root.parent / 'outside.loom'
        outside.write_text('outside')
        (self.catalog / 'provider.loom').unlink()
        (self.catalog / 'provider.loom').symlink_to(outside)
        with self.assertRaises(HemError):
            pipelines_qwen.get('qwen-08b')

    def test_source_internal_shared_symlink_is_allowed(self):
        self.source()
        shared = self.catalog.parent / 'shared.loom'
        shared.write_text('reviewed internal source')
        (self.catalog / 'control.loom').unlink()
        (self.catalog / 'control.loom').symlink_to(shared)
        self.assertEqual(pipelines_qwen.get('qwen-08b').programs.control, 'control.loom')

    def test_unknown_and_missing_selected_catalog_are_domain_errors(self):
        with self.assertRaises(InvalidInput):
            pipelines_qwen.get('moonshine-streaming-tiny')
        with self.assertRaisesRegex(HemError, 'No pipeline catalog'):
            pipelines_qwen.get('qwen')


class QwenArtifactsTest(unittest.TestCase):
    def setUp(self):
        self.binding = pipelines.get('qwen-08b')
        self.recipe = recipes.load('qwen35-08b-q5-k-xl')

    def test_actual_artifacts_and_changed_safe_source_values_are_admitted(self):
        self.binding.validate_artifacts(self.recipe.artifacts)
        first, second = self.recipe.artifacts
        changed = replace(first, id='different-id', path='different.gguf',
                          source=HFSource('other/repo', 'a' * 40, 'other.gguf'))
        self.binding.validate_artifacts((changed, second))

    def test_generated_wrong_roles_formats_and_parser_suffixes_rejected(self):
        first, second = self.recipe.artifacts
        for entries in ((first,), (first, first), (replace(first, source=GeneratedSource()), second),
                        (replace(first, format='safetensors'), second),
                        (replace(first, path='weights.bin'), second), (first, replace(second, path='t.bin'))):
            with self.subTest(entries=entries), self.assertRaises(InvalidInput):
                self.binding.validate_artifacts(entries)

    def test_noop_preflight_and_no_generated_preparation(self):
        plan = AcquisitionPlan(self.recipe.id, 'qwen-08b', self.recipe.artifacts, (), None)
        deadline = unittest.mock.Mock()
        self.assertIsNone(self.binding.preflight(plan, deadline))
        deadline.assert_not_called()
        self.assertEqual(self.binding.prepare(plan, {}, Path('/must-not-be-created'), deadline), ())
        self.assertEqual(deadline.mock_calls, [])
        with self.assertRaises(InvalidInput):
            self.binding.prepare(replace(plan, pipeline='qwen'), {}, Path('/unused'), deadline)
        with self.assertRaises(InvalidInput):
            self.binding.prepare(replace(plan, generated=(self.recipe.artifacts[0],)), {}, Path('/unused'), deadline)


if __name__ == '__main__':
    unittest.main()
