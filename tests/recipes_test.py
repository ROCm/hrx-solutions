"""Authored recipe admission, deterministic discovery and selected binding isolation."""
from dataclasses import FrozenInstanceError
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from binding.python.hem_common import HemError
from binding.python.models import pipelines, recipes
from binding.python.models.recipe_types import (
    AcquisitionPlan, ArtifactRequirement, BundleFile, BundleManifest, Configurable,
    Constraints, Fixed, GeneratedSource, HFSource, PipelineDescriptor,
    ProtocolDescriptor, RecognitionPrograms, TextPrograms, Unsupported, json_document,
)

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = (
    ROOT / 'loom-pipeline/llm/qwen/qwen_08b/recipes/qwen35-08b-q5-k-xl.json',
    ROOT / 'loom-pipeline/llm/qwen/qwen_27b/recipes/qwen38-27b-q5-k-xl.json',
    ROOT / 'loom-pipeline/recognition/moonshine/streaming_tiny/recipes/moonshine-streaming-tiny.json',
)


def document():
    return json.loads(EXAMPLES[0].read_bytes())


def admit(value):
    return recipes.parse(json.dumps(value).encode())


def descriptor(identifier='qwen-08b', modality='chat'):
    return PipelineDescriptor(
        id=identifier, modality=modality,
        parameters={'rows': Configurable('integer', 4, Constraints(minimum=1)),
                    'mtp': Fixed('boolean', False),
                    'draft_depth': Unsupported('integer', 'no draft support')},
        protocol=ProtocolDescriptor('chat', 'text', '/v1/chat/completions',
                                    'json', ['text'], ['text']),
        artifacts=[ArtifactRequirement('weights', ['gguf'], 1, 1)],
        preparation=None, programs=TextPrograms('prepare.loom', 'control.loom', ['provider.loom']),
        bundle=BundleManifest('llm/qwen/qwen_08b', [BundleFile('prepare.loom', 'prepare')]),
    )


class ParserTest(unittest.TestCase):
    def test_actual_chat_and_asr_files_have_one_representation_and_byte_identity(self):
        values = [recipes.parse(path.read_bytes()) for path in EXAMPLES]
        self.assertEqual([value.modality for value in values], ['chat', 'chat', 'asr'])
        self.assertEqual([value.pipeline for value in values], ['qwen-08b', 'qwen', 'moonshine-streaming-tiny'])
        for path, value in zip(EXAMPLES, values):
            self.assertEqual(value.id, path.stem)
            self.assertEqual(value.identity, hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertEqual(json_document(value.configuration), json.loads(path.read_bytes())['configuration'])
            self.assertIsInstance(value.artifacts[0].source, HFSource)
        self.assertIsInstance(values[2].artifacts[-1].source, GeneratedSource)
        raw = EXAMPLES[0].read_bytes()
        self.assertNotEqual(recipes.parse(raw).identity, recipes.parse(raw + b'\n').identity)

    def test_nested_configuration_is_copied_and_frozen(self):
        source = document()
        source['configuration'] = {'nested': {'items': [1, {'flag': True}]}}
        value = admit(source)
        source['configuration']['nested']['items'].clear()
        self.assertEqual(value.configuration['nested']['items'][0], 1)
        with self.assertRaises(TypeError):
            value.configuration['nested']['items'][1]['flag'] = False
        with self.assertRaises(FrozenInstanceError):
            value.name = 'changed'

    def test_exact_envelope_and_scalar_types(self):
        cases = [('format', True), ('format', 1.0), ('format', 2), ('id', '../bad'),
                 ('name', ''), ('name', 4), ('modality', 'tts'), ('pipeline', '-bad'),
                 ('artifacts', []), ('artifacts', {}), ('configuration', [])]
        for key, item in cases:
            with self.subTest(key=key, item=item), self.assertRaises(HemError):
                value = document()
                value[key] = item
                admit(value)
        for key in ('extra', 'revision'):
            value = document()
            value[key] = 1
            with self.assertRaises(HemError):
                admit(value)
        value = document()
        del value['configuration']
        with self.assertRaises(HemError):
            admit(value)

    def test_strict_json_at_every_depth_and_document_limits(self):
        raw = EXAMPLES[0].read_bytes()
        invalid = [raw.replace(b'"format": 1', b'"format": 1, "format": 1'),
                   raw.replace(b'"rows": 4', b'"rows": 4, "rows": 4'),
                   raw.replace(b'"rows": 4', b'"rows": NaN'),
                   raw.replace(b'"rows": 4', b'"rows": Infinity'),
                   raw.replace(b'"rows": 4', b'"rows": 1e999'),
                   b'\xff', b'{} trailing', b'x' * (recipes.MAX_BYTES + 1), raw.decode()]
        nested = document()
        child = nested['configuration']
        for _ in range(recipes.MAX_DEPTH + 1):
            child['deep'] = {}
            child = child['deep']
        invalid.append(json.dumps(nested).encode())
        for index, data in enumerate(invalid):
            with self.subTest(index=index), self.assertRaises(HemError):
                recipes.parse(data)

    def test_source_and_artifact_shapes_are_exact(self):
        changes = [dict(kind='local', path='x'), dict(kind='generated', helper='command'),
                   dict(kind='huggingface', repo_id='org/repo', revision='main', filename='model'),
                   dict(kind='huggingface', repo_id='org/repo', revision='A' * 40, filename='model')]
        for source in changes:
            value = document()
            value['artifacts'][0]['source'] = source
            with self.subTest(source=source), self.assertRaises(HemError):
                admit(value)
        for field, data in [('extra', 1), ('format', 'bin'), ('role', '../bad'), ('id', True)]:
            value = document()
            value['artifacts'][0][field] = data
            with self.subTest(field=field), self.assertRaises(HemError):
                admit(value)

    def test_paths_and_optionlike_source_locators(self):
        bad_paths = ['', '/', '/absolute', 'a//b', '.', '..', 'a/../b', 'a/./b',
                     'a/', 'a\\b', 'a\0b', 'C:/absolute', 'C:relative']
        for key in ('path', 'repo_id', 'filename'):
            for bad in bad_paths + (['-repo', 'org/-repo'] if key != 'path' else []):
                value = document()
                target = value['artifacts'][0] if key == 'path' else value['artifacts'][0]['source']
                target[key] = bad
                with self.subTest(key=key, bad=bad), self.assertRaises(HemError):
                    admit(value)

    def test_duplicate_ids_coordinates_and_overlapping_paths(self):
        for change in ('id', 'source', 'path', 'parent', 'child'):
            value = document()
            first, second = value['artifacts']
            if change in ('id', 'source', 'path'):
                second[change] = copy.deepcopy(first[change])
            elif change == 'parent':
                second['path'] = 'model.gguf/nested'
            else:
                first['path'] = 'tokenizer.json/nested'
            with self.subTest(change=change), self.assertRaises(HemError):
                admit(value)

    def test_shards_preserve_authored_order_and_explicit_destinations(self):
        value = document()
        shard = copy.deepcopy(value['artifacts'][0])
        shard.update(id='shard2', path='shards/second.gguf')
        shard['source']['filename'] = 'second.gguf'
        value['artifacts'].append(shard)
        parsed = admit(value)
        self.assertEqual(tuple(item.id for item in parsed.artifacts), ('weights', 'tokenizer', 'shard2'))
        self.assertEqual(parsed.artifacts[2].role, 'weights')
        self.assertEqual(parsed.artifacts[2].path, 'shards/second.gguf')


class LookupTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        roots = patch.object(pipelines, 'catalog_roots', return_value=(self.root,))
        roots.start()
        self.addCleanup(roots.stop)
        self.selected = []

    def write(self, value, directory='leaf', filename=None):
        path = self.root / directory / 'recipes' / (filename or (value['id'] + '.json'))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        return path

    def binding(self, pipeline):
        self.selected.append(pipeline)
        modality = 'asr' if pipeline == 'moonshine-streaming-tiny' else 'chat'
        return SimpleNamespace(describe=lambda: descriptor(pipeline, modality),
                               validate_artifacts=lambda items: None)

    def test_selected_lookup_ignores_unrelated_broken_json_and_implementations(self):
        value = document()
        self.write(value)
        bad = self.root / 'other/recipes/broken.json'
        bad.parent.mkdir(parents=True)
        bad.write_text('not JSON')
        with patch.object(pipelines, 'get', side_effect=self.binding):
            result = recipes.load(value['id'])
        self.assertEqual(result.id, value['id'])
        self.assertEqual(self.selected, ['qwen-08b'])

    def test_list_is_sorted_and_filters_by_modality(self):
        for index, path in enumerate(reversed(EXAMPLES)):
            self.write(json.loads(path.read_bytes()), str(index))
        with patch.object(pipelines, 'get', side_effect=self.binding):
            found = recipes.list()
            self.assertEqual(tuple(value.id for value in found), tuple(sorted(path.stem for path in EXAMPLES)))
            self.assertEqual(len(recipes.list('chat')), 2)
            self.assertEqual(len(recipes.list('asr')), 1)
        with self.assertRaises(HemError):
            recipes.list('tts')

    def test_unknown_duplicate_and_filename_mismatch_fail(self):
        with self.assertRaisesRegex(HemError, 'Unknown recipe'):
            recipes.load('unknown')
        value = document()
        self.write(value)
        self.write(value, 'second')
        with self.assertRaisesRegex(HemError, 'Duplicate'):
            recipes.load(value['id'])
        with patch.object(pipelines, 'get', side_effect=self.binding):
            with self.assertRaisesRegex(HemError, 'Duplicate'):
                recipes.list()
        for path in self.root.glob('**/*.json'):
            path.unlink()
        self.write(value, filename='wrong-name.json')
        with self.assertRaisesRegex(HemError, 'filename'):
            recipes.load('wrong-name')
        with self.assertRaisesRegex(HemError, 'filename'):
            recipes.list()

    def test_broken_binding_and_modality_mismatch_fail_discovery(self):
        self.write(document())
        with patch.object(pipelines, 'get', side_effect=HemError('missing implementation')):
            with self.assertRaisesRegex(HemError, 'missing implementation'):
                recipes.list()
        binding = SimpleNamespace(describe=lambda: descriptor(modality='asr'))
        with patch.object(pipelines, 'get', return_value=binding):
            with self.assertRaisesRegex(HemError, 'modalities differ'):
                recipes.list()

    def test_selected_binding_validates_artifact_contract(self):
        value = document()
        self.write(value)
        binding = SimpleNamespace(describe=lambda: descriptor(),
                                  validate_artifacts=lambda items: (_ for _ in ()).throw(HemError('unsupported-preparation')))
        with patch.object(pipelines, 'get', return_value=binding):
            with self.assertRaisesRegex(HemError, 'unsupported-preparation'):
                recipes.load(value['id'])

    def test_support_queries_never_invoke_acquisition_preflight(self):
        value = document()
        self.write(value)
        binding = self.binding(value['pipeline'])
        binding.preflight = lambda plan, deadline: self.fail('support query ran helper preflight')
        with patch.object(pipelines, 'get', return_value=binding):
            self.assertEqual(recipes.load(value['id']).id, value['id'])
            self.assertEqual(tuple(item.id for item in recipes.list()), (value['id'],))
            self.assertEqual(tuple(item.id for item in recipes.list('chat')), (value['id'],))
            self.assertEqual(pipelines.get(value['pipeline']).describe().id, value['pipeline'])

    def test_symlink_cannot_escape_catalog(self):
        with tempfile.TemporaryDirectory() as external:
            target = Path(external) / 'recipe.json'
            target.write_text(json.dumps(document()))
            link = self.root / 'leaf/recipes' / (document()['id'] + '.json')
            link.parent.mkdir(parents=True)
            link.symlink_to(target)
            with self.assertRaisesRegex(HemError, 'escapes'):
                recipes.load(document()['id'])


class InterfaceTest(unittest.TestCase):
    def test_parameter_and_descriptor_values_are_ordered_and_immutable(self):
        value = descriptor()
        self.assertEqual(tuple(value.parameters), ('rows', 'mtp', 'draft_depth'))
        self.assertEqual([item.state for item in value.parameters.values()], ['configurable', 'fixed', 'unsupported'])
        self.assertEqual(value.protocol.input_modalities, ('text',))
        self.assertEqual(value.artifacts[0].formats, ('gguf',))
        self.assertIsInstance(value.programs.providers, tuple)
        self.assertIsInstance(value.bundle.files, tuple)
        with self.assertRaises(TypeError):
            value.parameters['rows'] = None
        default = {'nested': [1]}
        configurable = Configurable('object', default, active_when={'parameter': 'mtp', 'equals': True})
        default['nested'].append(2)
        self.assertEqual(configurable.default['nested'], (1,))
        with self.assertRaises(TypeError):
            configurable.active_when['equals'] = False
        plan = AcquisitionPlan('recipe', 'pipeline', [], [], None)
        self.assertEqual(plan.inputs, ())
        self.assertEqual(RecognitionPrograms('p', 'c', ['s']).sources, ('s',))

    def test_selector_imports_only_requested_module_and_checks_identity(self):
        binding = SimpleNamespace(describe=lambda: descriptor())
        module = SimpleNamespace(get=lambda identifier: binding)
        with patch.object(pipelines, 'import_module', return_value=module) as imported:
            self.assertIs(pipelines.get('qwen-08b'), binding)
            imported.assert_called_once_with('binding.python.models.pipelines_qwen')
        with self.assertRaisesRegex(HemError, 'Unknown pipeline'):
            pipelines.get('unknown')
        with patch.object(pipelines, 'import_module', side_effect=ImportError('optional dependency')):
            with self.assertRaisesRegex(HemError, 'unavailable'):
                pipelines.get('qwen')
        with patch.object(pipelines, 'import_module', return_value=module):
            with self.assertRaisesRegex(HemError, 'inconsistent'):
                pipelines.get('qwen')

    def test_parser_import_and_parse_do_not_load_acquisition_state_or_concrete_bindings(self):
        code = '''
import sys
from pathlib import Path
from binding.python.models import recipes
recipes.parse(Path(sys.argv[1]).read_bytes())
for name in sys.modules:
    assert not any(part in name for part in (
        'huggingface', 'model_download', 'moonshine_preparation',
        'hem_state', 'hem_models', 'pipelines_qwen', 'pipelines_moonshine', 'modalities')) , name
'''
        subprocess.run([sys.executable, '-s', '-c', code, str(EXAMPLES[0])], cwd=ROOT, check=True)


if __name__ == '__main__':
    unittest.main()
