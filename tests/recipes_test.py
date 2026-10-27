"""Format-2 authored JSON, selected admission and finite lazy module lookup."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from binding.python.hem_common import Deadline, HemError
from binding.python.models import pipelines, recipes

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


class ParserTest(unittest.TestCase):
    def test_actual_three_documents_are_owned_plain_json_with_exact_byte_identity(self):
        for path in EXAMPLES:
            raw = path.read_bytes()
            value, identity = recipes.parse(raw)
            self.assertEqual(type(value), dict)
            self.assertEqual(value, json.loads(raw))
            self.assertEqual(value['id'], path.stem)
            self.assertEqual(identity, hashlib.sha256(raw).hexdigest())
            self.assertNotEqual(identity, recipes.parse(raw + b'\n')[1])
            value['configuration']['extra'] = {'items': [1]}
            self.assertNotIn('extra', recipes.parse(raw)[0]['configuration'])
        qwen08, qwen27, asr = [recipes.parse(p.read_bytes())[0] for p in EXAMPLES]
        self.assertEqual([v['executable'] for v in (qwen08, qwen27, asr)],
                         ['llm/qwen/qwen_08b/executable.json', 'llm/qwen/qwen_27b/executable.json',
                          'recognition/moonshine/streaming_tiny/executable.json'])
        self.assertEqual(tuple(qwen08['files']), ('weights', 'tokenizer'))
        self.assertEqual(qwen27['configuration']['mtp_depth'], 3)
        self.assertEqual(asr['files']['rotary']['source'], {'kind': 'generated'})

    def test_exact_envelope_scalar_and_file_types(self):
        cases = [('format', True), ('format', 3.0), ('format', 1), ('format', 2), ('id', '../bad'),
                 ('name', ''), ('name', 4), ('executable', '../bad'), ('executable', []),
                 ('files', []), ('files', {}), ('configuration', [])]
        for key, item in cases:
            with self.subTest(key=key, item=item), self.assertRaises(HemError):
                value = document()
                value[key] = item
                admit(value)
        for key in ('extra', 'revision', 'modality', 'pipeline', 'artifacts'):
            value = document()
            value[key] = 1
            with self.subTest(key=key), self.assertRaises(HemError):
                admit(value)
        for key in document():
            value = document()
            del value[key]
            with self.subTest(missing=key), self.assertRaises(HemError):
                admit(value)
        for item in ([], None, {'path': 'model.gguf'},
                     dict(document()['files']['weights'], format='gguf')):
            value = document()
            value['files']['weights'] = item
            with self.subTest(file=item), self.assertRaises(HemError):
                admit(value)

    def test_strict_json_at_every_depth_and_document_limits(self):
        raw = EXAMPLES[0].read_bytes()
        invalid = [raw.replace(b'"format": 3', b'"format": 3, "format": 3'),
                   raw.replace(b'"weights": {', b'"weights": {}, "weights": {'),
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

    def test_exact_source_tags_immutable_pins_and_purpose_identifiers(self):
        changes = [dict(kind='local', path='x'), dict(kind='generated', helper='command'),
                   dict(kind='huggingface', repo_id='org/repo', revision='main', filename='model'),
                   dict(kind='huggingface', repo_id='org/repo', revision='A' * 40, filename='model'),
                   dict(kind='huggingface', repo_id='org/repo', revision=4, filename='model'),
                   None, [], {'kind': False}]
        for source in changes:
            value = document()
            value['files']['weights']['source'] = source
            with self.subTest(source=source), self.assertRaises(HemError):
                admit(value)
        for purpose in ('../weights', '', '-weights'):
            value = document()
            value['files'][purpose] = value['files'].pop('weights')
            with self.subTest(purpose=purpose), self.assertRaises(HemError):
                admit(value)

    def test_paths_and_optionlike_source_locators(self):
        bad_paths = ['', '/', '/absolute', 'a//b', '.', '..', 'a/../b', 'a/./b',
                     'a/', 'a\\b', 'a\0b', 'C:/absolute', 'C:relative', None, 4]
        for key in ('path', 'repo_id', 'filename'):
            for bad in bad_paths + (['-repo', 'org/-repo'] if key != 'path' else []):
                value = document()
                target = value['files']['weights'] if key == 'path' else value['files']['weights']['source']
                target[key] = bad
                with self.subTest(key=key, bad=bad), self.assertRaises(HemError):
                    admit(value)

    def test_duplicate_coordinates_and_overlapping_destinations(self):
        for change in ('source', 'path', 'parent', 'child'):
            value = document()
            first, second = value['files'].values()
            if change in ('source', 'path'):
                second[change] = copy.deepcopy(first[change])
            elif change == 'parent':
                second['path'] = 'model.gguf/nested'
            else:
                first['path'] = 'tokenizer.json/nested'
            with self.subTest(change=change), self.assertRaises(HemError):
                admit(value)

    def test_parser_defers_concrete_purpose_and_configuration_policy(self):
        value = document()
        value['files']['weights']['source'] = {'kind': 'generated'}
        value['files']['custom'] = value['files'].pop('weights')
        value['configuration'] = {'nested': [True, {'value': None}]}
        self.assertEqual(admit(value)[0], value)


def executable_document():
    return json.loads((ROOT / 'loom-pipeline' / document()['executable']).read_bytes())


class ActualDiscoveryTest(unittest.TestCase):
    def test_all_current_recipes_executables_and_program_hash_closures(self):
        found = recipes.list()
        self.assertEqual([r['id'] for r, _ in found], sorted(p.stem for p in EXAMPLES))
        inventory = pipelines.list()
        self.assertEqual(len(inventory), 3)
        for path in EXAMPLES:
            recipe, identity = recipes.load(path.stem)
            self.assertEqual(recipe, json.loads(path.read_bytes()))
            self.assertEqual(identity, hashlib.sha256(path.read_bytes()).hexdigest())
            selected = recipes.list(executable=recipe['executable'])
            self.assertEqual(selected, ((recipe, identity),))
            executable, catalog, checksum = pipelines.load(recipe['executable'])
            # Product imports may be the source package or the ordinary installed
            # wheel. Exact byte identity belongs to the selected declaration;
            # compiled normalization is separately bound to authored source bytes.
            raw = (pipelines._root() / recipe['executable']).read_bytes()
            self.assertEqual(checksum, hashlib.sha256(raw).hexdigest())
            authored = (ROOT / 'loom-pipeline' / recipe['executable']).read_bytes()
            source = json.loads(authored)
            if executable['programs']['encoding'] == 'source':
                self.assertEqual(raw, authored)
                self.assertIsNone(executable['provenance'])
            else:
                self.assertEqual(executable['provenance'], {
                    'source_declaration_sha256': hashlib.sha256(authored).hexdigest(),
                    'source_files': source['programs']['files'],
                })
                for field in ('format', 'id', 'modality', 'native_contract', 'serving', 'assets', 'preparation'):
                    self.assertEqual(executable[field], source[field])
            native = pipelines.programs(executable, catalog, Deadline.after(10))
            self.assertEqual(native['prepare'], executable['programs']['prepare'])
            self.assertEqual(native['sources'], executable['programs']['sources'])
        self.assertEqual(len(pipelines.list(modality='chat')), 2)
        self.assertEqual(len(pipelines.list(modality='asr')), 1)
        with self.assertRaises(HemError):
            pipelines.list(modality='tts')

    def test_validate_files_returns_fresh_selected_operation_data_or_none(self):
        from binding.python.models.preparation.moonshine_contract import load_operation
        for path in EXAMPLES:
            recipe, _ = recipes.load(path.stem)
            executable, _, _ = pipelines.load(recipe['executable'])
            admitted = pipelines.validate_files(executable, recipe['files'])
            if executable['preparation'] is None:
                self.assertIsNone(admitted)
            else:
                self.assertEqual(admitted, load_operation(executable['preparation']['operation']))
                admitted['tool_revision'] = '0' * 40
                fresh = pipelines.validate_files(executable, recipe['files'])
                self.assertNotEqual(fresh['tool_revision'], admitted['tool_revision'])

    def test_nonpreparing_selection_does_not_load_unrelated_operation_data(self):
        from binding.python.models.preparation import moonshine_contract as operation
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(operation, 'operation_path', return_value=Path(directory) / 'absent.json'):
                self.assertEqual(recipes.load(EXAMPLES[0].stem)[0]['id'], EXAMPLES[0].stem)
                with self.assertRaises(HemError):
                    recipes.load(EXAMPLES[2].stem)


class ExecutableParserTest(unittest.TestCase):
    def reject(self, value):
        with self.assertRaises(HemError):
            pipelines.parse(json.dumps(value).encode())

    def test_closed_envelope_native_pair_and_serving_record(self):
        for key, bad in (('format', True), ('format', 1.0), ('format', 2),
                         ('id', '../bad'), ('modality', 'tts'), ('modality', []),
                         ('native_contract', 'recognition-v1'), ('serving', {})):
            value = executable_document()
            value[key] = bad
            with self.subTest(key=key):
                self.reject(value)
        for key in executable_document():
            value = executable_document()
            del value[key]
            with self.subTest(missing=key):
                self.reject(value)
        value = executable_document()
        value['capabilities'] = {}
        self.reject(value)
        value = executable_document()
        value['serving']['mtp_depth_max'] = True
        self.reject(value)
        value = executable_document()
        value['serving']['bounds']['rows'] = [8, 1]
        self.reject(value)
        value = executable_document()
        value['serving']['defaults']['rows'] = 100
        self.reject(value)

    def test_asset_policy_is_native_contract_specific_without_model_ids(self):
        for bad in ({}, [], {'new': {'format': 'json', 'path': None}}):
            value = executable_document()
            value['assets'] = bad
            self.reject(value)
        for key, bad in (('format', 'arbitrary'), ('path', '../bad'), ('path', 'weights.json')):
            value = executable_document()
            value['assets']['weights'][key] = bad
            self.reject(value)
        value = executable_document()
        value['id'] = 'new-model-unmentioned-by-product'
        value['serving']['bounds']['rows'] = [1, 8]
        self.assertEqual(pipelines.parse(json.dumps(value).encode())[0], value)

    def test_closed_source_program_roles_paths_inventory_hash_and_provenance(self):
        cases = [('root', '../escape'), ('encoding', 'plugin'), ('prepare', '/absolute'),
                 ('sources', []), ('sources', ['x', 'x']), ('support', None),
                 ('files', {}), ('files', {'../escape.loom': 'a' * 64}),
                 ('files', {'prepare.loom': 'A' * 64})]
        for key, bad in cases:
            value = executable_document()
            value['programs'][key] = bad
            with self.subTest(key=key, bad=bad):
                self.reject(value)
        value = executable_document()
        del value['programs']['files'][value['programs']['support'][0]]
        self.reject(value)
        value = executable_document()
        value['programs']['support'].append(value['programs']['prepare'])
        self.reject(value)
        value = executable_document()
        value['provenance'] = {'source_files': {}}
        self.reject(value)

    def test_preparation_binding_shape_and_duplicate_purpose_fail(self):
        for bad in ({}, {'operation': 'command', 'inputs': {}, 'outputs': {}},
                    {'operation': 'op', 'inputs': {'a': 'weights'}, 'outputs': {'b': 'weights'}}):
            value = executable_document()
            value['preparation'] = bad
            self.reject(value)


class CatalogIsolationTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        patched = patch.object(pipelines, 'catalog_roots', return_value=(self.root,))
        patched.start()
        self.addCleanup(patched.stop)

    def write(self, path, value):
        target = self.root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(value))
        return target

    def pair(self, folder='new/chat', *, modality='chat', compiled=False):
        value = executable_document()
        value['id'] = folder.replace('/', '-')
        value['modality'] = modality
        value['native_contract'] = 'text-v1' if modality == 'chat' else 'recognition-v1'
        if modality == 'asr':
            value['serving'] = {'pending_requests': [1, 8], 'defaults': {'pending_requests': 2, 'options': {}}}
            value['assets'] = {'samples': {'format': 'json', 'path': 'sample.json'}}
        extension = 'loombc' if compiled else 'loom'
        name = 'native.' + extension
        content = b'LOOM-test-fixture' if compiled else b'// admission fixture, not inference\n'
        value['programs'] = {'root': folder, 'encoding': 'bytecode' if compiled else 'source',
                             'prepare': name, 'control': name, 'sources': [name], 'support': [],
                             'files': {name: hashlib.sha256(content).hexdigest()}}
        value['preparation'] = None
        value['provenance'] = {'source_declaration_sha256': 'a' * 64, 'source_files': {'native.loom': 'b' * 64}} if compiled else None
        target = self.root / folder / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        reference = folder + '/executable.json'
        self.write(reference, value)
        recipe = {'format': 3, 'id': folder.replace('/', '-') + '-model', 'name': 'New data-only model',
                  'executable': reference, 'configuration': copy.deepcopy(value['serving']['defaults']),
                  'files': {p: {'path': a['path'] or ('weights.gguf' if p == 'weights' else 'tokenizer.json'),
                                'source': {'kind': 'huggingface', 'repo_id': 'example/data', 'revision': 'a' * 40,
                                           'filename': a['path'] or ('weights.gguf' if p == 'weights' else 'tokenizer.json')}}
                            for p, a in value['assets'].items()}}
        self.write(folder + '/recipes/' + recipe['id'] + '.json', recipe)
        return reference, value, recipe

    def test_new_chat_and_asr_source_and_compiled_onboard_with_data_only(self):
        for compiled in (False, True):
            for modality in ('chat', 'asr'):
                reference, executable, recipe = self.pair('unlisted/' + modality + ('-bc' if compiled else '-src'), modality=modality, compiled=compiled)
                selected, root, _ = pipelines.load(reference)
                self.assertEqual(selected, executable)
                self.assertEqual(recipes.load(recipe['id'])[0], recipe)
                self.assertEqual(recipes.list(executable=reference)[0][0], recipe)
                pipelines.programs(selected, root, Deadline.after(10))
        self.assertEqual(len(pipelines.list()), 4)

    def test_selected_lookup_ignores_unrelated_malformed_recipe_and_manifest(self):
        reference, _, recipe = self.pair()
        broken = self.write('unrelated/recipes/broken.json', {})
        broken.write_text('not JSON')
        manifest = self.write('unrelated/executable.json', {})
        manifest.write_text('not JSON')
        self.assertEqual(recipes.load(recipe['id'])[0], recipe)
        self.assertEqual(recipes.list(executable=reference)[0][0], recipe)
        self.assertEqual(pipelines.load(reference)[0]['id'], 'new-chat')
        with self.assertRaises(HemError):
            recipes.list()
        with self.assertRaises(HemError):
            pipelines.list()

    def test_unknown_duplicate_filename_and_selected_reference_mismatch(self):
        reference, value, recipe = self.pair()
        with self.assertRaises(HemError):
            recipes.load('unknown')
        self.write('elsewhere/recipes/' + recipe['id'] + '.json', recipe)
        with self.assertRaisesRegex(HemError, 'Duplicate'):
            recipes.load(recipe['id'])
        (self.root / 'elsewhere/recipes' / (recipe['id'] + '.json')).unlink()
        self.write('new/chat/recipes/wrong.json', recipe)
        with self.assertRaisesRegex(HemError, 'filename'):
            recipes.load('wrong')
        (self.root / 'new/chat/recipes/wrong.json').unlink()
        value2 = copy.deepcopy(value)
        value2['programs']['root'] = 'other'
        self.write('other/executable.json', value2)
        with self.assertRaisesRegex(HemError, 'Duplicate'):
            pipelines.list()
        recipe['executable'] = 'other/executable.json'
        self.write('new/chat/recipes/' + recipe['id'] + '.json', recipe)
        with self.assertRaisesRegex(HemError, 'selected catalog'):
            recipes.list(executable=reference)

    def test_misplaced_recipe_rejects_direct_and_global_without_polluting_selected_inventory(self):
        reference, value, recipe = self.pair()
        misplaced = copy.deepcopy(recipe)
        misplaced['id'] = 'misplaced-recipe'
        self.write('unrelated/recipes/misplaced-recipe.json', misplaced)
        with self.assertRaisesRegex(HemError, 'recipe location'):
            recipes.load('misplaced-recipe')
        with self.assertRaisesRegex(HemError, 'recipe location'):
            recipes.list()
        self.assertEqual([r['id'] for r, _ in recipes.list(executable=reference)], [recipe['id']])

    def test_catalog_root_resolution_errors_stay_domain_errors_with_original_cause(self):
        reference, _, recipe = self.pair()
        for failure in (PermissionError('catalog denied'), RuntimeError('catalog resolution loop')):
            for lookup in (lambda: pipelines.load(reference), lambda: recipes.load(recipe['id'])):
                with self.subTest(failure=type(failure).__name__), patch.object(Path, 'resolve', side_effect=failure):
                    with self.assertRaises(HemError) as raised:
                        lookup()
                    self.assertIs(raised.exception.__cause__, failure)

    def test_direct_reference_and_program_root_cannot_escape(self):
        reference, value, _ = self.pair()
        for bad in ('../escape/executable.json', '/abs/executable.json', 'C:/executable.json',
                    'new\\executable.json', 'new//executable.json', 'new/-x/executable.json', 'id', None, []):
            with self.subTest(reference=bad), self.assertRaises(HemError):
                pipelines.load(bad)
        with tempfile.TemporaryDirectory() as external:
            target = Path(external) / 'executable.json'
            target.write_text(json.dumps(value))
            link = self.root / 'escaped/executable.json'
            link.parent.mkdir()
            link.symlink_to(target)
            with self.assertRaises(HemError):
                pipelines.load('escaped/executable.json')
        value['programs']['root'] = 'other'
        (self.root / 'other').mkdir()
        self.write(reference, value)
        with self.assertRaises(HemError):
            pipelines.load(reference)

    def test_recipe_less_executable_rejects_unknown_operation_in_selected_and_global_admission(self):
        reference, value, recipe = self.pair(modality='asr')
        (self.root / 'new/chat/recipes' / (recipe['id'] + '.json')).unlink()
        value['preparation'] = {'operation': 'unsupported-operation-v9',
                                'inputs': {'input.json': 'samples'},
                                'outputs': {'result.json': 'other'}}
        value['assets']['other'] = {'format': 'json', 'path': 'result.json'}
        self.write(reference, value)
        with self.assertRaisesRegex(HemError, 'Unsupported trusted preparation'):
            pipelines.load(reference)
        with self.assertRaisesRegex(HemError, 'Unsupported trusted preparation'):
            pipelines.list()

    def test_program_hash_missing_extra_encoding_and_deadline_negatives(self):
        reference, value, _ = self.pair(compiled=True)
        selected, root, _ = pipelines.load(reference)
        program = root / selected['programs']['prepare']
        original = program.read_bytes()
        for bad in (b'LOOM-corrupted', b'bad-header'):
            program.write_bytes(bad)
            with self.assertRaises(HemError):
                pipelines.programs(selected, root, Deadline.after(10))
        program.write_bytes(original)
        extra = root / 'extra.loombc'
        extra.write_bytes(b'LOOM-extra')
        with self.assertRaises(HemError):
            pipelines.programs(selected, root, Deadline.after(10))
        extra.unlink()
        program.unlink()
        with self.assertRaises(HemError):
            pipelines.programs(selected, root, Deadline.after(10))
        program.write_bytes(original)
        with self.assertRaises(HemError):
            pipelines.programs(selected, root, Deadline(0, clock=lambda: 1))
        selected['programs']['files'][program.name] = hashlib.sha256(b'bad-header').hexdigest()
        program.write_bytes(b'bad-header')
        with self.assertRaisesRegex(HemError, 'not Loom bytecode'):
            pipelines.programs(selected, root, Deadline.after(10))

    def test_selected_operation_and_asset_negatives_use_real_admission(self):
        reference, value, recipe = self.pair()
        for change in ('unknown-purpose', 'suffix', 'generated'):
            bad = copy.deepcopy(recipe)
            if change == 'unknown-purpose':
                bad['files']['other'] = bad['files'].pop('weights')
            elif change == 'suffix':
                bad['files']['weights']['path'] = 'weights.json'
            else:
                bad['files']['weights']['source'] = {'kind': 'generated'}
            self.assertEqual(recipes.parse(json.dumps(bad).encode())[0], bad)
            self.write('new/chat/recipes/' + recipe['id'] + '.json', bad)
            with self.subTest(change=change), self.assertRaises(HemError):
                recipes.load(recipe['id'])


class ParserIsolationTest(unittest.TestCase):
    def test_parse_has_no_model_preparation_registry_or_hf_imports(self):
        code = '''
import sys
from pathlib import Path
from binding.python.models import recipes
recipes.parse(Path(sys.argv[1]).read_bytes())
for name in sys.modules:
    assert not any(part in name for part in (
        'huggingface', 'model_download', 'moonshine_preparation', 'recipe_types',
        'hem_state', 'hem_models', 'pipelines_qwen', 'pipelines_moonshine', 'modalities')) , name
'''
        subprocess.run([sys.executable, '-s', '-E', '-c', code, str(EXAMPLES[0])], cwd=ROOT, check=True)


if __name__ == '__main__':
    unittest.main()
