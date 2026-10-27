"""Distinct modality, executable and recipe discovery through the public grammar."""
import argparse
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
import io
import json
import hashlib
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from binding.python import hem_cli
from binding.python.cli import hem_arguments, hem_executable_cli, hem_help, hem_modality_cli, hem_render
from binding.python.hem_common import HemError, InvalidInput
from binding.python.models import hem_config, pipelines, recipes

REF = 'unseen/chat/compact/executable.json'
OTHER = 'unreferenced/chat/wide/executable.json'


def executable(reference=REF, identifier='compact-program', capacity=7):
    return {'executable': reference, 'executable_id': identifier, 'modality': 'chat',
            'native_contract': 'text-v1', 'catalog': '/package/' + reference.rsplit('/', 1)[0],
            'protocol': {'operation': 'chat', 'endpoint': '/v1/chat/completions', 'request_encoding': 'json'},
            'parameters': {'rows': {'state': 'configurable', 'type': 'integer', 'default': capacity,
                                    'constraints': {'minimum': 1, 'maximum': capacity + 1}}},
            'programs': {'prepare': 'main.loombc', 'control': 'main.loombc', 'providers': ['main.loombc']},
            'telemetry': {'raw_events': 'unverified', 'metrics': []}}


def modality(name='chat'):
    return {'modality': name, 'native_contract': 'text-v1' if name == 'chat' else 'recognition-v1',
            'protocol': {'operation': 'chat' if name == 'chat' else 'transcription',
                         'endpoint': '/v1/chat/completions' if name == 'chat' else '/v1/audio/transcriptions',
                         'request_encoding': 'json' if name == 'chat' else 'multipart'},
            'executables': [executable(), executable(OTHER, 'wide-program', 23)] if name == 'chat' else []}


class DiscoveryCliTest(unittest.TestCase):
    def invoke(self, argv):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            try:
                status = hem_cli.main(argv)
            except SystemExit as exc:
                status = exc.code
        return status, out.getvalue(), err.getvalue()

    def test_modality_show_never_names_a_concrete_model(self):
        for name in ('chat', 'asr'):
            args = hem_cli.parse_args(['modality', 'show', name])
            self.assertEqual(args.modality, name)
            self.assertFalse(hasattr(args, 'pipeline'))
        for name in ('compact-program', 'qwen-08b', REF):
            with self.subTest(name=name), self.assertRaises(InvalidInput):
                hem_cli.parse_args(['modality', 'show', name])
        with self.assertRaises(InvalidInput):
            hem_cli.parse_args(['pipeline', 'list'])

    def test_modality_list_uses_one_complete_view_without_id_relookup(self):
        budget, expected = object(), [modality(), modality('asr')]
        args = hem_cli.parse_args(['modality', 'list'])
        with patch.object(hem_config, 'modality_list', return_value=expected, create=True) as listing, \
             patch.object(hem_config, 'modality_info', create=True) as lookup:
            result = hem_modality_cli.run(args, budget)
        listing.assert_called_once_with(deadline=budget)
        lookup.assert_not_called()
        self.assertEqual(result, {'state': 'available', 'completed': True, 'items': expected})

    def test_modality_show_preserves_original_deadline_and_contract_view(self):
        budget = object()
        args = hem_cli.parse_args(['modality', 'show', 'asr'])
        with patch.object(hem_config, 'modality_info', return_value=modality('asr'), create=True) as info:
            result = hem_modality_cli.run(args, budget)
        info.assert_called_once_with('asr', deadline=budget)
        self.assertEqual(result['modality'], 'asr')
        self.assertNotIn('parameters', result)

    def test_executable_list_is_independent_of_recipe_membership_and_filters_only_modality(self):
        rows = [executable(), executable(OTHER, 'wide-program', 23)]
        budget = object()
        args = hem_cli.parse_args(['executable', 'list', '--modality', 'chat'])
        with patch.object(hem_config, 'executable_list', return_value=rows, create=True) as listing, \
             patch.object(recipes, 'list', side_effect=AssertionError('CLI must not derive executable inventory from recipes')):
            result = hem_executable_cli.run(args, budget)
        listing.assert_called_once_with('chat', deadline=budget)
        self.assertEqual([row['executable'] for row in result['items']], [REF, OTHER])
        self.assertEqual(result['items'][1]['parameters']['rows']['default'], 23)

    def test_executable_show_forwards_direct_reference_without_name_alias(self):
        args = hem_cli.parse_args(['executable', 'show', REF])
        budget = object()
        with patch.object(hem_config, 'executable_info', return_value=executable(), create=True) as info:
            result = hem_executable_cli.run(args, budget)
        info.assert_called_once_with(REF, deadline=budget)
        self.assertEqual(result['executable_id'], 'compact-program')

    def test_executable_invalid_modality_and_missing_reference_fail_in_parser(self):
        for argv in (['executable', 'list', '--modality', 'compact-program'],
                     ['executable', 'show']):
            with self.subTest(argv=argv), self.assertRaises(InvalidInput):
                hem_cli.parse_args(argv)

    def test_reader_domain_error_remains_json_error_without_state_initialization(self):
        with patch.object(hem_config, 'executable_info', side_effect=HemError('unsafe executable reference'), create=True), \
             patch('binding.python.cli.hem_model_cli.Registry', side_effect=AssertionError('Registry')):
            code, out, err = self.invoke(['executable', 'show', '../executable.json', '--json'])
        self.assertNotEqual(code, 0)
        self.assertIn('unsafe executable reference', json.loads(out)['error'])

    def test_json_keeps_each_selected_capability_and_all_identities(self):
        expected = {'state': 'available', 'completed': True, 'items': [executable(), executable(OTHER, 'wide-program', 23)]}
        before = deepcopy(expected)
        with patch.object(hem_config, 'executable_list', return_value=expected['items'], create=True):
            code, out, err = self.invoke(['executable', 'list', '--json'])
        self.assertEqual(code, 0, err)
        self.assertEqual(json.loads(out), expected)
        self.assertEqual(expected, before)

    def test_modality_rendering_has_no_universal_capacity_or_default(self):
        args = SimpleNamespace(command='show', resource='modality')
        data = modality()
        data['vocabulary'] = {'rows': {'type': 'integer'}}
        result = hem_render.render(args, data, width=100)
        self.assertIn('Modality', result)
        self.assertIn('chat', result)
        self.assertIn('not to the modality', result)
        self.assertNotIn('Default / fixed value', result)
        self.assertIn('compact-program', result)
        self.assertIn('wide-program', result)
        self.assertIn('Setting', result)
        self.assertIn('rows', result)

    def test_executable_rendering_labels_concrete_identity_and_actual_limits(self):
        args = SimpleNamespace(command='show', resource='executable')
        result = hem_render.render(args, executable(), width=120)
        self.assertIn('Executable reference', result)
        self.assertIn('Executable ID', result)
        self.assertIn('compact-program', result)
        self.assertIn('maximum=8', result)
        self.assertIn('not acquired-model or native readiness', result)

    def test_recipe_and_local_model_views_label_all_entities_without_model_dispatch(self):
        args = SimpleNamespace(command='available', resource='model')
        row = {'id': 'new-recipe', 'name': 'new quantization', 'modality': 'chat',
               'executable_id': 'compact-program', 'executable': REF}
        result = hem_render.render(args, {'items': [row]}, width=140)
        for label in ('Recipe', 'Modality', 'Executable ID', 'Executable reference', 'new-recipe'):
            self.assertIn(label, result)
        model = {'model': 'local-model', 'pipeline': REF, 'modality': 'chat', 'assets': [],
                 'provenance': {'recipe_id': 'new-recipe', 'recipe_identity': 'a' * 64}}
        result = hem_render.render(SimpleNamespace(command='show', resource='model'), model, width=100)
        self.assertIn('Executable reference', result)
        self.assertNotIn('Pipeline:', result)
        self.assertIn('new-recipe', result)

    def test_discovery_help_is_pure_and_explains_distinct_selection(self):
        for resource in ('modality', 'executable'):
            with patch.object(hem_cli.DOMAINS[resource], 'run', side_effect=AssertionError('help must not execute')):
                code, out, err = self.invoke([resource, '--help'])
            self.assertEqual(code, 0, err)
            self.assertIn('Examples:', out)
        self.assertIn('no modality chooses', hem_help.ACTIONS['modality']['list'][0])
        self.assertIn('--recipe', hem_help.ACTIONS['model']['import'][1])
        self.assertIn('selected executable defaults', hem_help.configuration_reference())

    def test_shared_assertion_flags_are_optional_and_never_select_a_recipe(self):
        parser = argparse.ArgumentParser()
        hem_arguments.selection_assertions(parser)
        args = parser.parse_args([])
        self.assertIsNone(args.executable)
        self.assertIsNone(args.modality)
        args = parser.parse_args(['--executable', REF, '--modality', 'chat'])
        self.assertEqual((args.executable, args.modality), (REF, 'chat'))
        self.assertFalse(hasattr(args, 'recipe'))

    def test_actual_reader_inventory_includes_unreferenced_executable_and_distinct_capabilities(self):
        # Only the package location is redirected. Reader, admission, config
        # projections and public CLI all execute their real implementations.
        # Source metadata fixtures prove discovery, not compilation/inference.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / 'catalogs'
            state = Path(temporary) / 'must-not-exist'
            references = ('novel/compact/executable.json', 'novel/wide/executable.json')
            for reference, capacity in zip(references, (7, 23)):
                directory = root / Path(reference).parent
                directory.mkdir(parents=True)
                content = b'func.def public @discovery_fixture() {}\n'
                (directory / 'main.loom').write_bytes(content)
                document = dict(format=1, id='program-' + str(capacity), modality='chat',
                    native_contract='text-v1', assets={
                        'weights': {'format': 'gguf', 'path': None},
                        'tokenizer': {'format': 'json', 'path': None}},
                    serving={'bounds': {key: [1, capacity * 4] for key in
                        ('rows', 'context_capacity', 'pool_capacity', 'prefill_capacity')},
                        'mtp_depth_max': 0, 'defaults': {'rows': capacity,
                            'context_capacity': capacity, 'pool_capacity': capacity,
                            'prefill_capacity': capacity, 'mtp': False}},
                    programs={'root': Path(reference).parent.as_posix(), 'encoding': 'source',
                        'prepare': 'main.loom', 'control': 'main.loom', 'sources': ['main.loom'],
                        'support': [], 'files': {'main.loom': hashlib.sha256(content).hexdigest()}},
                    preparation=None, provenance=None)
                (directory / 'executable.json').write_text(json.dumps(document))
            with patch.object(pipelines, 'catalog_roots', return_value=(root,)), \
                 patch.dict('os.environ', {'LOOM_SERVE_DIR': str(state)}):
                self.assertEqual(recipes.list(), ())
                status, output, errors = self.invoke(['executable', 'list', '--json'])
                self.assertEqual(status, 0, errors)
                rows = json.loads(output)['items']
                self.assertEqual([row['executable'] for row in rows], list(references))
                self.assertEqual([row['parameters']['rows']['default'] for row in rows], [7, 23])
                status, output, errors = self.invoke(['modality', 'show', 'chat', '--json'])
                self.assertEqual(status, 0, errors)
                view = json.loads(output)
                self.assertNotIn('parameters', view)
                self.assertEqual(len(view['executables']), 2)
                status, output, errors = self.invoke(['executable', 'show', references[0], '--json'])
                self.assertEqual(status, 0, errors)
                self.assertEqual(json.loads(output)['executable_id'], 'program-7')
                # Selected metadata ignores unrelated malformed entries; strict
                # global inventory rejects them, even under a modality filter.
                (root / references[1]).write_text('{broken')
                self.assertEqual(self.invoke(['executable', 'show', references[0], '--json'])[0], 0)
                self.assertNotEqual(self.invoke(['executable', 'list', '--modality', 'chat', '--json'])[0], 0)
            self.assertFalse(state.exists())


if __name__ == '__main__':
    unittest.main()
