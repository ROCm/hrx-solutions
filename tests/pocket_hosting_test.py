"""Prepared Pocket imports exercise ordinary ownership and checked configuration."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from binding.python.control.hem_state import Registry
from binding.python.hem_common import InvalidInput
from binding.python.models import hem_config, hem_models, pocket, recipes
from binding.python.models.modalities import tts


class PocketHostingTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.prepared = self.root / 'prepared'
        self.prepared.mkdir()
        for name in ('model.safetensors', 'tokenizer.json', 'rotary.safetensors', 'voices.safetensors'):
            (self.prepared / name).write_bytes(b'opaque native payload fixture')
        self.geometry = dict(context_capacity=512, rotary_capacity=4096, chunk_tokens=50,
                             voices=[dict(id='alba', frames=126)])
        self.write_helper()

    def write_helper(self):
        (self.prepared / 'model.json').write_text(json.dumps({'name': 'untrusted name',
            'base_directory': '/untrusted', 'definition': {'prepare': '/untrusted.loom',
            'options': self.geometry}}))

    def test_copy_alias_keep_owned_metadata_and_export_matches_resolve(self):
        registry = Registry(self.root / 'state')
        for alias in (False, True):
            name = 'alias' if alias else 'copy'
            with patch('binding.python.models.preparation.preflight', side_effect=AssertionError('no preparation')):
                hem_models.import_assets(registry, name, directory=self.prepared,
                                         alias=alias, recipe_id='pocket-tts-alba')
            record, paths, dependencies, evidence = hem_models.qualify(registry, name, modality='tts')
            self.assertFalse(paths['prepared_metadata'].is_symlink())
            self.assertEqual(paths['checkpoint'].is_symlink(), alias)
            resolved = hem_config.resolve(record, paths, dependencies, evidence)
            definition = hem_config.json_document(resolved.definition)
            self.assertEqual(definition['options'], {**self.geometry, **tts.defaults(
                record['provenance']['executable']['declaration']['serving'])['options']})
            self.assertNotEqual(definition['prepare'], '/untrusted.loom')
            exported = self.root / (name + '.json')
            hem_config.export_deployment(record, paths, dependencies, evidence, exported, 'reading-demo')
            self.assertEqual(json.loads(exported.read_text())['definition'], definition)
            paths['prepared_metadata'].write_text('{}')
            with self.assertRaisesRegex(InvalidInput, 'identity is unproven'):
                hem_models.qualify(registry, name)

    def test_variable_geometry_and_portable_input_conflict(self):
        self.geometry.update(context_capacity=1024, rotary_capacity=8192)
        self.geometry['voices'][0]['frames'] = 200
        self.write_helper()
        self.assertEqual(pocket.helper_geometry(self.prepared / 'model.json'), self.geometry)
        portable = dict(format=1, asset_contract='pocket-alba-assets-v1', profile='alba', geometry=self.geometry)
        (self.prepared / 'prepared-metadata.json').write_text(json.dumps(portable))
        pocket.import_metadata(self.prepared)
        portable['geometry'] = {**self.geometry, 'context_capacity': 2048}
        (self.prepared / 'prepared-metadata.json').write_text(json.dumps(portable))
        with self.assertRaisesRegex(InvalidInput, 'Conflicting'):
            pocket.import_metadata(self.prepared)
        (self.prepared / 'model.json').unlink()
        self.assertIn(b'2048', pocket.import_metadata(self.prepared))

    def test_malformed_metadata_and_readonly_settings_rejected(self):
        serving = recipes.load('pocket-tts-alba')[0]['configuration']
        # Settings are checked through the declaration, not metadata scheduling fields.
        from binding.python.models import pipelines
        serving = pipelines.load('speech/pocket_tts/executable.json')[0]['serving']
        for values in ({'context_capacity': 512}, {'pending_requests': True},
                       {'options': {'context_capacity': 512}}, {'rows': 4}):
            with self.subTest(values=values), self.assertRaises(InvalidInput):
                tts.validate_layer(serving, values)
        for mutation in (lambda x: x.update(context_capacity=True),
                         lambda x: x['voices'][0].update(id='anna'),
                         lambda x: x.update(rotary_capacity=16)):
            values = deepcopy(self.geometry)
            mutation(values)
            with self.assertRaises(InvalidInput):
                pocket.geometry(values)
        (self.prepared / 'model.json').write_text('{"definition":{},"definition":{}}')
        with self.assertRaisesRegex(InvalidInput, 'duplicate'):
            pocket.import_metadata(self.prepared)


if __name__ == '__main__':
    unittest.main()
