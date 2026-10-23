"""ASR uses common checked resolution, independent of acquisition or native load."""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from binding.python.models import hem_config
from binding.python.hem_common import InvalidInput
from tests.recipe_config_fixture import ConfigurationFixture


class CuratedAssetsTest(ConfigurationFixture, unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.configure('moonshine-streaming-tiny')

    def test_matching_fixed_config_and_all_dependencies_use_common_resolve(self):
        path = self.config_file('{"pending_requests":4,"options":{}}')
        checked = hem_config.resolve(self.recipe, self.model, config_path=path)
        self.assertEqual(checked.settings, {'pending_requests': 4, 'options': {}})
        self.assertEqual(checked.modality, 'asr')
        self.assertEqual(set(checked.dependencies), {item.path for item in self.model.files})
        self.assertEqual(checked.definition['kind'], 'recognition')

    def test_missing_generated_assets_and_bad_artifact_inventory_fail(self):
        generated = next(item for item in self.model.files if item.ownership == 'generated-owned')
        generated.path.unlink()
        with self.assertRaisesRegex(InvalidInput, 'unreadable'):
            hem_config.resolve(self.recipe, self.model)

    def test_complete_import_never_calls_preparation_or_preflight(self):
        self.model = replace(self.model, files=tuple(replace(item, ownership='caller-owned') for item in self.model.files))
        with patch('subprocess.Popen', side_effect=AssertionError('helper spawned')):
            result = hem_config.resolve(self.recipe, self.model)
        self.assertEqual(result.settings['pending_requests'], 4)

    def test_fixed_unknown_and_generated_path_overrides_fail(self):
        for layer in ({'pending_requests': 8}, {'options': {'unknown': True}}, {'rows': 2}, {'rotary': 'other'}):
            with self.subTest(layer=layer), self.assertRaises(InvalidInput):
                hem_config.resolve(self.recipe, self.model, layer)
