"""The old manager is retired without adopting or deleting its user state."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from binding.python import runtime_paths
from binding.python.models import hem_models, recipes, model_store
from binding.python.control.hem_state import Registry
from binding.python.hem_common import HemError


class MigrationBoundaryTest(unittest.TestCase):
    def test_legacy_inventory_is_not_adopted_or_modified(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / 'loom'
            legacy = {
                'models/old/manifest.json': {'model_id': 'old', 'pipeline': 'qwen-08b'},
                'models/old/model.gguf': 'immutable external input',
                'servers/old.json': {'pid': os.getpid(), 'status': 'running'},
                'connections/old.json': {'pid': os.getpid()},
                'remotes.json': {'old': {'host': 'localhost'}},
            }
            for name, value in legacy.items():
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(value))
            before = {name: (root / name).read_bytes() for name in legacy}
            with patch('subprocess.Popen', side_effect=AssertionError('no legacy process adoption')), patch('os.kill', side_effect=AssertionError('no legacy process termination')):
                registry = Registry(root)
                self.assertEqual(registry.list_models(), [])
                self.assertEqual(registry.list_servers(), [])
                self.assertEqual(registry.list_instances(), [])
                self.assertEqual(hem_models.inventory(registry), [])
            self.assertEqual({name: (root / name).read_bytes() for name in legacy}, before)

    def test_runtime_cache_resolves_per_call_without_creating_directories(self):
        with tempfile.TemporaryDirectory() as temporary:
            home = Path(temporary)
            root = home / 'isolated'
            with patch.object(Path, 'home', return_value=home), patch.dict(os.environ, {}, clear=True):
                self.assertEqual(runtime_paths.state_root(), home / '.loom')
                self.assertEqual(runtime_paths.runtime_cache_dir(), home / '.cache/hrx-solutions')
                with patch.dict(os.environ, {'LOOM_SERVE_DIR': str(root)}):
                    self.assertEqual(runtime_paths.runtime_cache_dir(), root / 'cache')
            self.assertEqual(list(home.iterdir()), [])

    def test_shared_asset_checks_reject_invalid_ids_and_broken_aliases(self):
        for model_id in ('../old', '.', '', '/absolute'):
            with self.subTest(model_id=model_id), self.assertRaises(model_store.ModelImportError):
                model_store.validate_model_id(model_id)
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / 'missing'
            alias = Path(temporary) / 'alias.gguf'
            alias.symlink_to(target)
            self.assertFalse(model_store.readable_file(alias))
            target.write_bytes(b'GGUF')
            self.assertTrue(model_store.readable_file(alias))

    def test_unknown_recipe_is_actionable_from_common_loader(self):
        with self.assertRaisesRegex(HemError, 'Unknown recipe'):
            recipes.load('missing')
