"""Loom state isolation without changing HOME or user cache settings."""
import io
import os
from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stderr
from unittest.mock import patch

from binding.python import (connection_manager, model_download, model_store,
                            pipeline_validation, remote_registry, server_manager, state_store)
from tests.gguf_fixtures import write_gguf


class StateDirectoryTest(unittest.TestCase):
    def test_unset_defaults_and_dynamic_override(self):
        with tempfile.TemporaryDirectory() as temporary:
            home = Path(temporary) / 'existing-home'
            with patch.dict(os.environ), patch.object(Path, 'home', return_value=home):
                os.environ.pop('LOOM_SERVE_DIR', None)
                self.assertEqual(state_store.state_root(), home / '.loom')
                self.assertEqual(model_store.default_model_root(), home / '.loom/models')
                self.assertEqual(state_store.runtime_cache_dir(), home / '.cache/hrx-solutions')
                for name in ('first', 'second'):
                    root = Path(temporary) / name
                    os.environ['LOOM_SERVE_DIR'] = str(root)
                    self.assertEqual(state_store.state_root(), root)
                    self.assertEqual(model_store.default_model_root(), root / 'models')
                    self.assertEqual(state_store.runtime_cache_dir(), root / 'cache')
                self.assertFalse(home.exists())

    def test_models_recipe_state_and_runtime_files_are_isolated(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root = base / 'loom'
            hf = base / 'hf-cache'
            environment = dict(os.environ)
            with patch.dict(os.environ, {'LOOM_SERVE_DIR': str(root), 'HF_HUB_CACHE': str(hf)}), \
                 patch.object(Path, 'home', side_effect=AssertionError('must not access HOME')):
                weights = base / 'weights.gguf'
                tokenizer = base / 'tokenizer.json'
                write_gguf(weights)
                tokenizer.write_text('{"model":{"type":"BPE","vocab":{"a":0}}}')
                with redirect_stderr(io.StringIO()):
                    model_store.import_model('example', gguf=weights, tokenizer=tokenizer, alias=True)
                    explicit = base / 'explicit-models'
                    model_store.import_model('other', gguf=weights, tokenizer=tokenizer,
                                             model_root=explicit, alias=True)
                self.assertEqual(model_store.model_files('example').weights.parent, root / 'models/example')
                self.assertEqual(model_store.model_files('other', explicit).weights.parent, explicit / 'other')
                model_store.record_pipeline('example', 'qwen-08b')
                self.assertEqual(model_store.read_manifest('example')['pipeline'], 'qwen-08b')
                self.assertEqual([model.model_id for model in model_store.model_inventory()], ['example'])
                self.assertTrue((root / 'models/.example.lock').is_file())
                plan = model_download.download('qwen35-08b-q5-k-xl', dry_run=True, offline=True)
                self.assertEqual(Path(plan['directory']).parent, root / 'models')
                self.assertEqual(model_download.cache_root(), hf)
                remote_registry.register_remote('test', 'localhost')
                self.assertTrue((root / 'remotes/test.json').is_file())
                self.assertTrue((root / 'state.lock').is_file())
                self.assertEqual(server_manager._record_path('test'), root / 'servers/test.json')
                self.assertEqual(connection_manager._record_path('test'), root / 'connections/test.json')
                with state_store.private_append_log(state_store.state_dir() / 'logs/test.log') as log:
                    log.write(b'log')
                source = pipeline_validation.resolve_catalog('qwen-08b')
                catalog = pipeline_validation.catalog_for_model('qwen-08b', source, 'example')
                self.assertTrue(catalog.is_relative_to(root / 'cache/catalogs'))
                deployment = pipeline_validation.deployment_for_model(
                    catalog, model_store.model_files('example'), 'example')
                self.assertEqual(deployment.parent, root / 'cache/deployments')
                self.assertTrue(deployment.is_file())
                self.assertTrue((root / 'logs/test.log').is_file())
            self.assertEqual(dict(os.environ), environment)
