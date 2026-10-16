"""Acquisition/storage contracts use tiny synthetic files; no model downloads."""
from contextlib import redirect_stderr
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from binding.python import hem_models as models, hem_config, hem_state
from binding.python.hem_common import HemError, InvalidInput
from tests.gguf_fixtures import write_gguf, write_matching_gguf, profile


class HemModelsTest(unittest.TestCase):
    def setUp(self):
        # Acquisition fixtures explicitly control offline behavior; a caller's
        # real-model offline policy must not change mocked retry expectations.
        offline = patch.dict(os.environ, {'HF_HUB_OFFLINE': '0'})
        offline.start()
        self.addCleanup(offline.stop)
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.registry = hem_state.Registry(self.root / 'state')
        self.gguf = self.root / 'input.gguf'
        write_gguf(self.gguf, tensors=[('tiny', [1], 0, 0)])
        self.tokenizer = self.root / 'tokenizer.json'
        self.tokenizer.write_text('{"model":{"fixture":true}}')
        self.qualifier = patch.object(hem_config, 'select_pipeline', return_value='qwen-08b')
        self.select = self.qualifier.start()
        self.addCleanup(self.qualifier.stop)
        self.output = redirect_stderr(io.StringIO())
        self.output.__enter__()
        self.addCleanup(self.output.__exit__, None, None, None)

    def acquire(self, name='tiny', **kwargs):
        return models.import_model(self.registry, name, ggufs=[self.gguf], tokenizer=self.tokenizer, **kwargs)

    def test_owned_copy_and_immutable_retry(self):
        row = self.acquire()
        target = Path(row['directory']) / 'model.gguf'
        self.assertFalse(target.is_symlink())
        original = target.read_bytes()
        self.gguf.write_bytes(b'changed external input')
        again = self.acquire()
        self.assertEqual(target.read_bytes(), original)
        self.assertEqual(len(self.registry.list_models()), 1)
        self.assertEqual(again['assets'], row['assets'])
        self.assertEqual(models.show(self.registry, 'tiny')['state'], 'ready')

    def test_alias_ownership_and_independent_tokenizers(self):
        first = self.acquire('first', alias=True)
        second_tokenizer = self.root / 'other' / 'tokenizer.json'
        second_tokenizer.parent.mkdir()
        second_tokenizer.write_text('{"different":true}')
        second = models.import_model(self.registry, 'second', ggufs=[self.gguf], tokenizer=second_tokenizer, alias=True)
        for row, token in ((first, self.tokenizer), (second, second_tokenizer)):
            path = Path(row['directory']) / 'tokenizer.json'
            self.assertTrue(path.is_symlink())
            self.assertEqual(path.resolve(), token)
        self.assertEqual(self.tokenizer.read_text(), '{"model":{"fixture":true}}')
        self.assertEqual(len(models.inventory(self.registry)), 2)

    def test_conflicting_id_preserves_original(self):
        self.acquire()
        before = self.registry.get_model('tiny')
        other = self.root / 'other.gguf'
        other.write_bytes(b'other')
        with self.assertRaisesRegex(HemError, 'different mappings'):
            models.import_model(self.registry, 'tiny', ggufs=[other], tokenizer=self.tokenizer)
        self.assertEqual(self.registry.get_model('tiny'), before)

    def test_invalid_and_ambiguous_assets_are_published_and_retryable(self):
        for name, diagnostic in [('invalid', 'incompatible'), ('ambiguous', 'Ambiguous compatible pipelines')]:
            self.select.side_effect = InvalidInput(diagnostic)
            with self.assertRaisesRegex(InvalidInput, diagnostic):
                self.acquire(name)
            row = self.registry.get_model(name)
            self.assertEqual(row['pending']['stage'], 'published-invalid')
            self.assertTrue((Path(row['directory']) / 'model.gguf').is_file())
            self.assertEqual(models.show(self.registry, name)['state'], 'invalid')
            self.select.side_effect = None
            self.acquire(name)
            self.assertIsNone(self.registry.get_model(name)['pending'])

    def test_missing_alias_inspectable_and_revalidation_fails(self):
        self.acquire(alias=True)
        self.gguf.unlink()
        self.assertEqual(models.show(self.registry, 'tiny')['state'], 'missing')
        with self.assertRaisesRegex(InvalidInput, 'missing'):
            models.validate(self.registry, 'tiny')
        self.assertFalse(self.registry.get_model('tiny')['validation']['valid'])
        self.assertTrue((self.registry.root / 'models/tiny/model.gguf').is_symlink())

    def test_incomplete_copy_retry_reuses_completed_members(self):
        contents = self.gguf.read_bytes()
        self.tokenizer.unlink()
        with self.assertRaisesRegex(HemError, 'incomplete acquisition retained'):
            self.acquire()
        row = self.registry.get_model('tiny')
        stage = Path(row['pending']['path'])
        inode = (stage / 'model.gguf').stat().st_ino
        self.gguf.unlink()
        self.tokenizer.write_text('{}')
        self.acquire()
        target = self.registry.root / 'models/tiny/model.gguf'
        self.assertEqual(target.stat().st_ino, inode)
        self.assertEqual(target.read_bytes(), contents)

    def test_atomic_partial_copy_is_not_reused(self):
        def fail(incoming, stream, **kwargs):
            stream.write(b'partial')
            raise OSError('interrupted copy')
        with patch.object(models.shutil, 'copyfileobj', side_effect=fail):
            with self.assertRaisesRegex(HemError, 'interrupted'):
                self.acquire()
        row = self.registry.get_model('tiny')
        self.assertFalse((Path(row['pending']['path']) / 'model.gguf').exists())
        (Path(row['pending']['path']) / '.model.gguf.partial').write_bytes(b'abrupt death')
        self.acquire()
        self.assertFalse((Path(row['directory']) / '.model.gguf.partial').exists())
        self.assertEqual((Path(row['directory']) / 'model.gguf').read_bytes(), self.gguf.read_bytes())

    def test_crash_after_rename_resumes_without_adopting_orphan(self):
        original = self.registry.update_model
        def crash(model_id, changes, **kwargs):
            if changes.get('pending') == {'stage': 'qualification'}:
                raise RuntimeError('crash after rename')
            return original(model_id, changes, **kwargs)
        with patch.object(self.registry, 'update_model', side_effect=crash):
            with self.assertRaisesRegex(RuntimeError, 'crash after rename'):
                self.acquire()
        row = self.registry.get_model('tiny')
        self.assertFalse(Path(row['pending']['path']).exists())
        self.assertTrue(Path(row['directory']).exists())
        self.acquire()
        self.assertEqual(models.show(self.registry, 'tiny')['state'], 'ready')
        orphan = self.registry.root / 'models/orphan'
        orphan.mkdir()
        with self.assertRaisesRegex(HemError, 'untracked'):
            self.acquire('orphan')

    def test_fragment_index_is_scoped_and_has_native_suffix(self):
        second = self.root / 'second.gguf'
        write_gguf(second, tensors=[('other', [1], 0, 0)])
        row = models.import_model(self.registry, 'split', ggufs=[self.gguf, second], tokenizer=self.tokenizer, alias=True)
        files = models.model_files(self.registry, 'split')
        self.assertTrue(str(files.weights).endswith('.safetensors.index.json'))
        self.assertEqual(json.loads(files.weights.read_text())['weight_map'], {
            'tiny': 'model-00001-of-00002.gguf', 'other': 'model-00002-of-00002.gguf'})
        self.assertTrue(all(path.suffix == '.gguf' for path in files.gguf_paths))
        self.assertEqual(Path(row['directory']), files.weights.parent)

    def test_bad_fragment_mapping_retains_invalid_assets(self):
        second = self.root / 'duplicate.gguf'
        second.write_bytes(self.gguf.read_bytes())
        with self.assertRaises(InvalidInput):
            models.import_model(self.registry, 'split', ggufs=[self.gguf, second], tokenizer=self.tokenizer)
        self.assertEqual(self.registry.get_model('split')['pending']['stage'], 'published-invalid')
        self.assertTrue((self.registry.root / 'models/split/model-00002-of-00002.gguf').exists())
        for inputs in ([self.gguf, self.gguf], [self.tokenizer]):
            with self.assertRaises(InvalidInput):
                models.import_model(self.registry, 'bad', ggufs=inputs, tokenizer=self.tokenizer)

    def recipe(self):
        return dict(id='fixture', pipeline='qwen-08b', gguf=dict(repo_id='test/weights', revision='a'*40, filenames=['weights.gguf']),
                    tokenizer=dict(repo_id='test/tokenizer', revision='b'*40, filename='tokenizer.json'))

    def download(self, **kwargs):
        return models.download(self.registry, 'fixture', model_id='downloaded', **kwargs)

    def cache(self, source):
        return models.model_download._cache_path(source)

    def test_offline_preserves_record_without_network_and_retry_uses_cache(self):
        with patch.dict(os.environ, {'HF_HUB_CACHE': str(self.root / 'cache')}), \
             patch.object(models.model_download, 'recipe_by_id', return_value=self.recipe()), \
             patch.object(models.model_download, '_run_hf') as network:
            with self.assertRaisesRegex(HemError, 'Offline cache is missing'):
                self.download(offline=True)
            self.assertEqual(self.registry.get_model('downloaded')['pending']['stage'], 'staging')
            for source in models.model_download._sources(self.recipe()):
                path = self.cache(source)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(self.gguf.read_bytes() if source['role'] == 'gguf' else self.tokenizer.read_bytes())
            row = self.download(offline=True)
            network.assert_not_called()
            self.assertEqual(row['pipeline'], 'qwen-08b')
            self.assertTrue(all(item['ownership'] == 'reference' for item in row['assets']))
            missing = self.cache(models.model_download._sources(self.recipe())[0])
            missing.unlink()
            self.assertEqual(models.show(self.registry, 'downloaded')['state'], 'missing')
            network.side_effect = lambda command: missing.write_bytes(self.gguf.read_bytes())
            self.download()
            self.assertEqual(network.call_count, 1)
            self.assertEqual(models.show(self.registry, 'downloaded')['state'], 'ready')

    def test_dryrun_never_initializes_state_or_executes_hf(self):
        with patch.object(models.model_download, 'recipe_by_id', return_value=self.recipe()), \
             patch.object(models.model_download, '_run_hf') as network:
            result = models.download(None, 'fixture', model_id='dry', dry_run=True)
            network.assert_not_called()
            self.assertEqual(result['state'], 'preview')
            self.assertEqual(self.registry.list_models(), [])

    def test_download_passes_pinned_command_and_preserves_partial_cache(self):
        calls = []
        def hf(command):
            calls.append(command)
            if len(calls) == 2:
                raise models.ModelImportError('network interrupted')
            source = models.model_download._sources(self.recipe())[0]
            path = self.cache(source)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(self.gguf.read_bytes())
        with patch.dict(os.environ, {'HF_HUB_CACHE': str(self.root / 'cache')}), \
             patch.object(models.model_download, 'recipe_by_id', return_value=self.recipe()), \
             patch.object(models.model_download, '_run_hf', side_effect=hf):
            with self.assertRaisesRegex(HemError, 'network interrupted'):
                self.download()
            self.assertEqual(len(calls), 2)
            self.assertEqual(calls[0], ['hf', 'download', 'test/weights', 'weights.gguf', '--revision', 'a'*40, '--cache-dir', str(self.root / 'cache')])
            self.assertTrue(self.cache(models.model_download._sources(self.recipe())[0]).exists())
            self.assertEqual(len(self.registry.list_models()), 1)

    def test_consumer_helper_rechecks_even_saved_valid(self):
        self.acquire(alias=True)
        self.select.side_effect = InvalidInput('modified descriptors')
        with self.assertRaisesRegex(InvalidInput, 'modified descriptors'):
            models.model_files(self.registry, 'tiny')
        self.assertFalse(self.registry.get_model('tiny')['validation']['valid'])

    def test_changed_assets_are_unvalidated_and_detected_during_qualification(self):
        self.acquire(alias=True)
        self.gguf.write_bytes(b'changed')
        self.assertEqual(models.show(self.registry, 'tiny')['state'], 'unvalidated')
        def mutate(*args):
            self.tokenizer.write_text('{"changed":true}')
            return 'qwen-08b'
        self.select.side_effect = mutate
        with self.assertRaisesRegex(InvalidInput, 'changed during qualification'):
            models.validate(self.registry, 'tiny')

    def test_real_packaged_pipeline_checks_both_synthetic_profiles(self):
        self.qualifier.stop()
        original = hem_config.load_profile
        digest = hashlib.sha256(json.dumps(json.loads(self.tokenizer.read_text()), sort_keys=True,
                                          separators=(',', ':')).encode()).hexdigest()
        def fixture(catalog):
            contract = original(catalog)
            contract['tokenizer_canonical_sha256'] = digest
            return contract
        with patch.object(hem_config, 'load_profile', side_effect=fixture):
            for identifier, architecture in [('qwen-08b', 'qwen_08b'), ('qwen', 'qwen_27b')]:
                weights = self.root / (identifier + '.gguf')
                write_matching_gguf(weights, contract=profile(architecture))
                row = models.import_model(self.registry, identifier, ggufs=[weights], tokenizer=self.tokenizer, alias=True)
                self.assertEqual(row['pipeline'], identifier)
                with self.assertRaisesRegex(InvalidInput, 'incompatible'):
                    models.validate(self.registry, identifier, pipeline='qwen' if identifier == 'qwen-08b' else 'qwen-08b')


if __name__ == '__main__':
    unittest.main()
