"""Installation tests use sparse GGUF fixtures and a fake CLI, never real downloads."""
from contextlib import redirect_stdout, redirect_stderr
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from binding.python import gguf, model_download, model_store, pipeline_validation, remote_registry, setup_cli
from tests.gguf_fixtures import profile, write_matching_gguf, write_gguf


class DownloadTest(unittest.TestCase):
    def test_local_import_cli_publishes_and_reports_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            weights = root / 'weights.gguf'
            tokenizer = root / 'tokenizer.json'
            write_gguf(weights)
            tokenizer.write_text('{"model": {"type": "BPE", "vocab": {"a": 0}}}')
            for alias in (False, True):
                model_id = 'alias' if alias else 'copy'
                arguments = ['models', 'import', model_id, '--gguf', str(weights),
                             '--tokenizer', str(tokenizer), '--model-root', str(root / 'models')]
                if alias:
                    arguments.append('--alias')
                with redirect_stdout(io.StringIO()) as output:
                    self.assertEqual(setup_cli.main(arguments), 0)
                files = model_store.model_files(model_id, root / 'models')
                self.assertIn(str(files.weights), output.getvalue())
                self.assertIn(str(files.tokenizer), output.getvalue())
                self.assertEqual(files.tokenizer.is_symlink(), alias)

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.models = self.root / 'models'
        self.cache = self.root / 'cache'
        environment = patch.dict(os.environ, {'HF_HUB_CACHE': str(self.cache), 'HF_HUB_OFFLINE': '0'})
        environment.start()
        self.addCleanup(environment.stop)
        self.recipe = model_download.recipes()[0]
        self.sources = model_download._sources(self.recipe)
        self.tokenizer_data = {'model': {'type': 'BPE', 'vocab': {'a': 0}}}
        contract = profile()
        contract['tokenizer_canonical_sha256'] = hashlib.sha256(json.dumps(self.tokenizer_data, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()
        policy = patch('binding.python.pipeline_validation.load_profile', return_value=contract)
        policy.start()
        self.addCleanup(policy.stop)

    def cached(self, index):
        path = model_download._cache_path(self.sources[index])
        path.parent.mkdir(parents=True, exist_ok=True)
        blob = self.root / ('blob-' + str(index))
        if index == 0:
            write_matching_gguf(blob)
        else:
            blob.write_text(json.dumps(self.tokenizer_data))
        path.symlink_to(blob)
        return path

    def test_catalog_local_cache_status_and_alias_registration_separate(self):
        with patch('subprocess.Popen', side_effect=AssertionError('catalog must not run commands')):
            self.assertEqual(model_download.available(self.models)[0]['cache_status'], 'not downloaded')
            weight = self.cached(0)
            self.assertEqual(model_download.available(self.models)[0]['cache_status'], 'partially cached')
            self.cached(1)
            before = model_download.available(self.models)[0]
            self.assertEqual(before['cache_status'], 'cached')
            self.assertEqual(before['registration_status'], 'not registered')
            with redirect_stderr(io.StringIO()):
                model_download.download(self.recipe['id'], model_id='chosen', model_root=self.models, offline=True)
            after = model_download.available(self.models)[0]
            self.assertEqual(after['registrations'], [{'id': 'chosen', 'status': 'ready'}])
            weight.unlink()
            after = model_download.available(self.models)[0]
            self.assertEqual(after['cache_status'], 'partially cached')
            self.assertEqual(after['registration_status'], 'broken alias')
            self.assertEqual(remote_registry.local_model_inventory(self.models)['models'][0]['availability'], 'broken alias')

    def test_shared_inventory_keeps_validation_separate_and_recognizes_legacy_sources(self):
        weights = self.cached(0)
        self.cached(1)
        with redirect_stderr(io.StringIO()):
            model_download.download(self.recipe['id'], model_id='legacy', model_root=self.models, offline=True)
        path = self.models / 'legacy' / 'manifest.json'
        manifest = json.loads(path.read_text())
        manifest.pop('recipe')
        manifest.pop('pipeline')
        path.write_text(json.dumps(manifest))
        with redirect_stdout(io.StringIO()) as output:
            setup_cli.main(['models', 'list', '--model-root', str(self.models), '--json'])
        item = json.loads(output.getvalue())['models'][0]
        self.assertEqual(item['availability'], 'ready')
        self.assertFalse(item['validated'])
        self.assertEqual(model_download.available(self.models)[0]['registrations'],
                         [{'id': 'legacy', 'status': 'unvalidated'}])
        manifest['pipeline'] = 'qwen'
        path.write_text(json.dumps(manifest))
        weights.unlink()
        inventory = remote_registry.local_model_inventory(self.models)['models'][0]
        registration = model_download.available(self.models)[0]['registrations'][0]
        self.assertTrue(inventory['validated'])
        self.assertEqual(inventory['availability'], registration['status'])
        self.assertEqual(registration['status'], 'broken alias')

    def test_malformed_manifest_reported_consistently_without_crashing_catalog(self):
        directory = self.models / self.recipe['id']
        directory.mkdir(parents=True)
        path = directory / 'manifest.json'
        malformed = ['not json', '[]', json.dumps({'files': []}),
                     json.dumps({'files': {'gguf': {'path': '../outside'}}}),
                     json.dumps({'files': {'gguf': {'path': 'model.gguf', 'source': []}}}),
                     json.dumps({'files': {'gguf': {'path': 'model.gguf', 'source': {'repo_id': []}}}}),
                     json.dumps({'gguf_files': 'model.gguf'}), json.dumps({'pipeline': []})]
        for contents in malformed:
            with self.subTest(contents=contents):
                path.write_text(contents)
                inventory = remote_registry.local_model_inventory(self.models)
                self.assertEqual(inventory['schema_version'], 1)
                item = inventory['models'][0]
                registration = model_download.available(self.models)[0]['registrations'][0]
                self.assertEqual(item['availability'], 'invalid')
                self.assertEqual(registration['status'], item['availability'])
                self.assertEqual(registration['error'], item['error'])
                self.assertFalse(item['validated'])
                with self.assertRaises(model_store.ModelImportError):
                    model_store.model_files(self.recipe['id'], self.models)

    def test_offline_missing_lists_all_files_and_does_not_publish(self):
        with redirect_stderr(io.StringIO()), patch('subprocess.Popen', side_effect=AssertionError('offline must not invoke CLI')):
            with self.assertRaises(model_store.ModelImportError) as error:
                model_download.download(self.recipe['id'], model_root=self.models, offline=True)
            self.assertIn(self.sources[0]['filename'], str(error.exception))
            self.assertIn(self.sources[1]['filename'], str(error.exception))
            self.assertFalse(self.models.exists())
            os.environ['HF_HUB_OFFLINE'] = '1'
            with self.assertRaises(model_store.ModelImportError):
                model_download.download(self.recipe['id'], model_root=self.models)

    def test_dry_run_json_no_publication_or_cli_when_offline(self):
        self.cached(0)
        with patch('subprocess.Popen', side_effect=AssertionError('offline plan is local only')), redirect_stdout(io.StringIO()) as output, redirect_stderr(io.StringIO()):
            code = setup_cli.main(['models', 'download', self.recipe['id'], '--model-root', str(self.models), '--offline', '--dry-run', '--json'])
        self.assertEqual(code, 0)
        plan = json.loads(output.getvalue())
        self.assertTrue(plan['dry_run'])
        self.assertEqual(plan['storage'], 'alias')
        self.assertEqual(plan['cache_status'], 'partially cached')
        self.assertFalse(self.models.exists())

    def test_download_aliases_snapshot_extensions_no_hashes_and_repeat_validation(self):
        paths = [self.cached(0), self.cached(1)]
        with redirect_stderr(io.StringIO()):
            result = model_download.download(self.recipe['id'], model_root=self.models, offline=True)
            manifest = model_store.read_manifest(self.recipe['id'], self.models)
            files = model_store.model_files(self.recipe['id'], self.models)
            self.assertEqual(result['status'], 'registered')
            self.assertEqual(manifest['pipeline'], 'qwen')
            self.assertNotIn('sha256', manifest['files']['gguf'])
            self.assertEqual(os.readlink(files.weights), str(paths[0]))
            self.assertEqual(files.weights.suffix, '.gguf')
            self.assertEqual(model_download.download(self.recipe['id'], model_root=self.models, offline=True)['status'], 'already registered')
            write_matching_gguf(paths[0], missing=['blk.64.nextn.eh_proj.weight'])
            with self.assertRaisesRegex(model_store.ModelImportError, 'missing required tensor'):
                model_download.download(self.recipe['id'], model_root=self.models, offline=True)
            paths[0].unlink()
            with self.assertRaisesRegex(model_store.ModelImportError, 'unavailable'):
                model_store.model_files(self.recipe['id'], self.models)

    def test_failure_atomic_existing_preserved_and_other_id_conflict(self):
        weights = self.cached(0)
        self.cached(1)
        write_matching_gguf(weights, missing=['output.weight'])
        with redirect_stderr(io.StringIO()):
            with self.assertRaisesRegex(model_store.ModelImportError, 'missing required tensor'):
                model_download.download(self.recipe['id'], model_root=self.models, offline=True)
            self.assertEqual(list(self.models.iterdir()), [])
            destination = self.models / self.recipe['id']
            destination.mkdir()
            manifest = destination / 'manifest.json'
            manifest.write_text('{"model_id":"existing"}')
            original = manifest.read_bytes()
            with self.assertRaisesRegex(model_store.ModelImportError, 'preserved'):
                model_download.download(self.recipe['id'], model_root=self.models, offline=True)
            self.assertEqual(manifest.read_bytes(), original)

    def test_cli_available_json_details_and_import_migration(self):
        with redirect_stdout(io.StringIO()) as output:
            self.assertEqual(setup_cli.main(['models', 'available', '--json', '--model-root', str(self.models)]), 0)
        catalog = json.loads(output.getvalue())
        self.assertEqual({item['id']: item['pipeline'] for item in catalog},
                         {'qwen38-27b-q5-k-xl': 'qwen', 'qwen35-08b-q5-k-xl': 'qwen-08b'})
        for item in catalog:
            for key in ('variant', 'quantization', 'pipeline', 'gguf', 'tokenizer', 'registrations'):
                self.assertIn(key, item)
            self.assertRegex(item['gguf']['revision'], r'^[0-9a-f]{40}$')
            self.assertRegex(item['tokenizer']['revision'], r'^[0-9a-f]{40}$')
        with redirect_stderr(io.StringIO()) as error:
            with self.assertRaises(SystemExit):
                setup_cli.main(['models', 'import', 'old', '--hf-repo', 'repo'])
        self.assertIn('models download RECIPE', error.getvalue())
        with redirect_stdout(io.StringIO()) as output:
            self.assertEqual(setup_cli.main(['models', 'list', '--model-root', str(self.models)]), 0)
        self.assertIn('No models', output.getvalue())

    def test_streamed_cli_hides_tokens_and_keeps_stdout_json_clean(self):
        binary = self.root / 'bin'
        binary.mkdir()
        fake = binary / 'hf'
        # This fake download populates the expected HF snapshot. Its paths are
        # small placeholders; CPU compatibility is exercised independently.
        fake.write_text('#!/usr/bin/env python3\nimport os,sys\nfrom pathlib import Path\nprint("Downloading weights… " + os.environ["HF_TOKEN"], file=sys.stderr, flush=True)\nrepo,filename=sys.argv[2:4]\nrevision=sys.argv[sys.argv.index("--revision")+1]\ncache=Path(sys.argv[sys.argv.index("--cache-dir")+1])\np=cache/("models--"+repo.replace("/","--"))/"snapshots"/revision/filename\np.parent.mkdir(parents=True,exist_ok=True)\np.write_text("data")\nprint(p)\n')
        fake.chmod(0o755)
        with patch.dict(os.environ, {'PATH': str(binary) + os.pathsep + os.environ['PATH'], 'HF_TOKEN': 'hf_testsecrettoken'}), redirect_stdout(io.StringIO()) as out, redirect_stderr(io.StringIO()) as err:
            source = model_download._download_source(self.sources[0], False)
        self.assertTrue(source['path'].is_file())
        self.assertEqual(out.getvalue(), '')
        self.assertIn('Running: hf download', err.getvalue())
        self.assertIn('Downloading weights', err.getvalue())
        self.assertNotIn('hf_testsecrettoken', err.getvalue())
        self.assertIn('[REDACTED]', err.getvalue())

    def test_cli_download_failure_actionable(self):
        fake = self.root / 'hf'
        fake.write_text('#!/bin/sh\necho "Access denied" >&2\nexit 3\n')
        fake.chmod(0o755)
        with patch.dict(os.environ, {'PATH': str(self.root) + os.pathsep + os.environ['PATH']}), redirect_stderr(io.StringIO()) as output:
            with self.assertRaisesRegex(model_store.ModelImportError, 'hf auth login'):
                model_download._download_source(self.sources[0], False)
        self.assertIn('Access denied', output.getvalue())

    def test_local_copy_alias_and_backward_manifest_without_hash_checks(self):
        weights = self.root / 'local.gguf'
        write_gguf(weights)
        tokenizer = self.root / 'tokenizer.json'
        tokenizer.write_text(json.dumps(self.tokenizer_data))
        with redirect_stderr(io.StringIO()):
            imported = model_store.import_model('copy', gguf=weights, tokenizer=tokenizer, model_root=self.models)
            self.assertFalse(imported.files.weights.is_symlink())
            model_store.import_model('alias', gguf=weights, tokenizer=tokenizer, model_root=self.models, alias=True)
        legacy = self.models / 'copy' / 'manifest.json'
        manifest = json.loads(legacy.read_text())
        manifest.pop('gguf_files')
        manifest.pop('weights_path')
        manifest['format_version'] = 1
        manifest['files']['gguf']['sha256'] = 'obsolete-do-not-check'
        legacy.write_text(json.dumps(manifest))
        self.assertEqual(model_store.model_files('copy', self.models).weights, imported.files.weights)
        weights.unlink()
        self.assertTrue(model_store.model_files('copy', self.models).weights.is_file())
        with self.assertRaisesRegex(model_store.ModelImportError, 'alias'):
            model_store.model_files('alias', self.models)

    def test_fragments_merge_validate_index_and_all_links_checked(self):
        first = self.root / 'first.gguf'
        second = self.root / 'second.gguf'
        metadata0 = [('split.count', 2, 2), ('split.no', 2, 0), ('split.tensors.count', 10, 2)]
        metadata1 = [('split.count', 2, 2), ('split.no', 2, 1), ('split.tensors.count', 10, 2)]
        write_gguf(first, metadata=metadata0, tensors=[('a', [32], 0, 0)])
        write_gguf(second, metadata=metadata1, tensors=[('b', [32], 0, 0)])
        self.assertEqual(set(gguf.inspect_gguf_set([first, second]).tensors), {'a', 'b'})
        with self.assertRaisesRegex(gguf.GGUFError, 'Incomplete'):
            gguf.inspect_gguf_set([first])
        tokenizer = self.root / 'tokenizer.json'
        tokenizer.write_text(json.dumps(self.tokenizer_data))
        with redirect_stderr(io.StringIO()):
            imported = model_store.import_model('split', gguf=[first, second], tokenizer=tokenizer, model_root=self.models, alias=True)
        files = model_store.model_files('split', self.models)
        self.assertTrue(files.weights.name.endswith('.safetensors.index.json'))
        self.assertEqual(len(files.gguf_paths), 2)
        index = json.loads(files.weights.read_text())['weight_map']
        self.assertEqual(set(index), {'a', 'b'})
        self.assertTrue(all((files.weights.parent / name).is_symlink() for name in index.values()))
        self.assertTrue(all(name.endswith('.gguf') for name in index.values()))
        self.assertEqual(index, {'a': files.gguf_paths[0].name, 'b': files.gguf_paths[1].name})
        second.unlink()
        with self.assertRaisesRegex(model_store.ModelImportError, 'gguf_2'):
            model_store.model_files('split', self.models)

    def test_qualified_fragmented_recipe_publishes_after_combined_validation(self):
        from tests.gguf_fixtures import expected_tensors
        recipe = json.loads(json.dumps(self.recipe))
        recipe['gguf']['filenames'] = ['part-one.gguf', 'part-two.gguf']
        sources = model_download._sources(recipe)
        names = list(expected_tensors())
        half = len(names) // 2
        for source, missing in zip(sources[:2], (names[half:], names[:half])):
            path = model_download._cache_path(source)
            path.parent.mkdir(parents=True, exist_ok=True)
            write_matching_gguf(path, missing=missing)
        self.cached(1)
        with patch('binding.python.model_download.recipes', return_value=[recipe]), redirect_stderr(io.StringIO()):
            result = model_download.download(recipe['id'], model_root=self.models, offline=True)
            files = model_store.model_files(recipe['id'], self.models)
            self.assertEqual(result['status'], 'registered')
            self.assertEqual(len(files.gguf_paths), 2)
            self.assertTrue(files.weights.name.endswith('.safetensors.index.json'))
            self.assertTrue(pipeline_validation.validate_model(files, 'qwen').compatible)
            self.assertEqual(len(json.loads(files.weights.read_text())['weight_map']), len(names))
            self.assertEqual(model_download.download(recipe['id'], model_root=self.models, offline=True)['status'], 'already registered')
            self.assertEqual(remote_registry.local_model_inventory(self.models)['models'][0]['availability'], 'ready')

    def test_08b_download_selects_its_contract_and_keeps_recipe_cache_status_separate(self):
        recipe = model_download.recipe_by_id('qwen35-08b-q5-k-xl')
        sources = model_download._sources(recipe)
        contract = profile('qwen_08b')
        contract['tokenizer_canonical_sha256'] = hashlib.sha256(
            json.dumps(self.tokenizer_data, sort_keys=True, ensure_ascii=False,
                       separators=(',', ':')).encode()).hexdigest()

        def fetch(command):
            source = next(source for source in sources if
                          source['repo_id'] == command[2] and source['filename'] == command[3])
            self.assertEqual(command[command.index('--revision') + 1], source['revision'])
            path = model_download._cache_path(source)
            path.parent.mkdir(parents=True, exist_ok=True)
            if source['role'] == 'gguf':
                write_matching_gguf(path, contract=contract)
            else:
                path.write_text(json.dumps(self.tokenizer_data))
            return str(path)

        with patch('binding.python.pipeline_validation.load_profile', return_value=contract), \
                patch('binding.python.model_download._run_hf', side_effect=fetch) as run, \
                redirect_stderr(io.StringIO()), redirect_stdout(io.StringIO()) as output:
            code = setup_cli.main(['models', 'download', recipe['id'],
                                   '--model-root', str(self.models), '--json'])
            self.assertEqual(code, 0)
            self.assertEqual(run.call_count, 2)
            self.assertEqual(json.loads(output.getvalue())['pipeline'], 'qwen-08b')
            manifest = model_store.read_manifest(recipe['id'], self.models)
            self.assertEqual(manifest['pipeline'], 'qwen-08b')
            files = model_store.model_files(recipe['id'], self.models)
            self.assertTrue(files.weights.is_symlink())
            self.assertTrue(files.tokenizer.is_symlink())
            result = pipeline_validation.validate_model(files, 'qwen-08b')
            self.assertTrue(result.compatible)
            self.assertIn('24 base layers without MTP', result.message)
            self.assertIn('320 tensors', result.message)
            run.reset_mock()
            self.assertEqual(model_download.download(recipe['id'], model_root=self.models,
                                                    offline=True)['status'], 'already registered')
            run.assert_not_called()

        with patch('subprocess.Popen', side_effect=AssertionError('catalog must stay local')):
            catalog = {item['id']: item for item in model_download.available(self.models)}
        self.assertEqual(catalog[recipe['id']]['cache_status'], 'cached')
        self.assertEqual(catalog[recipe['id']]['registration_status'], 'ready')
        self.assertEqual(catalog[self.recipe['id']]['cache_status'], 'not downloaded')
        self.assertEqual(catalog[self.recipe['id']]['registration_status'], 'not registered')

    def test_hf_stream_preserves_metadata_and_forces_visible_progress(self):
        from unittest.mock import Mock
        process = Mock(stdout=io.StringIO('/cache/model.gguf\n'),
                       stderr=io.StringIO('weights: 50% 10G/20G\r'))
        process.wait.return_value = 0
        with patch('subprocess.Popen', return_value=process) as run, \
                redirect_stderr(io.StringIO()) as updates, \
                patch.dict(os.environ, {'TQDM_POSITION': '0'}):
            metadata = model_download._run_hf(['hf', 'download', 'repo', 'weights.gguf'])
        self.assertEqual(metadata, '/cache/model.gguf\n')
        self.assertIn('weights: 50% 10G/20G', updates.getvalue())
        self.assertEqual(run.call_args.kwargs['env']['TQDM_POSITION'], '-1')
        self.assertEqual(run.call_args.kwargs['env']['HF_HUB_DISABLE_PROGRESS_BARS'], '0')

    def test_online_dry_run_capabilities_remote_size_and_older_cli_fallback(self):
        with patch('binding.python.model_download._run_hf', side_effect=['--dry-run --json', '[{"file":"weights.gguf","size":"2.5G"}]', '[{"file":"tokenizer.json","size":"123B"}]']) as run, redirect_stderr(io.StringIO()):
            plan = model_download.download(self.recipe['id'], model_root=self.models, dry_run=True)
        self.assertEqual(plan['estimated_download_bytes'], 2500000123)
        self.assertEqual(plan['files'][0]['expected_size'], '2.5G')
        self.assertIsNone(plan['files'][1]['size_bytes'])
        self.assertEqual(plan['files'][1]['estimated_size_bytes'], 123)
        self.assertFalse(self.models.exists())
        for call in run.call_args_list[1:]:
            self.assertIn('--dry-run', call.args[0])
        with patch('binding.python.model_download._run_hf', side_effect=['--dry-run --json', '{"size_bytes":123}', '[{"size":"unknown"}]']), redirect_stderr(io.StringIO()):
            plan = model_download.download(self.recipe['id'], model_root=self.models, dry_run=True)
        self.assertIsNone(plan['download_bytes'])
        self.assertNotIn('estimated_download_bytes', plan)
        self.assertIn('unknown', plan['size_note'])
        self.assertFalse(self.models.exists())
        with patch('binding.python.model_download._run_hf', return_value='--quiet --cache-dir') as run, redirect_stderr(io.StringIO()):
            plan = model_download.download(self.recipe['id'], model_root=self.models, dry_run=True)
        self.assertEqual(run.call_count, 1)
        self.assertIn('unknown', plan['size_note'])
        self.assertIsNone(plan['download_bytes'])

