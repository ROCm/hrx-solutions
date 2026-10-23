"""Public model/pipeline CLI in isolated subprocesses with tiny CPU fixtures."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from binding.python import hem_cli, hem_display
from binding.python.models import hem_models, model_download, recipes
from binding.python.models.recipe_types import HFSource
from tests.gguf_fixtures import write_gguf


class ModelCliTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.state = self.root / 'state'
        self.environment = dict(os.environ, LOOM_SERVE_DIR=str(self.state),
                                HF_HUB_CACHE=str(self.root / 'cache'))
        self.tokenizer = self.root / 'tokenizer.json'
        self.tokenizer.write_text('{"model":{"fixture":true}}')
        self.weights = self.root / 'weights.gguf'
        write_gguf(self.weights, tensors=[('fixture', [1], 0, 0)])

    def invoke(self, *args, fixture=False, expected=0):
        package = hem_cli.__package__
        command = [sys.executable, '-m', package + '.hem_cli']
        process = subprocess.run([*command, *args, '--json'], env=self.environment,
                                 capture_output=True, text=True, check=False)
        self.assertEqual(process.returncode, expected, process.stderr)
        data = json.loads(process.stdout)
        self.assertIn('state', data)
        self.assertIn('completed', data)
        self.assertNotIn('Traceback', process.stderr)
        return data, process.stderr

    def acquire(self, name='tiny', *options, expected=0):
        return self.invoke('model', 'import', name, '--gguf', str(self.weights),
                           '--tokenizer', str(self.tokenizer), '--pipeline', 'qwen-08b', *options,
                           fixture=True, expected=expected)

    def test_observational_packaged_inventory_and_preview_do_not_create_registry(self):
        pipelines, _ = self.invoke('pipeline', 'list')
        self.assertEqual({item['pipeline'] for item in pipelines['items']}, {'qwen', 'qwen-08b', 'moonshine-streaming-tiny'})
        info, _ = self.invoke('pipeline', 'show', 'qwen-08b')
        self.assertEqual(info['parameters']['context_capacity']['default'], 8192)
        self.assertFalse(info['parameters']['mtp']['value'])
        self.assertNotIn('compatibility', info)
        recipes, _ = self.invoke('model', 'available')
        for recipe in recipes['items']:
            preview, diagnostics = self.invoke('model', 'download', recipe['id'], '--id', 'preview', '--dry-run')
            self.assertEqual(preview['state'], 'preview')
            self.assertTrue(preview['artifacts'])
            self.assertTrue(all('logical_path' in artifact for artifact in preview['artifacts']))
            self.assertIn('Preview only', diagnostics)
        self.assertFalse(self.state.exists())
        self.assertFalse((self.root / 'cache').exists())

    def test_dry_run_displays_typed_pinned_hf_commands_without_acquisition(self):
        # Any accidental HF execution is visible even when `hf` is installed.
        bin_dir = self.root / 'bin'
        bin_dir.mkdir()
        hf = bin_dir / 'hf'
        marker = self.root / 'hf-called'
        hf.write_text('#!/bin/sh\ntouch "' + str(marker) + '"\nexit 99\n')
        hf.chmod(0o755)
        self.environment['PATH'] = str(bin_dir) + os.pathsep + self.environment.get('PATH', '')
        for recipe in recipes.list():
            for options in ((), ('--offline',)):
                with self.subTest(recipe=recipe.id, options=options):
                    with mock.patch.dict(os.environ, self.environment):
                        typed = hem_models.download(None, recipe.id, model_id='preview', dry_run=True)
                        commands = [hem_display.command(model_download.command(item.source)).display
                                    for item in typed.artifacts if isinstance(item.source, HFSource)]
                    self.assertTrue(commands)
                    preview, diagnostics = self.invoke('model', 'download', recipe.id,
                                                       '--id', 'preview', '--dry-run', *options)
                    self.assertEqual(preview['state'], 'preview')
                    self.assertEqual(diagnostics.count('# Equivalent command'), len(commands))
                    for command in commands:
                        self.assertIn(command, diagnostics)
                    self.assertIn('Preview only', diagnostics)
                    self.assertFalse(marker.exists())
                    self.assertFalse(self.state.exists())
                    self.assertFalse((self.root / 'cache').exists())

    def test_dry_run_command_display_redacts_hf_credentials(self):
        token = 'hf_previewSecret123'
        cache = self.root / ('cache with spaces ' + token)
        self.environment.update(HF_TOKEN=token, HF_HUB_CACHE=str(cache))
        _, diagnostics = self.invoke('model', 'download', 'qwen35-08b-q5-k-xl',
                                     '--id', 'preview', '--dry-run')
        self.assertIn('hf download', diagnostics)
        self.assertIn('<redacted>', diagnostics)
        self.assertIn('cache with spaces', diagnostics)
        self.assertNotIn(token, diagnostics)
        self.assertFalse(cache.exists())
        self.assertFalse(self.state.exists())

    def test_owned_import_retry_validate_and_show_locations(self):
        record, diagnostic = self.acquire()
        self.assertEqual(record['state'], 'ready')
        self.assertEqual(record['pipeline'], 'qwen-08b')
        self.assertTrue(record['completed'])
        self.assertIn('native loading remains unprobed', diagnostic)
        target = self.state / 'models/tiny/model.gguf'
        self.assertFalse(target.is_symlink())
        self.assertEqual(record['artifacts'][0]['path'], str(target))
        before = target.stat().st_ino
        self.acquire()
        self.assertEqual(target.stat().st_ino, before)
        validated, diagnostic = self.invoke('model', 'validate', 'tiny', fixture=True)
        self.assertEqual(validated['state'], 'ready')
        self.assertIn('prepared-asset checks', diagnostic)
        shown, _ = self.invoke('model', 'show', 'tiny')
        self.assertEqual(shown['instances'], [])
        self.assertEqual(shown['assets'][0]['ownership'], 'owned')
        self.assertEqual(shown['provenance']['kind'], 'import')
        self.assertIn('recipe_identity', shown['provenance'])
        inventory, _ = self.invoke('model', 'list')
        self.assertEqual([item['model'] for item in inventory['items']], ['tiny'])

    def test_references_missing_assets_and_incompatible_selection_remain_visible(self):
        self.acquire('aliased', '--alias')
        self.assertTrue((self.state / 'models/aliased/model.gguf').is_symlink())
        data, _ = self.invoke('model', 'validate', 'aliased', '--pipeline', 'qwen', fixture=True, expected=2)
        self.assertFalse(data['completed'])
        self.assertEqual(data['model'], 'aliased')
        shown, _ = self.invoke('model', 'show', 'aliased')
        self.assertEqual(shown['state'], 'invalid')
        self.invoke('model', 'validate', 'aliased', fixture=True)
        self.weights.unlink()
        shown, _ = self.invoke('model', 'show', 'aliased')
        self.assertEqual(shown['state'], 'missing')
        self.invoke('model', 'validate', 'aliased', fixture=True, expected=2)
        self.assertTrue((self.state / 'models/aliased/tokenizer.json').is_symlink())

    def test_incomplete_import_and_conflict_are_preserved(self):
        original = self.tokenizer.read_text()
        self.tokenizer.unlink()
        self.acquire(expected=1)
        shown, _ = self.invoke('model', 'show', 'tiny')
        self.assertEqual(shown['state'], 'incomplete')
        self.invoke('model', 'validate', 'tiny', expected=1)
        self.tokenizer.write_text(original)
        self.acquire()
        other = self.root / 'other.gguf'
        other.write_bytes(b'other')
        data, _ = self.invoke('model', 'import', 'tiny', '--gguf', str(other),
                              '--tokenizer', str(self.tokenizer), '--pipeline', 'qwen-08b', expected=1)
        self.assertIn('different mappings', data['error'])
        shown, _ = self.invoke('model', 'show', 'tiny')
        self.assertEqual(shown['state'], 'ready')

    def test_offline_failure_retains_incomplete_record_without_cache_download(self):
        recipes = hem_models.available()
        recipe = next(item['id'] for item in recipes if item['pipeline'] == 'qwen-08b')
        data, _ = self.invoke('model', 'download', recipe, '--id', 'offline', '--offline', expected=1)
        self.assertEqual(data['model'], 'offline')
        self.assertIn('Offline cache is missing', data['error'])
        shown, _ = self.invoke('model', 'show', 'offline')
        self.assertEqual(shown['state'], 'incomplete')
        self.assertFalse((self.root / 'cache').exists())

    def test_cached_offline_download_qualifies_and_retries_without_hf(self):
        recipe = recipes.load('qwen35-08b-q5-k-xl')
        for artifact in recipe.artifacts:
            source = artifact.source
            destination = (self.root / 'cache' / ('models--' + source.repo_id.replace('/', '--')) /
                           'snapshots' / source.revision / source.filename)
            destination.parent.mkdir(parents=True, exist_ok=True)
            if artifact.format == 'gguf':
                write_gguf(destination, tensors=[('fixture', [1], 0, 0)])
            else:
                destination.write_text(self.tokenizer.read_text())
        for _ in range(2):
            data, diagnostics = self.invoke('model', 'download', recipe.id, '--id', 'cached',
                                            '--offline', fixture=True)
            self.assertEqual(data['state'], 'ready')
            self.assertEqual(data['pipeline'], 'qwen-08b')
            self.assertEqual(data['provenance']['recipe_id'], recipe.id)
            self.assertTrue(all(asset['ownership'] == 'reference' for asset in data['assets']))
            self.assertIn('prepared-asset checks', diagnostics)
            self.assertNotIn('hf download', diagnostics)

    def test_invalid_selectors_and_missing_models_have_structured_errors(self):
        for args, expected in ((('pipeline', 'show', 'unknown'), 2),
                               (('model', 'show', 'missing'), 1),
                               (('model', 'download', 'unknown', '--id', 'x', '--dry-run'), 1),
                               (('model', 'show', '../escape'), 2)):
            with self.subTest(args=args):
                data, diagnostics = self.invoke(*args, expected=expected)
                self.assertFalse(data['completed'])
                self.assertTrue(diagnostics)


if __name__ == '__main__':
    unittest.main()
