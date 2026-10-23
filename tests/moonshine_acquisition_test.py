"""Acquisition ownership and bounded preparer process tests; no model validator."""
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import unittest
from contextlib import redirect_stderr
from copy import deepcopy
from dataclasses import replace
from unittest.mock import patch

from binding.python import hem_cli
from binding.python.cli import hem_instance_cli
from binding.python.models import hem_models, model_download, moonshine_preparation as preparation
from binding.python.hem_common import Deadline, HemError
from binding.python.control.hem_state import Registry
from binding.python.models.model_store import NamedAsset, PreparationResult, RuntimeMapping


class MoonshineAcquisitionTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.cache = self.root / 'cache'
        self.environment = patch.dict(os.environ, {'HF_HUB_CACHE': str(self.cache)})
        self.environment.start(); self.addCleanup(self.environment.stop)
        self.registry = Registry(self.root / 'state')
        self.recipe = model_download.recipe_by_id('moonshine-streaming-tiny')
        for source in self.recipe['sources']:
            path = model_download._cache_path(source)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'opaque model input')
        self.rotary = self.root / 'generated'
        self.rotary.write_bytes(b'opaque prepared rotary')
        self.preflight = patch.object(preparation, 'preflight').start()
        self.prepare = patch.object(preparation, 'prepare', side_effect=self.completed).start()
        self.network = patch.object(model_download, '_run_hf', side_effect=AssertionError('unexpected network')).start()
        self.addCleanup(patch.stopall)
        self.stderr = redirect_stderr(io.StringIO())
        self.stderr.__enter__(); self.addCleanup(self.stderr.__exit__, None, None, None)

    def completed(self, inputs, attempt, deadline):
        deadline.remaining()
        attempt.output.mkdir()
        rotary = attempt.output / 'rotary.safetensors'
        rotary.write_bytes(self.rotary.read_bytes())
        return PreparationResult(attempt.directory, (NamedAsset('rotary', rotary.name, rotary),))

    def acquire(self, name='speech', **kwargs):
        kwargs.setdefault('offline', True)
        return hem_models.download(self.registry, self.recipe['id'], model_id=name, **kwargs)

    def test_offline_mapping_generated_owned_and_completed_retry(self):
        row = self.acquire()
        self.assertTrue(row['validation']['valid'])
        self.assertEqual([asset['ownership'] for asset in row['assets']], ['reference'] * 3 + ['owned'])
        final = Path(row['directory']) / 'rotary.safetensors'
        self.assertEqual(row['assets'][-1]['source_path'], str(final))
        self.assertFalse(final.is_symlink())
        self.assertTrue((Path(row['directory']) / 'model.safetensors').is_symlink())
        inode = final.stat().st_ino
        self.acquire()
        self.assertEqual(final.stat().st_ino, inode)
        self.preflight.assert_called_once(); self.prepare.assert_called_once()
        self.network.assert_not_called()
        final.write_bytes(b'changed')
        with self.assertRaisesRegex(HemError, 'rotary changed'):
            self.acquire()
        self.prepare.assert_called_once()

    def test_missing_cache_and_failed_helper_remain_incomplete_retryable(self):
        source = model_download._cache_path(self.recipe['sources'][0])
        source.unlink()
        with self.assertRaisesRegex(HemError, 'Offline cache is missing'):
            self.acquire()
        self.assertEqual(hem_models.show(self.registry, 'speech')['state'], 'incomplete')
        self.prepare.assert_not_called()
        source.write_bytes(b'restored')
        self.prepare.side_effect = HemError('preparer failed')
        with self.assertRaisesRegex(HemError, 'preparer failed'):
            self.acquire()
        row = self.registry.get_model('speech')
        self.assertFalse((Path(row['pending']['path']) / 'rotary.safetensors').exists())
        self.assertFalse(Path(row['directory']).exists())
        self.prepare.side_effect = self.completed
        self.acquire()
        self.assertEqual(hem_models.show(self.registry, 'speech')['state'], 'ready')

    def test_preflight_before_network_and_complete_import_bypasses_helper(self):
        self.preflight.side_effect = HemError('missing optional dependencies')
        with self.assertRaisesRegex(HemError, 'optional dependencies'):
            self.acquire(offline=False)
        self.network.assert_not_called(); self.prepare.assert_not_called()
        inputs = self.root / 'prepared'; inputs.mkdir()
        for asset in self.recipe['sources']:
            (inputs / asset['filename']).write_bytes(b'caller prepared')
        (inputs / 'rotary.safetensors').write_bytes(b'caller rotary')
        row = hem_models.import_assets(self.registry, 'local', directory=inputs,
                                       pipeline=self.recipe['pipeline'])
        self.assertTrue(row['validation']['valid'])
        self.prepare.assert_not_called()

    def test_preview_runs_no_preflight_process_network_or_registry(self):
        answer = hem_models.download(None, self.recipe['id'], model_id='preview', dry_run=True)
        self.assertEqual(answer['generated'], [{'role': 'rotary', 'path': 'rotary.safetensors'}])
        self.preflight.assert_not_called(); self.prepare.assert_not_called(); self.network.assert_not_called()

    def test_explicit_source_prefix_and_preparation_only_input_do_not_become_runtime_paths(self):
        recipe = deepcopy(self.recipe)
        recipe['sources'][0]['filename'] = 'upstream/model.safetensors'
        recipe['sources'].append(dict(role='preparation-reference', kind='huggingface',
            repo_id='example/reference', revision='a' * 40, filename='audio/reference.wav'))
        sources = model_download._sources(recipe)
        selected = tuple(RuntimeMapping(asset['role'], asset['filename'], 'reference',
                                       input_id=f'source-{index}')
                         for index, asset in enumerate(self.recipe['sources']))
        mappings = model_download.runtime_mappings(recipe, sources, mappings=selected)
        assets = hem_models._mapped_assets(model_download.acquisition_inputs(sources), mappings)
        self.assertEqual([asset['logical_path'] for asset in assets],
                         ['model.safetensors', 'tokenizer.json', 'config.json'])
        self.assertTrue(assets[0]['source_path'].endswith('/upstream/model.safetensors'))
        self.assertNotIn('reference.wav', str(assets))
        self.prepare.assert_not_called(); self.network.assert_not_called()

    def test_slow_successful_obtain_does_not_consume_helper_allowance(self):
        clock = [0.0]
        sources = model_download._sources(self.recipe)
        for source in sources:
            model_download._cache_path(source).unlink()
        events = []
        def preflight():
            events.append('preflight')
            clock[0] += 25
        def fetch(command):
            events.append('obtain')
            clock[0] += 400  # Successful fetch takes longer than the helper cap.
            source = next(item for item in sources if item['filename'] == command[3])
            model_download._cache_path(source).write_bytes(b'freshly fetched')
        def prepare(inputs, attempt, deadline):
            events.append('prepare')
            self.assertEqual(deadline.remaining(), preparation.TIMEOUT)
            return self.completed(inputs, attempt, deadline)
        def after(seconds):
            return Deadline(clock[0] + seconds, clock=lambda: clock[0])
        self.preflight.side_effect = preflight
        self.network.side_effect = fetch
        self.prepare.side_effect = prepare
        with patch.object(hem_models, 'Deadline') as deadline:
            deadline.after.side_effect = after
            row = self.acquire(offline=False)
        self.assertTrue(row['validation']['valid'])
        self.assertEqual(events, ['preflight', 'obtain', 'obtain', 'obtain', 'prepare'])
        self.preflight.assert_called_once_with()
        deadline.after.assert_called_once_with(preparation.TIMEOUT)

    def test_incomplete_foreign_or_aliased_result_never_stages_or_publishes(self):
        def invalid(kind):
            def prepare(inputs, attempt, deadline):
                result = self.completed(inputs, attempt, deadline)
                output = result.outputs[0]
                if kind == 'missing':
                    return replace(result, outputs=())
                if kind == 'duplicate':
                    return replace(result, outputs=(output, output))
                if kind == 'unknown':
                    return replace(result, outputs=(replace(output, role='unknown'),))
                if kind == 'foreign-attempt':
                    return replace(result, attempt_directory=attempt.directory.parent)
                if kind == 'outside':
                    return replace(result, outputs=(replace(output, path=self.rotary),))
                if kind == 'nonregular':
                    output.path.unlink(); output.path.mkdir()
                if kind == 'symlink':
                    output.path.unlink(); output.path.symlink_to(self.rotary)
                return result
            return prepare
        for kind in ('missing', 'duplicate', 'unknown', 'foreign-attempt', 'outside', 'nonregular', 'symlink'):
            with self.subTest(kind=kind):
                self.prepare.side_effect = invalid(kind)
                with self.assertRaisesRegex(HemError, 'Incomplete preparation result'):
                    self.acquire(name=kind)
                row = self.registry.get_model(kind)
                self.assertFalse(Path(row['directory']).exists())
                self.assertFalse((Path(row['pending']['path']) / 'rotary.safetensors').exists())

    def test_rotary_appearing_during_preparation_is_not_adopted_or_proven(self):
        def conflicting(inputs, attempt, deadline):
            result = self.completed(inputs, attempt, deadline)
            record = self.registry.get_model('speech')
            stage = Path(record['pending']['path'])
            (stage / 'rotary.safetensors').write_bytes(b'unrelated staged rotary')
            return result
        self.prepare.side_effect = conflicting
        with self.assertRaisesRegex(HemError, 'Conflicting staged generated'):
            self.acquire()
        record = self.registry.get_model('speech')
        stage = Path(record['pending']['path'])
        self.assertFalse(Path(record['directory']).exists())
        self.assertEqual(record['pending']['stage'], 'staging')
        self.assertEqual((stage / 'rotary.safetensors').read_bytes(), b'unrelated staged rotary')
        self.assertFalse((stage / hem_models._GENERATED_RECEIPT).exists())
        # Refusal precedes all population, including the unrelated input aliases.
        self.assertFalse(any((stage / source['filename']).exists() for source in self.recipe['sources']))

    def test_published_missing_generated_file_never_regenerates(self):
        row = self.acquire()
        (Path(row['directory']) / 'rotary.safetensors').unlink()
        with self.assertRaisesRegex(HemError, 'identity is unproven'):
            self.acquire()
        self.prepare.assert_called_once()

    def test_original_receipt_survives_failed_validation_and_guards_all_consumers(self):
        row = self.acquire()
        receipt = Path(row['directory']) / hem_models._GENERATED_RECEIPT
        original = receipt.read_bytes()
        (Path(row['directory']) / 'rotary.safetensors').write_bytes(b'edited')
        for operation in (lambda: hem_models.validate(self.registry, 'speech'),
                          lambda: hem_models.model_files(self.registry, 'speech'), self.acquire):
            with self.assertRaisesRegex(HemError, 'identity is unproven'):
                operation()
            self.assertEqual(receipt.read_bytes(), original)
        self.assertFalse(self.registry.get_model('speech')['validation']['valid'])
        self.assertNotEqual(hem_models.show(self.registry, 'speech')['state'], 'ready')
        self.prepare.assert_called_once()

    def test_managed_start_and_cli_export_cannot_bypass_generated_guard(self):
        row = self.acquire()
        (Path(row['directory']) / 'rotary.safetensors').write_bytes(b'edited')
        self.registry.create_server('local', saved_settings={})
        output = self.root / 'export.json'
        commands = (['instance', 'start', 'speech', '--server', 'local'],
                    ['instance', 'export', 'speech', '--output', str(output)])
        with patch.object(hem_instance_cli, 'Registry', return_value=self.registry):
            for command in commands:
                args = hem_cli.parse_args(command)
                with self.assertRaisesRegex(HemError, 'identity is unproven'):
                    hem_instance_cli.run(args, Deadline.after())
        self.assertFalse(output.exists())
        self.assertEqual(self.registry.list_instances(), [])

    def test_missing_published_receipt_is_never_recreated(self):
        row = self.acquire()
        receipt = Path(row['directory']) / hem_models._GENERATED_RECEIPT
        receipt.unlink()
        for operation in (lambda: hem_models.validate(self.registry, 'speech'), self.acquire):
            with self.assertRaisesRegex(HemError, 'identity is unproven'):
                operation()
            self.assertFalse(receipt.exists())
        self.prepare.assert_called_once()

    def test_rename_before_database_update_cannot_adopt_missing_receipt(self):
        update = self.registry.update_model
        def crash(model_id, changes, **kwargs):
            if changes.get('pending') == {'stage': 'qualification'}:
                raise RuntimeError('crash after rename')
            return update(model_id, changes, **kwargs)
        with patch.object(self.registry, 'update_model', side_effect=crash), self.assertRaisesRegex(RuntimeError, 'rename'):
            self.acquire()
        row = self.registry.get_model('speech')
        self.assertEqual(row['pending']['stage'], 'staging')
        self.assertFalse(Path(row['pending']['path']).exists())
        receipt = Path(row['directory']) / hem_models._GENERATED_RECEIPT
        self.assertTrue(receipt.is_file())
        receipt.unlink()
        with self.assertRaisesRegex(HemError, 'identity is unproven'):
            self.acquire()
        self.assertFalse(receipt.exists())
        self.prepare.assert_called_once()

    def test_rename_before_database_update_recovers_with_original_receipt(self):
        update = self.registry.update_model
        def crash(model_id, changes, **kwargs):
            if changes.get('pending') == {'stage': 'qualification'}:
                raise RuntimeError('crash after rename')
            return update(model_id, changes, **kwargs)
        with patch.object(self.registry, 'update_model', side_effect=crash), self.assertRaises(RuntimeError):
            self.acquire()
        row = self.registry.get_model('speech')
        receipt = Path(row['directory']) / hem_models._GENERATED_RECEIPT
        original = receipt.read_bytes()
        self.acquire()
        self.assertEqual(receipt.read_bytes(), original)
        self.prepare.assert_called_once()
        self.assertTrue(self.registry.get_model('speech')['validation']['valid'])

    def test_staged_receipt_drift_refuses_adoption(self):
        with patch.object(self.registry, 'publish_model', side_effect=RuntimeError('before publication')):
            with self.assertRaises(RuntimeError):
                self.acquire()
        row = self.registry.get_model('speech')
        stage = Path(row['pending']['path'])
        receipt = stage / hem_models._GENERATED_RECEIPT
        original = receipt.read_bytes()
        (stage / 'rotary.safetensors').write_bytes(b'changed staged rotary')
        with self.assertRaisesRegex(HemError, 'rotary changed'):
            self.acquire()
        self.assertEqual(receipt.read_bytes(), original)
        self.assertFalse(Path(row['directory']).exists())


class PreparationProcessTest(unittest.TestCase):
    def cached_inputs(self, root):
        environment = patch.dict(os.environ, {'HF_HUB_CACHE': str(root / 'cache')})
        environment.start(); self.addCleanup(environment.stop)
        sources = model_download._sources(model_download.recipe_by_id('moonshine-streaming-tiny'))
        inputs = model_download.acquisition_inputs(sources)
        for item in inputs:
            item.path.parent.mkdir(parents=True, exist_ok=True)
            item.path.write_bytes(b'opaque input')
        return inputs

    def test_private_attempt_completion_not_launch_document_and_fresh_retry(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs = self.cached_inputs(root)
            calls = []
            def run(arguments, **kwargs):
                self.assertEqual(arguments[0], sys.executable)
                self.assertEqual(arguments[-1], '--offline')
                output = Path(arguments[arguments.index('--output') + 1])
                self.assertFalse(output.exists())
                calls.append(output)
                output.mkdir()
                for name in ('model.safetensors', 'tokenizer.json', 'config.json',
                             'rotary.safetensors', 'provenance.json'):
                    (output / name).write_bytes(b'opaque output')
                if len(calls) > 1:
                    (output / 'model.json').write_text('{"not": "a managed deployment"}')
                return 0
            with patch.object(preparation, '_run', side_effect=run):
                with self.assertRaisesRegex(HemError, 'did not complete'):
                    preparation.prepare(inputs, hem_models._preparation_attempt(root, 'speech'), Deadline.after())
                result = preparation.prepare(inputs, hem_models._preparation_attempt(root, 'speech'), Deadline.after())
            self.assertNotEqual(calls[0], calls[1])
            self.assertEqual(result.outputs[0].logical_path, 'rotary.safetensors')
            self.assertEqual(result.attempt_directory, calls[1].parent)
            self.assertEqual(result.facts, ())
            self.assertFalse(json.loads((calls[0].parent / 'receipt.json').read_text())['completed'])
            self.assertTrue(json.loads((calls[1].parent / 'receipt.json').read_text())['completed'])

    def test_timeout_reaps_owned_group(self):
        with tempfile.TemporaryDirectory() as temporary:
            pid_file = Path(temporary) / 'pid'
            code = ('import os, signal, time; from pathlib import Path; '
                    'signal.signal(signal.SIGTERM, signal.SIG_IGN); '
                    f'Path({str(pid_file)!r}).write_text(str(os.getpid())); time.sleep(30)')
            with patch.object(preparation, 'GRACE', .1), patch.object(preparation, 'tool_directory', return_value=Path(temporary)), self.assertRaises(subprocess.TimeoutExpired):
                preparation._run([sys.executable, '-c', code], timeout=.3,
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            pid = int(pid_file.read_text())
            with self.assertRaises(ProcessLookupError):
                os.kill(pid, 0)

    def test_sigterm_reaps_child_and_restores_previous_handler(self):
        with tempfile.TemporaryDirectory() as temporary:
            pid_file = Path(temporary) / 'pid'
            code = ('import os, signal, time; from pathlib import Path; '
                    'signal.signal(signal.SIGTERM, signal.SIG_IGN); '
                    f'Path({str(pid_file)!r}).write_text(str(os.getpid())); time.sleep(30)')
            previous = signal.getsignal(signal.SIGTERM)
            # Deliver the real signal only once the owned child has started.
            def terminate():
                import time
                for _ in range(100):
                    if pid_file.exists():
                        os.kill(os.getpid(), signal.SIGTERM)
                        return
                    time.sleep(.01)
            sender = threading.Thread(target=terminate)
            sender.start()
            try:
                with patch.object(preparation, 'GRACE', .1), patch.object(preparation, 'tool_directory', return_value=Path(temporary)):
                    with self.assertRaisesRegex(HemError, 'SIGTERM'):
                        preparation._run([sys.executable, '-c', code], timeout=5,
                                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            finally:
                sender.join(timeout=2)
            self.assertEqual(signal.getsignal(signal.SIGTERM), previous)
            with self.assertRaises(ProcessLookupError):
                os.kill(int(pid_file.read_text()), 0)

    def test_preflight_imports_use_trusted_tool_cwd_not_callers_shadow_modules(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            caller = root / 'caller'; caller.mkdir()
            trusted = root / 'trusted'; trusted.mkdir()
            marker = root / 'shadow-executed'
            (caller/'huggingface_hub.py').write_text(f'from pathlib import Path; Path({str(marker)!r}).touch()')
            # Valid-looking trusted API stubs exercise import context, not versions.
            (trusted/'huggingface_hub.py').write_text('hf_hub_download = None\n')
            (trusted/'safetensors.py').write_text('TensorSpec = None\nserialize = None\n')
            prior = Path.cwd()
            try:
                os.chdir(caller)
                with patch.object(preparation, 'tool_directory', return_value=trusted), patch.object(preparation, 'TOOL_FILES', {}):
                    preparation.preflight()
            finally:
                os.chdir(prior)
            self.assertFalse(marker.exists())

    def test_child_environment_has_same_cache_without_python_overrides(self):
        with patch.dict(os.environ, {'PYTHONPATH': '/unrelated', 'PYTHONHOME': '/unrelated', 'HF_HOME': '/keep'}):
            environment = preparation.environment()
            self.assertNotIn('PYTHONPATH', environment)
            self.assertNotIn('PYTHONHOME', environment)
            self.assertEqual(environment['HF_HOME'], '/keep')
            self.assertEqual(environment['HF_HUB_OFFLINE'], '1')
            self.assertEqual(environment['HF_HUB_CACHE'], str(model_download.cache_root()))

    def test_unsupported_interpreter_has_actionable_guidance(self):
        with patch.object(preparation.sys, 'version_info', (3, 11)), self.assertRaisesRegex(HemError, 'assets-dir'):
            preparation.preflight()


if __name__ == '__main__':
    unittest.main()
