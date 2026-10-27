"""Preserved fixed-helper boundary/process tests, owned by acquisition."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock, patch

from binding.python.hem_common import Deadline, DeadlineExpired, HemError
from binding.python.control.hem_state import Registry
from binding.python.models import hem_models, model_download, pipelines, recipes
from binding.python.models.preparation import moonshine as helper
from binding.python.models.preparation.moonshine_contract import load_operation
from binding.python.models.modalities import asr


class MoonshineHelperBoundaryTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        isolated_cache = patch.dict(os.environ, {'HF_HUB_CACHE': str(self.root / 'cache')})
        isolated_cache.start()
        self.addCleanup(isolated_cache.stop)
        self.inputs = {}
        for name, source in load_operation()['inputs'].items():
            path = model_download._cache_path(source)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'immutable input fixture')
            self.inputs[name] = path

    def attempt(self, name):
        directory = self.root / name
        directory.mkdir(mode=0o700)
        return directory

    def complete(self, arguments, *, missing=None, alias=False, **kwargs):
        output = Path(arguments[arguments.index('--output') + 1])
        self.assertFalse(output.exists())
        self.assertEqual(arguments[-1], '--offline')
        output.mkdir()
        for name in (*load_operation()['inputs'], 'rotary.safetensors', 'provenance.json', 'model.json'):
            if name == missing:
                continue
            if name == 'rotary.safetensors' and alias:
                (output / name).symlink_to(self.inputs['model.safetensors'])
            else:
                (output / name).write_bytes(b'opaque helper completion fixture')
        return 0

    def test_complete_helper_outputs_return_destination_map_and_receipt(self):
        attempt = self.attempt('one')
        before = {name: path.read_bytes() for name, path in self.inputs.items()}
        with patch.object(helper, '_run', side_effect=self.complete):
            outputs = helper.prepare(self.inputs, attempt, Deadline.after())
        self.assertEqual(outputs, {'rotary.safetensors': attempt / 'output/rotary.safetensors'})
        self.assertTrue(json.loads((attempt / 'receipt.json').read_text())['completed'])
        self.assertEqual(before, {name: path.read_bytes() for name, path in self.inputs.items()})

    def test_failed_completion_keeps_evidence_and_fresh_attempt_can_succeed(self):
        failed = self.attempt('one')
        with patch.object(helper, '_run', side_effect=lambda *a, **k: self.complete(*a, missing='model.json', **k)):
            with self.assertRaisesRegex(HemError, 'did not complete'):
                helper.prepare(self.inputs, failed, Deadline.after())
        self.assertFalse(json.loads((failed / 'receipt.json').read_text())['completed'])
        fresh = self.attempt('two')
        with patch.object(helper, '_run', side_effect=self.complete):
            self.assertEqual(set(helper.prepare(self.inputs, fresh, Deadline.after())), {'rotary.safetensors'})

    def test_generated_alias_is_not_completion(self):
        attempt = self.attempt('one')
        with patch.object(helper, '_run', side_effect=lambda *a, **k: self.complete(*a, alias=True, **k)):
            with self.assertRaisesRegex(HemError, 'aliased'):
                helper.prepare(self.inputs, attempt, Deadline.after())
        self.assertFalse(json.loads((attempt / 'receipt.json').read_text())['completed'])

    def test_bad_inputs_and_private_attempt_refused_before_process(self):
        for name, inputs in (('empty', {}), ('foreign', {**self.inputs, 'config.json': self.root / 'foreign'})):
            with self.subTest(name=name), patch.object(helper, '_run') as run:
                with self.assertRaises(HemError):
                    helper.prepare(inputs, self.attempt(name), Deadline.after())
                run.assert_not_called()
        shared = self.attempt('shared')
        shared.chmod(0o755)
        with patch.object(helper, '_run') as run, self.assertRaisesRegex(HemError, 'private fresh'):
            helper.prepare(self.inputs, shared, Deadline.after())
        run.assert_not_called()

    def test_preflight_preserves_budget_and_tool_integrity_before_spawn(self):
        budget = Mock()
        budget.remaining.return_value = 2.5
        with patch.object(helper.sys, 'version_info', (3, 12)), \
             patch.object(helper, 'load_operation', return_value={'tool_files': {}}), patch.object(helper, '_run', return_value=0) as run:
            helper.preflight(budget)
        self.assertEqual(run.call_args.kwargs['timeout'], 2.5)
        tool = self.root / 'tool'
        tool.mkdir()
        (tool / 'prepare_assets.py').write_bytes(b'changed')
        with patch.object(helper.sys, 'version_info', (3, 12)), \
             patch.object(helper, 'tool_directory', return_value=tool), \
             patch.object(helper, 'load_operation', return_value={'tool_files': {'prepare_assets.py': '0' * 64}}), \
             patch.object(helper, '_run') as run, self.assertRaisesRegex(HemError, 'integrity differs'):
            helper.preflight(Deadline.after())
        run.assert_not_called()

    def test_expired_preparation_budget_never_spawns_and_records_failure(self):
        attempt = self.attempt('expired')
        with patch.object(helper, '_run') as run, self.assertRaises(DeadlineExpired):
            helper.prepare(self.inputs, attempt, Deadline(0, clock=lambda: 1))
        run.assert_not_called()
        receipt = json.loads((attempt / 'receipt.json').read_text())
        self.assertFalse(receipt['completed'])
        self.assertIn('deadline expired', receipt['error'])

    def test_helper_uses_unchanged_supplied_post_fetch_budget(self):
        budget = Mock()
        budget.remaining.return_value = 1.25
        with patch.object(helper, '_run', side_effect=self.complete) as run:
            helper.prepare(self.inputs, self.attempt('budget'), budget)
        self.assertEqual(run.call_args.kwargs['timeout'], 1.25)

    def test_actual_reader_acquisition_helper_publication_and_native_definition(self):
        """Real source seams and tiny fixtures; external helper spawn is simulated."""
        registry = Registry(self.root / 'state')
        recipe, _ = recipes.load('moonshine-streaming-tiny')
        before = {name: (path.stat().st_ino, path.stat().st_size, path.stat().st_mtime_ns)
                  for name, path in self.inputs.items()}
        with patch.object(model_download, '_run_hf', side_effect=AssertionError('network forbidden')), \
             patch.object(helper, 'preflight') as preflight, patch.object(helper, '_run', side_effect=self.complete) as run:
            view = hem_models.download(registry, recipe['id'], model_id='speech', offline=True)
        self.assertEqual(view['state'], 'ready')
        preflight.assert_called_once()
        run.assert_called_once()
        record = registry.get_model('speech')
        paths, dependencies, evidence = hem_models.inspect_record(record)
        self.assertEqual(set(paths), set(recipe['files']))
        self.assertEqual(set(dependencies), set(paths.values()))
        self.assertEqual(evidence, record['validation']['evidence'])
        for asset in record['assets']:
            path = paths[asset['role']]
            if asset['role'] == 'rotary':
                self.assertFalse(path.is_symlink())
                self.assertEqual(asset['prepared_ownership'], 'generated-owned')
            else:
                self.assertTrue(path.is_symlink())
                self.assertEqual(os.readlink(path), str(self.inputs[recipe['files'][asset['role']]['path']]))
        self.assertEqual(before, {name: (path.stat().st_ino, path.stat().st_size, path.stat().st_mtime_ns)
                                  for name, path in self.inputs.items()})
        selected, catalog, _ = pipelines.load(recipe['executable'])
        asr.validate_layer(selected['serving'], recipe['configuration'])
        asr.finalize(selected['serving'], deepcopy(recipe['configuration']), {}, {})
        programs = pipelines.programs(selected, catalog, Deadline.after())
        native = asr.definition(recipe['configuration'], paths, Path(record['directory']), programs)
        self.assertEqual(native['kind'], 'recognition')
        self.assertEqual(native['assets'], record['directory'])
        with patch.object(helper, 'preflight', side_effect=AssertionError('complete retry preflight')), \
             patch.object(helper, '_run', side_effect=AssertionError('complete retry helper')):
            self.assertEqual(hem_models.download(registry, recipe['id'], model_id='speech', offline=True)['state'], 'ready')

    def test_actual_complete_prepared_import_bypasses_cache_and_helper(self):
        registry = Registry(self.root / 'state')
        prepared = self.root / 'prepared'
        prepared.mkdir()
        for item in recipes.load('moonshine-streaming-tiny')[0]['files'].values():
            (prepared / item['path']).write_bytes(b'caller-prepared fixture')
        with patch.object(model_download, '_run_hf', side_effect=AssertionError('network')), \
             patch.object(helper, 'preflight', side_effect=AssertionError('import preflight')), \
             patch.object(helper, 'prepare', side_effect=AssertionError('import helper')):
            hem_models.import_assets(registry, 'imported', directory=prepared,
                                     recipe_id='moonshine-streaming-tiny', alias=True)
        paths, _, _ = hem_models.inspect_record(registry.get_model('imported'))
        self.assertTrue(all(path.is_symlink() for path in paths.values()))


class MoonshineProcessBoundaryTest(unittest.TestCase):
    def test_timeout_reaps_owned_group(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pid_file = root / 'pid'
            code = ('import os, signal, time; from pathlib import Path; '
                    'signal.signal(signal.SIGTERM, signal.SIG_IGN); '
                    f'Path({str(pid_file)!r}).write_text(str(os.getpid())); time.sleep(30)')
            with patch.object(helper, 'GRACE', .1), patch.object(helper, 'tool_directory', return_value=root):
                with self.assertRaises(subprocess.TimeoutExpired):
                    helper._run([sys.executable, '-c', code], timeout=.5,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            with self.assertRaises(ProcessLookupError):
                os.kill(int(pid_file.read_text()), 0)

    def test_sigterm_reaps_child_and_restores_previous_handler(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pid_file = root / 'pid'
            code = ('import os, signal, time; from pathlib import Path; '
                    'signal.signal(signal.SIGTERM, signal.SIG_IGN); '
                    f'Path({str(pid_file)!r}).write_text(str(os.getpid())); time.sleep(30)')
            previous = signal.getsignal(signal.SIGTERM)
            def terminate():
                for _ in range(200):
                    if pid_file.exists():
                        os.kill(os.getpid(), signal.SIGTERM)
                        return
                    time.sleep(.01)
            sender = threading.Thread(target=terminate)
            sender.start()
            try:
                with patch.object(helper, 'GRACE', .1), patch.object(helper, 'tool_directory', return_value=root):
                    with self.assertRaisesRegex(HemError, 'SIGTERM'):
                        helper._run([sys.executable, '-c', code], timeout=3,
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            finally:
                sender.join(timeout=3)
            self.assertFalse(sender.is_alive())
            self.assertEqual(signal.getsignal(signal.SIGTERM), previous)
            with self.assertRaises(ProcessLookupError):
                os.kill(int(pid_file.read_text()), 0)

    def test_preflight_imports_only_from_trusted_tool_context(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            caller, trusted = root / 'caller', root / 'trusted'
            caller.mkdir()
            trusted.mkdir()
            marker = root / 'shadow-executed'
            (caller / 'huggingface_hub.py').write_text(f'from pathlib import Path; Path({str(marker)!r}).touch()')
            (trusted / 'huggingface_hub.py').write_text('hf_hub_download = None\n')
            (trusted / 'safetensors.py').write_text('TensorSpec = None\nserialize = None\n')
            prior = Path.cwd()
            try:
                os.chdir(caller)
                with patch.object(helper, 'tool_directory', return_value=trusted), patch.object(helper, 'load_operation', return_value={'tool_files': {}}):
                    helper.preflight(Deadline.after())
            finally:
                os.chdir(prior)
            self.assertFalse(marker.exists())

    def test_unsupported_interpreter_and_child_environment(self):
        with patch.object(helper.sys, 'version_info', (3, 11)), self.assertRaisesRegex(HemError, 'assets-dir'):
            helper.preflight(Deadline.after())
        with patch.dict(os.environ, {'PYTHONPATH': '/unrelated', 'PYTHONHOME': '/unrelated', 'HF_HOME': '/keep'}):
            environment = helper.environment()
            self.assertNotIn('PYTHONPATH', environment)
            self.assertNotIn('PYTHONHOME', environment)
            self.assertEqual(environment['HF_HOME'], '/keep')
            self.assertEqual(environment['HF_HUB_CACHE'], str(model_download.cache_root()))
            self.assertEqual(environment['HF_HUB_OFFLINE'], '1')


if __name__ == '__main__':
    unittest.main()
