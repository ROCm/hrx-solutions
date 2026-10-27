"""Generated ownership and retry contracts through the common binding interface."""
import signal
import shutil
import subprocess
import sys
import tempfile
import threading
import json
import os
from pathlib import Path
from unittest.mock import patch

import unittest
from tests.hem_models_test import AcquisitionFixture
from binding.python.models import hem_models, model_download, recipes, pipelines
from binding.python.models.preparation import moonshine as preparation
from binding.python.hem_common import Deadline, HemError


class MoonshineAcquisitionTest(AcquisitionFixture, unittest.TestCase):
    # Reuse environment setup, not the CHAT-specific inherited test suite.
    def setUp(self):
        super().setUp()
        self.recipe = self.declarations['moonshine-streaming-tiny']
        self.binding = preparation
        self.cached()

    def acquire(self, name='speech', **kwargs):
        kwargs.setdefault('offline', True)
        return self.download(name=name, **kwargs)

    def test_offline_mapping_generated_owned_and_completed_retry(self):
        result = self.acquire()
        self.assertEqual(result['state'], 'ready')
        record, paths, _, _ = self.inspect('speech')
        self.assertEqual([a['prepared_ownership'] for a in record['assets']], ['hf-reference']*3+['generated-owned'])
        final = paths['rotary']; inode = final.stat().st_ino
        self.assertFalse(final.is_symlink())
        self.assertTrue(paths['weights'].is_symlink())
        self.acquire(); self.assertEqual(final.stat().st_ino, inode)
        self.assertEqual(self.events, ['preflight', 'prepare'])
        final.write_bytes(b'changed')
        with self.assertRaisesRegex(HemError, 'output changed'): self.acquire()
        self.assertEqual(self.events, ['preflight', 'prepare'])

    def test_missing_cache_and_failed_helper_remain_incomplete_retryable(self):
        source = model_download._cache_path(self.recipe['files']['weights']['source']); source.unlink()
        with self.assertRaisesRegex(HemError, 'Offline cache'): self.acquire()
        self.assertEqual(hem_models.show(self.registry, 'speech')['state'], 'incomplete')
        self.assertNotIn('prepare', self.events)
        source.write_bytes(b'restored')
        with patch.object(self.binding, 'prepare', side_effect=HemError('helper failed')):
            with self.assertRaisesRegex(HemError, 'helper failed'): self.acquire()
        row = self.registry.get_model('speech')
        self.assertFalse(Path(row['directory']).exists())
        self.assertFalse((Path(row['pending']['path']) / 'rotary.safetensors').exists())
        self.acquire(); self.assertEqual(hem_models.show(self.registry, 'speech')['state'], 'ready')

    def test_preflight_before_network_and_complete_import_bypasses_helper(self):
        model_download._cache_path(self.recipe['files']['weights']['source']).unlink()
        with patch.object(self.binding, 'preflight', side_effect=HemError('missing dependencies')):
            with self.assertRaisesRegex(HemError, 'missing dependencies'): self.acquire(offline=False)
            self.fetch.assert_not_called()
            root = self.root / 'prepared'; root.mkdir()
            for item in self.recipe['files'].values(): (root / item['path']).write_bytes(b'local prepared')
            row = hem_models.import_assets(self.registry, 'local', directory=root, recipe_id=self.recipe['id'])
        self.assertTrue(row['validation']['valid'])
        self.assertEqual(self.events, [])
        record, _, _, _ = self.inspect('local')
        self.assertTrue(all(a['prepared_ownership'] == 'caller-owned' for a in record['assets']))

    def test_preview_runs_no_preflight_process_network_or_registry(self):
        with patch.object(self.binding, 'preflight', side_effect=AssertionError('preflight')):
            result = hem_models.download(None, self.recipe['id'], model_id='preview', dry_run=True)
        self.assertEqual(result['state'], 'preview')
        self.assertEqual(result['artifacts'][-1]['ownership'], 'generated-owned')
        self.assertIsNone(result['artifacts'][-1]['cache_path'])
        self.assertEqual(self.registry.list_models(), [])
        self.assertEqual(self.events, []); self.fetch.assert_not_called()

    def test_slow_successful_obtain_does_not_consume_helper_allowance(self):
        clock, events = [0.0], []
        for item in self.recipe['files'].values():
            if item['source']['kind'] == 'huggingface': model_download._cache_path(item['source']).unlink()
        prepare = self.binding.prepare
        def preflight(deadline):
            self.assertEqual(deadline.remaining(), 30); clock[0] += 25; events.append('preflight')
        def fetch(command):
            events.append('obtain'); clock[0] += 400
            item = next(a for a in self.recipe['files'].values() if a['source'].get('filename') == command[3])
            model_download._cache_path(item['source']).write_bytes(b'input')
        def generate(inputs, attempt, deadline):
            events.append('prepare'); self.assertEqual(deadline.remaining(), 300)
            return prepare(inputs, attempt, deadline)
        with patch.object(hem_models, 'Deadline') as deadlines, \
             patch.object(self.binding, 'preflight', side_effect=preflight), \
             patch.object(self.binding, 'prepare', side_effect=generate):
            deadlines.after.side_effect = lambda seconds=300: Deadline(clock[0] + seconds, clock=lambda: clock[0])
            self.fetch.side_effect = fetch
            self.acquire(offline=False)
        self.assertEqual(events, ['preflight', 'obtain', 'obtain', 'obtain', 'prepare'])

    def test_incomplete_foreign_or_aliased_result_never_stages_or_publishes(self):
        prepare = self.binding.prepare
        for kind in ('missing','extra','duplicate-sequence','foreign','symlink','wrong-destination','list'):
            def invalid(inputs, attempt, deadline):
                result = prepare(inputs, attempt, deadline); path = result['rotary.safetensors']
                if kind == 'missing': return {}
                if kind == 'extra': return dict(result, extra=path)
                if kind == 'duplicate-sequence': return [('rotary.safetensors',path)]*2
                if kind == 'foreign': return {'rotary.safetensors':self.gguf}
                if kind == 'wrong-destination': return {'other.safetensors':path}
                if kind == 'list': return list(result.items())
                path.unlink(); path.symlink_to(self.gguf); return result
            with self.subTest(kind=kind), patch.object(self.binding,'prepare',side_effect=invalid):
                with self.assertRaisesRegex(HemError,'Incomplete preparation'): self.acquire(name=kind)
                row = self.registry.get_model(kind)
                self.assertFalse(Path(row['directory']).exists())
                self.assertFalse((Path(row['pending']['path'])/'rotary.safetensors').exists())

    def test_output_appearing_during_preparation_is_not_adopted(self):
        prepare = self.binding.prepare
        def raced(inputs, attempt, deadline):
            row = self.registry.get_model('speech')
            (Path(row['pending']['path']) / 'rotary.safetensors').write_bytes(b'foreign')
            return prepare(inputs, attempt, deadline)
        with patch.object(self.binding, 'prepare', side_effect=raced):
            with self.assertRaisesRegex(HemError, 'Conflicting staged generated'): self.acquire()
        row = self.registry.get_model('speech')
        self.assertFalse(Path(row['directory']).exists())
        self.assertEqual((Path(row['pending']['path']) / 'rotary.safetensors').read_bytes(), b'foreign')

    def test_unproven_staged_generated_output_is_not_adopted_on_retry(self):
        with patch.object(self.binding, 'preflight', side_effect=HemError('interrupted')):
            with self.assertRaises(HemError): self.acquire()
        row = self.registry.get_model('speech'); stage = Path(row['pending']['path'])
        (stage / 'rotary.safetensors').write_bytes(b'foreign')
        with self.assertRaisesRegex(HemError, 'identity is unproven'): self.acquire()
        self.assertFalse((stage / hem_models._GENERATED_RECEIPT).exists())
        self.assertFalse(Path(row['directory']).exists())

    def test_published_missing_generated_file_never_regenerates(self):
        self.acquire()
        root = Path(self.registry.get_model('speech')['directory'])
        (root / 'rotary.safetensors').unlink()
        with self.assertRaisesRegex(HemError, 'identity is unproven'): self.acquire()
        self.assertEqual(self.events, ['preflight', 'prepare'])

    def test_original_receipt_guards_all_consumers(self):
        self.acquire(); row = self.registry.get_model('speech'); root = Path(row['directory'])
        receipt = root / hem_models._GENERATED_RECEIPT; original = receipt.read_bytes()
        (root / 'rotary.safetensors').write_bytes(b'drift')
        for consume in (lambda: hem_models.validate(self.registry, 'speech'),
                        lambda: self.inspect('speech'), self.acquire):
            with self.assertRaisesRegex(HemError, 'identity is unproven'): consume()
        self.assertFalse(hem_models.show(self.registry, 'speech')['accessible'])
        self.assertEqual(receipt.read_bytes(), original)
        self.assertEqual(self.events, ['preflight', 'prepare'])

    def test_missing_published_receipt_never_recreated(self):
        self.acquire(); root = Path(self.registry.get_model('speech')['directory'])
        receipt = root / hem_models._GENERATED_RECEIPT; receipt.unlink()
        with self.assertRaisesRegex(HemError, 'identity is unproven'): self.acquire()
        self.assertFalse(receipt.exists())

    def test_rename_before_database_update_recovers_with_original_receipt(self):
        update = self.registry.update_model
        def crash(model_id, changes, **kwargs):
            if changes.get('pending') == {'stage': 'qualification'}: raise RuntimeError('crash')
            return update(model_id, changes, **kwargs)
        with patch.object(self.registry, 'update_model', side_effect=crash):
            with self.assertRaises(RuntimeError): self.acquire()
        row = self.registry.get_model('speech'); root = Path(row['directory'])
        receipt = (root / hem_models._GENERATED_RECEIPT).read_bytes()
        self.acquire()
        self.assertEqual((root / hem_models._GENERATED_RECEIPT).read_bytes(), receipt)
        self.assertEqual(self.events, ['preflight', 'prepare'])

    def test_rename_before_database_update_cannot_adopt_missing_receipt(self):
        update = self.registry.update_model
        def crash(model_id, changes, **kwargs):
            if changes.get('pending') == {'stage': 'qualification'}: raise RuntimeError('crash')
            return update(model_id, changes, **kwargs)
        with patch.object(self.registry, 'update_model', side_effect=crash):
            with self.assertRaises(RuntimeError): self.acquire()
        row = self.registry.get_model('speech'); root = Path(row['directory'])
        receipt = root / hem_models._GENERATED_RECEIPT; receipt.unlink()
        with self.assertRaisesRegex(HemError, 'identity is unproven'): self.acquire()
        self.assertFalse(receipt.exists()); self.assertFalse(Path(row['pending']['path']).exists())

    def test_staged_receipt_drift_refuses_adoption(self):
        with patch.object(self.registry, 'publish_model', side_effect=RuntimeError('before publication')):
            with self.assertRaises(RuntimeError): self.acquire()
        row = self.registry.get_model('speech'); root = Path(row['pending']['path'])
        receipt = root / hem_models._GENERATED_RECEIPT; original = receipt.read_bytes()
        (root / 'rotary.safetensors').write_bytes(b'drift')
        with self.assertRaisesRegex(HemError, 'output changed'): self.acquire()
        self.assertEqual(receipt.read_bytes(), original)
        self.assertFalse(Path(row['directory']).exists())

    def test_extra_generated_purpose_is_rejected_before_effects(self):
        from copy import deepcopy
        data = deepcopy(self.recipe)
        data['files']['rotary2'] = dict(path='nested/second.safetensors',source={'kind':'generated'})
        source = next(root for root in pipelines.catalog_roots() if root.is_dir())
        root = self.root/'catalog'; shutil.copytree(source, root)
        selected = next(root.glob(f"**/recipes/{data['id']}.json"))
        selected.write_text(json.dumps(data))
        with patch.object(pipelines,'catalog_roots',return_value=(root,)):
            with self.assertRaisesRegex(HemError,'asset'): self.acquire()
        self.assertEqual(self.events,[]); self.fetch.assert_not_called()
        self.assertEqual(self.registry.list_models(), [])

    def test_generated_result_does_not_reuse_one_inode_for_two_destinations(self):
        from binding.python.models.model_store import generated_outputs, ModelImportError
        attempt = self.root/'attempt'; attempt.mkdir()
        first = attempt/'first'; first.write_bytes(b'owned')
        second = attempt/'second'; os.link(first,second)
        with self.assertRaisesRegex(ModelImportError,'aliased'):
            generated_outputs(attempt, {'one':first,'two':second}, ('one','two'))


class PreparationProcessTest(unittest.TestCase):
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
                with patch.object(preparation, 'tool_directory', return_value=trusted), patch.object(preparation, 'load_operation', return_value={'tool_files': {}}):
                    preparation.preflight(Deadline.after(30))
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
            preparation.preflight(Deadline.after(30))
