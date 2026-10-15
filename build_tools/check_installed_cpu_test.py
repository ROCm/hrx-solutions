"""Installed CPU gate rejects stale artifacts and reports contract failures."""
from contextlib import redirect_stderr
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import zipfile

import check_installed_cpu as gate


class InstalledPayloadTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.package = self.root / 'package'
        self.package.mkdir()
        self.files = {'__init__.py': b'', '_native/bin/loom-serve': b'ELF fixture',
                      'loom-pipeline/qwen/control.loom': b'catalog fixture'}
        self.wheel = self.root / 'fixture.whl'
        with zipfile.ZipFile(self.wheel, 'w') as archive:
            for name, content in self.files.items():
                archive.writestr('hrx_solutions/' + name, content)
                target = self.package / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)

    def test_exact_wheel_identity_and_installed_payload_are_recorded(self):
        evidence = gate.verify_wheel_payload(self.wheel, self.package)
        self.assertEqual(evidence['wheel_sha256'], hashlib.sha256(self.wheel.read_bytes()).hexdigest())
        self.assertEqual(set(evidence['installed_files']), set(self.files))

    def test_changed_or_missing_native_source_or_catalog_is_rejected(self):
        for name in self.files:
            with self.subTest(name=name):
                target = self.package / name
                target.write_bytes(b'wrong build')
                with self.assertRaisesRegex(AssertionError, 'installed wheel mismatch'):
                    gate.verify_wheel_payload(self.wheel, self.package)
                target.unlink()
                with self.assertRaisesRegex(AssertionError, 'installed wheel mismatch'):
                    gate.verify_wheel_payload(self.wheel, self.package)
                target.write_bytes(self.files[name])

    def test_snapshot_ignores_only_bytecode_not_shipped_payloads(self):
        before = gate.snapshot(self.package)
        cache = self.package / '__pycache__'
        cache.mkdir()
        (cache / 'fixture.pyc').write_bytes(b'bytecode')
        self.assertEqual(gate.snapshot(self.package), before)
        (self.package / '_native/bin/loom-serve').write_bytes(b'changed')
        self.assertNotEqual(gate.snapshot(self.package), before)

    def test_source_parity_rejects_stale_code_and_checkout_imports(self):
        source = self.root / 'source/binding/python'
        source.mkdir(parents=True)
        (source / 'fixture.py').write_text('current = True\n')
        installed = self.package / 'fixture.py'
        installed.write_text('current = True\n')
        module = types.ModuleType('hrx_solutions.fixture')
        module.__file__ = str(installed)
        with patch.dict(sys.modules, {module.__name__: module}):
            checked = gate.source_parity(self.root / 'source', self.package)
            self.assertIn('binding/python/fixture.py', checked)
            installed.write_text('stale = True\n')
            with self.assertRaisesRegex(AssertionError, 'stale installed module'):
                gate.source_parity(self.root / 'source', self.package)
            module.__file__ = str(source / 'fixture.py')
            with self.assertRaisesRegex(AssertionError, 'import escaped installed package'):
                gate.source_parity(self.root / 'source', self.package)


class RunnerFailureTest(unittest.TestCase):
    def test_failed_contract_is_not_reported_as_success(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'pyvenv.cfg').write_text('include-system-site-packages = false\n')
            package = root / 'site-packages/hrx_solutions'
            package.mkdir(parents=True)
            (package / '__init__.py').write_text('')
            installed = types.ModuleType('hrx_solutions')
            installed.__file__ = str(package / '__init__.py')
            evidence = root / 'evidence.json'
            def fails():
                raise AssertionError('intentional contract failure')
            suite = unittest.TestSuite([unittest.FunctionTestCase(fails)])
            environment = dict(os.environ, LOOM_SERVE_DIR=str(root / 'state'))
            environment.pop('PYTHONPATH', None)
            environment.pop('PYTHONHOME', None)
            with patch.object(sys, 'prefix', str(root)), patch.object(gate.site, 'ENABLE_USER_SITE', False), \
                 patch.dict(sys.modules, {'hrx_solutions': installed}), \
                 patch.dict(os.environ, environment, clear=True), \
                 patch.object(gate, 'source_parity', return_value={}), \
                 patch.object(gate, 'load_contracts', return_value=suite), redirect_stderr(io.StringIO()):
                with self.assertRaisesRegex(AssertionError, 'installed CPU contracts failed'):
                    gate.main(['--source-root', str(root / 'source'), '--evidence', str(evidence)])
            result = json.loads(evidence.read_text())
            self.assertFalse(result['success'])
            self.assertEqual(result['failures'], 1)
            self.assertEqual(result['tests_run'], 1)


if __name__ == '__main__':
    unittest.main()
