"""Installed CPU gate rejects stale artifacts and reports contract failures."""
from contextlib import redirect_stderr
import hashlib
import io
import json
import os
import shutil
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

    def test_generated_catalog_parity_rejects_stale_sources_or_modified_bytecode(self):
        from build_tools import loom_catalog
        from binding.python.pipeline_validation import catalog_modules
        repository = Path(__file__).resolve().parents[1]
        source_root = self.root / 'catalog-source'
        shutil.copytree(repository / 'loom-pipeline', source_root / 'loom-pipeline', symlinks=True)
        inventory = source_root / 'loom-pipeline/catalog_sources.bzl'
        products = {}
        for architecture, variant in loom_catalog.variants(inventory).items():
            output = self.root / 'bc' / variant['output']
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(b'LOOMfixture')
            products[architecture] = output
        with patch.object(loom_catalog, 'validate_bytecode_artifact'):
            loom_catalog.stage_bytecode_catalog(self.package / 'loom-pipeline/llm', products, inventory, decoder=Path('fixture-decoder'))
        (self.package / 'loom-pipeline/qwen/control.loom').unlink()
        adapter = types.ModuleType('hrx_solutions.pipeline_validation')
        adapter.catalog_modules = catalog_modules
        with patch.dict(sys.modules, {adapter.__name__:adapter}):
            self.assertIn('loom-pipeline/llm/qwen/qwen_08b/prepare.loom', gate.source_parity(source_root,self.package))
            source = source_root / 'loom-pipeline/llm/qwen/qwen_08b/prepare.loom'
            original = source.read_bytes()
            source.write_bytes(original+b'\n// current source changed')
            with self.assertRaisesRegex(AssertionError,'stale installed catalog input'):
                gate.source_parity(source_root,self.package)
            source.write_bytes(original)
            output = self.package / 'loom-pipeline/llm/qwen/qwen_08b/pipeline.loombc'
            output.write_bytes(b'LOOMchanged')
            with self.assertRaisesRegex(AssertionError,'installed bytecode mismatch'):
                gate.source_parity(source_root,self.package)
            output.write_bytes(b'LOOMfixture')
            local = output.parent
            mutations = (
                ('pipeline.json', 'installed pipeline descriptor mismatch'),
                ('compatibility.json', 'installed compatibility profile mismatch'),
                ('weights.loombc', 'unexpected executable modules'),
            )
            for filename, error in mutations:
                target = local / filename
                original = target.read_bytes() if target.exists() else None
                target.write_bytes(b'{}' if filename.endswith('.json') else b'LOOMextra')
                with self.subTest(filename=filename), self.assertRaisesRegex((AssertionError, ValueError), error):
                    gate.source_parity(source_root, self.package)
                if original is None:
                    target.unlink()
                else:
                    target.write_bytes(original)


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
