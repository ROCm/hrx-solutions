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

from build_tools.validation import check_installed_cpu as gate


class InstalledPayloadTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.package = self.root / 'package'
        self.package.mkdir()
        self.files = {'__init__.py': b'', '_native/bin/loom-serve': b'ELF fixture',
                      'loom-pipeline/qwen/control.loom': b'catalog fixture'}
        source = Path(__file__).resolve().parents[2] / 'binding/python'
        for name in (
            'integrations/__init__.py',
            'integrations/omp/__init__.py',
            'integrations/omp/hem_omp.py',
            'integrations/omp/hem_omp_yaml.py',
            'models/__init__.py',
            'models/curated_catalog.py',
            'models/gguf.py',
            'models/hem_config.py',
            'models/hem_models.py',
            'models/hem_model_operations.py',
            'models/model_download.py',
            'models/model_store.py',
            'models/moonshine_preparation.py',
            'models/moonshine_preparer_identity.py',
            'models/pipeline_validation.py',
            'models/qwen_validation.py',
            'pipeline_validation.py',
):
            self.files[name] = (source / name).read_bytes()
        for relative in ('tui/__init__.py', 'tui/hem_tui.py', 'tui/hem_tui_details.py',
                     'tui/hem_tui_logs.py', 'tui/hem_tui_provider.py'):
            self.files[relative] = (source / relative).read_bytes()
        from build_tools.validation.check_installed_cpu import CLI_FILES
        for name in CLI_FILES:
            self.files[name] = (source / name).read_bytes()
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

    def test_models_require_unique_regular_members_in_the_wheel(self):
        import warnings
        for target in self.files:
            if not target.startswith('models/'):
                continue
            for failure in ('missing', 'changed', 'duplicate', 'symlink', 'directory'):
                with self.subTest(target=target, failure=failure):
                    with zipfile.ZipFile(self.wheel, 'w') as archive:
                        for name, data in self.files.items():
                            if name == target and failure == 'missing':
                                continue
                            member = zipfile.ZipInfo('hrx_solutions/' + name)
                            if name == target and failure in ('symlink', 'directory'):
                                member.external_attr = (0o120777 if failure == 'symlink' else 0o040777) << 16
                            archive.writestr(member, data + b'changed' if name == target and failure == 'changed' else data)
                            if name == target and failure == 'duplicate':
                                with warnings.catch_warnings():
                                    warnings.simplefilter('ignore', UserWarning)
                                    archive.writestr(member, data)
                    with self.assertRaises(AssertionError):
                        gate.verify_wheel_payload(self.wheel, self.package)

    def test_models_refuse_installed_symlinks_even_with_identical_bytes(self):
        for name in self.files:
            if not name.startswith('models/'):
                continue
            with self.subTest(name=name):
                target = self.package / name
                backup = target.with_suffix('.owned-copy')
                target.rename(backup)
                target.symlink_to(backup)
                try:
                    with self.assertRaisesRegex(AssertionError, 'not regular'):
                        gate.verify_wheel_payload(self.wheel, self.package)
                finally:
                    target.unlink()
                    backup.rename(target)

    def test_omp_require_unique_regular_members_in_the_wheel(self):
        import warnings
        for target in self.files:
            if not target.startswith('integrations/'):
                continue
            for failure in ('missing', 'changed', 'duplicate', 'symlink', 'directory'):
                with self.subTest(target=target, failure=failure):
                    with zipfile.ZipFile(self.wheel, 'w') as archive:
                        for name, data in self.files.items():
                            if name == target and failure == 'missing':
                                continue
                            member = zipfile.ZipInfo('hrx_solutions/' + name)
                            if name == target and failure in ('symlink', 'directory'):
                                member.external_attr = (0o120777 if failure == 'symlink' else 0o040777) << 16
                            archive.writestr(member, data + b'changed' if name == target and failure == 'changed' else data)
                            if name == target and failure == 'duplicate':
                                with warnings.catch_warnings():
                                    warnings.simplefilter('ignore', UserWarning)
                                    archive.writestr(member, data)
                    with self.assertRaises(AssertionError):
                        gate.verify_wheel_payload(self.wheel, self.package)

    def test_omp_refuse_installed_symlinks_even_with_identical_bytes(self):
        for name in self.files:
            if not name.startswith('integrations/'):
                continue
            with self.subTest(name=name):
                target = self.package / name
                backup = target.with_suffix('.owned-copy')
                target.rename(backup)
                target.symlink_to(backup)
                try:
                    with self.assertRaisesRegex(AssertionError, 'not regular'):
                        gate.verify_wheel_payload(self.wheel, self.package)
                finally:
                    target.unlink()
                    backup.rename(target)

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
        for relative, name in (('fixture.py', 'hrx_solutions.fixture'),
                               ('models/fixture.py', 'hrx_solutions.models.fixture'),
                               ('models/__init__.py', 'hrx_solutions.models'),
                               ('cli/fixture.py', 'hrx_solutions.cli.fixture'),
                               ('cli/__init__.py', 'hrx_solutions.cli'),
                               ('tui/fixture.py', 'hrx_solutions.tui.fixture'),
                               ('tui/__init__.py', 'hrx_solutions.tui')):
            with self.subTest(relative=relative):
                authored = source / relative
                authored.parent.mkdir(parents=True, exist_ok=True)
                authored.write_text('current = True\n')
                installed = self.package / relative
                installed.parent.mkdir(parents=True, exist_ok=True)
                installed.write_text('current = True\n')
                module = types.ModuleType(name)
                module.__file__ = str(installed)
                with patch.dict(sys.modules, {name: module}):
                    checked = gate.source_parity(self.root / 'source', self.package)
                    self.assertIn('binding/python/' + relative, checked)
                    installed.write_text('stale = True\n')
                    with self.assertRaisesRegex(AssertionError, 'stale installed module'):
                        gate.source_parity(self.root / 'source', self.package)
                    installed.write_text('current = True\n')
                    module.__file__ = str(authored)
                    with self.assertRaisesRegex(AssertionError, 'import escaped installed package'):
                        gate.source_parity(self.root / 'source', self.package)
                authored.unlink()

    def test_generated_catalog_parity_rejects_stale_sources_or_modified_bytecode(self):
        from build_tools.model_tools import loom_catalog
        from binding.python.models.pipeline_validation import catalog_modules
        repository = Path(__file__).resolve().parents[2]
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
        adapter = types.ModuleType('hrx_solutions.models.pipeline_validation')
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


class FixtureAdmissionTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        (self.root / 'tests').mkdir()

    def write(self, name, code):
        (self.root / 'tests' / (name + '.py')).write_text(code)

    def test_late_transitive_fixture_is_admitted_and_only_selected_test_runs(self):
        self.write('helper', 'value = 17\n')
        self.write('instances_test', 'from tests.helper import value\n')
        self.write('assets_test', '''import unittest
class Assets(unittest.TestCase):
    def test_selected(self):
        from tests.instances_test import value
        self.assertEqual(value, 17)
    def test_unselected(self):
        raise AssertionError("unselected test ran")
''')
        with patch.object(gate, 'HELPERS', ('helper',)), \
             patch.object(gate, 'CONTRACTS', ('instances_test', 'assets_test')), \
             patch.dict(sys.modules):
            complete = gate.load_contracts(self.root, self.root / 'package')
            self.assertEqual(complete.countTestCases(), 2)
            selected = gate.select_contracts(complete, ['tests.assets_test.Assets.test_selected'])
            result = unittest.TestResult()
            selected.run(result)
            self.assertEqual(result.testsRun, 1)
            self.assertTrue(result.wasSuccessful(), result.errors)

    def test_missing_late_fixture_rejects_before_any_fixture_code_executes(self):
        marker = self.root / 'executed'
        self.write('first', f'from pathlib import Path\nPath({str(marker)!r}).touch()\n')
        self.write('assets_test', 'def late():\n    from tests.instances_test import value\n')
        with patch.object(gate, 'HELPERS', ('first',)), \
             patch.object(gate, 'CONTRACTS', ('assets_test',)):
            with self.assertRaisesRegex(AssertionError, 'incomplete installed-contract fixtures.*instances_test'):
                gate.load_contracts(self.root, self.root / 'package')
        self.assertFalse(marker.exists())

    def test_missing_transitive_fixture_file_is_rejected(self):
        self.write('assets_test', 'from tests import instances_test as fixture\n')
        with self.assertRaisesRegex(AssertionError, 'missing installed-contract fixture: tests.instances_test'):
            gate.admit_fixtures(self.root, ('assets_test', 'instances_test'))

    def test_transitive_dependency_of_fixture_is_checked(self):
        self.write('assets_test', 'def late():\n    from tests import instances_test\n')
        self.write('instances_test', 'def helper():\n    import tests.absent\n')
        with self.assertRaisesRegex(AssertionError, 'incomplete installed-contract fixtures.*absent'):
            gate.admit_fixtures(self.root, ('assets_test', 'instances_test'))

    def test_exact_selection_rejects_unknown_and_duplicate_ids_before_execution(self):
        test = unittest.FunctionTestCase(lambda: self.fail('must not run'))
        for ids, message in ((['does.not.exist'], 'unknown'),
                             ([test.id(), test.id()], 'duplicate')):
            with self.subTest(ids=ids), self.assertRaisesRegex(ValueError, message):
                gate.select_contracts(unittest.TestSuite([test]), ids)

    def test_default_selection_preserves_complete_suite(self):
        suite = unittest.TestSuite([unittest.FunctionTestCase(lambda: None)])
        self.assertIs(gate.select_contracts(suite, []), suite)


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
