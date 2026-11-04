# Copyright 2026 The HRX Authors
# Licensed under the MIT License.
# SPDX-License-Identifier: MIT
"""Guard the shared source-test commands and hosted consumer contract."""
from pathlib import Path
import os
import runpy
import sys
import tempfile
import subprocess
from types import SimpleNamespace
import unittest
from unittest import mock

from build_tools.ci_tools import source_tests
from build_tools.ci_tools import onboarding_fixture


class SourceTestsContractTest(unittest.TestCase):
    def test_decoder_lookup_uses_wheel_configuration_and_requires_built_output(self):
        from build_tools.model_tools.loom_catalog import DECODER_TARGET
        from build_tools.wheel_tools import python_packaging as packaging
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            decoder = root / 'loom-format'
            decoder.write_bytes(b'query locator fixture; never executed')
            for enabled in ('0', '1'):
                with self.subTest(cache_enabled=enabled), mock.patch.dict(
                        os.environ, {'HRX_BAZEL_CACHE_ENABLED': enabled}):
                    with mock.patch.object(packaging, 'discover_bazel', return_value=root / 'bazel'), \
                            mock.patch.object(packaging.subprocess, 'run', return_value=
                                              subprocess.CompletedProcess([], 0, str(decoder))) as query:
                        self.assertEqual(onboarding_fixture.built_decoder(root / 'rocm'), decoder)
                    self.assertEqual(query.call_args.args[0], [
                        str(root / 'bazel'), 'cquery', *packaging.wheel_build_options(root / 'rocm'),
                        '--output=files', DECODER_TARGET])
                    self.assertEqual(query.call_args.kwargs['env'], packaging.bazel_environment())
                    self.assertEqual(query.call_args.kwargs['cwd'], packaging.REPO_ROOT)
            with mock.patch.object(packaging, 'discover_bazel', return_value=root / 'bazel'), \
                    mock.patch.object(packaging.subprocess, 'run', return_value=
                                      subprocess.CompletedProcess([], 0, str(root / 'not-built'))):
                with self.assertRaisesRegex(RuntimeError, 'does not exist'):
                    onboarding_fixture.built_decoder(root / 'rocm')

    def test_onboarding_rejects_corrupt_bundle_and_unusable_decoder(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / 'native-fixture.zip'
            archive.write_bytes(b'corrupted native fixture')
            (root / 'native-fixture.json').write_text(
                '{"archive": "native-fixture.zip", "sha256": "incorrect"}')
            with mock.patch.object(onboarding_fixture, 'FIXTURES', root):
                with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
                    onboarding_fixture.prepare(root / 'extracted', Path(sys.executable))
            self.assertFalse((root / 'extracted').exists())
            with self.assertRaises(FileNotFoundError):
                onboarding_fixture.prepare(root / 'extracted', root / 'missing-decoder')

    def test_onboarding_rejects_existing_fixture_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(FileExistsError):
                onboarding_fixture.prepare(Path(temporary), Path(sys.executable))

    def test_onboarding_runner_rejects_empty_suite_and_skipped_cases(self):
        script = Path(__file__).resolve().parents[2] / 'tests/generic_modality_onboarding.py'
        suite = mock.Mock()
        with mock.patch.object(unittest.defaultTestLoader, 'loadTestsFromTestCase', return_value=suite):
            suite.countTestCases.return_value = 0
            with self.assertRaisesRegex(SystemExit, 'Expected all eight'):
                runpy.run_path(str(script), run_name='__main__')
            suite.countTestCases.return_value = 8
            result = mock.Mock(skipped=[('case', 'unexpected skip')])
            result.wasSuccessful.return_value = True
            with mock.patch.object(unittest.TextTestRunner, 'run', return_value=result):
                with self.assertRaises(SystemExit) as stopped:
                    runpy.run_path(str(script), run_name='__main__')
                self.assertEqual(stopped.exception.code, True)

    def test_onboarding_provisions_frozen_inputs_and_replaces_ambient_overrides(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Path(temporary) / 'fixture'
            decoder = Path(sys.executable).resolve()
            with mock.patch.dict(os.environ, {
                    'GENERIC_MODALITY_FIXTURE_WHEEL': '/foreign/wheel',
                    'GENERIC_MODALITY_FIXTURE_SOURCE_ROOT': '/foreign/source',
                    'GENERIC_MODALITY_FIXTURE_DECODER': '/foreign/decoder',
                    'GENERIC_MODALITY_FIXTURE_ARCHIVE': '/foreign/archive'}, clear=False):
                with mock.patch.object(source_tests.os, 'execv') as execute:
                    source_tests.main(['onboarding', '--decoder', str(decoder),
                                       '--fixture-directory', str(fixture)])
                self.assertNotIn('GENERIC_MODALITY_FIXTURE_WHEEL', os.environ)
                self.assertEqual(os.environ['GENERIC_MODALITY_FIXTURE_SOURCE_ROOT'],
                                 str(fixture / 'source'))
                self.assertEqual(os.environ['GENERIC_MODALITY_FIXTURE_DECODER'], str(decoder))
                archive = Path(os.environ['GENERIC_MODALITY_FIXTURE_ARCHIVE'])
                self.assertTrue(archive.is_file())
                self.assertTrue(any((fixture / 'source').rglob('*.loom')))
                execute.assert_called_once_with(sys.executable, [
                    sys.executable, 'tests/generic_modality_onboarding.py'])

    def test_onboarding_is_required_by_both_build_consumers(self):
        root = Path(__file__).resolve().parents[2]
        container = (root / '.github/scripts/build_python_wheel.sh').read_text()
        self.assertIn('set -euo pipefail', container)
        self.assertIn('-m build_tools.ci_tools.source_tests onboarding', container)
        self.assertIn('from build_tools.ci_tools.onboarding_fixture import built_decoder', container)
        self.assertNotIn('"$BAZEL" cquery', container)
        self.assertIn('-m venv "$HRX_OUTPUT_DIR/onboarding-venv"', container)
        self.assertIn('source_tests verify', container)
        for path in ('.github/workflows/build_python_wheel.yml', 'build_tools/ci_tools/local_ci.py'):
            self.assertIn('build_tools/build-wheel.sh', (root / path).read_text())

    def test_install_keeps_source_dependency_ranges(self):
        with mock.patch.object(source_tests.os, 'execv') as execute:
            source_tests.main(['install'])
        execute.assert_called_once_with(sys.executable, [
            sys.executable, '-m', 'pip', 'install', '--disable-pip-version-check',
            'setuptools>=77', 'wheel>=0.45', 'packaging>=24.2', 'build>=1.2', 'textual>=0.47,<2'])

    def test_verify_rejects_build_from_outside_standalone_venv(self):
        foreign_build = SimpleNamespace(__file__='/outside-venv/build/__init__.py')
        with mock.patch.dict(sys.modules, {'build': foreign_build}):
            with self.assertRaises(AssertionError):
                source_tests.verify_provenance()

    def test_each_discovery_replaces_process_with_original_unittest_command(self):
        # execv preserves unittest's exit/signal status and the caller's process
        # group, so the local supervisor still owns cancellation and stage logs.
        for directory in ('build_tools', 'tests'):
            with self.subTest(directory=directory), mock.patch.object(source_tests.os, 'execv') as execute:
                source_tests.main([directory])
                execute.assert_called_once_with(sys.executable, [
                    sys.executable, '-m', 'unittest', 'discover', '-s', directory,
                    '-p', '*_test.py', '-v'])

    def test_hosted_consumer_uses_shared_actions_and_cleans_fixture_environment(self):
        root = Path(__file__).resolve().parents[2]
        workflow = (root / '.github/workflows/build_python_wheel.yml').read_text()
        install = workflow.split('- name: Install Python test requirements', 1)[1].split('\n      - name:', 1)[0]
        run = workflow.split('- name: Run Python unit tests', 1)[1].split('\n      - name:', 1)[0]
        self.assertIn('env -u PYTHONPATH -u PYTHONHOME', install)
        self.assertIn('-m build_tools.ci_tools.source_tests install', install)
        self.assertIn('unset PYTHONPATH PYTHONHOME GITHUB_RUN_ID GITHUB_RUN_ATTEMPT', run)
        actions = [line.strip() for line in run.splitlines() if '-m build_tools.ci_tools.source_tests ' in line]
        self.assertEqual(actions, [
            f'"$python" -m build_tools.ci_tools.source_tests {action}'
            for action in ('verify', 'build_tools', 'tests')])
        self.assertNotIn('pip install', install)
        self.assertNotIn('unittest discover', run)
        self.assertNotIn('import os,site,sys', run)
        for name in ('LOOM_SERVE_DIR', 'LOOM_OPENCODE_TEST_NODE', 'LOOM_OMP_TEST_BUN', 'LOOM_OMP_TEST_PACKAGE'):
            self.assertIn(name + ':', run)


if __name__ == '__main__':
    unittest.main()
