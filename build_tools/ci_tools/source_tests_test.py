# Copyright 2026 The HRX Authors
# Licensed under the MIT License.
# SPDX-License-Identifier: MIT
"""Guard the shared source-test commands and hosted consumer contract."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

from build_tools.ci_tools import source_tests


class SourceTestsContractTest(unittest.TestCase):
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
