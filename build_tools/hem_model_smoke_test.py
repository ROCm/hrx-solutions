"""Failure propagation in the installed real-model smoke harness."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import tempfile
import unittest
from contextlib import redirect_stderr
import io

from build_tools import check_installed_wheel_test as checker_fixtures
from binding.python import hem_process
from binding.python.hem_common import Deadline
from binding.python.hem_shutdown import ShutdownManager
from unittest import mock

import hem_model_smoke


class SmokeFailureTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.gate = hem_model_smoke.Gate(argparse.Namespace(run=self.root))
        self.gate.states['a'] = self.root / 'venv/bin/python'

    def test_failed_child_is_recorded_and_fails_gate(self):
        failed = subprocess.CompletedProcess([], 1, '{"completed":false}', 'native failure')
        with mock.patch.object(hem_model_smoke.subprocess, 'run', return_value=failed):
            with self.assertRaises(AssertionError):
                self.gate.command('a', 'instance', 'start', 'model')
        evidence = json.loads((self.root / 'results.json').read_text())
        self.assertEqual(evidence['commands'][0]['exit'], 1)
        self.assertFalse(evidence['success'])

    def test_successful_exit_cannot_mask_invalid_native_order(self):
        diagnostic = '\n'.join('curl --request POST http://localhost/v1/models/' + action
                               for action in ('register', 'undrain', 'activate'))
        child = subprocess.CompletedProcess([], 0, '{"completed":true}', diagnostic)
        with mock.patch.object(hem_model_smoke.subprocess, 'run', return_value=child):
            with self.assertRaises(AssertionError):
                self.gate.command('a', 'instance', 'start', 'model')
        self.assertFalse(self.gate.results['success'])


class SmokePendingLaunchTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        checker_fixtures.CheckerCleanupTest.setUpClass()

    @classmethod
    def tearDownClass(cls):
        checker_fixtures.CheckerCleanupTest.tearDownClass()

    def setUp(self):
        self.fixture = checker_fixtures.CheckerCleanupTest('test_timed_out_start_reconciles_and_aborts_only_its_owned_server')
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.case = self.fixture.case
        cli = self.case.root / 'loom-hem'
        cli.write_text(self.fixture.cli.read_text().replace('binding.python', 'hrx_solutions'))
        cli.chmod(0o700)
        self.gate = hem_model_smoke.Gate(argparse.Namespace(run=self.case.root))
        python = self.case.root / 'python'
        python.write_text('#!' + sys.executable + '\nimport os,sys\nos.execv(sys.executable,[sys.executable,*sys.argv[1:]])\n')
        python.chmod(0o700)
        self.gate.states['a'] = python
        self.gate.namespace_roots['a'] = '.'
        self.gate.setup = lambda: None

    def assert_failed_launch_cleaned(self, outer_timeout=False, cleanup_failure=False):
        other = self.case.start('unrelated')
        original = self.gate.command
        captured = {}
        if outer_timeout:
            cli = self.case.root / 'loom-hem'
            cli.write_text(cli.read_text().replace('["--timeout","0.15"]', '["--timeout","5"]'))

        def command(namespace, *args, **kwargs):
            if cleanup_failure and args[:2] == ('server', 'reconcile'):
                raise RuntimeError('injected cleanup observation failure')
            try:
                if outer_timeout and args[:2] == ('server', 'start'):
                    kwargs['command_timeout'] = .3
                return original(namespace, *args, **kwargs)
            except (AssertionError, subprocess.TimeoutExpired) as error:
                if args[:2] == ('server', 'start'):
                    owned = self.case.registry.get_server('server-a')
                    captured['process'] = owned['process']
                    self.case.owned.append((owned['process'], owned['process']['executable']))
                    captured['error'] = error
                    raise error
                raise

        diagnostic = io.StringIO()
        with mock.patch.object(self.gate, 'command', side_effect=command), redirect_stderr(diagnostic):
            with self.assertRaises(subprocess.TimeoutExpired if outer_timeout else AssertionError) as caught:
                self.gate.run()
        self.assertIs(caught.exception, captured['error'])
        if cleanup_failure:
            self.assertIn('Cleanup unresolved', diagnostic.getvalue())
            self.assertIn('injected cleanup observation failure', diagnostic.getvalue())
            self.assertFalse(hem_process._confirmed_exit(captured['process']))
            self.assertIn('error', self.gate.results['cleanup'][-1])
            budget = Deadline.after(3)
            while self.case.manager.reconcile('server-a', budget)['phase'] != 'running':
                time.sleep(min(.01, budget.remaining()))
            ShutdownManager(self.case.registry, processes=self.case.manager).shutdown('server-a', budget, abort=True)
        self.assertTrue(hem_process._confirmed_exit(captured['process']))
        self.assertTrue(hem_process._confirmed_exit(captured['process']['logger']))
        self.assertTrue(self.gate.results['cleanup'])
        self.assertFalse(self.gate.results['success'])
        self.assertFalse(hem_process._confirmed_exit(other['process']))
        with self.case.manager.pin(other) as pinned:
            self.assertEqual(self.case.manager.client(other, pinned, Deadline.after()).status()['data'], [])
        ShutdownManager(self.case.registry, processes=self.case.manager).shutdown('unrelated', Deadline.after(3), abort=True)

    def test_accepted_slow_elf_start_failure_reconciles_and_aborts(self):
        self.assert_failed_launch_cleaned()

    def test_outer_timeout_after_submission_preserves_error_and_cleans_real_elf(self):
        self.assert_failed_launch_cleaned(outer_timeout=True)

    def test_unknown_cleanup_preserves_start_error_and_reports_live_process(self):
        self.assert_failed_launch_cleaned(cleanup_failure=True)

    def test_prelaunch_failure_does_not_abort_unrelated_server(self):
        other = self.case.start('unrelated')
        failure = FileNotFoundError('injected CLI launch failure')
        with mock.patch.object(self.gate, 'command', side_effect=failure):
            with self.assertRaises(FileNotFoundError) as caught:
                self.gate.run()
        self.assertIs(caught.exception, failure)
        self.assertFalse(self.gate.attempts)
        self.assertTrue(self.gate.results['cleanup'][-1]['absent'])
        self.assertFalse(hem_process._confirmed_exit(other['process']))
        ShutdownManager(self.case.registry, processes=self.case.manager).shutdown('unrelated', Deadline.after(3), abort=True)


if __name__ == '__main__':
    unittest.main()
