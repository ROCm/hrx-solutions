"""Failure propagation in the installed real-model smoke harness."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import tempfile
import sqlite3
import socket
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


class ShutdownVerificationTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.gate = hem_model_smoke.Gate(argparse.Namespace(run=self.root))
        python = self.root / 'python'
        source = Path(__file__).resolve().parents[1]
        python.write_text('#!' + sys.executable + '\nimport sys\n'
                          f'sys.path.insert(0, {str(source)!r})\n'
                          'code=sys.argv[2].replace("hrx_solutions", "binding.python")\n'
                          'sys.argv=["-c",*sys.argv[3:]]\nexec(code)\n')
        python.chmod(0o700)
        self.gate.states['a'] = python
        self.listener = socket.socket()
        self.listener.bind(('127.0.0.1', 0))
        self.port = self.listener.getsockname()[1]
        self.addCleanup(self.listener.close)
        self.command = mock.patch.object(self.gate, 'command', return_value={
            'completed': True, 'state': 'stopped'})
        self.command.start()
        self.addCleanup(self.command.stop)

    def child(self):
        child = subprocess.Popen([sys.executable, '-c', 'import sys; sys.stdin.read()'],
                                 stdin=subprocess.PIPE)
        def cleanup():
            child.stdin.close()
            child.wait(timeout=3)
        self.addCleanup(cleanup)
        return child, hem_process.identity(child.pid)

    def zombie(self):
        child, process = self.child()
        child.stdin.close()
        os.waitid(os.P_PID, child.pid, os.WEXITED | os.WNOWAIT)
        self.assertTrue(Path(f'/proc/{child.pid}').exists())
        return process

    def configure(self, server, logger):
        process = dict(server, logger=logger)
        self.gate.servers['a'] = ('server-a', f'http://127.0.0.1:{self.port}', process, self.port)
        self.gate.attempts['a'] = {'process': process}

    def test_shutdown_accepts_real_unreaped_server_and_logger(self):
        self.configure(self.zombie(), self.zombie())
        self.listener.close()
        self.gate.shutdown('a')
        self.assertTrue(self.gate.results['cleanup'][0]['processes_exited'])
        self.assertNotIn('a', self.gate.attempts)

    def test_live_server_or_logger_times_out_and_preserves_recovery(self):
        exited = self.zombie()
        _, live = self.child()
        for server, logger in ((live, exited), (exited, live)):
            with self.subTest(server=server['pid'], logger=logger['pid']):
                self.configure(server, logger)
                # Exercise one real installed-inspection equivalent before the bound.
                self.assertFalse(self.gate.inspect_server('a', 'server-a',
                                 self.gate.servers['a'][2])['exited'])
                with mock.patch.object(hem_model_smoke.time, 'monotonic', side_effect=[0, 0, 11, 11]):
                    with self.assertRaises(TimeoutError):
                        self.gate.shutdown('a')
                self.assertIn('a', self.gate.attempts)
                self.assertFalse(self.gate.results['cleanup'])
                self.assertTrue((self.root / 'shutdown-a.json').exists())

    def test_open_port_fails_even_after_confirmed_exit(self):
        self.configure(self.zombie(), self.zombie())
        self.listener.listen()
        with self.assertRaisesRegex(AssertionError, 'owned port still open'):
            self.gate.shutdown('a')
        self.assertIn('a', self.gate.attempts)
        self.assertFalse(self.gate.results['cleanup'])

    def test_inspection_error_cannot_become_success(self):
        self.configure(self.zombie(), self.zombie())
        with mock.patch.object(self.gate, 'inspect_server', side_effect=RuntimeError('unreadable')):
            with self.assertRaisesRegex(RuntimeError, 'unreadable'):
                self.gate.shutdown('a')
        self.assertIn('a', self.gate.attempts)
        self.assertFalse(self.gate.results['cleanup'])
        evidence = json.loads((self.root / 'shutdown-a.json').read_text())
        self.assertEqual(evidence['server']['state'], 'Z')


class AcquisitionPathTest(unittest.TestCase):
    def test_download_subprocess_uses_selected_venv_hf_with_clean_or_hostile_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            gate = hem_model_smoke.Gate(argparse.Namespace(run=root))
            source = Path(__file__).resolve().parents[1]
            empty = root / 'empty-bin'
            global_bin = root / 'global-bin'
            empty.mkdir()
            global_bin.mkdir()
            global_hf = global_bin / 'hf'
            global_hf.write_text('#!' + sys.executable + '\nraise SystemExit(73)\n')
            global_hf.chmod(0o700)
            for namespace in ('a', 'b'):
                binary = root / namespace / 'venv/bin'
                binary.mkdir(parents=True)
                gate.states[namespace] = binary / 'python'
                hf = binary / 'hf'
                hf.write_text(
                    '#!' + sys.executable + '\nimport json,os,sys,site\n'
                    'print(json.dumps(dict(hf=sys.argv[0], home=os.environ["HOME"],'
                    'xdg=os.environ["XDG_CONFIG_HOME"], cache=os.environ["HF_HUB_CACHE"],'
                    'credentials_preserved=os.environ.get("HF_TOKEN")=="synthetic-test-token",'
                    'python_overrides_absent="PYTHONPATH" not in os.environ and "PYTHONHOME" not in os.environ,'
                    'user_site=site.ENABLE_USER_SITE)))\n')
                hf.chmod(0o700)
                cli = binary / 'loom-control'
                cli.write_text(
                    '#!' + sys.executable + '\nimport json,sys\n'
                    f'sys.path.insert(0,{str(source)!r})\n'
                    'from binding.python.model_download import _run_hf\n'
                    'assert sys.argv[1:3]==["model","download"]\n'
                    '# Exercise the actual download subprocess primitive, without network or weights.\n'
                    'print(json.dumps(dict(completed=True,hf=json.loads(_run_hf(["hf","--help"])))))\n')
                cli.chmod(0o700)
                for inherited_path in (empty, global_bin):
                    with self.subTest(namespace=namespace, inherited_path=inherited_path), mock.patch.dict(
                            os.environ, dict(PATH=str(inherited_path), HOME=str(root / 'home'),
                            XDG_CONFIG_HOME=str(root / 'xdg'), HF_HUB_CACHE=str(root / 'cache'),
                            HF_TOKEN='synthetic-test-token', PYTHONPATH='must-be-cleared',
                            PYTHONHOME='must-be-cleared')):
                        result = gate.command(namespace, 'model', 'download', 'qwen35-08b-q5-k-xl')
                        self.assertEqual(result['hf']['hf'], str(hf))
                        self.assertEqual(result['hf']['home'], str(root / 'home'))
                        self.assertEqual(result['hf']['xdg'], str(root / 'xdg'))
                        self.assertEqual(result['hf']['cache'], str(root / 'cache'))
                        self.assertTrue(result['hf']['credentials_preserved'])
                        self.assertTrue(result['hf']['python_overrides_absent'])
                        self.assertFalse(result['hf']['user_site'])


class CiProfileTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        database = self.root / 'a/state/control.sqlite3'
        database.parent.mkdir(parents=True)
        with sqlite3.connect(database) as connection:
            connection.execute('CREATE TABLE instances (id TEXT)')
        self.gate = hem_model_smoke.Gate(argparse.Namespace(
            run=self.root, acquire08=False, model08=self.root / 'immutable-model'))

    def test_required_profile_imports_only_qualified08_and_retires_before_shutdown(self):
        calls = []
        def record(name):
            return lambda *args, **kwargs: calls.append((name, args))
        def command(*args, **kwargs):
            calls.append(('command', args))
            return {'items': []}
        with mock.patch.object(self.gate, 'start_server', side_effect=record('server')), \
             mock.patch.object(self.gate, 'import_model', side_effect=record('import')), \
             mock.patch.object(self.gate, 'start_instance', side_effect=record('activate')), \
             mock.patch.object(self.gate, 'query', side_effect=record('query')), \
             mock.patch.object(self.gate, 'shutdown', side_effect=record('shutdown')), \
             mock.patch.object(self.gate, 'command', side_effect=command):
            self.gate.run_ci08()
        self.assertEqual([name for name, args in calls],
                         ['server', 'import', 'activate', 'query', 'command', 'command', 'shutdown'])
        self.assertEqual(calls[1][1][-2:], ('Qwen3.5-0.8B-UD-Q5_K_XL.gguf', 'qwen-08b'))
        self.assertEqual(calls[4][1][1:3], ('instance', 'stop'))
        self.assertTrue(self.gate.results['success'])

    def test_ci_acquisition_selects_pinned08_recipe(self):
        self.gate.args.acquire08 = True
        def command(*args, **kwargs):
            if args[1:3] == ('model', 'download'):
                return {'completed': True, 'validation': {'valid': True}}
            return {'items': []}
        with mock.patch.object(self.gate, 'start_server'), \
             mock.patch.object(self.gate, 'import_model') as imported, \
             mock.patch.object(self.gate, 'start_instance'), \
             mock.patch.object(self.gate, 'query'), \
             mock.patch.object(self.gate, 'shutdown'), \
             mock.patch.object(self.gate, 'command', side_effect=command) as invoked:
            self.gate.run_ci08()
        imported.assert_not_called()
        self.assertEqual(invoked.call_args_list[0].args,
                         ('a', 'model', 'download', 'qwen35-08b-q5-k-xl',
                          '--id', 'small-a', '--pipeline', 'qwen-08b'))
        self.assertTrue(self.gate.results['success'])

    def test_diagnostics_are_bounded_redacted_and_exclude_weights(self):
        (self.root / 'query.sse').write_text('x' * 80000 + ' hf_secret123 token-value')
        (self.root / 'model.gguf').write_bytes(b'weights')
        with mock.patch.dict('os.environ', {'HF_TOKEN': 'token-value'}):
            self.gate.artifacts()
        diagnostic = self.root / 'diagnostics'
        text = (diagnostic / 'query.sse').read_text()
        self.assertLessEqual((diagnostic / 'query.sse').stat().st_size, 65536)
        self.assertNotIn('hf_secret123', text)
        self.assertNotIn('token-value', text)
        self.assertFalse((diagnostic / 'model.gguf').exists())
        with sqlite3.connect(diagnostic / 'a-control.sqlite3') as connection:
            self.assertEqual(connection.execute('PRAGMA integrity_check').fetchone()[0], 'ok')


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
        cli = self.case.root / 'loom-control'
        # Source-only fixture: the detached bootstrap must resolve binding.python
        # from this checkout too, without PYTHONPATH or an installed wheel.
        cli.write_text(self.fixture.cli.read_text().replace(
            'import functools,sys',
            'import functools,sys,os\nos.chdir(' + repr(str(Path(__file__).resolve().parents[1])) + ')'))
        cli.chmod(0o700)
        self.gate = hem_model_smoke.Gate(argparse.Namespace(run=self.case.root))
        python = self.case.root / 'python'
        python.write_text(
            '#!' + sys.executable + '\nimport sys\n'
            f'sys.path.insert(0,{str(Path(__file__).resolve().parents[1])!r})\n'
            'assert sys.argv[1] == "-c"\n'
            'code=sys.argv[2].replace("hrx_solutions", "binding.python")\n'
            'sys.argv=["-c",*sys.argv[3:]]\n'
            'exec(code)\n')
        python.chmod(0o700)
        self.gate.states['a'] = python
        self.gate.namespace_roots['a'] = '.'
        self.gate.setup = lambda: None

    def assert_failed_launch_cleaned(self, outer_timeout=False, cleanup_failure=False):
        other = self.case.start('unrelated')
        original = self.gate.command
        captured = {}
        if outer_timeout:
            cli = self.case.root / 'loom-control'
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
