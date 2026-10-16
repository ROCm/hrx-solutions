"""Crossdomain fault gates with real CLI children, owned ELF and minimal SQLite.

These are orchestration checks. Sparse synthetic GGUFs and a protocol fixture do
not qualify native Loom, model inference or an installed release wheel.
"""
from contextlib import closing
import json
from pathlib import Path
import shutil
import signal
import sqlite3
import subprocess
import tempfile
import time
import unittest

from binding.python import hem_process
from binding.python.hem_common import Deadline
from tests.hem_join_fixture import JoinFixture, NATIVE_JOIN


class JoinTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compiled = tempfile.TemporaryDirectory()
        source = Path(cls.compiled.name) / 'fixture.c'
        source.write_text(NATIVE_JOIN)
        cls.binary = source.with_name('loom-serve')
        subprocess.run(['cc', '-O0', '-Wall', '-o', str(cls.binary), str(source)], check=True,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    @classmethod
    def tearDownClass(cls):
        cls.compiled.cleanup()

    def setUp(self):
        temporary = Path(tempfile.mkdtemp())
        try:
            self.fixture = JoinFixture(temporary, self.binary)
        except BaseException:
            shutil.rmtree(temporary)
            raise
        def cleanup():
            # Preserve state for inspection if ownership/exit cannot be proved.
            # Separate unittest cleanups (or a TemporaryDirectory finalizer)
            # must never delete the namespace after a failed close().
            self.fixture.close()
            shutil.rmtree(temporary)
        self.addCleanup(cleanup)

    def test_launch_overlap_serializes_and_reuses_same_incarnation(self):
        f = self.fixture
        f.barrier('alpha', 'listen')
        first = f.spawn(*f.server_command('alpha'))
        f.wait_barrier('alpha', 'listen')
        f.remember('alpha')
        second = f.spawn('server', 'start', 'alpha')
        time.sleep(.08)
        self.assertIsNone(second.poll())
        f.release('alpha', 'listen')
        f.finish(first)
        f.finish(second)
        row = f.registry.get_server('alpha')
        self.assertEqual(row['process']['pid'], f.owned[0]['process']['pid'])
        self.assertEqual(len(f.registry.list_servers()), 1)
        self.assertIsNone(row['pending'])
        f.cli('server', 'shutdown', 'alpha')
        self.assertTrue(hem_process._confirmed_exit(row['process']))

    def test_caller_dies_after_exec_reconcile_observes_survivor_without_relaunch(self):
        f = self.fixture
        f.barrier('alpha', 'listen')
        caller = f.spawn(*f.server_command('alpha'))
        f.wait_barrier('alpha', 'listen')
        row = f.remember('alpha')
        self.assertEqual(hem_process.identity(row['process']['pid'])['executable'], row['build']['executable'])
        caller.kill()
        caller.communicate(timeout=3)
        f.release('alpha', 'listen')
        answer = f.cli('server', 'reconcile', 'alpha')
        self.assertEqual(answer['state'], 'running')
        current = f.registry.get_server('alpha')
        self.assertEqual(current['launch_token'], row['launch_token'])
        self.assertEqual(current['process']['pid'], row['process']['pid'])
        self.assertEqual(f.mutations('alpha'), [])
        f.cli('server', 'shutdown', 'alpha')

    def test_start_shutdown_overlap_and_other_server_progress(self):
        f = self.fixture
        f.acquire()
        f.start_server('alpha')
        f.start_server('beta')
        f.barrier('alpha', 'register')
        start = f.spawn('instance', 'start', 'tiny', '--name', 'chat', '--server', 'alpha', '--pipeline', 'qwen-08b')
        f.wait_barrier('alpha', 'register')
        stop = f.spawn('server', 'shutdown', 'alpha')
        time.sleep(.08)
        self.assertIsNone(stop.poll())
        # Separate model qualification does briefly share a model lock; server
        # observation must remain independent throughout the held alpha request.
        self.assertEqual(f.cli('server', 'status', 'beta')['state'], 'running')
        self.assertEqual(f.mutations('alpha'), ['POST /v1/models/register HTTP/1.1'])
        f.release('alpha', 'register')
        f.finish(start)
        f.finish(stop)
        self.assertEqual(f.registry.list_instances(), [])
        self.assertEqual(f.cli('server', 'status', 'beta')['state'], 'running')
        f.cli('server', 'shutdown', 'beta')

    def test_request_caller_crash_keeps_refs_and_reconcile_does_not_replay(self):
        f = self.fixture
        f.acquire()
        f.start_server('alpha')
        f.barrier('alpha', 'register')
        caller = f.spawn('instance', 'start', 'tiny', '--name', 'chat', '--server', 'alpha', '--pipeline', 'qwen-08b')
        f.wait_barrier('alpha', 'register')
        caller.kill()
        caller.communicate(timeout=3)
        f.release('alpha', 'register')
        row = f.registry.references('tiny')[0]
        self.assertIsNone(row['pending']['accepted'])
        self.assertEqual(row['pending']['stage'], 'register')
        before = f.mutations('alpha')
        for _ in range(2):
            f.cli('server', 'reconcile', 'alpha')
            shown = f.cli('instance', 'show', 'chat', '--server', 'alpha')
            self.assertIsNotNone(shown.get('tracked_submission') or shown.get('pending'))
        self.assertEqual(f.registry.references('tiny'), [row])
        f.cli('instance', 'start', 'tiny', '--name', 'chat', '--server', 'alpha', '--pipeline', 'qwen-08b', expected=1)
        self.assertEqual(f.mutations('alpha'), before)
        f.cli('instance', 'stop', 'chat', '--server', 'alpha')
        self.assertEqual(f.registry.references('tiny'), [])
        f.cli('server', 'shutdown', 'alpha')

    def test_preexec_crash_releases_lock_and_clears_only_confirmed_exit(self):
        f = self.fixture
        # Use the existing bootstrap checkpoint in this private source copy.
        # The crash occurs after both identities are durable but before ELF exec.
        bootstrap = f.package / 'hem_bootstrap.py'
        bootstrap.write_text(bootstrap.read_text().replace(
            '"""No-op seam for subprocess crash-window tests; no environment controls."""',
            '''"""Private fixture crash barrier, not a product setting."""
    import time
    if stage == 'before_exec':
        root = Path(os.environ['LOOM_SERVE_DIR']).parent
        (root / 'before-exec').touch()
        while (root / 'hold-exec').exists():
            time.sleep(.01)
        os._exit(73)'''))
        (f.root / 'hold-exec').touch()
        caller = f.spawn(*f.server_command('alpha'))
        f.until(lambda: (f.root / 'before-exec').exists())
        original = f.remember('alpha')
        self.assertEqual(original['pending'], {'action': 'start', 'stage': 'bootstrap'})
        self.assertNotEqual(hem_process.identity(original['process']['pid'])['executable'], original['build']['executable'])
        (f.root / 'hold-exec').unlink()
        f.finish(caller, expected=1)
        self.assertTrue(hem_process._confirmed_exit(original['process']))
        answer = f.cli('server', 'reconcile', 'alpha')
        self.assertEqual(answer['state'], 'stopped')
        self.assertIsNone(f.registry.get_server('alpha')['launch_token'])
        self.assertFalse((f.control('alpha') / 'seen-listen').exists())
        self.assertEqual(f.mutations('alpha'), [])
        # Restore the private bootstrap; the next start must be a fresh launch.
        checkout = Path(__file__).resolve().parents[1]
        bootstrap.write_bytes((checkout / 'binding/python/hem_bootstrap.py').read_bytes())
        f.cli('server', 'start', 'alpha')
        new = f.remember('alpha')
        self.assertNotEqual(new['launch_token'], original['launch_token'])
        f.cli('server', 'shutdown', 'alpha')

    def test_shutdown_caller_crash_does_not_replay_submitted_unregister(self):
        f = self.fixture
        f.acquire()
        f.start_server('alpha')
        f.start_instance('alpha')
        f.barrier('alpha', 'unregister')
        caller = f.spawn('server', 'shutdown', 'alpha')
        f.wait_barrier('alpha', 'unregister')
        caller.kill()
        caller.communicate(timeout=3)
        before = f.mutations('alpha')
        self.assertEqual(sum('/unregister ' in event for event in before), 1)
        self.assertEqual(len(f.registry.references('tiny')), 1)
        f.release('alpha', 'unregister')
        f.cli('server', 'reconcile', 'alpha')
        self.assertEqual(f.mutations('alpha'), before)
        # Native absence establishes retirement; an explicit shutdown can now
        # finish without a second unregister or an operation-history record.
        f.cli('server', 'shutdown', 'alpha')
        self.assertEqual(f.mutations('alpha'), before)
        self.assertEqual(f.registry.references('tiny'), [])

    def test_assertion_failure_at_preexec_barrier_cleans_helper_and_logger(self):
        f = self.fixture
        bootstrap = f.package / 'hem_bootstrap.py'
        bootstrap.write_text(bootstrap.read_text().replace(
            '"""No-op seam for subprocess crash-window tests; no environment controls."""',
            '''"""Private assertion-failure fixture barrier."""
    import time
    if stage == 'before_exec':
        root = Path(os.environ['LOOM_SERVE_DIR']).parent
        (root / 'before-exec').touch()
        while (root / 'hold-exec').exists():
            time.sleep(.01)
        # Remain Python after release, requiring verified bootstrap signaling.
        while True:
            time.sleep(.01)'''))
        (f.root / 'hold-exec').touch()
        caller = f.spawn(*f.server_command('alpha'))
        f.until(lambda: (f.root / 'before-exec').exists())
        record = f.remember('alpha')
        self.assertEqual(record['pending']['stage'], 'bootstrap')
        self.assertIn('logger', record['process'])

        class AbandonedCase(unittest.TestCase):
            def runTest(inner):
                inner.addCleanup(f.close)
                inner.fail('deterministic assertion failure while helper is paused')

        result = unittest.TestResult()
        try:
            AbandonedCase().run(result)
            self.assertEqual(len(result.failures), 1)
            self.assertEqual(result.errors, [])
            self.assertFalse((f.root / 'hold-exec').exists())
            self.assertTrue(hem_process._confirmed_exit(record['process']))
            self.assertTrue(hem_process._confirmed_exit(record['process']['logger']))
            self.assertIsNotNone(caller.returncode)
            with f.registry.lock('server', 'alpha', deadline=Deadline.after(.5)):
                pass
            # State remains until all these process-exit checks have passed.
            self.assertTrue(f.state.exists())
        finally:
            # Retain safe cleanup even when running this regression against the
            # unfixed helper in the intentional red phase.
            (f.root / 'hold-exec').unlink(missing_ok=True)
            if not hem_process._confirmed_exit(record['process']):
                with hem_process.PinnedProcess(record['process'], emit=lambda _: None) as pinned:
                    pinned.send(signal.SIGTERM, Deadline.after(3))
                    pinned.wait(Deadline.after(3))
            f.until(lambda: hem_process._confirmed_exit(record['process']))
            f.until(lambda: hem_process._confirmed_exit(record['process']['logger']))

    def test_crossserver_dependencies_remain_after_one_shutdown_and_package_replacement(self):
        f = self.fixture
        f.acquire()
        for name in ('alpha', 'beta'):
            f.start_server(name)
            f.start_instance(name)
        self.assertEqual(len(f.registry.references('tiny')), 2)
        replacement = f.root / 'replacement-tokenizer.json'
        replacement.write_text('{"model":{"replacement":true}}')
        refused = f.cli('model', 'import', 'tiny', '--gguf', str(f.root / 'weights.gguf'),
                        '--tokenizer', str(replacement), '--alias', expected=1)
        self.assertIn('different mappings', refused['error'])
        self.assertEqual(len(f.registry.references('tiny')), 2)
        beta = f.registry.references('tiny')[1]
        document = Path(beta['document'])
        sources = Path(json.loads(document.read_text())['base_directory'])
        f.cli('server', 'shutdown', 'alpha')
        self.assertEqual(f.registry.references('tiny'), [beta])
        self.assertTrue(document.exists())
        self.assertTrue((sources / 'prepare.loom').exists())
        # A future package source update must not affect an already submitted
        # server's executable or retained model source closure.
        (f.package / 'loom-pipeline/llm/qwen/qwen_08b/prepare.loom').write_text('package changed')
        self.assertEqual(f.cli('instance', 'status', 'chat', '--server', 'beta')['native']['loom']['parameters']['active'], True)
        f.cli('instance', 'stop', 'chat', '--server', 'beta')
        self.assertEqual(f.registry.references('tiny'), [])
        f.cli('server', 'shutdown', 'beta')

    def test_replacement_incarnation_is_observed_without_old_association_or_unregister(self):
        f = self.fixture
        f.acquire()
        f.start_server('alpha')
        f.start_instance('alpha')
        row = f.registry.references('tiny')[0]
        server = f.registry.get_server('alpha')
        manager = hem_process.ProcessManager(f.registry, package_root=f.package, emit=lambda _: None)
        document = f.registry.write_document('fixture-replace.json', '{}')
        with manager.pin(server) as pinned:
            manager.client(server, pinned, Deadline.after(2)).post('/fixture/replace', document)
        # Raw curl writers must coordinate with Hem. An observed new incarnation
        # is never promoted into the original submission's local association.
        shown = f.cli('instance', 'show', 'chat', '--server', 'alpha')
        self.assertNotEqual(shown['incarnation'], row['incarnation'])
        self.assertIsNone(shown['model'])
        self.assertEqual(shown['tracked_submission']['model'], 'tiny')
        before = f.mutations('alpha')
        f.cli('instance', 'stop', 'chat', '--server', 'alpha')
        self.assertEqual(f.mutations('alpha'), before)
        self.assertEqual(f.registry.references('tiny'), [])
        self.assertIsNotNone(f.cli('instance', 'show', 'chat', '--server', 'alpha')['native'])
        f.cli('server', 'shutdown', 'alpha')

    def test_wrong_executable_process_metadata_refuses_signal(self):
        f = self.fixture
        f.start_server('alpha')
        row = f.registry.get_server('alpha')
        corrupted = dict(row['build'], executable={'device': 0, 'inode': 0})
        f.registry.update_server('alpha', {'build': corrupted}, expected_token=row['launch_token'], expected_revision=row['revision'])
        try:
            answer = f.cli('server', 'shutdown', 'alpha', '--abort', expected=4)
            self.assertFalse(answer['completed'])
            self.assertFalse(hem_process._confirmed_exit(row['process']))
            self.assertEqual(f.mutations('alpha'), [])
        finally:
            current = f.registry.get_server('alpha')
            f.registry.update_server('alpha', {'build': row['build']}, expected_token=row['launch_token'], expected_revision=current['revision'])
        f.cli('server', 'shutdown', 'alpha')

    def test_repeated_complete_lifecycles_keep_only_bounded_current_rows(self):
        f = self.fixture
        f.acquire()
        tokens = set()
        for index in range(3):
            if index == 0:
                f.start_server('alpha')
            else:
                f.cli('server', 'start', 'alpha')
                f.remember('alpha')
            row = f.registry.get_server('alpha')
            tokens.add(row['launch_token'])
            f.start_instance('alpha')
            self.assertEqual(len(f.registry.references('tiny')), 1)
            f.cli('server', 'shutdown', 'alpha')
            self.assertFalse(Path(row['build']['directory']).exists())
            self.assertTrue(hem_process._confirmed_exit(row['process']))
            with closing(sqlite3.connect(f.registry.database)) as connection:
                tables = {name for name, in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
                self.assertEqual(tables, {'models', 'servers', 'instances', 'pi_target'})
                self.assertEqual([connection.execute(f'SELECT COUNT(*) FROM {table}').fetchone()[0]
                                  for table in ('models', 'servers', 'instances')], [1, 1, 0])
                self.assertEqual(connection.execute('PRAGMA integrity_check').fetchone()[0], 'ok')
        self.assertEqual(len(tokens), 3)


if __name__ == '__main__':
    unittest.main()
