"""Shared shutdown protocol using real HTTP and owned synthetic ELF processes."""
from contextlib import contextmanager, redirect_stderr
import io
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from binding.python import hem_process, hem_state, hem_shutdown
from binding.python.hem_common import Deadline, DeadlineExpired, HemError, UncertainOutcome
from binding.python.hem_http import NativeClient
from tests.hem_http_test import fixture, inventory, model, reply
from tests.hem_process_test import port, wait_exit
from tests.hem_instances_test import NATIVE


class Processes:
    def __init__(self):
        self.signals = []
        self.exited = False
        self.wait_pending = False
        self.cleanup_failure = False
        self.events = []

    @contextmanager
    def pin(self, record):
        yield self

    def verify(self, deadline):
        deadline.remaining()

    def client(self, record, pinned, deadline):
        return NativeClient(record['endpoint'], deadline, emit=lambda _: None)

    def send(self, number, deadline):
        deadline.remaining()
        self.signals.append(number)
        if not self.wait_pending:
            self.exited = True

    def wait(self, deadline):
        while not self.exited:
            time.sleep(min(.005, deadline.remaining()))

    def cleanup(self, record, deadline):
        deadline.remaining()
        assert self.exited
        if self.cleanup_failure:
            raise HemError('fixture cleanup failure')
        self.events.append(('cleanup', record['launch_token']))

    def reap(self):
        pass


class ShutdownTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.registry = hem_state.Registry(Path(temporary.name))
        self.processes = Processes()
        self.manager = hem_shutdown.ShutdownManager(self.registry, processes=self.processes)
        self.items = [model()]
        self.pending = False
        self.lost = False
        self.rejected = False
        self.posts = []
        self.polls = 0
        self.patch_exit = patch.object(hem_shutdown, '_confirmed_exit', side_effect=lambda _: self.processes.exited)
        self.patch_exit.start()
        self.addCleanup(self.patch_exit.stop)

    def server(self, endpoint):
        self.registry.create_server('local', saved_settings={})
        row = self.registry.reserve_launch('local', launch_token='launch', current_settings={}, endpoint=endpoint)
        return self.registry.update_server('local', {'phase':'running','pending':None,
                    'process':{'pid':123}, 'build':{'directory':'fixture'}},
                    expected_token='launch', expected_revision=row['revision'])

    def respond(self, handler, requests):
        method, path, body, _ = requests[-1]
        if method == 'GET':
            self.polls += 1
            reply(handler, inventory(*self.items))
            return
        payload = json.loads(body)
        self.posts.append(payload)
        self.assertEqual(path, '/v1/models/unregister')
        self.assertEqual(payload['mode'], 'drain')
        if self.lost:
            handler.connection.shutdown(socket.SHUT_RDWR)
            handler.connection.close()
            return
        if self.rejected:
            reply(handler, {'error':'release failed'}, 409)
            return
        if not self.pending:
            self.items = [item for item in self.items if item['id'] != payload['model']]
        reply(handler, {'accepted':True}, 202)

    def shutdown(self, **kwargs):
        return self.manager.shutdown('local', Deadline.after(kwargs.pop('timeout', 1)), **kwargs)

    def row(self):
        return self.registry.get_server('local')

    def test_drains_all_managed_and_external_then_signals(self):
        external = model(18)
        external['id'] = 'curl-model'
        self.items.append(external)
        with fixture(self.respond) as (endpoint, _):
            self.server(endpoint)
            self.registry.create_instance('local', 'launch', 'chat', incarnation=17)
            answer = self.shutdown()
        self.assertTrue(answer['completed'])
        self.assertEqual({p['model'] for p in self.posts}, {'chat', 'curl-model'})
        self.assertEqual(self.processes.signals, [signal.SIGTERM])
        self.assertEqual(self.registry.list_instances(), [])
        self.assertIsNone(self.row()['launch_token'])
        self.assertEqual(self.processes.events, [('cleanup','launch')])

    def test_timeout_blocks_start_keeps_alive_and_retry_observes(self):
        self.pending = True
        with fixture(self.respond) as (endpoint, _):
            self.server(endpoint)
            with self.assertRaises(DeadlineExpired) as raised:
                self.shutdown(timeout=.07)
            self.assertIs(raised.exception.accepted, True)
            self.assertFalse(self.processes.exited)
            self.assertEqual(self.row()['phase'], 'stopping')
            with self.assertRaisesRegex(HemError, 'not accepting'):
                self.registry.create_instance('local','launch','new')
            self.assertEqual(len(self.posts), 1)
            self.items = []
            self.assertTrue(self.shutdown()['completed'])
            self.assertEqual(len(self.posts), 1)

    def test_lost_response_is_uncertain_and_not_replayed(self):
        self.lost = True
        with fixture(self.respond) as (endpoint, _):
            self.server(endpoint)
            with self.assertRaises(UncertainOutcome):
                self.shutdown()
            with self.assertRaisesRegex(UncertainOutcome, 'instead of replaying'):
                self.shutdown()
            self.assertEqual(len(self.posts), 1)
            self.assertEqual(self.processes.signals, [])
            self.items = []
            self.assertTrue(self.shutdown()['completed'])

    def test_failed_retry_observation_preserves_unknown_unregister_acceptance(self):
        for mixed in (False, True):
            with self.subTest(mixed=mixed):
                self.items = [model()]
                if mixed:
                    second = model(18)
                    second['id'] = 'second'
                    self.items.append(second)
                unavailable = False
                self.pending = True
                self.posts = []
                def respond(handler, requests):
                    method, _, body, _ = requests[-1]
                    if method == 'GET' and unavailable:
                        reply(handler, {'error':'status unavailable'}, 503)
                        return
                    self.lost = method == 'POST' and (not mixed or json.loads(body)['model'] == 'second')
                    self.respond(handler, requests)
                with fixture(respond) as (endpoint, _):
                    if mixed:
                        # The previous subcase retains its unresolved launch;
                        # this new isolated record exercises mixed acknowledgments.
                        self.registry = hem_state.Registry(self.registry.root / 'mixed')
                        self.manager = hem_shutdown.ShutdownManager(self.registry, processes=self.processes)
                    self.server(endpoint)
                    with self.assertRaises(UncertainOutcome) as first:
                        self.shutdown()
                    self.assertIsNone(first.exception.accepted)
                    before = list(self.posts)
                    unavailable = True
                    with self.assertRaises(UncertainOutcome) as retry:
                        self.shutdown()
                    self.assertEqual(retry.exception.exit_code, 4)
                    self.assertIsNone(retry.exception.accepted)
                    self.assertEqual(self.posts, before)
                    self.assertEqual(self.processes.signals, [])

    def test_retry_observation_reports_only_unanimous_known_acceptance(self):
        cases = ((['accepted'], True), (['rejected'], False),
                 (['planned'], None), (['accepted', 'rejected'], None))
        for stages, expected in cases:
            with self.subTest(stages=stages):
                self.registry = hem_state.Registry(self.registry.root / 'next')
                self.manager = hem_shutdown.ShutdownManager(self.registry, processes=self.processes)
                with fixture(lambda handler, _: reply(handler, {'error':'unavailable'}, 503)) as (endpoint, _):
                    record = self.server(endpoint)
                    self.registry.update_server('local', {'phase':'stopping', 'pending':{
                        'action':'shutdown', 'mode':'drain', 'stage':'retirement',
                        'targets':{str(index):{'incarnation':index+1,'stage':stage}
                                   for index, stage in enumerate(stages)}}},
                        expected_token='launch', expected_revision=record['revision'])
                    with self.assertRaises(UncertainOutcome) as raised:
                        self.shutdown()
                    self.assertIs(raised.exception.accepted, expected)
                    self.assertEqual(self.posts, [])
                    self.assertEqual(self.processes.signals, [])

    def test_deadline_observation_preserves_target_acceptance(self):
        cases = ((['accepted'], True, DeadlineExpired), (['rejected'], False, DeadlineExpired),
                 (['planned'], None, DeadlineExpired),
                 (['accepted', 'submitting'], None, UncertainOutcome))
        for stages, expected, error_type in cases:
            with self.subTest(stages=stages):
                self.registry = hem_state.Registry(self.registry.root / 'next')
                self.manager = hem_shutdown.ShutdownManager(self.registry, processes=self.processes)
                record = self.server('http://127.0.0.1:1')
                self.registry.update_server('local', {'phase':'stopping', 'pending':{
                    'action':'shutdown', 'mode':'drain', 'stage':'retirement',
                    'targets':{str(index):{'incarnation':index+1,'stage':stage}
                               for index, stage in enumerate(stages)}}},
                    expected_token='launch', expected_revision=record['revision'])
                with patch.object(NativeClient, 'status', side_effect=DeadlineExpired('expired')):
                    with self.assertRaises(error_type) as raised:
                        self.shutdown()
                self.assertIs(raised.exception.accepted, expected)
                self.assertEqual(self.posts, [])
                self.assertEqual(self.processes.signals, [])

    def test_abort_escalates_pending_drain_without_replaying_unregister(self):
        self.pending = True
        with fixture(self.respond) as (endpoint, _):
            self.server(endpoint)
            with self.assertRaises(DeadlineExpired):
                self.shutdown(timeout=.05)
            self.assertTrue(self.shutdown(abort=True)['completed'])
            self.assertEqual(len(self.posts), 1)
            self.assertEqual(self.processes.signals, [signal.SIGTERM])

    def test_abort_timeout_keeps_marker_never_kills_and_retry_waits(self):
        self.processes.wait_pending = True
        with fixture(self.respond) as (endpoint, _):
            self.server(endpoint)
            with self.assertRaises(DeadlineExpired) as raised:
                self.shutdown(abort=True, timeout=.05)
            self.assertIs(raised.exception.accepted, True)
            self.assertEqual(self.posts, [])
            self.assertFalse(self.processes.exited)
            self.processes.exited = True
            self.assertTrue(self.shutdown()['completed'])
            self.assertEqual(self.processes.signals, [signal.SIGTERM])

    def test_unknown_ownership_never_mutates(self):
        with fixture(self.respond) as (endpoint, _):
            self.server(endpoint)
            with patch.object(self.processes, 'verify', side_effect=UncertainOutcome('unknown ownership')):
                with self.assertRaises(UncertainOutcome):
                    self.shutdown()
            self.assertIsNone(self.row()['pending'])
            self.assertEqual(self.posts, [])
            self.assertEqual(self.processes.signals, [])

    def test_failed_release_does_not_signal_and_known_rejection_is_retryable(self):
        self.rejected = True
        with fixture(self.respond) as (endpoint, _):
            self.server(endpoint)
            with self.assertRaises(HemError) as raised:
                self.shutdown()
            self.assertIs(raised.exception.accepted, False)
            self.assertEqual(self.processes.signals, [])
            self.rejected = False
            self.assertTrue(self.shutdown()['completed'])

    def test_existing_acknowledged_instance_stop_is_observed_only(self):
        with fixture(self.respond) as (endpoint, _):
            self.server(endpoint)
            self.registry.create_instance('local','launch','chat',incarnation=17,
                    pending={'action':'stop','accepted':True,'mode':'drain'})
            self.pending = True
            with self.assertRaises(DeadlineExpired):
                self.shutdown(timeout=.05)
            self.assertEqual(self.posts, [])
            self.items = []
            self.assertTrue(self.shutdown()['completed'])

    def test_caller_crash_during_unregister_keeps_no_replay_marker(self):
        with fixture(self.respond) as (endpoint, _):
            self.server(endpoint)
            original = NativeClient.post
            def crash(client, path, body):
                original(client, path, body)
                raise KeyboardInterrupt()
            self.pending = True
            with patch.object(NativeClient, 'post', new=crash):
                with self.assertRaises(KeyboardInterrupt):
                    self.shutdown()
            manager = hem_shutdown.ShutdownManager(hem_state.Registry(self.registry.root), processes=self.processes)
            with self.assertRaises(UncertainOutcome):
                manager.shutdown('local', Deadline.after(1))
            self.assertEqual(len(self.posts), 1)
            self.items = []
            self.assertTrue(manager.shutdown('local', Deadline.after(1))['completed'])

    def test_cleanup_failure_preserves_launch_snapshot_for_retry(self):
        self.items = []
        self.processes.cleanup_failure = True
        with fixture(self.respond) as (endpoint, _):
            self.server(endpoint)
            with self.assertRaisesRegex(HemError, 'cleanup failure') as raised:
                self.shutdown()
            self.assertIs(raised.exception.accepted, True)
            self.assertEqual(self.row()['launch_token'], 'launch')
            self.processes.cleanup_failure = False
            self.assertTrue(self.shutdown()['completed'])

    def test_admitted_request_completes_before_graceful_signal(self):
        admitted, complete = threading.Event(), threading.Event()
        failures = []
        def respond(handler, requests):
            if requests[-1][1] == '/query':
                admitted.set()
                time.sleep(.08)
                self.items = []
                complete.set()
                reply(handler, {'answer':'bounded known query'})
            else:
                if requests[-1][1] == '/v1/models/unregister':
                    self.assertFalse(complete.is_set())
                self.respond(handler, requests)
        def query(endpoint):
            try:
                answer = NativeClient(endpoint, Deadline.after(1), emit=lambda _: None).get('/query').json()
                self.assertEqual(answer['answer'], 'bounded known query')
            except BaseException as error:
                failures.append(error)
        self.pending = True
        with fixture(respond) as (endpoint, _):
            self.server(endpoint)
            worker = threading.Thread(target=query, args=(endpoint,))
            worker.start()
            self.assertTrue(admitted.wait(1))
            try:
                self.assertTrue(self.shutdown()['completed'])
                self.assertTrue(complete.is_set())
            finally:
                worker.join(timeout=2)
            self.assertFalse(worker.is_alive())
            self.assertEqual(failures, [])

    def test_unknown_existing_stop_is_not_replayed(self):
        with fixture(self.respond) as (endpoint, _):
            self.server(endpoint)
            self.registry.create_instance('local','launch','chat',incarnation=17,
                    pending={'action':'stop','accepted':None,'mode':'drain'})
            with self.assertRaises(UncertainOutcome):
                self.shutdown()
            self.assertEqual(self.posts, [])
            self.assertEqual(self.processes.signals, [])

    def test_crash_before_signal_ack_does_not_replay_signal(self):
        self.items = []
        with fixture(self.respond) as (endpoint, _):
            self.server(endpoint)
            with patch.object(self.processes, 'send', side_effect=KeyboardInterrupt()):
                with self.assertRaises(KeyboardInterrupt):
                    self.shutdown()
            with self.assertRaisesRegex(UncertainOutcome, 'termination submission'):
                self.shutdown()
            self.assertEqual(self.processes.signals, [])


class ExitRaceTest(unittest.TestCase):
    def test_same_pinned_fd_confirms_exit_after_procfs_disappears(self):
        process = {'pid':123, 'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip()}
        with patch.object(os, 'pidfd_open', return_value=1234), \
             patch.object(os, 'close') as close, \
             patch.object(hem_process, '_exited', side_effect=[False, True]) as exited, \
             patch.object(hem_process, 'identity', side_effect=UncertainOutcome('procfs vanished')):
            self.assertTrue(hem_process._confirmed_exit(process))
            self.assertEqual(exited.call_args_list[0].args, (1234,))
            self.assertEqual(exited.call_args_list[1].args, (1234,))
            close.assert_called_once_with(1234)

    def test_unreadable_live_identity_remains_uncertain(self):
        process = {'pid':123, 'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip()}
        with patch.object(os, 'pidfd_open', return_value=1234), \
             patch.object(os, 'close'), \
             patch.object(hem_process, '_exited', side_effect=[False, False]), \
             patch.object(hem_process, 'identity', side_effect=UncertainOutcome('permission denied')):
            with self.assertRaises(UncertainOutcome):
                hem_process._confirmed_exit(process)


class OwnedShutdownTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compiled = tempfile.TemporaryDirectory()
        cls.binary = Path(cls.compiled.name) / 'loom-serve'
        source = Path(cls.compiled.name) / 'native.c'
        source.write_text(NATIVE)
        subprocess.run(['cc', '-O0', '-o', str(cls.binary), str(source)], check=True,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    @classmethod
    def tearDownClass(cls):
        cls.compiled.cleanup()

    def test_two_owned_elf_servers_shutdown_isolated(self):
        with tempfile.TemporaryDirectory() as temporary, redirect_stderr(io.StringIO()):
            root = Path(temporary)
            package = root / 'package'
            (package / '_native/bin').mkdir(parents=True)
            (package / '_native/lib').mkdir()
            (package / 'loom-pipeline').mkdir()
            shutil.copy2(self.binary, package / '_native/bin/loom-serve')
            (package / '_native/lib/libhsa-runtime64.so.1').write_bytes(b'CPU fixture only')
            registry = hem_state.Registry(root / 'state')
            processes = hem_process.ProcessManager(registry, package_root=package, emit=lambda _: None)
            manager = hem_shutdown.ShutdownManager(registry, processes=processes)
            servers = []
            try:
                for name in ('first','second'):
                    settings = dict(device='cpu-fixture',listen=f'127.0.0.1:{port()}')
                    registry.create_server(name,saved_settings=settings)
                    servers.append(processes.start(name,settings,Deadline.after(5)))
                document = registry.write_document('register.json','{}')
                for server in servers:
                    with processes.pin(server) as pinned:
                        processes.client(server,pinned,Deadline.after(2)).post('/v1/models/register',document)
                self.assertTrue(manager.shutdown('first',Deadline.after(3))['completed'])
                self.assertTrue(hem_process._confirmed_exit(servers[0]['process']))
                with processes.pin(servers[1]) as pinned:
                    self.assertEqual(len(processes.client(servers[1],pinned,Deadline.after(2)).status()['data']),1)
                self.assertTrue(manager.shutdown('second',Deadline.after(3),abort=True)['completed'])
                for server in servers:
                    self.assertTrue(hem_process._confirmed_exit(server['process']))
                    self.assertFalse(Path(server['build']['directory']).exists())
                    wait_exit(server['process']['logger'])
                self.assertTrue(all(row['launch_token'] is None for row in registry.list_servers()))
            finally:
                for server in servers:
                    if not hem_process._confirmed_exit(server['process']):
                        manager.shutdown(server['id'],Deadline.after(3),abort=True)
                for child in processes.children.values():
                    child.wait(timeout=3)


if __name__ == '__main__':
    unittest.main()
