"""Real Linux subprocess/socket tests using a tiny C fake native, never GPU serving."""
from contextlib import contextmanager
import json
import errno
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from binding.python import hem_bootstrap, hem_process
from binding.python._launcher import native_environment
from binding.python.hem_common import Deadline, DeadlineExpired, HemError, InvalidInput, UncertainOutcome
from binding.python.hem_http import NativeClient
from binding.python.hem_state import Registry

# A real independently exec'd ELF is essential: a shebang fixture would retain
# Python's /proc/exe and never prove the bootstrap-to-native identity transition.
NATIVE = r'''
#include <arpa/inet.h>
#include <netinet/in.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>
int main(int argc, char **argv) {
 signal(SIGPIPE,SIG_IGN);
 int port=0, fd, client, yes=1; char request[2048];
 int unhealthy=0;
 for(int i=1;i<argc;i++) if(!strncmp(argv[i],"--listen=",9)) port=atoi(strrchr(argv[i], ':')+1);
 for(int i=1;i<argc;i++) { if(!strcmp(argv[i],"--device=slow")) usleep(500000); if(!strcmp(argv[i],"--device=unhealthy")) unhealthy=1; }
 if(!port) return 5;
 fd=socket(AF_INET,SOCK_STREAM,0); setsockopt(fd,SOL_SOCKET,SO_REUSEADDR,&yes,sizeof(yes));
 struct sockaddr_in a={.sin_family=AF_INET,.sin_port=htons(port),.sin_addr.s_addr=htonl(INADDR_LOOPBACK)};
 if(bind(fd,(void*)&a,sizeof(a)) || listen(fd,16)) return 6;
 fprintf(stderr,"fake-native ready %d\n",port); fflush(stderr);
 while((client=accept(fd,NULL,NULL))>=0) {
   int count=read(client,request,sizeof(request)-1); if(count<0) count=0; request[count]=0;
   const char *body="{\"object\":\"list\",\"data\":[],\"server\":{\"worker_count\":1}}";
   if(strstr(request,"/noise")) { for(int i=0;i<100000;i++) fputs("bounded diagnostic output 1234567890\n",stderr); fflush(stderr); }
   if(unhealthy || strstr(request,"/fail")) body="{}";
   char reply[4096]; int n=snprintf(reply,sizeof(reply),"HTTP/1.1 200 OK\r\nContent-Length: %zu\r\nConnection: close\r\n\r\n%s",strlen(body),body);
   write(client,reply,n); close(client);
 }
 return 0;
}
'''


def port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


def wait_exit(process, seconds=3):
    until = time.monotonic() + seconds
    while time.monotonic() < until:
        if hem_process._confirmed_exit(process):
            return
        time.sleep(.01)
    raise AssertionError(f'owned process did not exit: {process}')


class ProcessTests(unittest.TestCase):
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

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.package = self.root / 'package'
        (self.package / '_native/bin').mkdir(parents=True)
        (self.package / '_native/lib').mkdir()
        (self.package / 'loom-pipeline').mkdir()
        shutil.copy2(self.binary, self.package / '_native/bin/loom-serve')
        (self.package / '_native/lib/libhsa-runtime64.so.1').write_bytes(b'CPU fixture only')
        (self.package / 'loom-pipeline/compatibility.json').write_text('{"test":"fixture"}')
        self.registry = Registry(self.root / 'state')
        self.equivalents = []
        self.manager = hem_process.ProcessManager(self.registry, package_root=self.package,
                                                 emit=self.equivalents.append)
        self.owned = []

    def tearDown(self):
        # Cleanup acts only on identities produced by this test's registry.
        for process, executable in self.owned:
            if not hem_process._confirmed_exit(process):
                with hem_process.PinnedProcess(process, executable, emit=lambda _: None) as pinned:
                    pinned.send(signal.SIGKILL, Deadline.after(2))
                    pinned.wait(Deadline.after(2))
            if process.get('logger'):
                wait_exit(process['logger'])
        for child in self.manager.children.values():
            child.wait(timeout=2)
            if child.stdout:
                child.stdout.close()
            if child.stderr:
                child.stderr.close()
        self.temp.cleanup()

    def settings(self):
        return {'device': 'local-task', 'listen': f'127.0.0.1:{port()}'}

    def start(self, name='one'):
        settings = self.settings()
        self.registry.create_server(name, saved_settings=settings)
        try:
            record = self.manager.start(name, settings, Deadline.after(3))
        finally:
            current = self.registry.get_server(name)
            if current['process']:
                # Store actual current executable even for bootstrap failure.
                self.owned.append((current['process'], hem_process.identity(current['process']['pid'])['executable']))
        return record

    def reserve(self, name='one'):
        settings = self.settings()
        self.registry.create_server(name, saved_settings=settings)
        record = self.registry.reserve_launch(name, launch_token='test-token', current_settings=settings,
                                              endpoint='http://' + settings['listen'])
        build = hem_process.retain_payload(self.registry, name, 'test-token', self.package, Deadline.after(3))
        return self.registry.update_server(name, {'build': build}, expected_token='test-token',
                                           expected_revision=record['revision'])

    def test_two_servers_are_independent_reuse_and_shutdown(self):
        first = self.start()
        second = self.start('two')
        self.assertNotEqual(first['process']['pid'], second['process']['pid'])
        with self.manager.pin(first) as pinned:
            self.assertEqual(self.manager.client(first, pinned, Deadline.after(2)).status()['data'], [])
            pinned.send(signal.SIGTERM, Deadline.after(2))
            pinned.wait(Deadline.after(2))
        stopped = self.manager.reconcile('one', Deadline.after(2))
        self.assertIsNone(stopped['launch_token'])
        self.assertIsNotNone(stopped['saved_settings'])
        self.assertTrue(Path(first['build']['directory']).exists())  # reconcile never deletes artifacts
        with self.manager.pin(second) as pinned:
            self.assertEqual(self.manager.client(second, pinned, Deadline.after(2)).status()['data'], [])
        reused = self.manager.start('two', second['current_settings'], Deadline.after(2))
        self.assertEqual(reused['launch_token'], second['launch_token'])
        self.assertIn('background via identity bootstrap', self.equivalents[0])

    def test_payload_upgrade_does_not_remove_running_dependencies_and_mismatch_refuses(self):
        record = self.start()
        retained = hem_process.require_build(record, self.package, Deadline.after(2))
        self.assertTrue(retained.is_dir())
        (self.package / 'loom-pipeline/compatibility.json').write_text('{"test":"changed"}')
        with self.assertRaises(HemError):
            hem_process.require_build(record, self.package, Deadline.after(2))
        shutil.rmtree(self.package)
        with self.manager.pin(record) as pinned:
            self.assertEqual(self.manager.client(record, pinned, Deadline.after(2)).status()['data'], [])
        self.assertTrue((retained / 'compatibility.json').is_file())

    def test_pid_boot_and_executable_mismatch_never_signal_replacement(self):
        record = self.start()
        for change in ({'start_ticks': 0}, {'boot_id': 'prior-boot'},
                       {'executable': {'device': 0, 'inode': 0}}):
            expected = dict(record['process'], executable=record['build']['executable'])
            expected.update(change)
            with patch('signal.pidfd_send_signal') as send:
                with self.assertRaises(UncertainOutcome):
                    with hem_process.PinnedProcess(expected) as pinned:
                        pinned.send(signal.SIGTERM, Deadline.after(1))
                send.assert_not_called()
        with self.manager.pin(record) as pinned:
            self.manager.client(record, pinned, Deadline.after(2)).status()

    def test_reconcile_reboot_and_reused_pid_clears_old_record_without_signal(self):
        record = self.start()
        for key, value in [('boot_id', 'old-boot'), ('start_ticks', 0)]:
            current = self.registry.get_server('one')
            if current['launch_token'] is None:
                # Restore test-owned record through a fresh reservation only.
                current = self.registry.reserve_launch('one', launch_token='next-token',
                            current_settings=record['current_settings'], endpoint=record['endpoint'])
            self.registry.update_server('one', {'process': dict(record['process'], **{key: value}), 'build': record['build']},
                        expected_token=current['launch_token'], expected_revision=current['revision'])
            with patch('signal.pidfd_send_signal') as send:
                self.assertIsNone(self.manager.reconcile('one', Deadline.after(1))['launch_token'])
                send.assert_not_called()
        # Actual fixture is still alive and will be cleaned using its real identity.
        with self.manager.pin(record) as pinned:
            pinned.verify()

    def test_unrelated_actual_connection_refused_before_request(self):
        first, second = self.start(), self.start('two')
        document = self.root / 'mutation.json'
        document.write_text('{}')
        with self.manager.pin(first) as pinned:
            client = NativeClient(second['endpoint'], Deadline.after(.1), pinned.connection,
                                  ownership_after=pinned.verify, emit=lambda _: None)
            with self.assertRaises(UncertainOutcome):
                client.post('/noise', document)
        # No request bytes reached second: /noise would write megabytes of logs.
        self.assertLess(len(hem_process.read_logs(second)), 1000)
        with self.manager.pin(second) as pinned:
            self.assertEqual(self.manager.client(second, pinned, Deadline.after(1)).status()['data'], [])

    def test_bounded_rotating_logs_and_logger_exits_with_native(self):
        record = self.start()
        with self.manager.pin(record) as pinned:
            self.manager.client(record, pinned, Deadline.after(3)).get('/noise')
            pinned.send(signal.SIGTERM, Deadline.after(1))
            pinned.wait(Deadline.after(2))
        wait_exit(record['process']['logger'])
        directory = Path(record['build']['directory'])
        self.assertLessEqual((directory / 'process.log').stat().st_size, hem_bootstrap.LOG_BYTES)
        self.assertLessEqual((directory / 'process.log.1').stat().st_size, hem_bootstrap.LOG_BYTES)
        self.assertEqual(len(hem_process.read_logs(record, limit=100)), 100)

    def test_log_storage_failures_keep_reader_alive_and_discard_until_eof(self):
        for failure in ('initial_open', 'rotation_open', 'write'):
            with self.subTest(failure=failure):
                directory = self.root / failure
                directory.mkdir()
                read_fd, write_fd = os.pipe()
                errors = []
                failed = threading.Event()
                original_open, original_write = os.open, os.write
                opened = []
                def log_open(path, flags, *args):
                    if failure == 'initial_open' or (failure == 'rotation_open' and opened):
                        failed.set()
                        raise OSError(errno.ENOSPC, 'injected log storage full')
                    descriptor = original_open(path, flags, *args)
                    opened.append(descriptor)
                    return descriptor
                def log_write(descriptor, content):
                    if failure == 'write' and descriptor in opened:
                        failed.set()
                        raise OSError(errno.EDQUOT, 'injected log quota exhausted')
                    return original_write(descriptor, content)
                def drain():
                    try:
                        hem_bootstrap._drain(read_fd, directory, maximum=512)
                    except BaseException as error:
                        errors.append(error)
                with patch.object(hem_bootstrap.os, 'open', side_effect=log_open), \
                     patch.object(hem_bootstrap.os, 'write', side_effect=log_write):
                    reader = threading.Thread(target=drain, daemon=True)
                    reader.start()
                    try:
                        if failure == 'initial_open':
                            self.assertTrue(failed.wait(1), 'initial log open was not attempted')
                            self.assertTrue(reader.is_alive(), 'log failure closed the only native pipe reader')
                        # More than the pipe capacity proves continuing draining,
                        # rather than merely keeping an unread descriptor open.
                        original_write(write_fd, b'native diagnostics' * 8192)
                        self.assertTrue(failed.wait(1), 'storage failure was not exercised')
                        self.assertTrue(reader.is_alive(), 'reader must wait for native EOF')
                    finally:
                        os.close(write_fd)
                        reader.join(timeout=2)
                self.assertFalse(reader.is_alive(), 'logger did not exit on EOF')
                self.assertEqual(errors, [])

    def test_first_retained_payload_flushes_every_new_ancestor_before_publication(self):
        synced = []
        original_fsync = os.fsync
        def fsync(descriptor):
            synced.append(os.readlink(f'/proc/self/fd/{descriptor}'))
            return original_fsync(descriptor)
        with patch.object(hem_process.os, 'fsync', side_effect=fsync):
            build = hem_process.retain_payload(self.registry, 'new-server', 'new-launch',
                                               self.package, Deadline.after(3))
        expected = [self.registry.root, self.registry.root / 'servers',
                    self.registry.root / 'servers/new-server', Path(build['directory'])]
        for directory in expected:
            self.assertIn(str(directory), synced, f'new directory entry is not durable: {directory}')

    def test_timeout_leaves_accepted_process_pending_then_reconciliation_recovers(self):
        settings = dict(self.settings(), device='slow')
        self.registry.create_server('one', saved_settings=settings)
        with self.assertRaises(DeadlineExpired) as raised:
            self.manager.start('one', settings, Deadline.after(.2))
        self.assertTrue(raised.exception.accepted)
        current = self.registry.get_server('one')
        self.assertEqual(current['pending']['action'], 'start')
        self.owned.append((current['process'], current['build']['executable']))
        deadline = Deadline.after(2)
        while self.manager.reconcile('one', deadline)['phase'] != 'running':
            time.sleep(min(.01, deadline.remaining()))
        current = self.registry.get_server('one')
        self.assertIsNone(current['pending'])
        with self.manager.pin(current) as pinned:
            self.manager.client(current, pinned, Deadline.after(1)).status()

    def test_unhealthy_or_unknown_server_never_gets_a_duplicate(self):
        settings = dict(self.settings(), device='unhealthy')
        self.registry.create_server('one', saved_settings=settings)
        with self.assertRaises(UncertainOutcome):
            self.manager.start('one', settings, Deadline.after(2))
        current = self.registry.get_server('one')
        self.owned.append((current['process'], current['build']['executable']))
        token = current['launch_token']
        with patch('subprocess.Popen') as launch:
            with self.assertRaises(UncertainOutcome):
                self.manager.start('one', settings, Deadline.after(1))
            launch.assert_not_called()
        self.assertEqual(self.registry.get_server('one')['launch_token'], token)

    def test_exit_after_http_response_and_before_signal_never_redirects_control(self):
        record = self.start()
        with self.manager.pin(record) as pinned:
            def exit_before_postcheck(deadline):
                pinned.send(signal.SIGTERM, deadline)
                pinned.wait(deadline)
                pinned.verify(deadline)
            client = NativeClient(record['endpoint'], Deadline.after(2), pinned.connection,
                                  ownership_after=exit_before_postcheck, emit=lambda _: None)
            with self.assertRaises(UncertainOutcome):
                client.status()
            with patch('signal.pidfd_send_signal') as send:
                with self.assertRaises(UncertainOutcome):
                    pinned.send(signal.SIGTERM, Deadline.after(1))
                send.assert_not_called()

    def test_explicit_cleanup_refuses_live_process_and_reconcile_never_deletes(self):
        record = self.start()
        with self.registry.lock('server', 'one'):
            with self.assertRaises(HemError):
                self.manager.cleanup(record, Deadline.after(1))
        with self.manager.pin(record) as pinned:
            pinned.send(signal.SIGTERM, Deadline.after(1))
            pinned.wait(Deadline.after(1))
        self.manager.reconcile('one', Deadline.after(1))
        target = Path(record['build']['directory'])
        self.assertTrue(target.exists())
        with self.registry.lock('server', 'one'):
            self.manager.cleanup(record, Deadline.after(1))
        self.assertFalse(target.exists())

    def test_native_environment_retained_root_owns_library_closure(self):
        native = self.package / '_native'
        environment = native_environment({'HOME': '/normal-home', 'HF_TOKEN': 'private',
                         'IREE_HAL_AMDGPU_LIBHSA_PATH': '/external/libhsa.so',
                         'LD_LIBRARY_PATH': '/external/libs', 'LD_PRELOAD': '/external/inject.so'}, native_root=native)
        self.assertEqual(environment['HOME'], '/normal-home')
        self.assertEqual(environment['HF_TOKEN'], 'private')
        self.assertNotIn('LD_PRELOAD', environment)
        self.assertNotIn('/external', environment['LD_LIBRARY_PATH'])
        self.assertEqual(environment['IREE_HAL_AMDGPU_LIBHSA_PATH'], str(native / 'lib/libhsa-runtime64.so.1'))

    def test_bad_managed_endpoint_and_unknown_capability_fail_closed(self):
        self.registry.create_server('one', saved_settings={})
        for listen in ('0.0.0.0:8080', 'localhost:8080', '127.0.0.1:0'):
            with self.assertRaises(InvalidInput):
                self.manager.start('one', {'device': 'task', 'listen': listen}, Deadline.after(1))
        with patch('os.pidfd_open', side_effect=PermissionError('restricted')):
            with self.assertRaises(UncertainOutcome):
                with hem_process.PinnedProcess(hem_process.identity(os.getpid())):
                    pass

    def bootstrap_child(self, record, lock, *, checkpoint=None, token=None):
        program = '''
import json, os, sys
from pathlib import Path
from binding.python import hem_bootstrap as b
from binding.python.hem_process import identity
from binding.python.hem_common import Deadline
stage = sys.argv[6]
if stage:
    def checkpoint(current):
        if current == stage:
            children = Path(f'/proc/self/task/{os.getpid()}/children').read_text().split()
            Path(sys.argv[1], 'checkpoint-' + sys.argv[2] + '.json').write_text(json.dumps([identity(int(pid)) for pid in children]))
            os._exit(77)
    b._checkpoint = checkpoint
b.bootstrap(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), Deadline(float(sys.argv[5])))
'''
        return subprocess.Popen([sys.executable, '-c', program, str(self.registry.root), record['id'],
                                token or record['launch_token'], str(lock.fileno()),
                                str(Deadline.after(3).expires_at), checkpoint or ''],
                               pass_fds=(lock.fileno(),), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               env=dict(os.environ, LOOM_SERVE_DIR=str(self.registry.root)))

    def test_all_bootstrap_crash_windows_release_lock_and_preserve_truth(self):
        for index, checkpoint in enumerate(('before_identity', 'after_identity', 'after_log_fork', 'after_logger', 'before_exec')):
            with self.subTest(checkpoint=checkpoint):
                record = self.reserve(f'crash-{index}')
                with self.registry.lock('server', record['id']) as lock:
                    child = self.bootstrap_child(record, lock, checkpoint=checkpoint)
                    output, error = child.communicate(timeout=3)
                    self.assertEqual(child.returncode, 77, error)
                current = self.registry.get_server(record['id'])
                evidence = self.registry.root / ('checkpoint-' + record['id'] + '.json')
                for logger in json.loads(evidence.read_text()):
                    wait_exit(logger)
                if current['process'] and current['process'].get('logger'):
                    wait_exit(current['process']['logger'])
                cleared = self.manager.reconcile(record['id'], Deadline.after(2))
                self.assertIsNone(cleared['launch_token'])
                self.assertTrue(Path(record['build']['directory']).exists())

    def test_stale_token_never_executes_and_exec_failure_never_reports_running(self):
        record = self.reserve()
        with self.registry.lock('server', 'one') as lock:
            child = self.bootstrap_child(record, lock, token='stale-token')
            _, error = child.communicate(timeout=3)
            self.assertNotEqual(child.returncode, 0)
            self.assertIn(b'token/intent changed', error)
        self.assertIsNone(self.registry.get_server('one')['process'])
        executable = Path(record['build']['directory']) / '_native/bin/loom-serve'
        executable.chmod(0o600)  # Same inode but exec is now forbidden.
        with self.registry.lock('server', 'one') as lock:
            child = self.bootstrap_child(record, lock)
            _, error = child.communicate(timeout=3)
            self.assertNotEqual(child.returncode, 0)
        current = self.registry.get_server('one')
        self.assertNotEqual(current['phase'], 'running')
        wait_exit(current['process']['logger'])
        self.assertIsNone(self.manager.reconcile('one', Deadline.after(2))['launch_token'])

    def test_live_caller_disconnect_recovers_native_without_relaunch(self):
        settings = self.settings()
        self.registry.create_server('one', saved_settings=settings)
        program = '''
import json, os, subprocess, sys
from binding.python.hem_state import Registry
from binding.python.hem_process import ProcessManager
from binding.python.hem_common import Deadline
original = subprocess.Popen
def spawn(*a, **kw):
    original(*a, **kw)
    os._exit(88)
subprocess.Popen = spawn
ProcessManager(Registry(sys.argv[1]), package_root=sys.argv[2], emit=lambda _:None).start('one', json.loads(sys.argv[3]), Deadline.after(3))
'''
        caller = subprocess.Popen([sys.executable, '-c', program, str(self.registry.root),
                                   str(self.package), json.dumps(settings)],
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                  env=dict(os.environ, LOOM_SERVE_DIR=str(self.registry.root)))
        output, error = caller.communicate(timeout=3)
        self.assertEqual(caller.returncode, 88, error)
        until = Deadline.after(3)
        while True:
            current = self.manager.reconcile('one', until)
            if current['phase'] == 'running':
                break
            time.sleep(min(.01, until.remaining()))
        self.owned.append((current['process'], current['build']['executable']))
        token = current['launch_token']
        self.assertEqual(self.manager.start('one', settings, Deadline.after(2))['launch_token'], token)

    def test_inherited_lock_survives_parent_descriptor_close_until_exec(self):
        record = self.reserve()
        marker = self.root / 'helper-waiting'
        release = self.root / 'helper-release'
        program = '''
import os, sys, time
from pathlib import Path
from binding.python import hem_bootstrap as b
from binding.python.hem_common import Deadline
def checkpoint(stage):
    if stage == 'after_identity':
        Path(sys.argv[6]).touch()
        while not Path(sys.argv[7]).exists(): time.sleep(.005)
b._checkpoint = checkpoint
b.bootstrap(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), Deadline(float(sys.argv[5])))
'''
        with self.registry.lock('server', 'one') as lock:
            child = subprocess.Popen([sys.executable, '-c', program, str(self.registry.root), 'one',
                                      record['launch_token'], str(lock.fileno()),
                                      str(Deadline.after(4).expires_at), str(marker), str(release)],
                                     pass_fds=(lock.fileno(),), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            deadline = Deadline.after(2)
            while not marker.exists():
                time.sleep(min(.005, deadline.remaining()))
        with self.assertRaises(DeadlineExpired):
            with self.registry.lock('server', 'one', deadline=Deadline.after(.05)):
                pass
        release.touch()
        deadline = Deadline.after(3)
        while True:
            current = self.manager.reconcile('one', deadline)
            if current['phase'] == 'running':
                break
            time.sleep(min(.01, deadline.remaining()))
        self.owned.append((current['process'], current['build']['executable']))
        self.manager.children[record['launch_token']] = child

    def test_bootstrap_exec_between_pin_rejection_and_handler_recheck_is_retried(self):
        record = self.reserve()
        marker, release = self.root / 'exec-waiting', self.root / 'exec-release'
        program = '''
import os, sys, time
from pathlib import Path
from binding.python import hem_bootstrap as b
from binding.python.hem_common import Deadline
def checkpoint(stage):
    if stage == 'before_exec':
        Path(sys.argv[6]).touch()
        while not Path(sys.argv[7]).exists(): time.sleep(.001)
b._checkpoint = checkpoint
b.bootstrap(sys.argv[1],sys.argv[2],sys.argv[3],int(sys.argv[4]),Deadline(float(sys.argv[5])))
'''
        with self.registry.lock('server', 'one') as lock:
            child = subprocess.Popen([sys.executable, '-c', program, str(self.registry.root), 'one',
                                      record['launch_token'], str(lock.fileno()),
                                      str(Deadline.after(4).expires_at), str(marker), str(release)],
                                     pass_fds=(lock.fileno(),), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                     env=dict(os.environ, LOOM_SERVE_DIR=str(self.registry.root)))
            self.manager.children[record['launch_token']] = child
            deadline = Deadline.after(3)
            while not marker.exists():
                time.sleep(min(.001, deadline.remaining()))
            current = self.registry.get_server('one')
            self.owned.append((current['process'], record['build']['executable']))
            original_identity = hem_process.identity
            observed_bootstrap = []
            def exec_after_snapshot(pid):
                snapshot = original_identity(pid)
                if pid == child.pid and not observed_bootstrap:
                    self.assertEqual(snapshot['executable'], current['process']['executable'])
                    observed_bootstrap.append(snapshot)
                    release.touch()
                    # Force the real same-PID exec before verify() rejects the
                    # older Python snapshot. A second inspection now sees native.
                    while original_identity(pid)['executable'] != record['build']['executable']:
                        time.sleep(min(.001, deadline.remaining()))
                return snapshot
            with patch.object(hem_process, 'identity', side_effect=exec_after_snapshot):
                running = self.manager._wait_start('one', child, deadline)
            self.assertEqual(len(observed_bootstrap), 1)
            self.assertEqual(running['phase'], 'running')
            self.assertEqual(running['launch_token'], record['launch_token'])
            for key in ('pid', 'boot_id', 'start_ticks'):
                self.assertEqual(running['process'][key], current['process'][key])
            with self.manager.pin(running) as pinned:
                self.assertEqual(self.manager.client(running, pinned, Deadline.after(1)).status()['data'], [])
