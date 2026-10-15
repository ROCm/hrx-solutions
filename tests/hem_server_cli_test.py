"""Public server CLI CPU gates with a separately exec'd native ELF fixture.

Only package payload discovery is injected for source checks. Installed checks
can set HEM_TEST_PACKAGE=hrx_solutions; no behavior/HTTP/process action is mocked.
"""
from contextlib import contextmanager
import importlib
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest

PACKAGE = os.environ.get('HEM_TEST_PACKAGE', 'binding.python')
process_module = importlib.import_module(PACKAGE + '.hem_process')
Registry = importlib.import_module(PACKAGE + '.hem_state').Registry
Deadline = importlib.import_module(PACKAGE + '.hem_common').Deadline
InvalidInput = importlib.import_module(PACKAGE + '.hem_common').InvalidInput
resolve_server = importlib.import_module(PACKAGE + '.hem_server_cli').resolve_server
NativeClient = importlib.import_module(PACKAGE + '.hem_http').NativeClient
NATIVE = r'''
#include <arpa/inet.h>
#include <netinet/in.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>
int main(int argc,char **argv) {
 signal(SIGPIPE,SIG_IGN); int port=0,yes=1,unhealthy=0;
 for(int i=1;i<argc;i++) if(!strncmp(argv[i],"--listen=",9)) port=atoi(strrchr(argv[i],':')+1);
 int s=socket(AF_INET,SOCK_STREAM,0); setsockopt(s,SOL_SOCKET,SO_REUSEADDR,&yes,sizeof(yes));
 struct sockaddr_in a={.sin_family=AF_INET,.sin_port=htons(port),.sin_addr.s_addr=htonl(INADDR_LOOPBACK)};
 if(!port || bind(s,(void*)&a,sizeof(a)) || listen(s,16)) return 6;
 fprintf(stderr,"fake-native ready %d\n",port); fflush(stderr);
 int c; while((c=accept(s,NULL,NULL))>=0) {
  char request[4096]; int n=read(c,request,sizeof(request)-1); if(n<0)n=0; request[n]=0;
  fprintf(stderr,"%.*s\n",(int)strcspn(request,"\r\n"),request); fflush(stderr);
  if(strstr(request,"/fail")) unhealthy=1;
  if(strstr(request,"/noise")) {for(int i=0;i<100000;i++) fputs("bounded fixture log output abcdefghijklmnopqrstuvwxyz\n",stderr); fflush(stderr);}
  const char *body=unhealthy?"{}":"{\"object\":\"list\",\"data\":[],\"server\":{\"worker_count\":1}}";
  dprintf(c,"HTTP/1.1 200 OK\r\nContent-Length: %zu\r\nConnection: close\r\n\r\n%s",strlen(body),body); close(c);
 }
 return 0;
}
'''


def available_port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


class ServerCliTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compiled = tempfile.TemporaryDirectory()
        source = Path(cls.compiled.name) / 'fixture.c'
        source.write_text(NATIVE)
        cls.binary = source.with_name('loom-serve')
        subprocess.run(['cc', '-O0', '-o', str(cls.binary), str(source)], check=True, capture_output=True)

    @classmethod
    def tearDownClass(cls):
        cls.compiled.cleanup()

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.state = self.root / 'state'
        self.package = self.root / 'payload'
        (self.package / '_native/bin').mkdir(parents=True)
        (self.package / '_native/lib').mkdir()
        (self.package / 'loom-pipeline').mkdir()
        shutil.copy2(self.binary, self.package / '_native/bin/loom-serve')
        (self.package / '_native/lib/libhsa-runtime64.so.1').write_bytes(b'CPU fixture only')
        (self.package / 'loom-pipeline/compatibility.json').write_text('{"test":"CPU fixture"}')
        self.environment = dict(os.environ, LOOM_SERVE_DIR=str(self.state), PYTHONNOUSERSITE='1')
        self.environment.pop('PYTHONPATH', None)
        self.environment.pop('PYTHONHOME', None)
        # Exercise the public entry point in a distinct interpreter. The sole
        # injection selects fixture payload discovery, leaving lifecycle intact.
        self.launcher = [sys.executable, '-c',
            'import functools,sys; '
            f'from {PACKAGE} import hem_cli,hem_server_cli; '
            'hem_server_cli.ProcessManager=functools.partial(hem_server_cli.ProcessManager,package_root=sys.argv[1]); '
            'raise SystemExit(hem_cli.main(sys.argv[2:]))', str(self.package)]
        if PACKAGE == 'hrx_solutions':
            self.launcher = [str(Path(sys.executable).with_name('loom-hem'))]
        self.records = []

    def tearDown(self):
        if self.state.exists():
            registry = Registry(self.state)
            for record in registry.list_servers():
                if record.get('process'):
                    actual = record['process']
                    if not process_module._confirmed_exit(actual):
                        with process_module.PinnedProcess(actual, process_module.identity(actual['pid'])['executable'], emit=lambda _:None) as pinned:
                            pinned.send(signal.SIGTERM, Deadline.after(2))
                            pinned.wait(Deadline.after(2))
                    logger = actual.get('logger')
                    end = time.monotonic() + 2
                    while logger and not process_module._confirmed_exit(logger):
                        self.assertLess(time.monotonic(), end)
                        time.sleep(.01)
        self.temporary.cleanup()

    def invoke(self, *argv, expected=0):
        completed = subprocess.run([*self.launcher, 'server', *argv, '--json'],
                                   env=self.environment, capture_output=True, text=True, timeout=15)
        self.assertEqual(completed.returncode, expected, completed.stderr + completed.stdout)
        return json.loads(completed.stdout), completed.stderr

    def start(self, name='one', port=None):
        return self.invoke('start', '--name', name, '--device', 'local-task', '--listen',
                           f'127.0.0.1:{port or available_port()}')

    def stop_owned(self, name):
        record = Registry(self.state).get_server(name)
        with process_module.PinnedProcess(record['process'], record['build']['executable'], emit=lambda _: None) as pinned:
            pinned.send(signal.SIGTERM, Deadline.after(2))
            pinned.wait(Deadline.after(2))
        return record

    def test_start_show_status_reuse_and_restart_empty(self):
        first, diagnostic = self.start()
        self.assertTrue(first['accepted'])
        self.assertIn('background via identity bootstrap', diagnostic)
        self.assertIn('curl ', diagnostic)
        self.assertTrue(first['log_paths'])
        shown, _ = self.invoke('show', 'one')
        self.assertEqual(shown['saved_settings'], first['record']['saved_settings'])
        self.assertNotIn('process', shown)
        status, _ = self.invoke('status', 'one')
        self.assertEqual(status['native']['data'], [])
        self.assertEqual(status['http_health'], 'healthy')
        self.assertEqual(status['record']['current_settings'], shown['saved_settings'])
        reused, _ = self.invoke('start', 'one')
        self.assertEqual(reused['process']['pid'], first['process']['pid'])
        self.invoke('start', '--name', 'one', '--device', 'local-task', '--listen',
                    first['record']['saved_settings']['listen'], expected=1)
        self.stop_owned('one')
        stopped, _ = self.invoke('reconcile', 'one')
        self.assertEqual(stopped['state'], 'stopped')
        self.assertIsNone(stopped['record']['process'])
        self.assertTrue(Path(first['record']['build']['directory']).exists())
        restarted, _ = self.invoke('start', 'one')
        self.assertNotEqual(restarted['record']['launch_token'], first['record']['launch_token'])
        self.assertEqual(self.invoke('status', 'one')[0]['native']['data'], [])

    def test_two_servers_inventory_and_stopping_one_leaves_other_usable(self):
        first, _ = self.start()
        second, _ = self.start('two')
        self.assertNotEqual(first['process']['pid'], second['process']['pid'])
        self.stop_owned('one')
        inventory, _ = self.invoke('list')
        states = {item['server']: item['state'] for item in inventory['items']}
        self.assertEqual(states, {'one':'stopped', 'two':'running'})
        self.assertEqual(self.invoke('status', 'two')[0]['http_health'], 'healthy')
        self.assertEqual(len(Registry(self.state).list_servers()), 2)

    def test_unhealthy_and_unknown_refuse_duplicate_but_other_items_survive(self):
        first, _ = self.start()
        self.start('two')
        NativeClient(first['endpoint']).get('/fail')
        unhealthy, _ = self.invoke('status', 'one', expected=1)
        self.assertEqual(unhealthy['state'], 'unhealthy')
        self.assertEqual(unhealthy['process_health'], 'verified')
        failed, _ = self.invoke('start', 'one', expected=4)
        self.assertIsNone(failed['accepted'])
        registry = Registry(self.state)
        record = registry.get_server('one')
        build = dict(record['build'], executable=dict(device=0,inode=0))
        registry.update_server('one', {'build':build}, expected_token=record['launch_token'], expected_revision=record['revision'])
        unknown, _ = self.invoke('status', 'one', expected=4)
        self.assertEqual(unknown['state'], 'unknown')
        self.invoke('start', 'one', expected=4)
        inventory, _ = self.invoke('list', expected=4)
        self.assertEqual({item['server']:item['state'] for item in inventory['items']}, {'one':'unknown','two':'running'})
        self.assertEqual(registry.get_server('one')['process']['pid'], first['process']['pid'])

    def test_reconcile_only_observes_and_never_cleans_files(self):
        first, _ = self.start()
        current = Registry(self.state).get_server('one')
        registry = Registry(self.state)
        registry.update_server('one', {'phase':'unknown'}, expected_token=current['launch_token'], expected_revision=current['revision'])
        result, diagnostic = self.invoke('reconcile', 'one')
        self.assertEqual(result['state'], 'running')
        self.assertEqual(result['record']['phase'], 'running')
        self.assertNotIn('POST', diagnostic)
        self.assertTrue(Path(first['log_paths'][-1]).exists())
        self.stop_owned('one')
        self.invoke('reconcile', 'one')
        self.assertTrue(Path(first['record']['build']['directory']).exists())
        self.assertEqual(Registry(self.state).list_instances(), [])

    def test_bounded_logs_and_follow_release_lock(self):
        first, _ = self.start()
        NativeClient(first['endpoint']).get('/noise')
        log, diagnostic = self.invoke('logs', 'one')
        self.assertLessEqual(len(log['logs'].encode()), 65536)
        self.assertIn('bounded fixture', log['logs'])
        self.assertIn('tail -c 65536', diagnostic)
        follow = subprocess.Popen([*self.launcher, 'server', 'logs', 'one', '--follow', '--json'],
                                  env=self.environment, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            time.sleep(.2)
            with Registry(self.state).lock('server', 'one', deadline=Deadline.after(.5)):
                pass
            self.stop_owned('one')
            stdout, stderr = follow.communicate(timeout=3)
            self.assertEqual(follow.returncode, 0, stderr)
            self.assertTrue(json.loads(stdout)['followed'])
            self.assertIn('tail -c 65536 -F', stderr)
        finally:
            if follow.poll() is None:
                follow.terminate()
                follow.communicate(timeout=3)

    def test_invalid_new_settings_missing_server_and_shutdown(self):
        for args in (('start','--name','bad'), ('start','--name','bad','--device','task','--listen','0.0.0.0:8080')):
            data, _ = self.invoke(*args, expected=2)
            self.assertEqual(data['state'], 'failed')
            self.assertFalse(self.state.exists())
        self.invoke('show', 'missing', expected=1)
        self.invoke('shutdown', 'missing', expected=1)

    def test_selector_external_origin_and_exactly_one_saved_definition(self):
        registry = Registry(self.state)
        with self.assertRaises(InvalidInput):
            resolve_server(registry)
        external = resolve_server(registry, url='http://127.0.0.1:12345/')
        self.assertEqual(external, dict(id=None,endpoint='http://127.0.0.1:12345',external=True))
        self.assertEqual(registry.list_servers(), [])
        with self.assertRaises(InvalidInput):
            resolve_server(registry, url='http://secret@localhost:80/')
        registry.create_server('one', saved_settings={'device':'task','listen':'127.0.0.1:12345'})
        self.assertEqual(resolve_server(registry)['id'], 'one')
        registry.create_server('two', saved_settings={'device':'task','listen':'127.0.0.1:12346'})
        self.assertEqual(resolve_server(registry,'two')['id'], 'two')
        with self.assertRaises(InvalidInput):
            resolve_server(registry)
        with self.assertRaises(InvalidInput):
            resolve_server(registry,'one','http://localhost:80')


if __name__ == '__main__':
    unittest.main()
