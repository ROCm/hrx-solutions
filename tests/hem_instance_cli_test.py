"""Public instance commands with sparse CPU assets and an owned native ELF.

The only injections select the CPU payload and tokenizer fingerprint. All CLI,
configuration, HTTP, socket/process ownership and registry behavior stays real.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest

from binding.python import hem_config, hem_process, hem_state
from binding.python.hem_common import Deadline
from binding.python.hem_http import NativeClient
from tests.gguf_fixtures import install_lifecycle_profile, lifecycle_profile, write_matching_gguf
from tests.hem_http_test import fixture, inventory, model, reply
from tests.hem_instances_test import NATIVE
from tests.hem_process_test import port, wait_exit

# Keep the existing native-shaped metadata-only registration fixture, adding
# diagnostics and controllable externally-created/replaced/error registrations.
NATIVE = NATIVE.replace('registered=0, active=0, open=0;',
    'registered=0, active=0, open=0, incarnation=17, pending=0, broken=0;')
NATIVE = NATIVE.replace('   int code=200;', '''
   int code=200;
   if(strstr(request,"GET /fixture/external ")) {registered=1; active=1; open=1; incarnation=18;}
   if(strstr(request,"GET /fixture/pending ")) pending=1;
   if(strstr(request,"GET /fixture/broken ")) broken=1;
''')
NATIVE = NATIVE.replace('registered=0; code=202;', 'if(!pending) registered=0; code=202;')
NATIVE = NATIVE.replace('   n=snprintf(response,', r'''
   if(incarnation==18) { char *p=strstr(body,"\"incarnation\":17"); if(p) p[15]='8'; }
   if(broken && !strncmp(request,"GET /v1/status",14)) strcpy(body,"{}");
   if(strstr(request,"GET /logz ")) {
     if(strstr(request,"X-Loom-Cursor:")) strcpy(body,"{\"event\":\"cursor\",\"value\":\"1:1:0\"}\n");
     else strcpy(body,"{\"event\":\"log\",\"model\":\"chat\",\"message\":\"CPU model diagnostic\"}\n{\"event\":\"cursor\",\"value\":\"1:1:0\"}\n");
   }
   n=snprintf(response,''')


class InstanceCliTest(unittest.TestCase):
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
        shutil.copy2(self.binary, self.package / '_native/bin/loom-serve')
        (self.package / '_native/lib/libhsa-runtime64.so.1').write_bytes(b'CPU fixture only')
        catalog = Path(hem_config.pipeline_info('qwen-08b')['catalog'])
        shutil.copytree(catalog.parents[2], self.package / 'loom-pipeline')
        install_lifecycle_profile(self.package / 'loom-pipeline')
        self.weights = self.root / 'weights.gguf'
        write_matching_gguf(self.weights, contract=lifecycle_profile())
        self.tokenizer = self.root / 'tokenizer.json'
        self.tokenizer.write_text('{"model":{"fixture":true}}')
        digest = hashlib.sha256(json.dumps({'model':{'fixture':True}}, sort_keys=True,
                                          separators=(',', ':')).encode()).hexdigest()
        package = hem_config.__package__
        script = (
            'import functools,sys; '
            f'from {package} import hem_cli,hem_config,hem_instance_cli,hem_server_cli; '
            'from pathlib import Path; '
            'original_resolve=hem_config.resolve_catalog; '
            'hem_config.resolve_catalog=lambda identifier:Path(sys.argv[1])/"loom-pipeline"/Path(original_resolve(identifier)).relative_to(Path(original_resolve(identifier)).parents[2]); '
            'original=hem_config.load_profile; '
            f'hem_config.load_profile=lambda catalog:dict(original(catalog), tokenizer_canonical_sha256={digest!r}); '
            'hem_instance_cli.ProcessManager=functools.partial(hem_instance_cli.ProcessManager,package_root=sys.argv[1]); '
            'hem_server_cli.ProcessManager=functools.partial(hem_server_cli.ProcessManager,package_root=sys.argv[1]); '
            'raise SystemExit(hem_cli.main(sys.argv[2:]))')
        self.launcher = [sys.executable, '-c', script, str(self.package)]
        self.environment = dict(os.environ, LOOM_SERVE_DIR=str(self.state), PYTHONNOUSERSITE='1')
        self.environment.pop('PYTHONPATH', None)
        self.environment.pop('PYTHONHOME', None)
        self.server = None

    def tearDown(self):
        if self.state.exists():
            for record in hem_state.Registry(self.state).list_servers():
                actual = record.get('process')
                if actual and not hem_process._confirmed_exit(actual):
                    with hem_process.PinnedProcess(actual, record['build']['executable'], emit=lambda _:None) as pinned:
                        pinned.send(signal.SIGTERM, Deadline.after(2))
                        pinned.wait(Deadline.after(2))
                if actual and actual.get('logger'):
                    wait_exit(actual['logger'])
        self.temporary.cleanup()

    def invoke(self, *args, expected=0):
        completed = subprocess.run([*self.launcher, *args, '--json'], env=self.environment,
                                  capture_output=True, text=True, timeout=15)
        self.assertEqual(completed.returncode, expected, completed.stderr + completed.stdout)
        self.assertNotIn('Traceback', completed.stderr)
        return json.loads(completed.stdout), completed.stderr

    def setup_model(self):
        return self.invoke('model', 'import', 'tiny', '--gguf', str(self.weights),
                           '--tokenizer', str(self.tokenizer), '--alias')

    def setup_server(self, name='one'):
        response, _ = self.invoke('server', 'start', '--name', name, '--device', 'CPU-fixture',
                                 '--listen', f'127.0.0.1:{port()}')
        self.server = response
        return response

    def start(self, *args):
        self.setup_model()
        self.setup_server()
        return self.invoke('instance', 'start', 'tiny', '--name', 'chat', *args)

    def test_eager_start_show_retains_submission_and_stop(self):
        config = self.root / 'serving.json'
        config.write_text('{"context_capacity":4096,"rows":2}')
        started, diagnostic = self.start('--config', str(config), '--rows', '3', '--pipeline', 'qwen-08b')
        self.assertTrue(started['accepted'])
        self.assertEqual(started['settings']['rows'], 3)
        self.assertEqual(started['settings']['context_capacity'], 4096)
        self.assertFalse(started['settings']['mtp'])
        routes = ['/v1/models/register','/v1/models/activate','/v1/models/undrain']
        self.assertEqual(sorted(routes, key=diagnostic.index), routes)
        self.assertIn('curl ', diagnostic)
        self.assertIn('--data-binary @', diagnostic)
        config.write_text('{"rows":15}')
        shown, _ = self.invoke('instance', 'show', 'chat')
        self.assertEqual(shown['settings'], started['settings'])
        self.assertEqual(shown['model'], 'tiny')
        document = json.loads(Path(shown['document']).read_text())
        self.assertEqual(document['definition']['rows'], 3)
        self.assertEqual(shown['native']['loom']['status']['admission'], 'open')
        status, _ = self.invoke('instance', 'status', 'chat', '--server', 'one')
        self.assertTrue(status['native']['loom']['parameters']['active'])
        stopped, diagnostic = self.invoke('instance', 'stop', 'chat')
        self.assertTrue(stopped['completed'])
        self.assertIn('/v1/models/unregister', diagnostic)
        self.assertEqual(hem_state.Registry(self.state).list_instances(), [])
        self.assertEqual(self.invoke('instance', 'list')[0]['items'], [])

    def test_external_registration_and_replacement_have_unknown_association(self):
        self.start()
        NativeClient(self.server['endpoint'], emit=lambda _:None).get('/fixture/external')
        listed, _ = self.invoke('instance', 'list')
        self.assertEqual(len(listed['items']), 1)
        shown = listed['items'][0]
        self.assertEqual(shown['incarnation'], 18)
        self.assertIsNone(shown['model'])
        self.assertIsNone(shown['settings'])
        self.assertEqual(shown['tracked_submission']['incarnation'], 17)
        self.assertEqual(shown['tracked_submission']['model'], 'tiny')
        self.assertEqual(len(hem_state.Registry(self.state).list_instances()), 1)
        # Explicit URL never borrows a local mapping even at a managed endpoint.
        shown, _ = self.invoke('instance', 'show', 'chat', '--url', self.server['endpoint'])
        self.assertIsNone(shown['model'])
        self.assertEqual(shown['native']['loom']['incarnation'], 18)

    def test_external_only_inventory_stop_and_url_start_refusal(self):
        self.setup_server()
        NativeClient(self.server['endpoint'], emit=lambda _:None).get('/fixture/external')
        self.assertIsNone(self.invoke('instance', 'show', 'chat')[0]['model'])
        self.setup_model()
        self.invoke('instance', 'start', 'tiny', '--url', self.server['endpoint'], expected=2)
        stopped, _ = self.invoke('instance', 'stop', 'chat', '--url', self.server['endpoint'])
        self.assertTrue(stopped['completed'])
        self.assertEqual(hem_state.Registry(self.state).list_instances(), [])
        self.assertEqual(len(hem_state.Registry(self.state).list_servers()), 1)

    def test_stop_no_wait_pending_timeout_and_abort_are_explicit(self):
        self.start()
        NativeClient(self.server['endpoint'], emit=lambda _:None).get('/fixture/pending')
        stopped, _ = self.invoke('instance', 'stop', 'chat', '--no-wait')
        self.assertFalse(stopped['completed'])
        self.assertTrue(stopped['accepted'])
        pending, _ = self.invoke('instance', 'stop', 'chat', '--timeout', '.1', expected=3)
        self.assertTrue(pending['accepted'])
        shown, _ = self.invoke('instance', 'show', 'chat')
        self.assertEqual(shown['pending']['mode'], 'drain')
        # This fixture remains pending even on abort, proving accepted truth.
        aborted, _ = self.invoke('instance', 'stop', 'chat', '--abort', '--no-wait')
        self.assertFalse(aborted['completed'])
        self.assertEqual(self.invoke('instance', 'show', 'chat')[0]['pending']['mode'], 'abort')

    def test_config_conflicts_and_invalid_pipeline_fail_before_registration(self):
        self.setup_model()
        self.setup_server()
        config = self.root / 'bad.json'
        config.write_text('{"mtp_depth":2}')
        self.invoke('instance', 'start', 'tiny', '--config', str(config), '--no-mtp', expected=2)
        self.invoke('instance', 'start', 'tiny', '--pipeline', 'qwen', expected=2)
        self.invoke('instance', 'start', 'tiny', '--rows', '17', expected=2)
        self.assertEqual(self.invoke('instance', 'list')[0]['items'], [])
        self.assertEqual(hem_state.Registry(self.state).list_instances(), [])

    def test_export_offline_atomic_body_no_reservation_and_responsibility(self):
        self.setup_model()
        output = self.root / 'registration.json'
        exported, diagnostic = self.invoke('instance', 'export', 'tiny', '--name', 'chat',
                                          '--output', str(output), '--rows', '6')
        document = json.loads(output.read_text())
        self.assertEqual(document['name'], 'chat')
        self.assertEqual(document['admission'], 'drained')
        self.assertEqual(document['definition']['rows'], 6)
        self.assertEqual(document['definition']['draft_depth'], 0)
        self.assertIn('does not reserve', diagnostic)
        self.assertIn('compatible server', diagnostic)
        self.assertIn('curl ', diagnostic)
        self.assertIn(str(output), diagnostic)
        registry = hem_state.Registry(self.state)
        self.assertEqual(registry.list_servers(), [])
        self.assertEqual(registry.list_instances(), [])
        self.assertEqual(registry.references('tiny'), [])
        before = output.read_bytes()
        self.invoke('instance', 'export', 'tiny', '--output', str(output), '--mtp-depth', '2', expected=2)
        self.assertEqual(output.read_bytes(), before)
        self.assertEqual(exported['settings']['rows'], 6)

    def test_logs_are_native_model_diagnostics_and_follow_has_no_lifecycle_lock(self):
        self.start()
        logs, diagnostic = self.invoke('instance', 'logs', 'chat')
        self.assertIn('CPU model diagnostic', logs['logs'])
        self.assertIn('/logz', diagnostic)
        self.assertIn('X-Loom-Model: chat', diagnostic)
        process = subprocess.Popen([*self.launcher, 'instance','logs','chat','--follow','--json'],
                                   env=self.environment, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            time.sleep(.4)
            with hem_state.Registry(self.state).lock('server','one',deadline=Deadline.after(.5)):
                pass
            process.send_signal(signal.SIGINT)
            stdout, stderr = process.communicate(timeout=3)
            self.assertEqual(process.returncode, 0, stderr)
            self.assertTrue(json.loads(stdout)['followed'])
            self.assertEqual(stderr.count('CPU model diagnostic'), 1)
            self.assertIn('X-Loom-Cursor: 1:1:0', stderr)
        finally:
            if process.poll() is None:
                process.terminate()
                process.communicate(timeout=3)

    def test_unhealthy_observation_preserves_rows_and_nonzero_payload(self):
        self.start()
        NativeClient(self.server['endpoint'], emit=lambda _:None).get('/fixture/broken')
        shown, _ = self.invoke('instance', 'show', 'chat', expected=1)
        self.assertEqual(shown['settings']['rows'], 4)
        self.assertEqual(shown['state'], 'unhealthy')
        self.assertIsNone(shown['native'])
        listed, _ = self.invoke('instance', 'list', expected=1)
        self.assertEqual(listed['items'][0]['model'], 'tiny')
        self.assertEqual(listed['server_health'], 'unhealthy')
        self.invoke('instance', 'list', '--url', self.server['endpoint'], expected=4)
        self.assertEqual(len(hem_state.Registry(self.state).list_instances()), 1)

    def test_unknown_ownership_preserves_current_configuration_and_does_not_mutate(self):
        self.start()
        registry = hem_state.Registry(self.state)
        original = registry.get_server('one')
        build = dict(original['build'], executable={'device':0,'inode':0})
        registry.update_server('one', {'build':build}, expected_token=original['launch_token'],
                               expected_revision=original['revision'])
        try:
            shown, diagnostic = self.invoke('instance','show','chat',expected=4)
            self.assertEqual(shown['state'], 'unknown')
            self.assertEqual(shown['settings']['rows'], 4)
            self.assertNotIn('POST', diagnostic)
            listed, _ = self.invoke('instance','list',expected=4)
            self.assertEqual(listed['items'][0]['model'], 'tiny')
            self.assertEqual(len(registry.list_instances()), 1)
        finally:
            current = registry.get_server('one')
            registry.update_server('one', {'build':original['build']}, expected_token=current['launch_token'],
                                   expected_revision=current['revision'])

    def test_native_model_errors_and_unreachable_empty_inventory_keep_nonzero_json(self):
        def respond(handler, requests):
            reply(handler, inventory(model(error='fixture native failure')))
        with fixture(respond) as (endpoint, _):
            shown, _ = self.invoke('instance','status','chat','--url',endpoint,expected=1)
            self.assertEqual(shown['native']['loom']['error'], 'fixture native failure')
            self.assertEqual(shown['state'], 'unhealthy')
        listed, _ = self.invoke('instance','list','--url',endpoint,expected=4)
        self.assertEqual(listed['items'], [])
        self.assertEqual(listed['server_health'], 'unknown')

    def test_selectors_deferred_commands_and_missing_instances(self):
        for args in (('start','tiny','--no-wait'), ('restart','chat'), ('configure','chat'),
                     ('export','tiny','--output','x','--server','one'),
                     ('list','--server','one','--url','http://127.0.0.1:1')):
            self.invoke('instance', *args, expected=2)
        self.assertFalse(self.state.exists())
        self.invoke('instance', 'list', expected=2)
        self.setup_server()
        self.invoke('instance','show','missing',expected=1)
        self.setup_server('two')
        self.invoke('instance','list',expected=2)
        self.assertEqual(self.invoke('instance','list','--server','one')[0]['items'], [])


if __name__ == '__main__':
    unittest.main()
