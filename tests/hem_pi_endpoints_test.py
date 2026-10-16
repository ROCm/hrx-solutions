"""Durable projection and native-success/publication-failure separation."""
import importlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import uuid

PACKAGE = os.environ.get('HEM_TEST_PACKAGE', 'binding.python')
endpoints = importlib.import_module(PACKAGE + '.hem_pi_endpoints')
contracts = importlib.import_module(PACKAGE + '.hem_pi_contracts')
state = importlib.import_module(PACKAGE + '.hem_pi_state')
Registry = importlib.import_module(PACKAGE + '.hem_state').Registry
common = importlib.import_module(PACKAGE + '.hem_common')
assets_module = importlib.import_module(PACKAGE + '.hem_pi_assets')
from tests import hem_server_cli_test as fixtures


class ProjectionTest(unittest.TestCase):
    def test_saved_definitions_sorted_distinct_and_active_endpoint(self):
        target = dict(target='/unused', integration_id=str(uuid.uuid4()))
        rows = [dict(id=name, saved_settings={'listen':listen}, launch_token=None, endpoint=None)
                for name,listen in [('a-b','127.0.0.1:8080'),('a.b','[::1]:8081'),('A','127.0.0.1:8082')]]
        rows[0].update(launch_token='launch', endpoint='http://127.0.0.1:8090')
        result = endpoints.project_endpoints('/tmp',target,rows)
        self.assertEqual([r['server'] for r in result['servers']], ['A','a-b','a.b'])
        self.assertEqual(len({r['provider'] for r in result['servers']}),3)
        self.assertEqual(result['servers'][1]['base_url'],'http://127.0.0.1:8090/v1')
        self.assertEqual(result['servers'][2]['base_url'],'http://[::1]:8081/v1')
        self.assertNotIn('models',json.dumps(result))
        for row in result['servers']:
            self.assertEqual(row['native_profile'],contracts.NATIVE_PROFILE)
        rows[0].update(launch_token=None, endpoint=None)
        self.assertEqual(endpoints.project_endpoints('/tmp',target,rows)['servers'][1]['base_url'],
                         'http://127.0.0.1:8080/v1')

    def test_invalid_saved_origin_refused(self):
        for listen in ('0.0.0.0:90','localhost:90','127.0.0.1:0','user@127.0.0.1:90','127.0.0.1:90/path'):
            with self.assertRaises(common.InvalidInput):
                endpoints.project_endpoints('/tmp', {'integration_id':'id'},
                    [dict(id='one', saved_settings=dict(listen=listen))])

    def test_count_and_name_are_bounded_by_document_bytes_only(self):
        target=dict(integration_id=str(uuid.uuid4()))
        servers=[dict(id=f'server-{i:03}',saved_settings=dict(listen='127.0.0.1:8000')) for i in range(129)]
        servers.append(dict(id='long-'+'x'*300,saved_settings=dict(listen='127.0.0.1:8001')))
        document=endpoints.project_endpoints('/tmp',target,servers)
        self.assertEqual(len(document['servers']),130)
        self.assertEqual(document['servers'][0]['server'],servers[-1]['id'])

    def test_exact_serialized_reader_boundary_includes_escapes_and_newline(self):
        files=importlib.import_module(PACKAGE+'.hem_pi_files')
        target=dict(integration_id=str(uuid.uuid4()))
        # Find a large valid name just below the byte limit. A few extra root
        # characters then give an exact one-MiB document without filesystem I/O.
        size=449000
        row=dict(id='x'*size,saved_settings=dict(listen='127.0.0.1:8000'))
        document=endpoints.project_endpoints('/tmp',target,[row])
        remaining=1024*1024-len(files.encoded(document))
        self.assertGreater(remaining,0)
        # Avoid creating a huge path component: increase the name toward its
        # 7/3 encoded expansion, leaving only a handful of root characters.
        size+=max(0,(remaining-10)*3//7)
        row['id']='x'*size
        document=endpoints.project_endpoints('/tmp',target,[row])
        remaining=1024*1024-len(files.encoded(document))
        self.assertGreater(remaining,0)
        self.assertLess(remaining,20)
        root='/tmp/'+'r'*(remaining-1)
        document=endpoints.project_endpoints(root,target,[row])
        self.assertEqual(len(files.encoded(document)),1024*1024)
        self.assertEqual(endpoints._encoded(document),files.encoded(document))
        with self.assertRaisesRegex(common.InvalidInput,'1 MiB'):
            endpoints.project_endpoints(root+'r',target,[row])
        unicode_document=endpoints.project_endpoints('/tmp/é',target,[])
        self.assertIn(b'\\u00e9',endpoints._encoded(unicode_document))
        self.assertTrue(endpoints._encoded(unicode_document).endswith(b'\n'))


class PublicationTest(unittest.TestCase):
    """Uses the installer's frozen shared helper contract, no installer mocks."""
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.registry = Registry(self.root/'state')
        self.target = self.root/'pi'
        self.identity = dict(target=str(self.target),integration_id=str(uuid.uuid4()))
        self.registry.create_server('one',saved_settings=dict(device='task',listen='127.0.0.1:12345'))

    def configured(self):
        self.target.mkdir(mode=0o700)
        (self.target/'extensions').mkdir(mode=0o700)
        assets = assets_module.bundled_assets()
        for relative,data in assets.items():
            path=self.target/relative
            path.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
            path.write_bytes(data)
            path.chmod(0o600)
        manifest=dict(schema_version=contracts.MANIFEST_SCHEMA_VERSION,
            integration_version=contracts.INTEGRATION_VERSION,
            integration_id=self.identity['integration_id'],enabled=True,
            hem_root=str(self.registry.root),target=str(self.target),
            extension_version=contracts.EXTENSION_VERSION,native_profile=contracts.NATIVE_PROFILE,
            native_commit=contracts.NATIVE_COMMIT,pi_package=contracts.PI_PACKAGE,
            pi_versions=list(contracts.PI_VERSIONS),
            endpoint_file=str(self.registry.root/contracts.ENDPOINT_PATH),
            artifacts=assets_module.asset_digests(assets))
        (self.target/'loom-hem').mkdir(mode=0o700)
        (self.target/contracts.MANIFEST_PATH).write_text(json.dumps(manifest))
        (self.target/contracts.MANIFEST_PATH).chmod(0o600)
        state.set_target(self.registry,self.identity)

    def test_absent_and_reservation_only_no_creation(self):
        self.assertFalse(endpoints.publish_endpoints(self.registry,common.Deadline.after())['changed'])
        state.set_target(self.registry,self.identity)
        self.assertFalse(endpoints.publish_endpoints(self.registry,common.Deadline.after())['changed'])
        self.assertFalse(self.target.exists())
        self.assertFalse((self.registry.root/contracts.ENDPOINT_PATH).exists())

    def test_publish_saved_stopped_definition_idempotent_and_shutdown_survives(self):
        self.configured()
        first=endpoints.publish_endpoints(self.registry,common.Deadline.after())
        path=self.registry.root/contracts.ENDPOINT_PATH
        self.assertTrue(first['changed'])
        self.assertEqual(json.loads(path.read_text())['servers'][0]['base_url'],'http://127.0.0.1:12345/v1')
        timestamp=path.stat().st_mtime_ns
        self.assertFalse(endpoints.publish_endpoints(self.registry,common.Deadline.after())['changed'])
        self.assertEqual(path.stat().st_mtime_ns,timestamp)
        self.registry.create_server('new',saved_settings=dict(device='task',listen='[::1]:12346'))
        update=endpoints.publish_endpoints(self.registry,common.Deadline.after())
        self.assertEqual(update['new_providers'],[contracts.provider_id('new')])
        self.assertIsNone(self.registry.get_server('one')['launch_token'])

    def test_foreign_projection_and_edited_dependency_refused_without_overwrite(self):
        self.configured()
        endpoints.publish_endpoints(self.registry,common.Deadline.after())
        path=self.registry.root/contracts.ENDPOINT_PATH
        foreign=json.loads(path.read_text()); foreign['integration_id']=str(uuid.uuid4())
        path.write_text(json.dumps(foreign)); before=path.read_bytes()
        with self.assertRaises(common.HemError):
            endpoints.publish_endpoints(self.registry,common.Deadline.after())
        self.assertEqual(path.read_bytes(),before)
        (self.target/'extensions/loom-hem/index.ts').write_text('edited')
        with self.assertRaises(common.HemError):
            endpoints.publish_endpoints(self.registry,common.Deadline.after())
        self.assertEqual(path.read_bytes(),before)

    def test_disabled_and_pending_never_publish_or_clear_reservation(self):
        self.configured()
        manifest_path=self.target/contracts.MANIFEST_PATH
        manifest=json.loads(manifest_path.read_text())
        manifest['enabled']=False
        manifest_path.write_text(json.dumps(manifest))
        pending=self.target/'loom-hem/.pending'
        pending.mkdir(mode=0o700)
        self.assertFalse(endpoints.publish_endpoints(self.registry,common.Deadline.after())['changed'])
        self.assertFalse((self.registry.root/contracts.ENDPOINT_PATH).exists())
        self.assertEqual(state.get_target(self.registry),self.identity)
        manifest['enabled']=True
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaises(common.HemError):
            endpoints.publish_endpoints(self.registry,common.Deadline.after())
        self.assertEqual(state.get_target(self.registry),self.identity)

    def test_oversized_projection_never_replaces_last_valid_map(self):
        self.configured()
        endpoints.publish_endpoints(self.registry,common.Deadline.after())
        path=self.registry.root/contracts.ENDPOINT_PATH
        before=path.read_bytes()
        self.registry.create_server('x'*450000,saved_settings=dict(device='task',listen='127.0.0.1:12346'))
        with self.assertRaisesRegex(common.InvalidInput,'1 MiB'):
            endpoints.publish_endpoints(self.registry,common.Deadline.after())
        self.assertEqual(path.read_bytes(),before)
        self.assertEqual(state.get_target(self.registry),self.identity)
        payload=endpoints.update_result(self.registry,common.Deadline.after(),common.result('running',completed=True))
        self.assertEqual(payload['state'],'running')
        self.assertTrue(payload['completed'])
        self.assertEqual(payload['pi_update'],'failed')
        self.assertEqual(payload['pi_repair'],'loom-hem pi configure')
        self.assertEqual(path.read_bytes(),before)
        process=importlib.import_module(PACKAGE+'.hem_process')
        with patch.object(process.subprocess,'Popen',side_effect=AssertionError('must not launch')):
            with self.assertRaisesRegex(common.InvalidInput,'1 MiB'):
                process.ProcessManager(self.registry).start('one',self.registry.get_server('one')['saved_settings'],common.Deadline.after())
        self.assertIsNone(self.registry.get_server('one')['launch_token'])
        self.assertEqual(path.read_bytes(),before)

    def test_concurrent_publication_serializes_and_respects_deadline(self):
        import subprocess
        import sys
        self.configured()
        code=(f'from {PACKAGE}.hem_pi_endpoints import publish_endpoints; '
              f'from {PACKAGE}.hem_common import Deadline; '
              f'from {PACKAGE}.hem_state import Registry; '
              'import sys; r=Registry(sys.argv[1],deadline=Deadline.after(2)); '
              'publish_endpoints(r,Deadline.after(2))')
        children=[subprocess.Popen([sys.executable,'-c',code,str(self.registry.root)],
                                   stdout=subprocess.PIPE,stderr=subprocess.PIPE) for _ in range(3)]
        try:
            for child in children:
                stdout,stderr=child.communicate(timeout=5)
                self.assertEqual(child.returncode,0,stderr)
        finally:
            for child in children:
                if child.poll() is None:
                    child.terminate()
                child.communicate(timeout=2)
        path=self.registry.root/contracts.ENDPOINT_PATH
        self.assertEqual(len(json.loads(path.read_text())['servers']),1)
        child=None
        with state.integration_lock(self.registry.root,common.Deadline.after()):
            child=subprocess.Popen([sys.executable,'-c',code.replace('after(2)','after(.1)'),str(self.registry.root)],
                                   stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            stdout,stderr=child.communicate(timeout=2)
            self.assertNotEqual(child.returncode,0)
            self.assertIn(b'deadline',stderr)


class LifecyclePublicationTest(fixtures.ServerCliTest):
    # Inherited native ELF fixture and public subprocess lifecycle gates.
    def inject(self, code):
        self.launcher[2]=self.launcher[2].replace('raise SystemExit',code+'; raise SystemExit')

    def test_prelaunch_failure_does_not_spawn_or_reserve(self):
        self.inject(f'from {PACKAGE} import hem_pi_endpoints; '
                    "hem_pi_endpoints.publish_endpoints=lambda *a: (_ for _ in ()).throw(hem_pi_endpoints.HemError('publish failed'))")
        payload,_=self.invoke('start','--name','pi-prelaunch','--device','task','--listen',
                              f'127.0.0.1:{fixtures.available_port()}',expected=1)
        record=Registry(self.state).get_server('pi-prelaunch')
        self.assertIsNone(record['launch_token'])
        self.assertIsNone(record['process'])
        self.assertFalse(payload['completed'])

    def test_secondary_failure_keeps_start_status_reuse_shutdown_success(self):
        # Pass mandatory first publication; fail only subsequent start repair.
        self.inject(f'from {PACKAGE} import hem_pi_endpoints; '
                    "counter=[0]; "
                    "exec(\"def publish(*args):\\n counter[0]+=1\\n if counter[0]>1: raise hem_pi_endpoints.HemError('secondary failed')\\n return dict(changed=False,new_providers=[],pi_update='unchanged')\"); "
                    "hem_pi_endpoints.publish_endpoints=publish")
        payload,stderr=self.start()
        self.assertEqual(payload['state'],'running')
        self.assertEqual(payload['pi_update'],'failed')
        self.assertEqual(payload['pi_repair'],'loom-hem pi configure')
        self.assertIn('native result unchanged',stderr)
        self.assertIn('curl ',stderr)
        # Subsequent successful native operations each invoke only secondary publication.
        self.launcher[2]=self.launcher[2].replace('counter=[0]','counter=[1]')
        for args in [('status','one'),('reconcile','one'),('start','one'),('shutdown','one')]:
            result,stderr=self.invoke(*args)
            self.assertTrue(result['completed'])
            self.assertEqual(result['pi_update'],'failed')
            self.assertNotIn('retry',stderr)
        self.assertIsNone(Registry(self.state).get_server('one')['process'])

    def test_real_saved_projection_precedes_spawn_and_survives_shutdown(self):
        # Join the actual installer and projector; only payload selection and
        # a pre-spawn assertion are injected. Native lifecycle stays real.
        installer=importlib.import_module(PACKAGE+'.hem_pi')
        self.assertTrue(installer.configure_pi(self.state,self.root/'pi',common.Deadline.after())['configured'])
        projection=self.state/contracts.ENDPOINT_PATH
        assertion=("def checked_spawn(*a, **kw):\n import json\n from pathlib import Path\n "
                   f"document=json.loads(Path({str(projection)!r}).read_text())\n "
                   "assert document['servers'][0]['server']=='one'\n return original_popen(*a,**kw)")
        self.inject(f'from {PACKAGE} import hem_pi_endpoints; '
                    'original_popen=hem_server_cli.ProcessManager.func.__init__.__globals__["subprocess"].Popen; '
                    f'exec({assertion!r}); '
                    'hem_server_cli.ProcessManager.func.__init__.__globals__["subprocess"].Popen=checked_spawn')
        first,diagnostic=self.start()
        self.assertEqual(first['state'],'running')
        self.assertIn('/reload',diagnostic)
        document=json.loads(projection.read_text())
        self.assertEqual(document['servers'][0]['base_url'],first['endpoint']+'/v1')
        self.invoke('shutdown','one')
        self.assertEqual(json.loads(projection.read_text()),document)
        self.assertEqual(self.invoke('status','one')[0]['state'],'stopped')
        # Restart repairs an absent derived map before exposing a new process.
        projection.unlink()
        self.invoke('start','one')
        self.assertEqual(json.loads(projection.read_text()),document)
        self.invoke('shutdown','one')

    def test_concurrent_configure_and_starts_publish_all_saved_definitions(self):
        import subprocess
        import sys
        installer=importlib.import_module(PACKAGE+'.hem_pi')
        target=self.root/'pi'
        installer.configure_pi(self.state,target,common.Deadline.after())
        code=(f'from {PACKAGE}.hem_pi import configure_pi; '
              f'from {PACKAGE}.hem_common import Deadline; '
              'import sys; from pathlib import Path; '
              '[configure_pi(Path(sys.argv[1]),Path(sys.argv[2]),Deadline.after(3)) for _ in range(4)]')
        child=subprocess.Popen([sys.executable,'-c',code,str(self.state),str(target)],
                               env=self.environment,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        try:
            self.start('one')
            self.start('two')
            stdout,stderr=child.communicate(timeout=5)
            self.assertEqual(child.returncode,0,stderr)
            document=json.loads((self.state/contracts.ENDPOINT_PATH).read_text())
            self.assertEqual([r['server'] for r in document['servers']],['one','two'])
            self.assertEqual(document['integration_id'],state.get_target(Registry(self.state))['integration_id'])
            self.invoke('shutdown','one')
            self.assertEqual(self.invoke('status','two')[0]['state'],'running')
            self.invoke('shutdown','two')
        finally:
            if child.poll() is None:
                child.terminate()
            child.communicate(timeout=2)


if __name__=='__main__':
    unittest.main()
