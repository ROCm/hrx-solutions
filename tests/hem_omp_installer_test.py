"""Surgical activation ownership and recoverable multi-file publication."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from binding.python import hem_omp as installer
from binding.python.hem_omp_yaml import scan
from binding.python.hem_common import Deadline, InvalidInput


class OMPInstallerTest(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.base=Path(self.temp.name);self.root=self.base/'root';self.target=self.base/'agent'
        self.providers=['loom-hem-YQ','loom-hem-Yg']

    def run_install(self,**kw):
        # This suite isolates file/YAML transactions; public prerequisite parity
        # is exercised through hem_omp_cli_test's CLI and direct owner calls.
        assets=installer.bundled_assets()
        if kw.get('check'):
            return installer._check_owned(self.root,self.target,assets,self.providers)
        disable=kw.get('disable',False)
        inspected=installer._preflight(self.root,self.target,assets,self.providers,disable)
        return installer._apply_owned(self.root,self.target,Deadline.after(),assets,inspected,
                                      providers=self.providers,disable=disable)

    def files(self):
        return {str(p.relative_to(self.base)):(p.read_bytes(),p.stat().st_mtime_ns) for p in self.base.rglob('*') if p.is_file()}

    def yaml(self,text,name='models.yml'):
        self.target.mkdir(mode=0o700,exist_ok=True);(self.target/name).write_text(text);(self.target/name).chmod(0o600)

    def install_prior_generation(self):
        # Frozen reviewed source, not new discovery with its helper simply removed.
        fixture=json.loads((Path(__file__).parent/'client_protocol_legacy.json').read_text())
        self.assertEqual(fixture['source'],'344569eaab47ef07b3c7f1d5da0024f040d88bdb')
        assets=installer.bundled_assets();del assets['native_protocol.ts']
        import hashlib
        for name,record in fixture['assets']['omp_extension'].items():
            data=record['text'].encode()
            self.assertEqual(hashlib.sha256(data).hexdigest(),record['sha256'])
            assets[name]=data
        for name,expected in fixture['unchanged_sha256']['omp_extension'].items():
            self.assertEqual(hashlib.sha256(assets[name]).hexdigest(),expected)
        with patch.object(installer,'bundled_assets',return_value=assets):self.run_install()
        directory=self.target/installer.PLUGIN
        return json.loads((directory/'manifest.json').read_bytes())

    def test_prior_exact_generation_passive_stale_and_runtime_free_disable(self):
        self.install_prior_generation()
        directory=self.target/installer.PLUGIN
        self.assertFalse((directory/'native_protocol.ts').exists())
        before=self.files()
        self.assertEqual(self.run_install(check=True)['state'],'outdated')
        self.assertEqual(before,self.files())
        with patch.object(installer,'inspect_runtime',side_effect=AssertionError('disable must not inspect client')):
            installer.configure_omp(self.root,self.target,Deadline.after(),disable=True)
        self.assertFalse(directory.exists())

    def test_prior_to_shared_generation_upgrade_recovers_after_companion_publication(self):
        old=self.install_prior_generation()
        directory=self.target/installer.PLUGIN
        original=installer.atomic_replace
        def interrupted(path,*args,**kwargs):
            answer=original(path,*args,**kwargs)
            if path.name=='native_protocol.ts':raise OSError('after new companion publication')
            return answer
        with patch.object(installer,'atomic_replace',side_effect=interrupted):
            with self.assertRaises(OSError):self.run_install()
        before=self.files()
        self.assertEqual(self.run_install(check=True)['state'],'pending')
        self.assertEqual(before,self.files())
        self.run_install()
        current=json.loads((directory/'manifest.json').read_bytes())
        self.assertEqual(current['integration_id'],old['integration_id'])
        self.assertNotEqual(current['generation'],old['generation'])
        self.assertEqual(set(current['artifacts']),set(installer.ARTIFACTS))
        self.assertTrue(self.run_install(check=True)['configured'])
        self.assertFalse((directory/'pending.json').exists())

    def test_neither_partial_prior_set_nor_missing_current_companion_is_owned(self):
        self.install_prior_generation()
        directory=self.target/installer.PLUGIN
        path=directory/'discovery.ts';data=path.read_bytes();path.unlink()
        self.assertEqual(self.run_install(check=True)['state'],'conflict')
        with self.assertRaises(InvalidInput):self.run_install(disable=True)
        path.write_bytes(data+b'// external edit');path.chmod(0o600)
        self.assertEqual(self.run_install(check=True)['state'],'conflict')
        with self.assertRaises(InvalidInput):self.run_install(disable=True)
        path.write_bytes(data)
        self.run_install()
        (directory/'native_protocol.ts').unlink()
        self.assertEqual(self.run_install(check=True)['state'],'conflict')
        with self.assertRaises(InvalidInput):self.run_install(disable=True)

    def test_shared_fixture(self):
        for case in json.loads((Path(__file__).parent/'omp_activation_cases.json').read_text()):
            with self.subTest(case=case['name']):
                if case.get('error'):
                    with self.assertRaises(InvalidInput):scan(case['text'])
                else:self.assertEqual({k:v['valid'] for k,v in scan(case['text']).items()},case['providers'])

    def test_idempotent_and_owned_file_removal(self):
        self.assertTrue(self.run_install()['changed']);before=self.files()
        self.assertFalse(self.run_install()['changed']);self.assertEqual(before,self.files())
        self.assertTrue(self.run_install(check=True)['configured']);self.assertEqual(before,self.files())
        self.run_install(disable=True);self.assertFalse((self.target/'models.yml').exists())
        self.assertFalse((self.target/installer.PLUGIN).exists())
        self.assertFalse(self.run_install(disable=True)['changed'])

    def test_fallback_heading_restore_and_foreign_preservation(self):
        original='# comment\nproviders: {} # empty\nsettings:\n  custom: value\n'
        self.yaml(original,'models.yaml');self.run_install()
        self.assertFalse((self.target/'models.yml').exists())
        self.assertIn('providers: # empty\n',(self.target/'models.yaml').read_text())
        self.run_install(disable=True);self.assertEqual((self.target/'models.yaml').read_text(),original)

    def test_unterminated_heading_restored_exactly(self):
        for text in ('providers:', 'providers: # keep', 'providers: {} # keep',
                     'providers: {} # preserve literal {} here\n',
                     'providers: {} # keep\r\n', 'providers: {}\r\n', 'providers: # keep\r\n'):
            with self.subTest(text=text):
                self.yaml(text);self.run_install()
                if 'literal {}' in text:self.assertIn(b'# preserve literal {} here',(self.target/'models.yml').read_bytes())
                self.assertIn('# keep' if '# keep' in text else 'providers:',(self.target/'models.yml').read_text())
                before=self.files();self.assertTrue(self.run_install(check=True)['configured']);self.assertEqual(before,self.files())
                self.run_install(disable=True);self.assertEqual((self.target/'models.yml').read_bytes(),text.encode())

    def test_borrowed_entry_never_deleted(self):
        original='providers:\n  loom-hem-YQ:\n    auth: none # mine\n  foreign:\n    apiKey: my-secret\n'
        self.yaml(original);self.run_install();self.run_install(disable=True)
        self.assertEqual((self.target/'models.yml').read_text(),original)

    def test_foreign_comments_and_new_provider_survive_disable(self):
        self.run_install();path=self.target/'models.yml'
        path.write_text('# inserted\n'+path.read_text()+'  foreign:\n    auth: none\n')
        self.run_install(disable=True)
        self.assertEqual(path.read_text(),'# inserted\nproviders:\n  foreign:\n    auth: none\n')

    def test_edited_activation_and_asset_are_retained(self):
        self.run_install();path=self.target/'models.yml'
        path.write_text(path.read_text().replace('auth: none','auth: key',1));before=self.files()
        self.assertEqual(self.run_install(check=True)['state'],'conflict')
        with self.assertRaises(InvalidInput):self.run_install(disable=True)
        self.assertEqual(before,self.files())

    def test_new_server_requires_configure_and_generation_changes_once(self):
        self.run_install();manifest=self.target/installer.PLUGIN/'manifest.json';old=manifest.read_bytes()
        self.providers.append('loom-hem-Yw');self.assertEqual(self.run_install(check=True)['state'],'outdated')
        self.run_install();self.assertNotEqual(old,manifest.read_bytes());before=self.files()
        self.run_install();self.assertEqual(before,self.files())

    def test_conflicts_and_legacy_never_write(self):
        for text in ('providers:\n  loom-hem-YQ:\n    auth: none\n    baseUrl: http://localhost:9\n', 'providers:\n  local: {auth: none}\n'):
            self.yaml(text);before=self.files()
            with self.assertRaises(InvalidInput):self.run_install()
            self.assertEqual(before,self.files())
        (self.target/'models.yml').unlink();self.yaml('{}','models.json');before=self.files()
        with self.assertRaises(InvalidInput):self.run_install()
        self.assertEqual(before,self.files())

    def test_new_primary_after_fallback_is_conflict(self):
        self.yaml('providers: {}\n','models.yaml');self.run_install();self.yaml('providers: {}\n')
        before=self.files();self.assertEqual(self.run_install(check=True)['state'],'conflict');self.assertEqual(before,self.files())

    def test_symlink_refused(self):
        other=self.base/'other';other.write_text('providers: {}\n')
        self.target.mkdir(mode=0o700);(self.target/'models.yml').symlink_to(other)
        with self.assertRaises(InvalidInput):self.run_install()
        self.assertEqual(other.read_text(),'providers: {}\n')

    def test_every_publication_boundary_install_and_disable(self):
        # Inject after each durable mutation, including final pending removal.
        for disabling in (False,True):
            count=0
            while True:
                count+=1
                with tempfile.TemporaryDirectory(dir=self.base) as temp:
                    self.target=Path(temp)/'agent'
                    if disabling:self.run_install()
                    calls=0;original_replace=installer.atomic_replace;original_remove=installer.remove_owned
                    def wrap(operation):
                        def interrupted(*args,**kwargs):
                            nonlocal calls
                            value=operation(*args,**kwargs);calls+=1
                            if calls==count:raise OSError('injected after durable mutation')
                            return value
                        return interrupted
                    with patch.object(installer,'atomic_replace',wrap(original_replace)),patch.object(installer,'remove_owned',wrap(original_remove)):
                        try:self.run_install(disable=disabling)
                        except OSError:pass
                    # A repeated same operation must safely converge.
                    response=self.run_install(disable=disabling)
                    self.assertEqual(response['state'],'disabled' if disabling else 'configured')
                    self.assertFalse((self.target/installer.PLUGIN/'pending.json').exists())
                    if calls<count:break
                    self.assertLess(count,40)

    def test_pending_contains_no_foreign_credentials_and_external_edit_refuses(self):
        self.yaml('providers:\n  other:\n    apiKey: PRIVATE-CREDENTIAL\n')
        original=installer.atomic_replace
        def interrupt(path,*a,**kw):
            original(path,*a,**kw)
            if path.name=='pending.json':raise OSError('interrupted')
        with patch.object(installer,'atomic_replace',interrupt):
            with self.assertRaises(OSError):self.run_install()
        pending=(self.target/installer.PLUGIN/'pending.json').read_bytes()
        self.assertNotIn(b'PRIVATE-CREDENTIAL',pending)
        path=self.target/'models.yml';path.write_text(path.read_text()+'# external edit\n');before=self.files()
        with self.assertRaises(InvalidInput):self.run_install()
        self.assertEqual(before,self.files())

    def test_malformed_manifest_and_foreign_root_refused(self):
        self.run_install();manifest=self.target/installer.PLUGIN/'manifest.json'
        original=manifest.read_bytes();value=json.loads(original);value['native_activation']={}
        manifest.write_text(json.dumps(value))
        self.assertEqual(self.run_install(check=True)['state'],'conflict')
        manifest.write_bytes(original)
        with self.assertRaises(InvalidInput):installer.configure_omp(self.base/'other-root',self.target,Deadline.after(),disable=True)

    def test_native_partial_stage_and_stale_compare(self):
        original=installer.atomic_replace
        def interrupt(path,data,**kw):
            if path.name=='models.yml':
                temporary=installer.staged_path(path);temporary.write_bytes(data[:12]);temporary.chmod(0o600)
                raise OSError('native partial stage')
            return original(path,data,**kw)
        with patch.object(installer,'atomic_replace',interrupt):
            with self.assertRaises(OSError):self.run_install()
        self.run_install();self.assertTrue(self.run_install(check=True)['configured'])

    def test_first_plan_torn_stage_is_retained_with_concrete_guidance(self):
        for partial in (b'', b'{"schema_version":'):
            with tempfile.TemporaryDirectory(dir=self.base) as temp:
                self.target=Path(temp)/'agent'
                original=installer.atomic_replace
                def torn(path,data,**kw):
                    if path.name=='pending.json':
                        installer.private_path(path.parent,create=True)
                        temporary=installer.staged_path(path);temporary.write_bytes(partial);temporary.chmod(0o600)
                        raise OSError('first write torn')
                    return original(path,data,**kw)
                with patch.object(installer,'atomic_replace',torn):
                    with self.assertRaises(OSError):self.run_install()
                before=self.files();response=self.run_install(check=True)
                self.assertEqual(response['state'],'pending');self.assertIn('manually move',response['next_action'])
                for disabling in (False,True):
                    with self.assertRaisesRegex(InvalidInput,'Unpublished OMP plan temporary'):
                        self.run_install(disable=disabling)
                self.assertEqual(before,self.files());self.assertFalse((self.target/'models.yml').exists())
                self.assertFalse((self.target/installer.PLUGIN/'index.ts').exists())
                installer.staged_path(self.target/installer.PLUGIN/'pending.json').unlink()
                self.assertTrue(self.run_install()['configured'])

    def test_partial_stage_recovery(self):
        original=installer.atomic_replace
        def interrupt(path,data,**kw):
            if path.name=='index.ts':
                installer.staged_path(path).write_bytes(data[:20]);installer.staged_path(path).chmod(0o600)
                raise OSError('partial stage')
            return original(path,data,**kw)
        with patch.object(installer,'atomic_replace',interrupt):
            with self.assertRaises(OSError):self.run_install()
        self.run_install();self.assertTrue(self.run_install(check=True)['configured'])


if __name__=='__main__':unittest.main()
