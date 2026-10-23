"""Toolkit ownership/recovery tests; no vendor runtime or inference certification."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from binding.python.integrations.opencode import hem_opencode as installer
from binding.python.hem_common import Deadline, InvalidInput


def configure_files(root, target, deadline, *, check=False, disable=False):
    # Isolate transaction/recovery tests; public qualification and endpoint
    # ownership are covered by the CLI/direct-owner suite.
    for path in (root, target):
        if not path.is_absolute() or path.resolve() != path:
            raise InvalidInput('test transaction paths must be canonical and absolute')
        installer.private_path(path)
    assets = installer.bundled_assets()
    if check:
        return installer._check_owned(root, target, assets)
    installer._preflight(root, target)
    return installer._apply_owned(root, target, deadline, assets, disable=disable)


class InstallerTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.root, self.target = self.base / 'state', self.base / 'config'
        self.root.mkdir(mode=0o700)
        self.directory = self.target / installer.PLUGIN

    def run_install(self, **kwargs):
        return configure_files(self.root, self.target, Deadline.after(5), **kwargs)

    def contents(self):
        return {str(path.relative_to(self.base)): (path.lstat().st_mode, path.lstat().st_mtime_ns,
                path.read_bytes() if path.is_file() and not path.is_symlink() else None)
                for path in self.base.rglob('*')}

    def install_prior_generation(self):
        # Frozen reviewed source, not new discovery with its helper simply removed.
        fixture=json.loads((Path(__file__).parent/'client_protocol_legacy.json').read_text())
        self.assertEqual(fixture['source'],'344569eaab47ef07b3c7f1d5da0024f040d88bdb')
        assets=installer.bundled_assets();del assets['native_protocol.ts']
        import hashlib
        for name,record in fixture['assets']['opencode_plugin'].items():
            data=record['text'].encode()
            self.assertEqual(hashlib.sha256(data).hexdigest(),record['sha256'])
            assets[name]=data
        for name,expected in fixture['unchanged_sha256']['opencode_plugin'].items():
            self.assertEqual(hashlib.sha256(assets[name]).hexdigest(),expected)
        with patch.object(installer,'bundled_assets',return_value=assets):self.run_install()
        directory=self.target/installer.PLUGIN
        return json.loads((directory/'manifest.json').read_bytes())

    def test_prior_exact_generation_passive_stale_and_runtime_free_disable(self):
        self.install_prior_generation()
        directory=self.target/installer.PLUGIN
        self.assertFalse((directory/'native_protocol.ts').exists())
        before=self.contents()
        self.assertEqual(self.run_install(check=True)['state'],'outdated')
        self.assertEqual(before,self.contents())
        with patch.object(installer,'inspect_runtime',side_effect=AssertionError('disable must not inspect client')):
            installer.configure_opencode(self.root,self.target,Deadline.after(),disable=True)
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
        before=self.contents()
        self.assertEqual(self.run_install(check=True)['state'],'pending')
        self.assertEqual(before,self.contents())
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

    def test_absent_passive_and_idempotent(self):
        before = self.contents()
        self.assertEqual(self.run_install(check=True)['state'], 'absent')
        self.assertEqual(self.contents(), before)
        self.assertTrue(self.run_install()['changed'])
        before = self.contents()
        self.assertTrue(self.run_install(check=True)['configured'])
        self.assertEqual(self.contents(), before)
        self.assertFalse(self.run_install()['changed'])
        self.assertEqual(self.contents(), before)
        manifest = json.loads((self.directory / 'manifest.json').read_bytes())
        self.assertEqual(set(manifest['artifacts']), set(installer.ARTIFACTS))

    def test_disable_preserves_unrelated_and_root(self):
        self.run_install()
        unrelated = self.target / 'opencode.jsonc'
        unrelated.write_text('// keep credentials and defaults\n{}')
        root_file = self.root / 'unrelated'; root_file.write_text('keep')
        self.assertTrue(self.run_install(disable=True)['changed'])
        self.assertFalse(self.directory.exists())
        self.assertEqual(unrelated.read_text(), '// keep credentials and defaults\n{}')
        self.assertEqual(root_file.read_text(), 'keep')
        self.assertFalse(self.run_install(disable=True)['changed'])

    def test_foreign_partial_edited_unknown_and_symlink_refused(self):
        cases = ['foreign', 'edited', 'unknown', 'symlink', 'root']
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory(dir=self.base) as tmp:
                target = Path(tmp) / 'config'
                def run(**kw): return configure_files(self.root, target, Deadline.after(5), **kw)
                if case == 'foreign':
                    installer.private_path(target / installer.PLUGIN, create=True)
                    (target / installer.PLUGIN / 'index.ts').write_text('foreign')
                else:
                    run()
                    if case == 'edited': (target / installer.PLUGIN / 'index.ts').write_text('edited')
                    if case == 'unknown': (target / installer.PLUGIN / 'user.ts').write_text('user')
                    if case == 'symlink':
                        (target / installer.PLUGIN / 'index.ts').unlink()
                        (target / installer.PLUGIN / 'index.ts').symlink_to(self.root / 'other')
                    if case == 'root':
                        other = self.base / 'other-root'; other.mkdir(exist_ok=True)
                        with self.assertRaises(InvalidInput): configure_files(other, target, Deadline.after(5), disable=True)
                        continue
                before = {str(p): p.lstat().st_mtime_ns for p in target.rglob('*')}
                self.assertEqual(run(check=True)['state'], 'conflict')
                for disable in (False, True):
                    with self.assertRaises(InvalidInput): run(disable=disable)
                self.assertEqual(before, {str(p): p.lstat().st_mtime_ns for p in target.rglob('*')})

    def test_crash_after_each_replace_recovers_configure_and_disable(self):
        for disable in (False, True):
            # pending plan + seven assets + marker removal, covering every durable boundary.
            for boundary in range(1, 10):
                with self.subTest(disable=disable, boundary=boundary), tempfile.TemporaryDirectory(dir=self.base) as tmp:
                    target = Path(tmp) / 'config'
                    def run(**kw): return configure_files(self.root, target, Deadline.after(5), **kw)
                    if disable: run()
                    counter = [0]
                    replace, remove = installer.atomic_replace, installer.remove_owned
                    def after(operation, *args, **kw):
                        answer = operation(*args, **kw)
                        counter[0] += 1
                        if counter[0] == boundary: raise RuntimeError('crash')
                        return answer
                    with patch.object(installer, 'atomic_replace', side_effect=lambda *a, **k: after(replace, *a, **k)), patch.object(installer, 'remove_owned', side_effect=lambda *a, **k: after(remove, *a, **k)):
                        try: run(disable=disable)
                        except RuntimeError: pass
                    before = {str(p): (p.lstat().st_mtime_ns, p.read_bytes()) for p in (target / installer.PLUGIN).glob('*') if p.is_file()}
                    check = run(check=True)
                    self.assertEqual(before, {str(p): (p.lstat().st_mtime_ns, p.read_bytes()) for p in (target / installer.PLUGIN).glob('*') if p.is_file()})
                    run(disable=disable)
                    self.assertEqual(run(check=True)['state'], 'absent' if disable else 'configured')

    def test_interrupted_update_retains_identity_previous_bytes_and_user_edits(self):
        self.run_install()
        manifest = json.loads((self.directory / 'manifest.json').read_bytes())
        assets = installer.bundled_assets(); assets['discovery.ts'] += b'\n// upgrade\n'
        original = installer.atomic_replace
        def crash(path, data, **kw):
            answer = original(path, data, **kw)
            if path.name == 'discovery.ts': raise RuntimeError('crash')
            return answer
        with patch.object(installer, 'bundled_assets', return_value=assets), patch.object(installer, 'atomic_replace', side_effect=crash):
            with self.assertRaises(RuntimeError): self.run_install()
        plan = installer.document((self.directory / 'pending.json').read_bytes())
        self.assertIn('previous', plan)
        (self.directory / 'index.ts').write_text('later user edit')
        before = self.contents()
        with self.assertRaises(InvalidInput): self.run_install()
        self.assertEqual(self.contents(), before)
        # Restore only the exact previous known bytes for this test.
        import base64
        (self.directory / 'index.ts').write_bytes(base64.b64decode(plan['previous']['index.ts']))
        with patch.object(installer, 'bundled_assets', return_value=assets): self.run_install()
        new = json.loads((self.directory / 'manifest.json').read_bytes())
        self.assertEqual(manifest['integration_id'], new['integration_id'])
        self.assertNotEqual(manifest['generation'], new['generation'])
        self.assertTrue(self.run_install(check=True)['configured'] is False)  # current wheel differs

    def test_completed_pending_temp_and_incomplete_or_foreign_are_safe(self):
        desired = installer._desired(self.target, self.root, installer.bundled_assets(), {})
        plan = installer._plan({}, desired, self.target, self.root)
        installer.private_path(self.directory, create=True)
        stage = installer.staged_path(self.directory / 'pending.json')
        stage.write_bytes(plan); stage.chmod(0o600)
        self.assertEqual(self.run_install(check=True)['state'], 'pending')
        self.run_install()
        self.assertFalse(stage.exists())
        self.run_install(disable=True)
        self.directory.mkdir()
        stage.write_bytes(plan[:100]); stage.chmod(0o600)
        before = self.contents()
        with self.assertRaises(InvalidInput): self.run_install()
        self.assertEqual(self.contents(), before)

    def test_staged_owned_prefix_recovery_and_foreign_temp_refusal(self):
        for foreign in (False, True):
            with self.subTest(foreign=foreign), tempfile.TemporaryDirectory(dir=self.base) as tmp:
                target = Path(tmp) / 'config'
                directory = target / installer.PLUGIN
                installer.private_path(directory, create=True)
                desired = installer._desired(target, self.root, installer.bundled_assets(), {})
                plan = installer._plan({}, desired, target, self.root)
                (directory / 'pending.json').write_bytes(plan)
                (directory / 'pending.json').chmod(0o600)
                stage = installer.staged_path(directory / 'index.ts')
                stage.write_bytes(b'foreign' if foreign else desired['index.ts'][:20]); stage.chmod(0o600)
                if foreign:
                    with self.assertRaises(InvalidInput): configure_files(self.root, target, Deadline.after(5))
                    self.assertEqual(stage.read_bytes(), b'foreign')
                else:
                    configure_files(self.root, target, Deadline.after(5))
                    self.assertFalse(stage.exists())

    def test_duplicate_manifest_fields_and_directory_alias_refused(self):
        self.run_install()
        path = self.directory / 'manifest.json'
        path.write_bytes(path.read_bytes().replace(b'"enabled":true', b'"enabled":true,"enabled":true'))
        self.assertEqual(self.run_install(check=True)['state'], 'conflict')
        alias = self.base / 'alias'; alias.symlink_to(self.target)
        with self.assertRaises(InvalidInput): configure_files(self.root, alias, Deadline.after(5))

    def test_empty_pending_presence_blocks_passive_check_without_writes(self):
        self.run_install()
        marker = self.directory / 'pending.json'
        marker.write_bytes(b'')
        marker.chmod(0o600)
        before = self.contents()
        result = self.run_install(check=True)
        self.assertEqual(result['state'], 'pending')
        self.assertTrue(result['pending'])
        self.assertFalse(result['configured'])
        self.assertEqual(result.exit_code, 1)
        self.assertEqual(self.contents(), before)
        with self.assertRaises(InvalidInput): self.run_install(disable=True)
        self.assertEqual(self.contents(), before)
