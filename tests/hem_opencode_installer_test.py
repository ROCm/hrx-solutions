"""Toolkit ownership/recovery tests; no vendor runtime or inference certification."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from binding.python import hem_opencode as installer
from binding.python.hem_common import Deadline, InvalidInput


class InstallerTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.root, self.target = self.base / 'state', self.base / 'config'
        self.root.mkdir(mode=0o700)
        self.directory = self.target / installer.PLUGIN

    def run_install(self, **kwargs):
        return installer.configure_opencode(self.root, self.target, Deadline.after(5), **kwargs)

    def contents(self):
        return {str(path.relative_to(self.base)): (path.lstat().st_mode, path.lstat().st_mtime_ns,
                path.read_bytes() if path.is_file() and not path.is_symlink() else None)
                for path in self.base.rglob('*')}

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
                def run(**kw): return installer.configure_opencode(self.root, target, Deadline.after(5), **kw)
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
                        with self.assertRaises(InvalidInput): installer.configure_opencode(other, target, Deadline.after(5), disable=True)
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
                    def run(**kw): return installer.configure_opencode(self.root, target, Deadline.after(5), **kw)
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
                    with self.assertRaises(InvalidInput): installer.configure_opencode(self.root, target, Deadline.after(5))
                    self.assertEqual(stage.read_bytes(), b'foreign')
                else:
                    installer.configure_opencode(self.root, target, Deadline.after(5))
                    self.assertFalse(stage.exists())

    def test_duplicate_manifest_fields_and_directory_alias_refused(self):
        self.run_install()
        path = self.directory / 'manifest.json'
        path.write_bytes(path.read_bytes().replace(b'"enabled":true', b'"enabled":true,"enabled":true'))
        self.assertEqual(self.run_install(check=True)['state'], 'conflict')
        alias = self.base / 'alias'; alias.symlink_to(self.target)
        with self.assertRaises(InvalidInput): installer.configure_opencode(self.root, alias, Deadline.after(5))

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
