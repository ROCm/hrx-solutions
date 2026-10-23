"""Public CLI behavior with isolated owned targets; no real inference here."""
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from binding.python import hem_cli
from binding.python.integrations.opencode import hem_opencode as owner
from binding.python.hem_common import Deadline, InvalidInput
from binding.python.integrations.opencode.hem_opencode import PLUGIN
from binding.python.hem_state import Registry


class OpenCodeCliTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.root, self.target = self.base / 'state', self.base / 'config'
        self.bin = self.base / 'bin'; self.bin.mkdir(mode=0o700)
        self.executable = self.bin / 'opencode'
        self.executable.write_text("#!/bin/sh\n[ \"$1\" = --version ] || exit 29\nprintf 'opencode v2.0.31\\n'\n")
        self.executable.chmod(0o700)
        self.env = {key: value for key, value in os.environ.items() if key not in ('PYTHONPATH', 'PYTHONHOME', 'OPENCODE_CONFIG', 'OPENCODE_CONFIG_CONTENT')}
        self.env.update(PATH=str(self.bin), LOOM_SERVE_DIR=str(self.root), OPENCODE_CONFIG_DIR=str(self.target),
            OPENCODE_CONFIG_PROJECT_DISABLE='1', PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1')

    def invoke(self, *args, expected=0):
        completed = subprocess.run([sys.executable, '-m', 'binding.python.hem_cli', 'opencode', 'configure', '--json', *args],
            env=self.env, cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=15)
        self.assertEqual(completed.returncode, expected, completed.stdout + completed.stderr)
        self.assertNotIn('Traceback', completed.stderr)
        return json.loads(completed.stdout), completed.stderr

    def contents(self):
        return {str(p): (p.lstat().st_mtime_ns, p.read_bytes() if p.is_file() else None) for p in self.base.rglob('*')}

    def test_explicit_configure_refreshes_prior_owned_projection_after_passive_check(self):
        from binding.python.hem_pi_contracts import NATIVE_PROFILE, PREVIOUS_NATIVE_PROFILE
        from binding.python.hem_state import Registry
        Registry(self.root).create_server('local', saved_settings=dict(device='task', listen='127.0.0.1:18080'))
        self.invoke()
        path = self.root / 'clients/endpoints.json'
        prior = json.loads(path.read_bytes())
        prior['servers'][0]['native_profile'] = PREVIOUS_NATIVE_PROFILE
        path.write_text(json.dumps(prior))
        before = self.contents()
        self.invoke('--check', expected=1)
        self.assertEqual(self.contents(), before)
        self.invoke()
        self.assertEqual(json.loads(path.read_bytes())['servers'][0]['native_profile'], NATIVE_PROFILE)

    def test_configure_repeat_disable_absent_client_and_no_daemon(self):
        response, stderr = self.invoke()
        self.assertEqual(response['state'], 'configured')
        self.assertTrue((self.root / 'clients/endpoints.json').is_file())
        self.assertIn('No native Loom equivalent', stderr)
        self.assertIn('OPENCODE_CONFIG_DIR=', response['launch'])
        self.assertEqual(response['availability']['state'], 'not_observed')
        self.assertFalse(list(self.root.rglob('*.sock')))
        before = self.contents()
        response, _ = self.invoke()
        self.assertFalse(response['changed'])
        # SQLite housekeeping can change, but plugin bytes/mtimes must not.
        for path, info in before.items():
            if PLUGIN in path: self.assertEqual(self.contents()[path], info)
        self.executable.unlink()
        before = self.contents()
        response, _ = self.invoke('--check', expected=1)
        self.assertTrue(response['configured'])
        self.assertEqual(response['runtime_compatibility']['state'], 'unqualified')
        self.assertEqual(before, self.contents())
        response, _ = self.invoke('--disable')
        self.assertEqual(response['state'], 'disabled')
        self.assertTrue((self.root / 'clients/endpoints.json').exists())
        response, _ = self.invoke('--disable')
        self.assertFalse(response['changed'])

    def test_check_never_executes_client_and_configuration_findings_survive(self):
        self.executable.write_text('#!/bin/sh\nprintf invoked > ' + str(self.base / 'invoked') + '\nexit 99\n')
        before = self.contents()
        response, _ = self.invoke('--check', expected=1)
        self.assertEqual(response['state'], 'absent')
        self.assertEqual(response['runtime_compatibility']['state'], 'unqualified')
        self.assertEqual(before, self.contents())
        self.assertFalse(self.root.exists())
        self.assertFalse(self.target.exists())

    def test_reject_v1_before_writes(self):
        self.executable.write_text("#!/bin/sh\nprintf 'opencode v1.19.3\\n'\n")
        self.invoke(expected=2)
        self.assertFalse(self.root.exists()); self.assertFalse(self.target.exists())

    def test_override_precedence_and_launch_recipe(self):
        explicit = self.base / 'space target'
        response, _ = self.invoke('--opencode-dir', str(explicit))
        self.assertEqual(response['target'], str(explicit))
        self.assertIn("'" + str(explicit) + "'", response['launch'])
        self.assertFalse(self.target.exists())
        self.env['OPENCODE_CONFIG_DIR'] = 'relative'
        self.invoke('--opencode-dir', str(explicit), '--disable')
        self.invoke('--check', expected=2)

    def test_foreign_refusal_before_root_initialization_and_edited_disable(self):
        from binding.python.hem_pi_files import private_path
        directory = self.target / PLUGIN
        private_path(directory, create=True)
        (directory / 'index.ts').write_text('foreign')
        self.invoke(expected=2)
        self.assertFalse(self.root.exists())
        self.invoke('--disable', expected=2)
        self.assertEqual((directory / 'index.ts').read_text(), 'foreign')

    def test_host_policy_unassessed_and_qualified_check_succeeds(self):
        with patch.dict(os.environ, self.env, clear=True):
            Registry(self.root, deadline=Deadline.after(5)).create_server(
                'one', saved_settings=dict(device='cpu', listen='127.0.0.1:19001'))
        self.target.mkdir(mode=0o700)
        config = self.target / 'opencode.jsonc'
        original = '// user policy\n{"disabled_providers":["loom-hem-b25l"],"model":"other/default",}\n'
        config.write_text(original)
        response, _ = self.invoke()
        self.assertTrue(response['configured'])
        self.assertNotIn('policy_conflicts', response)
        self.assertNotIn('policy_warnings', response)
        self.assertIn('host policy is unassessed', response['guidance'])
        with patch.dict(os.environ, self.env, clear=True), patch.object(
                owner, 'QUALIFIED_SHA256', __import__('hashlib').sha256(self.executable.read_bytes()).hexdigest()):
            before = self.contents()
            checked = owner.configure_opencode(self.root, self.target, Deadline.after(5), check=True)
            self.assertEqual(checked.exit_code, 0)
            self.assertTrue(checked['configured'])
            self.assertNotIn('policy_conflicts', checked)
            self.assertNotIn('policy_warnings', checked)
            self.assertEqual(before, self.contents())
        self.assertEqual(config.read_text(), original)

    def test_no_host_config_reads_or_ancestor_walk(self):
        self.target.mkdir(mode=0o700)
        project = self.base / 'project'
        project.mkdir(mode=0o700)
        nested = project / '.opencode'
        nested.mkdir(mode=0o700)
        host_paths = [self.target / 'opencode.json', self.target / 'opencode.jsonc',
                      project / 'opencode.json', nested / 'opencode.jsonc', self.base / 'env.jsonc']
        for path in host_paths:
            path.write_text('malformed host policy with secret')
        environment = dict(self.env, OPENCODE_CONFIG=str(host_paths[-1]),
                           OPENCODE_CONFIG_CONTENT='malformed inline secret', OPENCODE_CONFIG_PROJECT_DISABLE='0')
        real_open, real_path_open = os.open, Path.open
        def guarded_open(path, *args, **kwargs):
            if Path(path) in host_paths:
                raise AssertionError('setup read unrelated host config: ' + str(path))
            return real_open(path, *args, **kwargs)
        def guarded_path_open(path, *args, **kwargs):
            if path in host_paths:
                raise AssertionError('setup opened unrelated host config: ' + str(path))
            return real_path_open(path, *args, **kwargs)
        with patch.object(Path, 'open', guarded_path_open), patch.dict(os.environ, environment, clear=True), patch.object(Path, 'cwd', return_value=project) as cwd, patch.object(
                os, 'open', side_effect=guarded_open), patch.object(owner, 'inspect_runtime', return_value={'qualification': 'test'}):
            response = owner.configure_opencode(None, None, Deadline.after(5))
            before = {str(path): path.stat().st_mtime_ns for path in self.base.rglob('*')}
            checked = owner.configure_opencode(None, None, Deadline.after(5), check=True)
            self.assertEqual(checked.exit_code, 0)
            self.assertEqual(before, {str(path): path.stat().st_mtime_ns for path in self.base.rglob('*')})
            cwd.assert_not_called()
        self.assertTrue(response['configured'])
        self.assertNotIn('policy_warnings', response)
        self.assertNotIn('secret', json.dumps(response))
        for path in host_paths:
            self.assertEqual(path.read_text(), 'malformed host policy with secret')

    def test_direct_owner_requires_runtime_and_disable_skips_it(self):
        with patch.object(owner, 'inspect_runtime', side_effect=InvalidInput('unsupported')):
            with self.assertRaises(InvalidInput):
                owner.configure_opencode(self.root, self.target, Deadline.after(5))
            self.assertFalse(self.root.exists())
            self.assertFalse(self.target.exists())
            checked = owner.configure_opencode(self.root, self.target, Deadline.after(5), check=True)
            self.assertEqual(checked['runtime_compatibility']['state'], 'unqualified')
            self.assertEqual(checked.exit_code, 1)
        with patch.object(owner, 'inspect_runtime', side_effect=AssertionError('disable runtime')):
            response = owner.configure_opencode(self.root, self.target, Deadline.after(5), disable=True)
            self.assertEqual(response['state'], 'disabled')
            self.assertFalse(self.root.exists())
            self.assertFalse(self.target.exists())

    def test_direct_owner_preflights_before_root_and_rechecks_after_publication(self):
        from binding.python.hem_pi_files import private_path
        private_path(self.target / PLUGIN, create=True)
        foreign = self.target / PLUGIN / 'index.ts'
        foreign.write_text('foreign')
        with patch.object(owner, 'inspect_runtime', return_value={'qualification': 'test'}), patch.object(
                owner, 'Registry', side_effect=AssertionError('root published before preflight')):
            with self.assertRaises(InvalidInput):
                owner.configure_opencode(self.root, self.target, Deadline.after(5))
        self.assertFalse(self.root.exists())
        foreign.unlink()
        publish = owner.ensure_endpoints
        def publish_then_edit(*args):
            publish(*args)
            foreign.write_text('concurrent foreign')
        with patch.object(owner, 'inspect_runtime', return_value={'qualification': 'test'}), patch.object(
                owner, 'ensure_endpoints', side_effect=publish_then_edit):
            with self.assertRaises(InvalidInput):
                owner.configure_opencode(self.root, self.target, Deadline.after(5))
        self.assertEqual(foreign.read_text(), 'concurrent foreign')

    def test_known_recovery_conflicts_refuse_before_root_publication(self):
        cases = ('malformed', 'foreign_root', 'orphan_stage', 'malformed_plan_stage',
                 'edited_current', 'foreign_artifact_stage', 'foreign_plan_stage')
        for case in cases:
            with self.subTest(case=case):
                root, target = self.base / (case + '-root'), self.base / (case + '-target')
                directory = target / PLUGIN
                owner.private_path(directory, create=True)
                desired = owner._desired(target, root, owner.bundled_assets(), {})
                plan = owner._plan({}, desired, target, root)
                if case == 'foreign_root':
                    other = self.base / 'other-root'
                    plan = owner._plan({}, owner._desired(target, other, owner.bundled_assets(), {}), target, other)
                if case == 'orphan_stage':
                    owner.staged_path(directory / 'index.ts').write_bytes(b'orphan')
                elif case == 'malformed_plan_stage':
                    owner.staged_path(directory / 'pending.json').write_bytes(b'foreign')
                else:
                    (directory / 'pending.json').write_bytes(b'foreign' if case == 'malformed' else plan)
                    if case == 'edited_current':
                        (directory / 'index.ts').write_bytes(b'foreign')
                    if case == 'foreign_artifact_stage':
                        owner.staged_path(directory / 'index.ts').write_bytes(b'foreign')
                    if case == 'foreign_plan_stage':
                        owner.staged_path(directory / 'pending.json').write_bytes(b'foreign')
                for path in directory.iterdir():
                    path.chmod(0o600)
                before = self.contents()
                with patch.object(owner, 'inspect_runtime', return_value={'qualification': 'test'}):
                    with self.assertRaises(InvalidInput):
                        owner.configure_opencode(root, target, Deadline.after(5))
                self.assertFalse(root.exists(), 'known recovery conflict published the root')
                self.assertEqual(before, self.contents())

    def test_public_owner_recovers_valid_published_and_unpublished_plans(self):
        for unpublished in (False, True):
            with self.subTest(unpublished=unpublished):
                root, target = self.base / ('recover-root-' + str(unpublished)), self.base / ('recover-target-' + str(unpublished))
                directory = target / PLUGIN
                owner.private_path(directory, create=True)
                desired = owner._desired(target, root, owner.bundled_assets(), {})
                plan = owner._plan({}, desired, target, root)
                marker = directory / 'pending.json'
                (owner.staged_path(marker) if unpublished else marker).write_bytes(plan)
                # A partial legitimate generation plus an interrupted artifact copy.
                (directory / 'discovery.ts').write_bytes(desired['discovery.ts'])
                owner.staged_path(directory / 'index.ts').write_bytes(desired['index.ts'][:20])
                for path in directory.iterdir():
                    path.chmod(0o600)
                before = self.contents()
                owner._preflight(root, target)
                self.assertEqual(before, self.contents())
                self.assertFalse(root.exists())
                with patch.object(owner, 'inspect_runtime', return_value={'qualification': 'test'}):
                    response = owner.configure_opencode(root, target, Deadline.after(5))
                    checked = owner.configure_opencode(root, target, Deadline.after(5), check=True)
                self.assertTrue(response['configured'])
                self.assertEqual(checked.exit_code, 0)
                self.assertFalse(marker.exists())
                self.assertFalse(owner.staged_path(marker).exists())
                self.assertEqual({name: (directory / name).read_bytes() for name in owner.FILES}, desired)

    def test_dispatch_calls_complete_owner_once_and_reads_assets_once(self):
        from binding.python.hem_integrations import configure_integration
        with patch.object(owner, 'inspect_runtime', return_value={'qualification': 'test'}), patch.object(
                owner, 'configure_opencode', wraps=owner.configure_opencode) as operation, patch.object(
                owner, 'bundled_assets', wraps=owner.bundled_assets) as assets:
            response = configure_integration('opencode', self.root, Deadline.after(5), opencode_dir=self.target)
            self.assertTrue(response['configured'])
            self.assertEqual(operation.call_count, 1)
            self.assertEqual(assets.call_count, 1)
        with patch.object(owner, 'inspect_runtime', return_value={'qualification': 'test'}):
            checked = configure_integration('opencode', self.root, Deadline.after(5), mode='check', opencode_dir=self.target)
            self.assertEqual(checked.exit_code, 0)
        with patch.object(owner, 'inspect_runtime', side_effect=AssertionError('disable runtime')):
            self.assertEqual(configure_integration('opencode', self.root, Deadline.after(5), mode='disable', opencode_dir=self.target)['state'], 'disabled')

    def test_fixed_dispatch_preserves_namespace_target_mode_and_deadline(self):
        from binding.python.hem_integrations import configure_integration
        deadline = Deadline.after(5)
        for mode in ('configure', 'check', 'disable'):
            with patch.object(owner, 'configure_opencode', return_value={}) as operation:
                configure_integration('opencode', self.root, deadline, mode=mode, opencode_dir=self.target)
                operation.assert_called_once_with(self.root, self.target, deadline,
                    check=mode == 'check', disable=mode == 'disable')
        with patch.object(owner, 'configure_opencode', side_effect=AssertionError('invalid mode dispatched')):
            with self.assertRaises(InvalidInput):
                configure_integration('opencode', self.root, deadline, mode='invalid', opencode_dir=self.target)

    def test_passive_qualified_binary_hash_and_no_subprocess(self):
        with patch.dict(os.environ, self.env, clear=True), patch.object(owner, 'QUALIFIED_SHA256', __import__('hashlib').sha256(self.executable.read_bytes()).hexdigest()), patch.object(owner.subprocess, 'run', side_effect=AssertionError('must not execute')):
            runtime = owner.inspect_runtime(Deadline.after(5), passive=True)
            self.assertEqual(runtime['version'], '2.0.31')

    def test_check_mutual_exclusion_and_help(self):
        self.invoke('--check', '--disable', expected=2)
        with contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(hem_cli.main(['opencode']), 0)
        self.assertIn('configure', output.getvalue())
