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

from binding.python import hem_cli, hem_opencode_cli as cli
from binding.python.hem_common import Deadline, InvalidInput
from binding.python.hem_opencode import PLUGIN
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

    def test_policy_is_read_only_conflict_and_warning(self):
        with patch.dict(os.environ, self.env, clear=True):
            registry = Registry(self.root, deadline=Deadline.after(5))
            registry.create_server('one', saved_settings=dict(device='cpu', listen='127.0.0.1:19001'))
        self.target.mkdir(mode=0o700)
        config = self.target / 'opencode.jsonc'
        original = '// user policy\n{"disabled_providers":["loom-hem-b25l"],"model":"other/default",}\n'
        config.write_text(original)
        self.invoke(expected=2)
        self.assertFalse((self.target / PLUGIN).exists())
        response, _ = self.invoke('--check', expected=1)
        self.assertTrue(response['policy_conflicts'])
        self.assertEqual(config.read_text(), original)

    def test_passive_qualified_binary_hash_and_no_subprocess(self):
        with patch.dict(os.environ, self.env, clear=True), patch.object(cli, 'QUALIFIED_SHA256', __import__('hashlib').sha256(self.executable.read_bytes()).hexdigest()), patch.object(cli.subprocess, 'run', side_effect=AssertionError('must not execute')):
            runtime = cli.inspect_runtime(Deadline.after(5), passive=True)
            self.assertEqual(runtime['version'], '2.0.31')

    def test_check_mutual_exclusion_and_help(self):
        self.invoke('--check', '--disable', expected=2)
        with contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(hem_cli.main(['opencode']), 0)
        self.assertIn('configure', output.getvalue())
