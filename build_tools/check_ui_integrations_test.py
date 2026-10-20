"""Grouped installed-wheel UI integration entrypoint."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


REPO = Path(__file__).resolve().parents[1]


class GroupedUiIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.scripts = self.root / 'scripts'
        self.scripts.mkdir()
        shutil.copy2(REPO / 'build_tools/check-ui-integrations.sh', self.scripts)
        self.package = self.root / 'package'
        for name in ('pi_extension', 'opencode_plugin', 'omp_extension'):
            (self.package / name).mkdir(parents=True)
        self.venv = self.root / 'venv'
        (self.venv / 'bin').mkdir(parents=True)
        python = self.venv / 'bin/python'
        python.write_text('#!/bin/sh\nprintf "%s\\n" "$UI_TEST_PACKAGE"\n')
        python.chmod(0o755)
        self.runtimes = [self.root / name for name in ('pi', 'opencode', 'omp')]
        for runtime in self.runtimes:
            runtime.mkdir()
        self.trace = self.root / 'trace'
        for client in ('pi', 'opencode', 'omp'):
            check = self.scripts / f'check-{client}-assets.sh'
            check.write_text('#!/bin/sh\necho "$1:$2" >> "$UI_TEST_TRACE"\n'
                             f'if [ "${{UI_TEST_FAIL:-}}" = {client!r} ]; then exit 19; fi\n')
            check.chmod(0o755)

    def invoke(self, fail=None):
        environment = dict(os.environ, UI_TEST_PACKAGE=str(self.package),
                           UI_TEST_TRACE=str(self.trace))
        if fail:
            environment['UI_TEST_FAIL'] = fail
        return subprocess.run(['bash', self.scripts / 'check-ui-integrations.sh',
                               self.venv, *self.runtimes], env=environment,
                              capture_output=True, text=True)

    def test_runs_all_integrations_in_one_installed_wheel_stage(self):
        result = self.invoke()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual([line.split(':', 1)[0] for line in self.trace.read_text().splitlines()],
                         [str(path.resolve()) for path in self.runtimes])
        self.assertEqual(result.stdout.splitlines(), [
            '[ui-integrations] Pi', '[ui-integrations] OpenCode',
            '[ui-integrations] OMP', '[ui-integrations] passed'])

    def test_failure_is_attributed_to_ui_integration_and_stops_group(self):
        result = self.invoke('pi')
        self.assertEqual(result.returncode, 19)
        self.assertEqual(result.stdout.splitlines(), ['[ui-integrations] Pi'])
        self.assertEqual(len(self.trace.read_text().splitlines()), 1)


if __name__ == '__main__':
    unittest.main()
