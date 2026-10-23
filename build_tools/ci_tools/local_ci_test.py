# Copyright 2026 The HRX Authors
# Licensed under the MIT License.
# SPDX-License-Identifier: MIT
import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import unittest
from unittest import mock

from build_tools.ci_tools import local_ci as local_ci
from build_tools.ci_tools import local_ci_source as source


class SnapshotTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name)
        self.root = self.base / 'caller'
        self.root.mkdir()
        self.git('init', '-q')
        self.git('config', 'user.email', 'test@example.invalid')
        self.git('config', 'user.name', 'Test')
        for name, text in {'tracked': 'original', 'deleted': 'remove', 'mode': 'mode',
                           'AGENTS.md': 'tracked guide', '.gitignore': 'ignored/\n.bazelrc.local\n'}.items():
            (self.root / name).write_text(text)
        (self.root / 'link').symlink_to('tracked')
        self.git('add', '.')
        self.git('commit', '-qm', 'fixture')

    def tearDown(self):
        self.temporary.cleanup()

    def git(self, *args):
        return source.git(self.root, *args)

    def test_dirty_overlay_preserves_current_bytes_and_index(self):
        (self.root / 'tracked').write_text('staged')
        self.git('add', 'tracked')
        (self.root / 'tracked').write_text('working')
        (self.root / 'deleted').unlink()
        (self.root / 'mode').chmod(0o755)
        self.git('update-index', '--skip-worktree', 'AGENTS.md')
        (self.root / 'AGENTS.md').write_text('current guide')
        (self.root / 'new.py').write_text('new')
        for directory in ('ignored', '.deps', 'build', 'dist', 'bazel-out', '.agents', '.codex', '.loom-serve'):
            (self.root / directory).mkdir()
            (self.root / directory / 'secret').write_text('omit')
        (self.root / '.bazelrc.local').write_text('host SDK')
        for name in ('model.gguf', 'model.pt', 'model.pth', 'model.ckpt'):
            (self.root / name).write_text('weights')
        index = (self.root / '.git/index').read_bytes()
        captured = source.capture(self.root, self.base / 'snapshot')
        snapshot = self.base / 'snapshot'
        self.assertEqual((snapshot / 'tracked').read_text(), 'working')
        self.assertEqual((snapshot / 'AGENTS.md').read_text(), 'current guide')
        self.assertFalse((snapshot / 'deleted').exists())
        self.assertTrue((snapshot / 'mode').stat().st_mode & 0o111)
        self.assertTrue((snapshot / 'link').is_symlink())
        self.assertTrue((snapshot / 'new.py').is_file())
        for name in ('ignored', '.deps', 'build', 'dist', 'bazel-out', '.agents', '.codex', '.loom-serve',
                     '.bazelrc.local', 'model.gguf', 'model.pt', 'model.pth', 'model.ckpt'):
            self.assertFalse((snapshot / name).exists(), name)
        self.assertTrue(captured['dirty'])
        self.assertEqual(source.git(snapshot, 'rev-parse', 'HEAD'), self.git('rev-parse', 'HEAD'))
        self.assertTrue((snapshot / '.git').is_dir())
        self.assertEqual((self.root / '.git/index').read_bytes(), index)
        (snapshot / 'tracked').write_text('mutated')
        with self.assertRaisesRegex(ValueError, 'Captured input changed'):
            source.verify(snapshot, captured['inputs'])

    def test_head_input_staged_out_of_index_still_uses_working_bytes(self):
        self.git('rm', '--cached', 'tracked')
        (self.root / '.gitignore').write_text('tracked\n')
        (self.root / 'tracked').write_text('current ignored-but-head-tracked bytes')
        captured = source.capture(self.root, self.base / 'snapshot')
        self.assertEqual((self.base / 'snapshot/tracked').read_text(),
                         'current ignored-but-head-tracked bytes')
        self.assertIn('tracked', captured['inputs'])

    def test_skip_worktree_alone_is_dirty(self):
        self.git('update-index', '--skip-worktree', 'AGENTS.md')
        (self.root / 'AGENTS.md').write_text('overlay')
        self.assertEqual(self.git('status', '--porcelain'), b'')
        self.assertTrue(source.capture(self.root, self.base / 'snapshot')['dirty'])

    def test_escape_and_drift_rejected(self):
        (self.root / 'escape').symlink_to('/etc/passwd')
        with self.assertRaisesRegex(ValueError, 'Escaping'):
            source.capture(self.root, self.base / 'escape-snapshot')
        (self.root / 'escape').unlink()
        original = source.shutil.copyfile
        def changing(*args, **kwargs):
            result = original(*args, **kwargs)
            (self.root / 'tracked').write_text('concurrent edit')
            return result
        with mock.patch.object(source.shutil, 'copyfile', side_effect=changing):
            with self.assertRaisesRegex(ValueError, 'Source changed during capture'):
                source.capture(self.root, self.base / 'drift-snapshot')

    def test_tracked_authored_inputs_survive_untracked_exclusions(self):
        (self.root / '.agents').mkdir()
        (self.root / '.agents/guide.md').write_text('tracked authoring guide')
        (self.root / 'fixture.pt').write_text('tiny tracked source fixture')
        self.git('add', '.agents/guide.md', 'fixture.pt')
        self.git('commit', '-qm', 'authored inputs')
        captured = source.capture(self.root, self.base / 'snapshot')
        self.assertIn('.agents/guide.md', captured['inputs'])
        self.assertEqual((self.base / 'snapshot/fixture.pt').read_text(), 'tiny tracked source fixture')

    def test_clean_identity(self):
        self.assertFalse(source.capture(self.root, self.base / 'snapshot')['dirty'])


class RunnerTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name)
        self.root = self.base / 'caller'
        self.root.mkdir()

    def tearDown(self):
        self.temporary.cleanup()

    def runner(self, *options):
        runner = local_ci.Runner(self.root, local_ci.parse_args(
            ['--run-dir', str(self.base / 'run'), *options]))
        self.addCleanup(local_ci.shutil.rmtree, runner.runtime)
        return runner

    def test_arguments_fail_before_run(self):
        for options in (['--nope'], ['--cpu-only', '--model08', '/x'],
                        ['--run-dir'], ['--cpu-only', '--cpu-only']):
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    local_ci.parse_args(options)
                self.assertEqual(error.exception.code, 2)
        self.assertFalse((self.base / 'run').exists())

    def test_external_new_destination_required(self):
        for path in (self.root / 'inside', self.root):
            with self.assertRaises(local_ci.Failure):
                local_ci.Runner(self.root, argparse.Namespace(run_dir=path, cpu_only=True, model08=None))
        (self.base / 'run').mkdir()
        with self.assertRaisesRegex(local_ci.Failure, 'exists'):
            self.runner()

    def test_full_order_and_cpu_skip(self):
        for cpu_only in (False, True):
            with self.subTest(cpu_only=cpu_only):
                if (self.base / 'run').exists():
                    import shutil
                    shutil.rmtree(self.base / 'run')
                runner = self.runner(*(['--cpu-only'] if cpu_only else []))
                observed = []
                for stage in local_ci.STAGES:
                    method = stage.replace('-', '_')
                    setattr(runner, method, lambda name=stage: observed.append(name))
                self.assertEqual(runner.execute(), 0)
                self.assertEqual(observed, list(local_ci.STAGES[:-2] if cpu_only else local_ci.STAGES))
                for name in ('gpu', 'openwebui'):
                    self.assertEqual(runner.summary['stages'][name]['status'], 'skipped' if cpu_only else 'passed')
                self.assertEqual(json.loads((runner.run / 'summary.json').read_text())['status'], 'passed')

    def test_gpu_preflight_failure_stops_before_capture(self):
        runner = self.runner()
        with mock.patch.object(runner, 'command'), \
             mock.patch.object(local_ci.shutil, 'which', return_value='/tool'), \
             mock.patch.object(local_ci.os, 'access', return_value=False):
            self.assertEqual(runner.execute(), 1)
        self.assertEqual(runner.summary['failure']['stage'], 'preflight')
        self.assertIn('/dev/kfd', runner.summary['failure']['message'])
        self.assertFalse(runner.snapshot.exists())

    def test_exact_wheel_and_pi_propagation(self):
        runner = self.runner('--model08', str(self.base / 'model'))
        wheelhouse = runner.snapshot / 'build/wheel/wheelhouse'
        wheelhouse.mkdir(parents=True)
        wheel = wheelhouse / 'candidate.whl'
        wheel.write_bytes(b'exact candidate')
        with mock.patch.object(runner, 'command') as command:
            runner.build()
            runner.installed_cpu()
            # GPU shares only the lock path, never adopts other run state.
            runner.workspace = self.base / 'workspace'
            runner.gpu()
        calls = command.call_args_list
        self.assertEqual(calls[1].args[0][2], wheel)
        self.assertEqual(calls[2].args[0][2], wheel)
        for client in ('pi', 'opencode', 'omp'):
            self.assertNotIn('LOOM_' + client.upper() + '_RUNTIME', calls[2].kwargs['env'])
        self.assertEqual(calls[2].args[0][-1], self.base / 'model')
        self.assertEqual(runner.summary['wheel']['sha256'], local_ci.hashlib.sha256(wheel.read_bytes()).hexdigest())
        (wheelhouse / 'stale.whl').write_bytes(b'other')
        with mock.patch.object(runner, 'command'):
            with self.assertRaisesRegex(local_ci.Failure, 'exactly one'):
                runner.build()

    def test_openwebui_uses_same_wheel_private_stage_and_explicit_fixture(self):
        runner = self.runner('--model08', str(self.base / 'model'))
        runner.args.model08.mkdir()
        for name in (local_ci.MODEL_NAME, 'tokenizer.json'):
            (runner.args.model08 / name).touch()
        runner.wheel = self.base / 'newly-repaired.whl'
        runner.workspace = self.base / 'workspace'
        with mock.patch.object(runner, 'command') as command:
            runner.openwebui()
        argv = command.call_args.args[0]
        self.assertEqual(argv, ['bash', runner.snapshot / 'build_tools/model-smoke.sh',
                               '--openwebui', runner.wheel, runner.run / 'openwebui', runner.args.model08])
        env = command.call_args.kwargs['env']
        self.assertEqual(env['HRX_LOCAL_CI_RUNTIME_ROOT'], str(runner.runtime / 'openwebui'))
        self.assertEqual(env['HRX_LOCAL_CI_CLEANUP_FILE'], str(runner.run / 'openwebui-cleanup.json'))
        self.assertTrue(command.call_args.kwargs['cooperative'])
        (runner.args.model08 / 'tokenizer.json').unlink()
        with mock.patch.object(runner, 'command') as command:
            with self.assertRaisesRegex(local_ci.Failure, 'Missing immutable WebUI gate fixture'):
                runner.openwebui()
        command.assert_not_called()

    def test_ui_source_groups_runtimes_before_python_tests(self):
        runner = self.runner()
        with mock.patch.object(runner, 'command') as command:
            runner.ui_source()
        self.assertEqual([str(call.args[0][1]) for call in command.call_args_list], [
            'build_tools/integrations/pi/stage-pi-runtime.sh', 'build_tools/integrations/opencode/stage-opencode-runtime.sh',
            'build_tools/integrations/omp/stage-omp-runtime.sh', 'build_tools/integrations/pi/check-pi-assets.sh',
            'build_tools/integrations/opencode/check-opencode-assets.sh', 'build_tools/integrations/omp/check-omp-assets.sh'])
        self.assertEqual(runner.env['LOOM_OPENCODE_TEST_NODE'],
                         str(runner.run / 'opencode-runtime/node-v24.10.0-linux-x64/bin/node'))
        self.assertEqual(runner.env['LOOM_OMP_TEST_PACKAGE'],
                         str(runner.run / 'omp-runtime/client/node_modules/@oh-my-pi/pi-coding-agent'))
        self.assertLess(local_ci.STAGES.index('ui-source'), local_ci.STAGES.index('source-tests'))

    def test_installed_ui_integrations_are_one_isolated_wheel_stage(self):
        runner = self.runner()
        runner.wheel = self.base / 'candidate.whl'
        with mock.patch.object(runner, 'command') as command:
            runner.ui_integrations()
        self.assertEqual(command.call_args.args[0], [
            'bash', runner.snapshot / 'build_tools/integrations/check-ui-integrations.sh',
            runner.run / 'installed-venv', runner.run / 'pi-runtime',
            runner.run / 'opencode-runtime', runner.run / 'omp-runtime'])
        self.assertEqual(command.call_args.kwargs['cwd'], runner.run)

    def test_source_tests_use_shared_setup_and_both_discovery_suites(self):
        runner = self.runner()
        with mock.patch.object(runner, 'command') as command:
            runner.source_tests()
        venv = runner.run / 'source-venv'
        self.assertEqual(command.call_args_list, [
            mock.call(['python3', '-m', 'venv', venv]),
            *(mock.call([venv / 'bin/python', '-m', 'build_tools.ci_tools.source_tests', action])
              for action in ('install', 'verify', 'build_tools', 'tests'))])

    def test_failure_preserves_original_code_and_stops(self):
        runner = self.runner('--cpu-only')
        runner.preflight = lambda: None
        runner.capture = lambda: None
        runner.ui_source = lambda: None
        runner.source_tests = lambda: runner.command(['bash', '-c', 'echo failure; exit 37'], cwd=self.base)
        self.assertEqual(runner.execute(), 1)
        self.assertEqual(runner.summary['failure']['exit_code'], 37)
        self.assertEqual(runner.summary['stages']['build']['status'], 'not-run')
        self.assertIn('failure', (runner.run / 'source-tests.log').read_text())

    def test_cache_precedence_and_environment(self):
        self.assertEqual(local_ci.cache_root({'HF_HUB_CACHE': '/one', 'HF_HOME': '/two'}), '/one')
        self.assertEqual(local_ci.cache_root({'HUGGINGFACE_HUB_CACHE': '/two'}), '/two')
        self.assertEqual(local_ci.cache_root({'HF_HOME': '/three'}), '/three/hub')
        self.assertEqual(local_ci.cache_root({'XDG_CACHE_HOME': '/four'}), '/four/huggingface/hub')
        with mock.patch.dict(os.environ, {'HF_HUB_CACHE': '/shared', 'PYTHONPATH': '/bad',
                                         'LD_LIBRARY_PATH': '/sdk', 'HOME': str(Path.home()),
                                         'SSH_AUTH_SOCK': '/ssh', 'GITHUB_RUN_ID': 'fixture'}):
            runner = self.runner()
        self.assertEqual(runner.env['HF_HUB_CACHE'], '/shared')
        self.assertEqual(runner.env['SSH_AUTH_SOCK'], '/ssh')
        for name in ('PYTHONPATH', 'LD_LIBRARY_PATH', 'GITHUB_RUN_ID'):
            self.assertNotIn(name, runner.env)

    def test_interruption_summary_and_repeated_signal(self):
        runner = self.runner()
        runner.preflight = lambda: runner.on_signal(signal.SIGTERM, None)
        self.assertEqual(runner.execute(), 143)
        self.assertEqual(runner.summary['status'], 'interrupted')
        self.assertEqual(runner.summary['stages']['capture']['status'], 'not-run')
        self.assertEqual(runner.summary['stages']['preflight']['status'], 'interrupted')
        runner.on_signal(signal.SIGINT, None)
        self.assertEqual(runner.interrupted, signal.SIGTERM)

    def test_signal_routing_uses_active_command_metadata_instead_of_stage_name(self):
        runner=self.runner()
        for stage,cooperative in (('build',False),('preflight',True)):
            with self.subTest(stage=stage,cooperative=cooperative):
                runner.interrupted=None
                runner.current=stage
                child=mock.Mock(pid=4321)
                child.poll.return_value=None
                runner.active=(child,cooperative)
                with mock.patch.object(os,'killpg') as group:
                    runner.on_signal(signal.SIGTERM,None)
                    runner.on_signal(signal.SIGINT,None)
                if cooperative:
                    child.send_signal.assert_called_once_with(signal.SIGTERM)
                    group.assert_not_called()
                else:
                    group.assert_called_once_with(child.pid,signal.SIGTERM)
                    child.send_signal.assert_not_called()

    def test_signal_during_launch_uses_new_command_metadata_after_publication(self):
        runner=self.runner()
        for cooperative in (False,True):
            with self.subTest(cooperative=cooperative):
                runner.interrupted=None
                runner.current='preflight'
                (runner.run/'preflight-cleanup.json').write_text('{"status":"complete"}')
                child=mock.Mock(pid=4321,returncode=-signal.SIGTERM)
                child.poll.return_value=child.returncode
                child.stdout.read1.return_value=b''
                def launch(*args,**kwargs):
                    runner.on_signal(signal.SIGTERM,None)
                    return child
                with mock.patch.object(subprocess,'Popen',side_effect=launch), \
                     mock.patch.object(os,'killpg') as group:
                    with self.assertRaises(local_ci.Failure) as error:
                        runner.command(['fixture'],cwd=self.base,cooperative=cooperative)
                self.assertEqual(error.exception.code,-signal.SIGTERM)
                self.assertIsNone(runner.active)
                if cooperative:
                    child.send_signal.assert_called_once_with(signal.SIGTERM)
                    group.assert_not_called()
                else:
                    group.assert_called_once_with(child.pid,signal.SIGTERM)
                    child.send_signal.assert_not_called()

    def test_actual_cooperative_signal_reaps_owned_child(self):
        import threading
        import time
        runner = self.runner()
        runner.preflight = lambda: None
        runner.capture = lambda: None
        runner.ui_source = lambda: None
        runner.source_tests = lambda: None
        ready = runner.run / 'ready'
        cleanup = runner.run / 'build-cleanup.json'
        program = ("import signal,time,pathlib,json; "
                   "signal.signal(signal.SIGTERM,lambda n,f: None); "
                   f"pathlib.Path({str(ready)!r}).write_text('ready'); "
                   "time.sleep(0.5); "
                   f"pathlib.Path({str(cleanup)!r}).write_text(json.dumps(dict(status='complete')))")
        runner.build = lambda: runner.command([local_ci.sys.executable, '-c', program],
                                              cwd=self.base, cooperative=True)
        def cancel():
            deadline = time.monotonic() + 5
            while not ready.exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            os.kill(os.getpid(), signal.SIGTERM)
            time.sleep(0.05)
            os.kill(os.getpid(), signal.SIGINT)
        sender = threading.Thread(target=cancel)
        sender.start()
        self.assertEqual(runner.execute(), 143)
        sender.join()
        self.assertIsNone(runner.active)
        self.assertEqual(runner.summary['cleanup']['status'], 'complete')
        self.assertEqual(runner.summary['stages']['build']['status'], 'interrupted')
        self.assertEqual(runner.summary['stages']['installed-cpu']['status'], 'not-run')

    def test_runtime_child_excludes_host_toolchain_and_preserves_user_context(self):
        unsafe = {'LD_PRELOAD': '/host/preload.so', 'LD_AUDIT': '/host/audit.so',
                  'IREE_ROCM_PATH': '/host/rocm', 'CC': '/host/cc', 'CXX': '/host/cxx',
                  'CMAKE_PREFIX_PATH': '/host/cmake', 'LD_LIBRARY_PATH': '/host/lib'}
        preserved = {'HF_HUB_CACHE': '/shared/hf', 'HF_HOME': '/shared/auth',
                     'HOME': str(Path.home()), 'XDG_CACHE_HOME': '/shared/xdg',
                     'SSH_AUTH_SOCK': '/shared/ssh-agent'}
        with mock.patch.dict(os.environ, {**unsafe, **preserved}):
            runner = self.runner()
        runner.current = 'installed-cpu'
        program = ('import os; '
                   f'assert all(name not in os.environ for name in {list(unsafe)!r}); '
                   f'assert all(os.environ.get(k)==v for k,v in {preserved!r}.items()); '
                   'print("runtime environment isolated; user context preserved")')
        runner.command([local_ci.sys.executable, '-c', program], cwd=self.base)

    def test_detached_test_fixture_is_reported_unresolved_and_retained(self):
        import threading
        import time
        runner = self.runner('--cpu-only')
        runner.preflight = lambda: None
        runner.capture = lambda: None
        runner.ui_source = lambda: None
        pid_file = runner.runtime / 'tmp/fixture.pid'
        token = 'local-ci-fixture-' + local_ci.uuid.uuid4().hex
        program = ("import subprocess,sys,time,pathlib; "
                   f"p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)',{token!r}],"
                   "start_new_session=True,stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL); "
                   f"pathlib.Path({str(pid_file)!r}).write_text(str(p.pid)); time.sleep(30)")
        runner.source_tests = lambda: runner.command([local_ci.sys.executable, '-c', program], cwd=self.base)
        def cancel():
            deadline = time.monotonic() + 5
            while not pid_file.exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            os.kill(os.getpid(), signal.SIGTERM)
        unrelated = subprocess.Popen([local_ci.sys.executable, '-c', 'import time; time.sleep(30)'],
                                     start_new_session=True)
        sender = threading.Thread(target=cancel)
        sender.start()
        try:
            self.assertEqual(runner.execute(), 143)
            sender.join()
            pid = int(pid_file.read_text())
            self.assertIn(token.encode(), Path(f'/proc/{pid}/cmdline').read_bytes())
            self.assertEqual(runner.summary['cleanup']['status'], 'unresolved')
            self.assertEqual(runner.summary['stages']['source-tests']['cleanup']['status'], 'unresolved')
            self.assertTrue(pid_file.exists())
            self.assertIsNone(unrelated.poll())
            self.assertEqual(runner.summary['stages']['source-tests']['status'], 'interrupted')
        finally:
            sender.join()
            if pid_file.exists():
                pid = int(pid_file.read_text())
                command_line = Path(f'/proc/{pid}/cmdline')
                if command_line.exists() and token.encode() in command_line.read_bytes():
                    os.kill(pid, signal.SIGTERM)
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline:
                    stat_path = Path(f'/proc/{pid}/stat')
                    if not stat_path.exists() or stat_path.read_text().split(') ', 1)[1].startswith('Z'):
                        break
                    time.sleep(0.02)
                else:
                    self.fail('exact owned detached fixture did not exit')
            unrelated.terminate()
            unrelated.wait(timeout=5)

    def test_group_writable_report_ancestor_keeps_private_pi_state(self):
        self.base.chmod(0o775)
        with mock.patch.dict(os.environ, {'HF_HUB_OFFLINE': '1'}):
            runner = self.runner()
        self.assertEqual(self.base.stat().st_mode & 0o777, 0o775)
        self.assertEqual(runner.env['HF_HUB_OFFLINE'], '1')
        self.assertFalse(runner.runtime.is_relative_to(self.base))
        self.assertEqual(runner.summary['runtime']['run_dir'], str(runner.run))
        self.assertTrue(runner.summary['runtime']['retained'])
        # Exercise the real unchanged Pi policy, not a duplicate predicate.
        from binding.python.hem_pi_files import private_path
        from binding.python.hem_common import InvalidInput
        with self.assertRaisesRegex(InvalidInput, 'unsafe managed directory'):
            private_path(runner.run)
        private_path(runner.runtime / 'source/state', create=True)
        private_path(runner.runtime / 'model/pi-agent', create=True)
        runner.current = 'installed-cpu'
        program = ('import os; from pathlib import Path; '
                   'from binding.python.hem_pi_files import private_path; '
                   'assert os.environ["HF_HUB_OFFLINE"] == "1"; '
                   'private_path(Path(os.environ["LOOM_SERVE_DIR"]),create=True); '
                   'private_path(Path(os.environ["TMPDIR"])); '
                   'print("private runtime accepted with offline caller and group writable reports")')
        runner.command([local_ci.sys.executable, '-c', program], cwd=Path(__file__).resolve().parents[2])
        with mock.patch.object(runner, 'command') as command:
            runner.wheel = self.base / 'candidate.whl'
            runner.installed_cpu()
            self.assertEqual(command.call_args.kwargs['env']['HRX_LOCAL_CI_RUNTIME_ROOT'],
                             str(runner.runtime / 'installed'))
        self.assertEqual(self.base.stat().st_mode & 0o777, 0o775)
        self.assertEqual(runner.env['HF_HUB_OFFLINE'], '1')

    def test_private_runtime_rejects_alias_tamper_and_untrusted_stage(self):
        runner = self.runner()
        runtime = local_ci.runtime
        stage = runner.runtime / 'model'
        self.assertEqual(runtime.validate_stage(stage, runner.run), stage)
        with self.assertRaisesRegex(ValueError, 'another run'):
            runtime.validate_stage(stage, self.base / 'another-run')
        alias = self.base / 'alias'
        alias.symlink_to(stage, target_is_directory=True)
        with self.assertRaises(ValueError):
            runtime.validate_stage(alias)
        stage.chmod(0o775)
        with self.assertRaises(ValueError):
            runtime.validate_stage(stage)
        stage.chmod(0o700)
        marker = runner.runtime / runtime.MARKER
        owner = json.loads(marker.read_text())
        owner['inode'] += 1
        marker.write_text(json.dumps(owner))
        with self.assertRaisesRegex(ValueError, 'ownership'):
            runtime.validate_stage(stage)

    def test_cleanup_failure_preserves_functional_failure(self):
        runner = self.runner()
        runner.current = 'build'
        runner.summary['stages']['build'] = {}
        atomic = runner.run / 'build-cleanup.json'
        atomic.write_text('{"status":"unresolved","details":"still alive"}')
        with self.assertRaises(local_ci.Failure) as error:
            runner.command(['bash', '-c', 'exit 17'], cwd=self.base, cooperative=True)
        self.assertEqual(error.exception.code, 17)
        self.assertEqual(runner.summary['cleanup']['status'], 'unresolved')


if __name__ == '__main__':
    unittest.main()
