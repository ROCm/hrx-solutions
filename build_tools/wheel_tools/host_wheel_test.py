# Copyright 2026 The HRX Authors
# Licensed under the MIT License.
# SPDX-License-Identifier: MIT
"""Focused safety/contract tests; no compiler, real Bazel, Docker or model assets."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock
import zipfile

from build_tools.wheel_tools import host_wheel as host
from build_tools.ci_tools import local_ci_source


class HostWheelTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def wheel(self, filename='hrx_solutions-1.0-py3-none-linux_x86_64.whl',
              tag='py3-none-linux_x86_64', pure='false', version='1.0'):
        path = self.root / filename
        with zipfile.ZipFile(path, 'w') as archive:
            archive.writestr('hrx_solutions-1.0.dist-info/WHEEL', f'Tag: {tag}\nRoot-Is-Purelib: {pure}\n')
            archive.writestr('hrx_solutions-1.0.dist-info/METADATA', f'Name: hrx-solutions\nVersion: {version}\n')
        return path

    def test_independent_filename_and_internal_tags(self):
        host.validate_host_tags(self.wheel())
        for kwargs in (
            {'filename': 'hrx_solutions-1.0-py3-none-manylinux_2_28_x86_64.whl'},
            {'tag': 'py3-none-manylinux_2_28_x86_64'},
            {'tag': 'py3-none-linux_x86_64\nTag: py3-none-linux_x86_64'},
            {'pure': 'true'}, {'version': '2.0'},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                host.validate_host_tags(self.wheel(**kwargs))

    def test_existing_and_in_checkout_outputs_rejected(self):
        for value in (str(self.root), str(self.root / 'child')):
            with self.assertRaises(ValueError):
                host.external_path(value, self.root, new=True)
        other = self.root / 'existing'
        other.mkdir()
        with self.assertRaises(ValueError):
            host.external_path(str(other), self.root / 'source', new=True)
        with self.assertRaises(ValueError):
            host.external_path('relative', self.root)

    def test_clean_environment_and_explicit_rejections(self):
        for name in host.REJECT_ENV:
            with self.subTest(name=name), self.assertRaises(ValueError):
                host.clean_environment({name: ''})
        original = {name: 'bad' for name in host.CLEAR_ENV}
        original.update(HOME='/home/preserved', XDG_CACHE_HOME='/cache/preserved',
                        BAZEL='foreign', BAZELISK_HOME='foreign', HRX_BAZEL_CACHE_DIR='foreign',
                        HRX_BAZEL_CACHE_ENABLED='1', HF_TOKEN='secret')
        actual = host.clean_environment(original)
        self.assertTrue(all(name not in actual for name in host.CLEAR_ENV))
        self.assertEqual(actual['HOME'], original['HOME'])
        self.assertEqual(actual['HF_TOKEN'], original['HF_TOKEN'])
        self.assertNotIn('BAZELISK_HOME', actual)
        self.assertEqual(actual['HRX_BAZEL_CACHE_ENABLED'], '0')

    def git_source(self):
        source = self.root / 'authored'
        source.mkdir()
        subprocess.run(['git', 'init', '-q', source], check=True)
        (source / 'MODULE.bazel').write_text('module(name="example")\n')
        (source / '.bazelrc').write_text('build --compilation_mode=opt\n')
        (source / 'payload.py').write_text('original\n')
        subprocess.run(['git', '-C', source, 'add', '.'], check=True)
        subprocess.run(['git', '-C', source, '-c', 'user.name=fixture', '-c', 'user.email=fixture@example.invalid',
                        'commit', '-qm', 'fixture'], check=True)
        return source

    def test_source_audit_refuses_local_packaging_and_routing(self):
        source = self.git_source()
        host.audit_source(source, 'selected\n')
        for name in ('setup.cfg', '.bazelrc.local', '.bazelrc.configured', 'MODULE.bazel.local'):
            path = source / name
            path.write_text('unsupported\n')
            with self.subTest(name=name), self.assertRaises(ValueError):
                host.audit_source(source, 'selected\n')
            path.unlink()
        (source / '.bazelrc.configured').write_text('selected\n')
        host.audit_source(source, 'selected\n')
        (source / '.bazelrc').write_text('build --remote_cache=foreign\n')
        with self.assertRaises(ValueError):
            host.audit_source(source, 'selected\n')

    def test_dirty_source_captured_and_verified_without_mutating_caller(self):
        source = self.git_source()
        (source / 'payload.py').write_text('dirty iteration marker\n')
        (source / 'new.py').write_text('new payload\n')
        captured = self.root / 'capture'
        manifest = local_ci_source.capture(source, captured)
        self.assertTrue(manifest['dirty'])
        self.assertEqual((captured / 'payload.py').read_text(), 'dirty iteration marker\n')
        self.assertEqual((captured / 'new.py').read_text(), 'new payload\n')
        local_ci_source.verify(captured, manifest['inputs'])
        (captured / 'payload.py').write_text('stale\n')
        with self.assertRaises(ValueError):
            local_ci_source.verify(captured, manifest['inputs'])
        self.assertEqual((source / 'payload.py').read_text(), 'dirty iteration marker\n')

    def test_launcher_overrides_ambient_rc_and_cache_for_all_backend_queries(self):
        executable = self.root / 'fake-bazel'
        executable.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n')
        executable.chmod(0o700)
        launcher = self.root / 'bazel'
        source = self.root / 'source with spaces'
        host.write_launcher(launcher, executable, self.root, source)
        output = subprocess.check_output([launcher, 'info', 'output_base'], text=True).splitlines()
        self.assertEqual(output[0], '--output_user_root=' + str(self.root / 'bazel-user'))
        self.assertIn('--output_base=' + str(self.root / 'bazel-output'), output)
        for flag in ('--nosystem_rc', '--nohome_rc', '--noworkspace_rc'):
            self.assertIn(flag, output)
        self.assertEqual(output[-2:], ['info', 'output_base'])
        rc = (self.root / 'private.bazelrc').read_text()
        for option in ('--repository_cache=', '--repo_contents_cache=', '--disk_cache=\n',
                       '--remote_cache=\n', '--remote_executor=\n'):
            self.assertIn(option, rc)
        self.assertNotIn('docker', launcher.read_text())
        self.assertNotIn('auditwheel', launcher.read_text())

    def test_command_records_real_failure_and_elapsed(self):
        evidence = {'phases': []}
        commands = host.Commands(self.root, self.root, os.environ.copy(), evidence)
        with self.assertRaises(subprocess.CalledProcessError) as caught:
            commands.command('validator', [sys.executable, '-c', 'raise SystemExit(17)'])
        self.assertEqual(caught.exception.returncode, 17)
        self.assertEqual(evidence['phases'][0]['exit'], 17)
        self.assertGreater(evidence['phases'][0]['elapsed'], 0)

    def test_cancellation_retires_owned_group_and_preserves_unrelated_process(self):
        foreign = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])
        self.addCleanup(lambda: foreign.poll() is None and foreign.kill())
        self.addCleanup(lambda: foreign.poll() is None and foreign.terminate())
        evidence = {'phases': []}
        commands = host.Commands(self.root, self.root, os.environ.copy(), evidence)
        timer = threading.Timer(.2, lambda: commands.interrupt(signal.SIGINT, None))
        timer.start()
        try:
            with self.assertRaises(InterruptedError):
                commands.command('owned', [sys.executable, '-c', 'import time; time.sleep(30)'])
        finally:
            timer.join()
        self.assertIsNone(foreign.poll())
        self.assertIsNotNone(evidence['phases'][0]['exit'])
        foreign.terminate()
        foreign.wait(timeout=5)
        # Cleanup remains callable after cancellation and does not replace its signal.
        self.assertEqual(commands.command('shutdown', [sys.executable, '-c', 'print("retired")'], cleanup=True).strip(), 'retired')
        self.assertEqual(commands.cancelled, signal.SIGINT)

    def test_failed_parent_retires_term_resistant_descendant_and_preserves_first_exit(self):
        foreign = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])
        pidfile = self.root / 'owned-descendant.json'
        parent = (
            'import json,pathlib,subprocess,sys,time; '
            'child=subprocess.Popen([sys.executable,"-c",'
            '"import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(30)"]); '
            'time.sleep(.1); '
            'p=pathlib.Path("/proc")/str(child.pid)/"stat"; '
            f'pathlib.Path({str(pidfile)!r}).write_text(json.dumps({{"pid":child.pid,"start":p.read_text().rsplit(")",1)[1].split()[19]}})); '
            'sys.exit(23)')
        evidence = {'phases': []}
        commands = host.Commands(self.root, self.root, os.environ.copy(), evidence)
        try:
            with self.assertRaises(subprocess.CalledProcessError) as caught:
                commands.command('failed-parent', [sys.executable, '-c', parent])
            self.assertEqual(caught.exception.returncode, 23)
            self.assertIsNone(foreign.poll())
            identity = json.loads(pidfile.read_text())
            stat = Path('/proc') / str(identity['pid']) / 'stat'
            if stat.exists():
                fields = stat.read_text().rsplit(')', 1)[1].split()
                self.assertTrue(fields[19] != identity['start'] or fields[0] == 'Z')
            phase = evidence['phases'][0]
            self.assertEqual(phase['exit'], 23)
            self.assertEqual(phase['cleanup'], 'complete')
            self.assertIn(identity['pid'], [entry['pid'] for entry in phase['owned_processes']])
            self.assertEqual(commands.unresolved, [])
        finally:
            foreign.terminate()
            foreign.wait(timeout=5)
            if pidfile.exists():
                identity = json.loads(pidfile.read_text())
                stat = Path('/proc') / str(identity['pid']) / 'stat'
                if stat.exists():
                    fields = stat.read_text().rsplit(')', 1)[1].split()
                    if fields[19] == identity['start'] and fields[0] != 'Z':
                        os.kill(identity['pid'], signal.SIGKILL)

    def test_unresolved_cleanup_does_not_replace_actual_phase_failure(self):
        evidence = {'phases': []}
        commands = host.Commands(self.root, self.root, os.environ.copy(), evidence)
        with mock.patch.object(commands, 'retire_group', side_effect=RuntimeError('fixture cleanup unresolved')):
            with self.assertRaises(subprocess.CalledProcessError) as caught:
                commands.command('first-error', [sys.executable, '-c', 'raise SystemExit(23)'])
        self.assertEqual(caught.exception.returncode, 23)
        self.assertEqual(evidence['phases'][0]['cleanup'], 'unresolved')
        self.assertEqual(commands.unresolved, ['first-error'])

    def test_existing_backend_receives_private_metrics_and_queries_preserve_them(self):
        from build_tools.wheel_tools import python_packaging as backend
        metrics = self.root / 'native-metrics'
        native = self.root / 'native-fixture'
        native.write_text('native output fixture')
        executable = self.root / 'fake-bazel'
        executable.write_text('#!' + sys.executable + '\n' +
            'import os,pathlib,sys\n'
            'args=sys.argv[1:]\n'
            'if args[0] == "build":\n'
            '  root=pathlib.Path(os.environ["HRX_BAZEL_METRICS_DIR"])\n'
            '  assert "--profile="+str(root/"native-build.profile.gz") in args\n'
            '  assert "--build_event_json_file="+str(root/"native-build.events.json") in args\n'
            '  (root/"native-build.profile.gz").write_bytes(b"preserved profile fixture")\n'
            '  (root/"native-build.events.json").write_text("preserved event fixture")\n'
            'elif args[0] == "cquery":\n'
            '  assert not any(a.startswith(("--profile=","--build_event_json_file=")) for a in args)\n'
            f'  print({str(native)!r})\n'
            'else: raise SystemExit(2)\n')
        executable.chmod(0o700)
        environment = host.clean_environment(dict(os.environ, HRX_BAZEL_METRICS_DIR='/foreign/metrics'))
        self.assertNotIn('HRX_BAZEL_METRICS_DIR', environment)
        environment['HRX_BAZEL_METRICS_DIR'] = str(metrics)
        context = backend.BazelBuildContext(executable, self.root, (), environment)
        result = backend.build_native_executables(context)
        self.assertEqual(result['loom-serve'], native)
        context.query('//:another-output-query')
        self.assertEqual((metrics / 'native-build.profile.gz').read_bytes(), b'preserved profile fixture')
        self.assertEqual((metrics / 'native-build.events.json').read_text(), 'preserved event fixture')

    def test_launcher_normalizes_exact_public_version_request(self):
        executable = self.root / 'strict-bazel'
        executable.write_text('#!' + sys.executable + '\nimport sys\n'
                              'args=sys.argv[1:]\n'
                              'if "--version" in args: raise SystemExit(2)\n'
                              'assert args[-2:] == ["version", "--gnu_format"], args\n'
                              'assert any(a.startswith("--output_base=") for a in args), args\n'
                              'print("bazel 9.2.0")\n')
        executable.chmod(0o700)
        launcher = self.root / 'bazel'
        host.write_launcher(launcher, executable, self.root, self.root / 'source')
        result = subprocess.run([launcher, '--version'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), 'bazel 9.2.0')
        # Additional arguments must not be silently normalized or dropped.
        malformed = subprocess.run([launcher, '--version', 'extra'], capture_output=True, text=True)
        self.assertEqual(malformed.returncode, 2)

    def test_cache_lease_contention_identity_and_normal_release(self):
        cache = self.root / 'private-cache'
        identity = {'host': 'fixture-host', 'compiler_sha256': 'selected-bytes'}
        lease = host.CacheLease(cache, identity, self.root / 'run-one')
        store = lease.acquire()
        self.assertEqual(store, cache / 'bazel-actions')
        self.assertTrue((cache / 'active.json').exists())
        code = ('import sys; from pathlib import Path; '
                f'sys.path.insert(0,{str(host.ROOT)!r}); '
                'from build_tools.wheel_tools.host_wheel import CacheLease; '
                f'CacheLease(Path({str(cache)!r}),{identity!r},Path({str(self.root / "run-two")!r})).acquire()')
        result = subprocess.run([sys.executable, '-E', '-s', '-c', code], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('leased by another invocation', result.stderr)
        lease.close(cleanup_complete=True)
        self.assertFalse((cache / 'active.json').exists())
        wrong = host.CacheLease(cache, dict(identity, compiler_sha256='changed'), self.root / 'run-two')
        with self.assertRaisesRegex(ValueError, 'identity differs'):
            wrong.acquire()
        same = host.CacheLease(cache, identity, self.root / 'run-two')
        same.acquire()
        same.close(cleanup_complete=True)

    def test_interrupted_cache_owner_blocks_reuse_until_confirmed_retired(self):
        cache = self.root / 'interrupted-cache'
        identity = {'host': 'fixture'}
        code = ('import os,sys; from pathlib import Path; '
                f'sys.path.insert(0,{str(host.ROOT)!r}); '
                'from build_tools.wheel_tools.host_wheel import CacheLease; '
                f'CacheLease(Path({str(cache)!r}),{identity!r},Path({str(self.root / "interrupted-run")!r})).acquire(); '
                'os._exit(0)')
        child = subprocess.Popen([sys.executable, '-E', '-s', '-c', code])
        self.assertEqual(child.wait(timeout=5), 0)
        owner = json.loads((cache / 'active.json').read_text())
        self.assertEqual(owner['pid'], child.pid)
        with self.assertRaisesRegex(ValueError, 'ownership is unresolved'):
            host.CacheLease(cache, identity, self.root / 'next').acquire()
        # This fixture created no compiler/server/descendants and its exact owner
        # has exited. Model the documented operator recovery after that proof.
        self.assertFalse((self.root / 'interrupted-run/bazel-output').exists())
        (cache / 'active.json').unlink()
        recovered = host.CacheLease(cache, identity, self.root / 'next')
        recovered.acquire()
        recovered.close(cleanup_complete=True)

    def test_selected_cache_identity_is_bounded_and_python_payload_is_not_a_key(self):
        source = self.root / 'source'
        source.mkdir()
        (source / '.bazelversion').write_text('9.2.0\n')
        sdk = self.root / 'sdk'
        sdk.mkdir()
        receipt = sdk / '.hrx-rocm-artifacts.json'
        receipt.write_text('{"artifact_identity":"fixture"}\n')
        evidence = {'sdk': str(sdk), 'bazel': {'path': '/selected/bazel', 'sha256': 'bazel-bytes'},
                    'compilers': {'/selected/clang': 'compiler-bytes'}}
        first = host.cache_identity(evidence, source)
        (source / 'packaged.py').write_text('changed packaged marker\n')
        self.assertEqual(host.cache_identity(evidence, source), first)
        evidence['bazel']['version'] = 'bazel 9.2.0'
        self.assertNotIn('version', first['bazel'])  # Retained identity is not mutated by later per-run evidence.
        receipt.write_text('{"artifact_identity":"changed"}\n')
        self.assertNotEqual(host.cache_identity(evidence, source), first)
        self.assertNotIn('source_commit', first)
        self.assertNotIn('host_fingerprint', first)

    def test_cache_mode_requires_directory_and_paths_cannot_overlap(self):
        arguments = argparse.Namespace(run_dir=str(self.root / 'run'), venv=sys.prefix,
                                       rocm=str(self.root / 'sdk'), bazel=str(self.root / 'bazel'),
                                       cache_dir=None, cache_mode='disabled')
        with self.assertRaisesRegex(ValueError, 'requires --cache-dir'):
            host.run_build(arguments, host.clean_environment(os.environ))
        arguments.cache_dir = arguments.run_dir
        with self.assertRaisesRegex(ValueError, 'must not overlap'):
            host.run_build(arguments, host.clean_environment(os.environ))
        arguments.cache_dir = ''
        with self.assertRaisesRegex(ValueError, 'absolute external path'):
            host.run_build(arguments, host.clean_environment(os.environ))

    def test_cache_rejects_unsafe_layout_and_symlinks(self):
        for name in ('unexpected', 'symlink', 'foreign-mode'):
            cache = self.root / name
            cache.mkdir(mode=0o700)
            if name == 'unexpected':
                (cache / 'foreign').write_text('do not adopt')
            elif name == 'symlink':
                (cache / 'identity.json').symlink_to(self.root / 'foreign-identity')
            else:
                cache.chmod(0o755)
            with self.subTest(name=name), self.assertRaises(ValueError):
                host.CacheLease(cache, {}, self.root / 'run').acquire()

    def test_cache_modes_share_empty_stamps_and_control_disables_disk_results(self):
        clean = host.clean_environment(dict(os.environ, HRX_BAZEL_CACHE_ENABLED='1',
                                             HRX_BAZEL_CACHE_DIR='/ambient',
                                             HRX_WHEEL_COMPILER_INPUTS='container-fingerprint'))
        self.assertEqual(clean['HRX_BAZEL_CACHE_ENABLED'], '0')
        self.assertNotIn('HRX_WHEEL_COMPILER_INPUTS', clean)
        self.assertNotIn('HRX_BAZEL_CACHE_DIR', clean)
        source = self.root / 'source'
        bodies = []
        for name, disk in (('disabled', None), ('read-write', self.root / 'cache/bazel-actions')):
            run = self.root / name
            run.mkdir()
            host.write_launcher(run / 'bazel', self.root / 'actual-bazel', run, source, disk)
            body = (run / 'private.bazelrc').read_text()
            bodies.append(body)
            self.assertIn('build --disk_cache=' + (str(disk) if disk else '') + '\n', body)
            self.assertIn('build --remote_cache=\n', body)
            self.assertIn('build --remote_executor=\n', body)
            self.assertIn('common --action_env=HRX_WHEEL_COMPILER_INPUTS=\n', body)
            self.assertIn('common --host_action_env=HRX_WHEEL_COMPILER_INPUTS=\n', body)
        executable = self.root / 'echo-bazel'
        executable.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n')
        executable.chmod(0o700)
        launcher = self.root / 'controlled-launcher'
        host.write_launcher(launcher, executable, self.root, source, self.root / 'cache/bazel-actions')
        arguments = subprocess.check_output([launcher, 'build', '--config=opt', '--disk_cache=/ambient', '//:target'], text=True).splitlines()
        self.assertEqual(arguments[-5:], ['--disk_cache=' + str(self.root / 'cache/bazel-actions'),
                                         '--remote_cache=', '--remote_executor=',
                                         '--action_env=HRX_WHEEL_COMPILER_INPUTS=',
                                         '--host_action_env=HRX_WHEEL_COMPILER_INPUTS='])
        from build_tools.wheel_tools.python_packaging import wheel_build_options
        with mock.patch.dict(os.environ, HRX_BAZEL_CACHE_ENABLED='1'):
            self.assertNotIn('--disk_cache=', wheel_build_options(self.root))
        with mock.patch.dict(os.environ, HRX_BAZEL_CACHE_ENABLED='0'):
            self.assertIn('--disk_cache=', wheel_build_options(self.root))

    def test_current_standalone_venv_contract_and_missing_module_failure(self):
        result = host.verify_venv(Path(sys.prefix).resolve())
        self.assertEqual(result['executable'], sys.executable)
        with mock.patch.object(host.importlib, 'import_module', side_effect=ImportError('missing build dependency')):
            with self.assertRaises(ImportError):
                host.verify_venv(Path(sys.prefix).resolve())

    def test_build_frontend_delivers_no_user_cfg_to_actual_backend(self):
        project = self.root / 'packaging-fixture'
        project.mkdir()
        home = self.root / 'fixture-home'
        home.mkdir()
        (home / '.pydistutils.cfg').write_text('[bdist_wheel]\nplat_name = unsupported_user_platform\n')
        (project / 'pyproject.toml').write_text('[build-system]\nrequires = ["setuptools>=77", "wheel>=0.45"]\nbuild-backend = "setuptools.build_meta"\n')
        (project / 'setup.py').write_text(
            'import sys\nfrom setuptools import setup\n'
            'print("BACKEND_ARGV=" + repr(sys.argv))\n'
            'setup(name="routing-fixture", version="1.0", py_modules=[])\n')
        environment = host.clean_environment(os.environ)
        environment['HOME'] = str(home)
        result = subprocess.run([sys.executable, '-m', 'build', '--wheel', '--no-isolation',
                                 '--config-setting=--global-option=--no-user-cfg'],
                                cwd=project, env=environment, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("'--no-user-cfg'", result.stdout)
        self.assertEqual([path.name for path in (project / 'dist').glob('*.whl')],
                         ['routing_fixture-1.0-py3-none-any.whl'])

    def test_runner_build_archive_failures_preserve_exit_and_always_shutdown(self):
        source = self.git_source()
        (source / '.bazelversion').write_text('9.2.0\n')
        subprocess.run(['git', '-C', source, 'add', '.'], check=True)
        subprocess.run(['git', '-C', source, '-c', 'user.name=fixture', '-c', 'user.email=fixture@example.invalid',
                        'commit', '-qm', 'pin fixture'], check=True)
        sdk = self.root / 'sdk'
        (sdk / 'include').mkdir(parents=True)
        (sdk / 'lib').mkdir()
        (sdk / 'lib/libhsa-runtime64.so.1').write_text('fixture')
        (sdk / 'bin').mkdir()
        for name in ('clang', 'clang++'):
            path = sdk / 'bin' / name
            path.write_text('never execute this fixture')
            path.chmod(0o700)
        bazel = self.root / 'actual-bazel-fixture'
        bazel.write_bytes(b'\x7fELF')
        with zipfile.ZipFile(bazel, 'a') as archive:
            archive.writestr('A-server.jar', 'fixture')
        bazel.chmod(0o700)
        for failed_phase, cache_mode in ((None, None), ('build', None), ('archive-validation', None), (None, 'disabled'), (None, 'read-write'), ('build', 'read-write'), ('archive-validation', 'disabled')):
            with self.subTest(failed_phase=failed_phase, cache_mode=cache_mode):
                run = self.root / ('run-' + str(failed_phase) + '-' + str(cache_mode))
                server = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])
                phases = []
                def fake_command(commands, phase, arguments, **kwargs):
                    phases.append((phase, list(map(str, arguments))))
                    if phase == failed_phase:
                        raise subprocess.CalledProcessError(23, arguments)
                    if phase == 'bazel-version':
                        return 'bazel 9.2.0\n'
                    if phase == 'bazel-root':
                        return str(run / 'bazel-output')
                    if phase == 'bazel-pid':
                        return str(server.pid)
                    if phase == 'build':
                        import shutil
                        shutil.copyfile(self.wheel(), run / 'wheel/hrx_solutions-1.0-py3-none-linux_x86_64.whl')
                    if phase == 'bazel-shutdown':
                        server.terminate()
                        server.wait(timeout=5)
                    return ''
                arguments = argparse.Namespace(run_dir=str(run), venv=sys.prefix, rocm=str(sdk), bazel=str(bazel),
                                               cache_dir=str(self.root / 'runner-cache') if cache_mode else None, cache_mode=cache_mode)
                try:
                    with mock.patch.object(host, 'ROOT', source), mock.patch.object(host.Commands, 'command', fake_command):
                        code = host.run_build(arguments, host.clean_environment(os.environ))
                    self.assertEqual(code, 0 if failed_phase is None else 23)
                    summary = json.loads((run / 'summary.json').read_text())
                    self.assertEqual(summary['cleanup']['status'], 'complete')
                    if cache_mode:
                        self.assertEqual(summary['cache']['mode'], cache_mode)
                        self.assertEqual(summary['environment']['HRX_BAZEL_CACHE_ENABLED'], '1')
                        self.assertFalse((self.root / 'runner-cache/active.json').exists())
                        cache_flag = 'build --disk_cache=' + (str(self.root / 'runner-cache/bazel-actions') if cache_mode == 'read-write' else '') + '\n'
                        self.assertIn(cache_flag, (run / 'private.bazelrc').read_text())
                    else:
                        self.assertNotIn('cache', summary)
                        self.assertEqual(summary['environment']['HRX_BAZEL_CACHE_ENABLED'], '0')
                    self.assertEqual(phases[-1][0], 'bazel-shutdown')
                    build = next(args for phase, args in phases if phase == 'build')
                    self.assertIn('--no-isolation', build)
                    self.assertIn('--config-setting=--global-option=--no-user-cfg', build)
                    self.assertNotIn('docker', json.dumps(phases))
                    self.assertNotIn('auditwheel', json.dumps(phases))
                    self.assertEqual(summary['environment']['PATH'], str(Path(sys.prefix) / 'bin') + ':/usr/bin:/bin')
                    self.assertEqual(summary['environment']['HRX_BAZEL_METRICS_DIR'], str(run / 'native-metrics'))
                    self.assertTrue((run / 'native-metrics').is_dir())
                    recorded = {phase['phase'] for phase in summary['phases']}
                    self.assertTrue({'preflight', 'capture'}.issubset(recorded))
                    if failed_phase is None:
                        self.assertTrue({'host-tag-validation', 'source-verification'}.issubset(recorded))
                finally:
                    if server.poll() is None:
                        server.kill()
                    server.wait(timeout=5)

    def test_bootstrap_help_ignores_python_import_overrides(self):
        environment = dict(os.environ, PYTHONPATH='/nonexistent/injected', PYTHONHOME='/nonexistent/home')
        result = subprocess.run(['bash', str(host.ROOT / 'build_tools/build-wheel.sh'), '--fast', '--help'],
                                env=environment, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--run-dir', result.stdout)


if __name__ == '__main__':
    unittest.main()
