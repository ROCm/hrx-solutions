# Copyright 2026 The HRX Authors
# Licensed under the MIT License.
# SPDX-License-Identifier: MIT
"""One local invocation of the shared CI checks; no publishing or host setup."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid

from build_tools.ci_tools import local_ci_source as source
from build_tools.ci_tools import local_ci_runtime as runtime

STAGES = ('preflight', 'capture', 'clients-source', 'source-tests', 'pi-source', 'build',
          'installed-cpu', 'installed-pi', 'installed-clients', 'gpu', 'openwebui')
NATIVE_OVERRIDES = ('HSA_OVERRIDE_GFX_VERSION', 'IREE_HAL_AMDGPU_LIBHSA_PATH',
                    'IREE_HAL_AMDGPU_LIBAQLPROFILE_PATH', 'LD_LIBRARY_PATH', 'LD_PRELOAD',
                    'LD_AUDIT', 'IREE_ROCM_PATH', 'CC', 'CXX', 'CMAKE_PREFIX_PATH')
MODEL_NAME = 'Qwen3.5-0.8B-UD-Q5_K_XL.gguf'


class Failure(Exception):
    def __init__(self, message, code=1):
        super().__init__(message)
        self.code = code


def atomic_json(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


def cache_root(env):
    return str(Path(env.get('HF_HUB_CACHE') or env.get('HUGGINGFACE_HUB_CACHE') or
                    (str(Path(env['HF_HOME']) / 'hub') if env.get('HF_HOME') else
                     str(Path(env.get('XDG_CACHE_HOME', str(Path.home() / '.cache'))) /
                         'huggingface/hub'))).expanduser().resolve())


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--cpu-only', action='store_true')
    parser.add_argument('--model08', type=Path)
    parser.add_argument('--run-dir', type=Path)
    # argparse normally accepts duplicates; a typo must not silently select one.
    options = [x.split('=', 1)[0] for x in argv if x.startswith('--')]
    if len(options) != len(set(options)):
        parser.error('duplicate options are not supported')
    args = parser.parse_args(argv)
    if args.cpu_only and args.model08:
        parser.error('--model08 does not apply to --cpu-only')
    return args


class Runner:
    def __init__(self, root, args):
        self.root = root.resolve()
        self.args = args
        workspace = Path(os.environ.get('CODEX_PROJECT_WORKSPACES',
                                         str(Path.home() / 'codex/project-workspaces'))) / 'hrx-solutions'
        self.workspace = workspace.resolve()
        self.run = (args.run_dir or self.workspace / 'gates/local-ci' /
                    (time.strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:8])).expanduser().resolve()
        if self.run == self.root or self.run.is_relative_to(self.root):
            raise Failure('Run directory must be new and outside the source checkout', 2)
        if self.run.exists():
            raise Failure(f'Run directory already exists: {self.run}', 2)
        self.run.mkdir(parents=True, mode=0o700)
        print(f'Local CI results: {self.run}', flush=True)
        self.snapshot = self.run / 'source'
        self.env = dict(os.environ)
        # Stage working directories differ. Keep relative caller cache locations
        # anchored to the invocation directory instead of relocating their data.
        for name in ('HF_HUB_CACHE', 'HUGGINGFACE_HUB_CACHE', 'HF_HOME', 'XDG_CACHE_HOME'):
            if self.env.get(name):
                self.env[name] = str(Path(self.env[name]).expanduser().absolute())
        for name in ('PYTHONPATH', 'PYTHONHOME', 'GITHUB_RUN_ID', 'GITHUB_RUN_ATTEMPT', *NATIVE_OVERRIDES):
            self.env.pop(name, None)
        self.env.pop('HRX_LOCAL_CI_RUNTIME_ROOT', None)
        self.runtime_owner = runtime.create(self.run)
        self.runtime = Path(self.runtime_owner['root'])
        print(f'Private runtime state (retained): {self.runtime}', flush=True)
        self.env.update(PYTHONNOUSERSITE='1', LOOM_SERVE_DIR=str(self.runtime / 'source/state'),
                        TMPDIR=str(self.runtime / 'tmp'))
        self.summary = {'schema_version': 1, 'mode': 'cpu-only' if args.cpu_only else 'full',
                        'status': 'running', 'run_dir': str(self.run), 'source': None,
                        'runtime': dict(self.runtime_owner, retained=True),
                        'wheel': None, 'failure': None, 'cleanup': {'status': 'complete'},
                        'invocation': {'github_run_id': os.environ.get('GITHUB_RUN_ID', 'local'),
                                       'github_run_attempt': os.environ.get('GITHUB_RUN_ATTEMPT', '1')},
                        'build': {name: os.environ.get(name, default) for name, default in
                                  (('HRX_RELEASE_TYPE', 'nightly'), ('HRX_RUN_ID', ''),
                                   ('HRX_ARTIFACT_SET', 'core'), ('HRX_BAZEL_CACHE_DIR', ''),
                                   ('HRX_BAZEL_CACHE_ENABLED', '1'))},
                        'model': {'mode': 'skipped' if args.cpu_only else
                                  ('local' if args.model08 else 'huggingface'),
                                  'cache_root': cache_root(self.env)},
                        'stages': {name: {'status': 'not-run'} for name in STAGES}}
        if args.cpu_only:
            for name in ('gpu', 'openwebui'):
                self.summary['stages'][name] = {'status': 'skipped', 'reason': 'explicit --cpu-only'}
        self.child = None
        self.interrupted = None
        self.current = None
        self.manifest = None
        self.save()

    def save(self):
        atomic_json(self.run / 'summary.json', self.summary)

    def on_signal(self, number, frame):
        if self.interrupted:
            return
        self.interrupted = number
        if self.current in ('source-tests', 'installed-cpu'):
            cleanup = {'status': 'unresolved', 'stage': self.current,
                       'details': 'Interrupted test fixtures may own detached native/logger sessions; '
                                  'driver exit cannot prove their retirement. Retain this run for inspection.',
                       'temporary_files': str(self.runtime / 'tmp'),
                       'source_state': str(self.runtime / 'source/state'),
                       'installed_state': str(self.runtime / 'installed'),
                       'log': str(self.run / f'{self.current}.log')}
            self.summary['cleanup'] = cleanup
            self.summary['stages'][self.current]['cleanup'] = cleanup
        if self.child and self.child.poll() is None:
            # Shared build/model wrappers reconcile their own native/container work.
            if self.current in ('build', 'gpu', 'openwebui'):
                self.child.send_signal(number)
            else:
                os.killpg(self.child.pid, number)

    def command(self, command, cwd=None, env=None, cooperative=False):
        if self.interrupted:
            raise Failure('Interrupted', 128 + self.interrupted)
        print('+ ' + ' '.join(map(str, command)), flush=True)
        log = self.run / f'{self.current}.log'
        command_env = dict(self.env)
        if env:
            command_env.update(env)
        self.child = subprocess.Popen(list(map(str, command)), cwd=cwd or self.snapshot,
                                      env=command_env, stdout=subprocess.PIPE,
                                      stderr=subprocess.STDOUT, start_new_session=True)
        child = self.child
        if self.interrupted:
            if cooperative:
                child.send_signal(self.interrupted)
            else:
                os.killpg(child.pid, self.interrupted)
        def pump():
            with log.open('ab') as output:
                while True:
                    data = child.stdout.read1(65536)
                    if not data:
                        break
                    output.write(data)
                    output.flush()
                    sys.stdout.buffer.write(data)
                    sys.stdout.buffer.flush()
        thread = threading.Thread(target=pump, daemon=True)
        thread.start()
        cancellation_started = None
        while child.poll() is None:
            if self.interrupted:
                cancellation_started = cancellation_started or time.monotonic()
                if time.monotonic() - cancellation_started > (90 if cooperative else 10):
                    if cooperative:
                        self.summary['cleanup'] = {'status': 'unresolved',
                                                  'details': 'Cooperative stage did not exit within 90s'}
                        raise Failure('Cleanup unresolved; stage may still own live work',
                                      128 + self.interrupted)
                    os.killpg(child.pid, signal.SIGKILL)
            time.sleep(0.05)
        code = child.returncode
        thread.join(timeout=5)
        child.stdout.close()
        self.child = None
        if cooperative:
            cleanup_path = self.run / f'{self.current}-cleanup.json'
            if cleanup_path.is_file():
                try:
                    cleanup = json.loads(cleanup_path.read_text())
                except (OSError, ValueError) as error:
                    cleanup = {'status': 'unresolved', 'error': str(error)}
                self.summary['stages'][self.current]['cleanup'] = cleanup
                if cleanup.get('status') != 'complete':
                    self.summary['cleanup'] = {'status': 'unresolved', 'evidence': str(cleanup_path)}
            else:
                self.summary['cleanup'] = {'status': 'unresolved', 'details': 'Missing owned cleanup evidence'}
        if code:
            raise Failure(f'Command failed with exit {code}', code)
        if self.interrupted:
            raise Failure('Interrupted', 128 + self.interrupted)
        if cooperative and self.summary['cleanup']['status'] != 'complete':
            raise Failure('Owned cleanup is unresolved')
        return code

    def stage(self, name, action):
        if self.interrupted:
            raise Failure('Interrupted', 128 + self.interrupted)
        self.current = name
        record = self.summary['stages'][name]
        record.update(status='running', started=time.time(), log=str(self.run / f'{name}.log'))
        self.save()
        print(f'[{name}] starting', flush=True)
        try:
            action()
            if self.interrupted:
                raise Failure('Interrupted', 128 + self.interrupted)
        except BaseException as error:
            record.update(status='interrupted' if self.interrupted else 'failed',
                          exit_code=getattr(error, 'code', 1), error=str(error))
            raise
        else:
            record.update(status='passed', exit_code=0)
        finally:
            record['seconds'] = round(time.time() - record['started'], 3)
            self.save()

    def preflight(self):
        if platform.system() != 'Linux' or platform.machine() != 'x86_64':
            raise Failure('Local CI requires Linux x86-64')
        for tool in ('bash', 'git', 'python3', 'docker', 'curl', 'tar', 'timeout', 'realpath', 'sha256sum'):
            if not shutil.which(tool):
                raise Failure(f'Missing prerequisite: {tool}')
        self.command(['docker', 'info', '--format', '{{.ServerVersion}}'], cwd=self.root)
        if not self.args.cpu_only:
            renders = list(Path('/dev/dri').glob('renderD*'))
            access = os.R_OK | os.W_OK
            if not os.access('/dev/kfd', access) or not any(os.access(p, access) for p in renders):
                raise Failure('Full mode requires accessible /dev/kfd and a render node; use --cpu-only explicitly')
        if self.args.model08:
            self.args.model08 = self.args.model08.expanduser().resolve()
            for name in (MODEL_NAME, 'tokenizer.json'):
                if not (self.args.model08 / name).is_file():
                    raise Failure(f'Missing explicit model input: {self.args.model08 / name}')
        print('Build, Python and Pi dependency acquisition may require network access.', flush=True)

    def capture(self):
        captured = source.capture(self.root, self.snapshot)
        self.manifest = captured.pop('inputs')
        atomic_json(self.run / 'source-inputs.json', self.manifest)
        self.summary['source'] = captured
        self.summary['source']['manifest'] = str(self.run / 'source-inputs.json')

    def clients_source(self):
        # Stage immutable, client-specific public runtimes before Python tests,
        # so optional genuine loader tests are exercised rather than skipped.
        for client, assets in (('opencode', 'opencode_plugin'), ('omp', 'omp_extension')):
            self.command(['bash', f'build_tools/stage-{client}-runtime.sh',
                          self.run / f'{client}-runtime'])
            if client == 'opencode':
                self.env['PATH'] = str(self.run / 'opencode-runtime/node-v24.10.0-linux-x64/bin') + os.pathsep + self.env.get('PATH', '')
            self.command(['bash', f'build_tools/check-{client}-assets.sh',
                          self.run / f'{client}-runtime', self.snapshot / 'binding/python' / assets])
        node_bin = self.run / 'opencode-runtime/node-v24.10.0-linux-x64/bin'
        self.env.update(
            PATH=str(node_bin) + os.pathsep + self.env.get('PATH', ''),
            LOOM_OPENCODE_TEST_NODE=str(node_bin / 'node'),
            LOOM_OMP_TEST_BUN=str(self.run / 'omp-runtime/runtime/node_modules/.bin/bun'),
            LOOM_OMP_TEST_PACKAGE=str(self.run / 'omp-runtime/client/node_modules/@oh-my-pi/pi-coding-agent'))

    def source_tests(self):
        venv = self.run / 'source-venv'
        self.command(['python3', '-m', 'venv', venv])
        python = venv / 'bin/python'
        self.command([python, '-m', 'pip', 'install', '--disable-pip-version-check',
                      'setuptools>=77', 'wheel>=0.45', 'packaging>=24.2'])
        self.command([python, '-c', 'import os,site,sys,setuptools,packaging; from pathlib import Path; '
                      'assert sys.prefix != sys.base_prefix and not site.ENABLE_USER_SITE; '
                      'assert "include-system-site-packages = false" in (Path(sys.prefix)/"pyvenv.cfg").read_text(); '
                      'assert not os.environ.get("PYTHONPATH") and not os.environ.get("PYTHONHOME"); '
                      'assert all(Path(m.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()) '
                      'for m in (setuptools,packaging)); print(sys.executable,setuptools.__file__,packaging.__file__)'])
        for directory in ('build_tools', 'tests'):
            self.command([python, '-m', 'unittest', 'discover', '-s', directory, '-p', '*_test.py', '-v'])

    def pi_source(self):
        self.command(['bash', 'build_tools/stage-pi-runtime.sh', self.run / 'pi-runtime'])
        self.command(['bash', 'build_tools/check-pi-assets.sh', self.run / 'pi-runtime',
                      self.snapshot / 'binding/python/pi_extension'])

    def build(self):
        self.command(['bash', 'build_tools/build-wheel.sh'], env={
            'GITHUB_RUN_ID': self.summary['invocation']['github_run_id'],
            'GITHUB_RUN_ATTEMPT': self.summary['invocation']['github_run_attempt'],
            'HRX_LOCAL_CI_CLEANUP_FILE': str(self.run / 'build-cleanup.json'),
            'HRX_LOCAL_CI_CONTAINER_STATE': str(self.run / 'build-container.json')}, cooperative=True)
        wheels = list((self.snapshot / 'build/wheel/wheelhouse').glob('*.whl'))
        if len(wheels) != 1:
            raise Failure(f'Expected exactly one newly repaired wheel, found {len(wheels)}')
        self.wheel = wheels[0].resolve()
        self.summary['wheel'] = {'path': str(self.wheel),
                                 'sha256': hashlib.sha256(self.wheel.read_bytes()).hexdigest()}

    def installed_cpu(self):
        self.command(['bash', self.snapshot / 'build_tools/check-wheel.sh', self.wheel,
                      self.run / 'installed-venv'], cwd=self.run,
                     env={'HRX_LOCAL_CI_RUNTIME_ROOT': str(self.runtime / 'installed')})

    def installed_pi(self):
        python = self.run / 'installed-venv/bin/python'
        # Resolve through that interpreter outside the checkout; then assert ownership.
        code = ('import hrx_solutions,sys; from pathlib import Path; '
                'p=Path(hrx_solutions.__file__).resolve(); '
                'assert p.is_relative_to(Path(sys.prefix).resolve()); '
                'print(p.parent / "pi_extension")')
        result = self.run / 'installed-assets.txt'
        self.command([python, '-c', code + f'; Path({str(result)!r}).write_text(str(p.parent / "pi_extension"))'],
                     cwd=self.run)
        self.command(['bash', self.snapshot / 'build_tools/check-pi-assets.sh', self.run / 'pi-runtime',
                      result.read_text()], cwd=self.run)

    def installed_clients(self):
        python = self.run / 'installed-venv/bin/python'
        for client, assets in (('opencode', 'opencode_plugin'), ('omp', 'omp_extension')):
            result = self.run / f'installed-{client}-assets.txt'
            code = ('import hrx_solutions,sys; from pathlib import Path; '
                    'p=Path(hrx_solutions.__file__).resolve(); '
                    'assert p.is_relative_to(Path(sys.prefix).resolve()); '
                    f'Path({str(result)!r}).write_text(str(p.parent / {assets!r}))')
            self.command([python, '-c', code], cwd=self.run)
            self.command(['bash', self.snapshot / f'build_tools/check-{client}-assets.sh',
                          self.run / f'{client}-runtime', result.read_text()], cwd=self.run)

    def gpu(self):
        self.locked_model_gate(False)

    def openwebui(self):
        self.locked_model_gate(True)

    def locked_model_gate(self, webui):
        lock = self.workspace / 'gates/qwen08-gpu.lock'
        lock.parent.mkdir(parents=True, exist_ok=True)
        with lock.open('a') as stream:
            print(f'Waiting for GPU stage lock: {lock}', flush=True)
            while True:
                if self.interrupted:
                    raise Failure('Interrupted waiting for GPU lock', 128 + self.interrupted)
                try:
                    fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    time.sleep(0.2)
            command = ['bash', self.snapshot / 'build_tools/model-smoke.sh', self.wheel, self.run / 'model']
            stage = 'openwebui' if webui else 'gpu'
            if webui:
                command = ['bash', self.snapshot / 'build_tools/model-smoke.sh', '--openwebui',
                           self.wheel, self.run / 'openwebui']
                fixture = self.args.model08 or (Path.home() / 'Downloads/gguf/models/multimodal/qwen/qwen3.5-0.8b')
                for name in (MODEL_NAME, 'tokenizer.json'):
                    if not (fixture / name).is_file():
                        raise Failure(f'Missing immutable WebUI gate fixture: {fixture / name}; use --model08 DIRECTORY')
                command.append(fixture)
            elif self.args.model08:
                command.append(self.args.model08)
            self.command(command, cwd=self.run, env={
                'LOOM_PI_RUNTIME': str(self.run / 'pi-runtime'),
                'LOOM_OPENCODE_RUNTIME': str(self.run / 'opencode-runtime'),
                'LOOM_OMP_RUNTIME': str(self.run / 'omp-runtime'),
                'HRX_LOCAL_CI_RUNTIME_ROOT': str(self.runtime / ('openwebui' if webui else 'model')),
                'HRX_LOCAL_CI_CLEANUP_FILE': str(self.run / f'{stage}-cleanup.json')}, cooperative=True)

    def execute(self):
        old_handlers = {number: signal.signal(number, self.on_signal)
                        for number in (signal.SIGINT, signal.SIGTERM)}
        exit_code = 0
        try:
            actions = (self.preflight, self.capture, self.clients_source, self.source_tests, self.pi_source,
                       self.build, self.installed_cpu, self.installed_pi, self.installed_clients, self.gpu, self.openwebui)
            for name, action in zip(STAGES, actions):
                if name in ('gpu', 'openwebui') and self.args.cpu_only:
                    continue
                self.stage(name, action)
            if self.summary['cleanup']['status'] != 'complete':
                raise Failure('Owned cleanup is unresolved')
            self.summary['status'] = 'passed'
        except (Exception, KeyboardInterrupt) as error:
            exit_code = 128 + self.interrupted if self.interrupted else 1
            self.summary.update(status='interrupted' if self.interrupted else 'failed',
                                failure={'stage': self.current, 'exit_code': getattr(error, 'code', 1),
                                         'message': str(error)})
            print(f'Local CI {self.summary["status"]}: {error}', file=sys.stderr, flush=True)
        finally:
            if self.manifest:
                try:
                    source.verify(self.snapshot, self.manifest)
                    self.summary['source']['inputs_unchanged'] = True
                except Exception as error:
                    self.summary['source']['inputs_unchanged'] = False
                    self.summary['source']['mutation_error'] = str(error)
                    if exit_code == 0:
                        exit_code = 1
                        self.summary.update(status='failed', failure={'stage': 'source-integrity',
                                            'exit_code': 1, 'message': str(error)})
            self.summary['exit_code'] = exit_code
            self.save()
            for number, handler in old_handlers.items():
                signal.signal(number, handler)
            print(f'Local CI {self.summary["status"]}; summary: {self.run / "summary.json"}', flush=True)
        return exit_code


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        return Runner(Path(__file__).resolve().parents[2], args).execute()
    except Failure as error:
        print(str(error), file=sys.stderr)
        return 2 if error.code == 2 else 1


if __name__ == '__main__':
    raise SystemExit(main())
