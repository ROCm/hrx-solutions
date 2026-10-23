#!/usr/bin/env python3
# Copyright 2026 The HRX Authors
# Licensed under the MIT License.
# SPDX-License-Identifier: MIT
"""Build an explicitly host-tagged wheel using private, disposable Bazel state."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import email.parser
import fcntl
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shlex
import signal
import stat
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[2]
REJECT_ENV = (
    '_PYTHON_HOST_PLATFORM', '_PYTHON_SYSCONFIGDATA_NAME', '_PYTHON_SYSCONFIGDATA_PATH',
    'PYTHONPLATLIBDIR', 'DIST_EXTRA_CONFIG', '_PYTHON_PROJECT_BASE', 'PLAT', 'PLAT_NAME',
    'CFLAGS', 'CPPFLAGS', 'CXXFLAGS', 'LDFLAGS', 'LDSHARED', 'AR', 'ARCHFLAGS',
    'SETUPTOOLS_USE_DISTUTILS', 'SETUPTOOLS_EXT_SUFFIX',
)
CLEAR_ENV = (
    'PYTHONPATH', 'PYTHONHOME', 'PYTHONUSERBASE', 'LD_PRELOAD', 'LD_AUDIT',
    'LD_LIBRARY_PATH', 'LIBRARY_PATH', 'CPATH', 'C_INCLUDE_PATH', 'CPLUS_INCLUDE_PATH',
    'GCC_EXEC_PREFIX', 'COMPILER_PATH', 'BASH_ENV', 'ENV', 'HRX_WHEEL_COMPILER_INPUTS',
)


def clean_environment(original):
    forbidden = [name for name in REJECT_ENV if name in original]
    if forbidden:
        raise ValueError('Unsupported host build environment: ' + ', '.join(forbidden))
    environment = {key: value for key, value in original.items()
                   if key not in CLEAR_ENV and not key.startswith(('BAZEL', 'HRX_BAZEL'))}
    environment.update(PYTHONNOUSERSITE='1', HRX_BAZEL_CACHE_ENABLED='0')
    return environment


def sha256(path):
    with path.open('rb') as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
        return digest.hexdigest()


def external_path(value, root, *, new=False):
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ValueError(f'An absolute external path is required: {value}')
    path = path.resolve()
    if path == root or path.is_relative_to(root):
        raise ValueError(f'Path must be outside the source checkout: {path}')
    if new and path.exists():
        raise ValueError(f'Run directory must not exist: {path}')
    return path


def audit_source(root, configured):
    for name in ('setup.cfg', '.bazelrc.local'):
        if (root / name).exists():
            raise ValueError(f'Unsupported local build configuration: {name}')
    path = root / '.bazelrc.configured'
    if path.exists() and path.read_text() != configured:
        raise ValueError('Unsupported .bazelrc.configured; only the selected setup-generated configuration is supported')
    # Local/module overrides omitted by capture must never silently disappear.
    if (root / 'MODULE.bazel.local').exists():
        raise ValueError('Unsupported MODULE.bazel.local')
    for name in ('.bazelrc', 'MODULE.bazel'):
        committed = subprocess.check_output(['git', '-C', str(root), 'show', 'HEAD:' + name])
        if (root / name).read_bytes() != committed:
            raise ValueError(f'Uncommitted build routing changes are unsupported: {name}')
    if 'local_path_override(' in (root / 'MODULE.bazel').read_text():
        raise ValueError('Local module overrides are not supported')


def verify_venv(venv):
    """Executed by the selected interpreter, before importing build dependencies."""
    if sys.prefix == sys.base_prefix or Path(sys.prefix).resolve() != venv:
        raise ValueError('Select a standalone venv with --venv')
    import site
    if site.ENABLE_USER_SITE or 'include-system-site-packages = false' not in (venv / 'pyvenv.cfg').read_text().lower():
        raise ValueError('The build venv must disable system and user site packages')
    from packaging.version import Version
    versions = {}
    for name, minimum in (('packaging', '24.2'), ('setuptools', '77'), ('wheel', '0.45'), ('build', '1.2')):
        module = importlib.import_module(name)
        if not Path(module.__file__).resolve().is_relative_to(venv):
            raise ValueError(f'{name} was imported outside the build venv')
        version = importlib.metadata.version(name)
        if Version(version) < Version(minimum):
            raise ValueError(f'{name}>={minimum} is required')
        versions[name] = {'version': version, 'module': module.__file__}
    import setuptools
    config = Path(setuptools.__file__).parent / '_distutils' / 'distutils.cfg'
    if config.exists() and config.read_text().strip():
        raise ValueError(f'Unsupported setuptools system configuration: {config}')
    # Fail closed if the frontend/backend no longer expose the reviewed routing.
    import setuptools.build_meta
    import inspect
    if '--global-option' not in inspect.getsource(setuptools.build_meta._ConfigSettingsTranslator):
        raise ValueError('Setuptools does not support the required --global-option routing')
    return {'executable': sys.executable, 'prefix': sys.prefix, 'base_prefix': sys.base_prefix,
            'dependencies': versions}


def validate_host_tags(wheel):
    from packaging.tags import Tag
    from packaging.utils import canonicalize_name, parse_wheel_filename
    from packaging.version import Version
    name, version, _, tags = parse_wheel_filename(wheel.name)
    expected = {Tag('py3', 'none', 'linux_x86_64')}
    if name != 'hrx-solutions' or tags != expected:
        raise ValueError('Host wheel filename must name hrx-solutions and exactly py3-none-linux_x86_64')
    with zipfile.ZipFile(wheel) as archive:
        paths = [name for name in archive.namelist() if name.endswith('.dist-info/WHEEL')]
        if len(paths) != 1:
            raise ValueError('Expected exactly one wheel metadata record')
        metadata = email.parser.Parser().parsestr(archive.read(paths[0]).decode())
        if metadata.get_all('Tag') != ['py3-none-linux_x86_64'] or metadata.get_all('Root-Is-Purelib') != ['false']:
            raise ValueError('Internal WHEEL must have the exact host tag and Root-Is-Purelib=false')
        package = email.parser.Parser().parsestr(archive.read(paths[0].removesuffix('WHEEL') + 'METADATA').decode())
        if canonicalize_name(package['Name']) != name or Version(package['Version']) != version:
            raise ValueError('Wheel filename and metadata name/version disagree')
        directory = paths[0].split('/')[-2].removesuffix('.dist-info')
        distribution, directory_version = directory.rsplit('-', 1)
        if canonicalize_name(distribution) != name or Version(directory_version) != version:
            raise ValueError('Wheel metadata directory disagrees with filename')


class CacheLease:
    """An ordinary same-stable-host developer cache, not a host fingerprint."""
    def __init__(self, path, identity, run):
        self.path, self.identity, self.run = path, identity, run
        self.fd = None
        self.active = False

    def acquire(self):
        self.path.mkdir(mode=0o700, parents=True, exist_ok=True)
        info = self.path.lstat()
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
            raise ValueError('Cache directory must be a same-user private directory (mode 0700)')
        self.inode = (info.st_dev, info.st_ino)
        self.fd = os.open(self.path / 'lock', os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        try:
            lock_info = os.fstat(self.fd)
            if not stat.S_ISREG(lock_info.st_mode) or lock_info.st_uid != os.getuid():
                raise ValueError('Cache lock must be a same-user regular file')
            try:
                fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise ValueError('Cache is leased by another invocation') from error
            allowed = {'lock', 'identity.json', 'active.json', 'bazel-actions'}
            for entry in self.path.iterdir():
                entry_info = entry.lstat()
                expected_kind = stat.S_ISDIR(entry_info.st_mode) if entry.name == 'bazel-actions' else stat.S_ISREG(entry_info.st_mode)
                if entry.name not in allowed or entry.is_symlink() or entry_info.st_uid != os.getuid() or not expected_kind:
                    raise ValueError('Cache has an unsupported entry or symlink')
            store = self.path / 'bazel-actions'
            if store.exists():
                if not store.is_dir():
                    raise ValueError('Cache action store is not a directory')
                for parent, directories, files in os.walk(store):
                    if any((Path(parent) / name).is_symlink() for name in directories + files):
                        raise ValueError('Cache action store contains a symlink')
            identity_path = self.path / 'identity.json'
            expected = dict(contract='host-wheel-stable-developer-v1', root=str(self.path),
                            uid=os.getuid(), device=info.st_dev, inode=info.st_ino,
                            selected_inputs=self.identity)
            if identity_path.exists():
                if json.loads(identity_path.read_text()) != expected:
                    raise ValueError('Cache selected host/Bazel/SDK/compiler identity differs; invalidate or use a fresh cache')
            elif store.exists() and any(store.iterdir()):
                raise ValueError('Cannot adopt an action store without its identity record')
            if (self.path / 'active.json').exists():
                raise ValueError('Prior cache invocation ownership is unresolved; confirm its owned lifecycle is retired before removing active.json')
            if not identity_path.exists():
                identity_path.write_text(json.dumps(expected, indent=2) + '\n')
            owner = dict(run_dir=str(self.run), pid=os.getpid(),
                         process_start=(Path('/proc') / str(os.getpid()) / 'stat').read_text().rsplit(')', 1)[1].split()[19])
            with (self.path / 'active.json').open('x') as stream:
                json.dump(owner, stream)
                stream.write('\n')
            self.active = True
            store.mkdir(mode=0o700, exist_ok=True)
            return store
        except Exception:
            self.close(cleanup_complete=True)
            raise

    def close(self, *, cleanup_complete):
        if self.fd is None:
            return
        try:
            if self.active and cleanup_complete:
                info = self.path.lstat()
                if (info.st_dev, info.st_ino) != self.inode or self.path.is_symlink():
                    raise RuntimeError('Cache root changed during its invocation lease')
                (self.path / 'active.json').unlink()
                self.active = False
        finally:
            os.close(self.fd)
            self.fd = None


def cache_identity(evidence, source):
    machine = Path('/etc/machine-id')
    receipt = Path(evidence['sdk']) / '.hrx-rocm-artifacts.json'
    return dict(host=platform.node(), machine_id_sha256=sha256(machine) if machine.is_file() else None,
                platform=[platform.system(), platform.machine()], bazel=dict(evidence['bazel']),
                bazel_version=(source / '.bazelversion').read_text().strip(),
                sdk=evidence['sdk'], sdk_receipt_sha256=sha256(receipt) if receipt.is_file() else None,
                compilers=evidence['compilers'])


def write_launcher(path, bazel, run, source, disk_cache=None):
    # This launcher owns startup flags even for backend info/mod/fetch queries.
    startup = [str(bazel), f'--output_user_root={run / "bazel-user"}',
               f'--output_base={run / "bazel-output"}', '--nosystem_rc', '--nohome_rc',
               '--noworkspace_rc', f'--bazelrc={source / ".bazelrc"}']
    # An explicit rc is loaded after startup rc suppression. No ambient cache or
    # repository content cache is borrowed; these paths start empty on every run.
    rc = run / 'private.bazelrc'
    rc.write_text(f'import {source / ".bazelrc"}\n' +
                  f'common --repository_cache={run / "repository-cache"}\n' +
                  f'common --repo_contents_cache={run / "repo-contents-cache"}\n' +
                  f'build --disk_cache={disk_cache or ""}\n' +
                  'build --remote_cache=\nbuild --remote_executor=\n' +
                  'common --action_env=HRX_WHEEL_COMPILER_INPUTS=\n' +
                  'common --host_action_env=HRX_WHEEL_COMPILER_INPUTS=\n')
    startup[-1] = f'--bazelrc={rc}'
    controlled = [f'--disk_cache={disk_cache or ""}', '--remote_cache=', '--remote_executor=',
                  '--action_env=HRX_WHEEL_COMPILER_INPUTS=', '--host_action_env=HRX_WHEEL_COMPILER_INPUTS=']
    path.write_text('#!/bin/sh\n'
                    'if [ "$#" -eq 1 ] && [ "$1" = --version ]; then\n'
                    '  set -- version --gnu_format\nfi\n'
                    # Command-line values follow --config expansion, so a source
                    # config cannot override the explicitly selected cache route.
                    'case "${1:-}" in build|cquery) set -- "$@" ' + shlex.join(controlled) + ';; esac\n'
                    'exec ' + shlex.join(startup) + ' "$@"\n')
    path.chmod(0o700)


class Commands:
    """Supervise only the current invocation's process groups; retain logs."""
    def __init__(self, run, cwd, environment, evidence):
        self.run, self.cwd, self.environment, self.evidence = run, cwd, environment, evidence
        self.cancelled = None
        self.unresolved = []

    def interrupt(self, signum, frame):
        if self.cancelled is None:
            self.cancelled = signum

    @staticmethod
    def group_members(group):
        members = []
        for path in Path('/proc').glob('[0-9]*/stat'):
            try:
                fields = path.read_text().rsplit(')', 1)[1].split()
                if int(fields[2]) == group:
                    members.append({'pid': int(path.parent.name), 'start': fields[19], 'state': fields[0]})
            except (FileNotFoundError, ProcessLookupError):
                continue
        return members

    def retire_group(self, child, record):
        # A failed parent can leave descendants after it exits. The PGID was
        # created by start_new_session for this exact phase, never inherited.
        record['owned_processes'] = self.group_members(child.pid)
        for sig, seconds in ((signal.SIGTERM, 1), (signal.SIGKILL, 3)):
            members = self.group_members(child.pid)
            if not any(member['state'] != 'Z' for member in members):
                break
            try:
                os.killpg(child.pid, sig)
            except ProcessLookupError:
                pass
            deadline = time.monotonic() + seconds
            while time.monotonic() < deadline:
                child.poll()  # Reap our direct child while descendants settle.
                if not any(member['state'] != 'Z' for member in self.group_members(child.pid)):
                    break
                time.sleep(.05)
        child.wait(timeout=5)
        remaining = self.group_members(child.pid)
        record['retired_processes'] = remaining  # Zombies have exited and cannot write state.
        if any(member['state'] != 'Z' for member in remaining):
            raise RuntimeError('Owned phase process group still has live work')
        record['cleanup'] = 'complete'

    def command(self, phase, args, *, cleanup=False, timeout=None):
        record = {'phase': phase, 'command': list(map(str, args)), 'start': time.time()}
        self.evidence['phases'].append(record)
        start = time.monotonic()
        child = None
        try:
            if self.cancelled and not cleanup:
                raise InterruptedError('Build interrupted')
            with (self.run / (phase + '.log')).open('w') as log, (self.run / (phase + '.stdout.log')).open('w') as stdout:
                child = subprocess.Popen(list(map(str, args)), cwd=self.cwd, env=self.environment,
                                         stdout=stdout, stderr=log, start_new_session=True)
                while child.poll() is None:
                    if (self.cancelled and not cleanup) or (timeout and time.monotonic() - start > timeout):
                        raise InterruptedError(f'{phase} interrupted or timed out')
                    time.sleep(.05)
                if child.returncode:
                    raise subprocess.CalledProcessError(child.returncode, args)
            return (self.run / (phase + '.stdout.log')).read_text()
        finally:
            original_error = sys.exc_info()[1]
            try:
                if child is not None:
                    self.retire_group(child, record)
            except Exception as error:
                record.update(cleanup='unresolved', cleanup_error=str(error))
                self.unresolved.append(phase)
                if original_error is None:
                    raise
            finally:
                record.update(end=time.time(), elapsed=time.monotonic() - start,
                              exit=child.returncode if child else None)


@contextmanager
def measured(evidence, phase):
    record = {'phase': phase, 'start': time.time(), 'exit': 0}
    evidence['phases'].append(record)
    start = time.monotonic()
    try:
        yield
    except Exception:
        record['exit'] = 1
        raise
    finally:
        record.update(end=time.time(), elapsed=time.monotonic() - start)


def run_build(args, environment):
    invocation_start, invocation_clock = time.time(), time.monotonic()
    root = ROOT.resolve()
    run = external_path(args.run_dir, root, new=True)
    venv = external_path(args.venv, root)
    sdk = external_path(args.rocm, root)
    bazel = external_path(args.bazel, root)
    cache = None
    cache_mode = getattr(args, 'cache_mode', None)
    if getattr(args, 'cache_dir', None) is not None:
        raw_cache = Path(args.cache_dir).expanduser()
        cache = external_path(args.cache_dir, root)
        if raw_cache != cache or root.is_relative_to(cache):
            raise ValueError('Cache must be a dedicated absolute path without symlink components')
        for selected in (run, venv, sdk, bazel):
            if selected.is_relative_to(cache) or cache.is_relative_to(selected):
                raise ValueError('Cache must not overlap the run, venv, SDK or Bazel')
        if any(character.isspace() for character in str(cache)):
            raise ValueError('Cache path must not contain whitespace')
        cache_mode = cache_mode or 'read-write'
    elif cache_mode is not None:
        raise ValueError('--cache-mode requires --cache-dir')
    if any(any(character.isspace() for character in str(path)) for path in (run, sdk, bazel, venv)):
        raise ValueError('Host build paths must not contain whitespace (Bazel rc configuration contract)')
    if platform.system() != 'Linux' or platform.machine() != 'x86_64':
        raise ValueError('Host wheel builds require Linux x86-64')
    identity = verify_venv(venv)
    from build_tools.wheel_tools.setup_python import validate_rocm_root, render_bazelrc, configured_values
    from build_tools.ci_tools import local_ci_source
    sdk, cc, cxx = validate_rocm_root(str(sdk))
    if not bazel.is_file() or not os.access(bazel, os.X_OK):
        raise ValueError('An executable pinned Bazel binary is required')
    with bazel.open('rb') as stream:
        native = stream.read(4) == b'\x7fELF'
    if not native or not zipfile.is_zipfile(bazel):
        raise ValueError('--bazel must be the actual self-extracting Linux Bazel executable, not Bazelisk or a wrapper')
    with zipfile.ZipFile(bazel) as archive:
        if 'A-server.jar' not in archive.namelist():
            raise ValueError('Bazel executable lacks its embedded server; Bazelisk is not supported')
    audit_source(root, render_bazelrc(configured_values(sdk, cc, cxx)))
    # Overlap could mutate supposedly immutable inputs, including through parent paths.
    for selected in (venv, sdk, bazel):
        if selected.is_relative_to(run) or run.is_relative_to(selected):
            raise ValueError('The fresh run directory must not overlap the venv, SDK or Bazel')
    run.mkdir(parents=True)
    source = run / 'source'
    evidence = {'status': 'failed', 'start': invocation_start, 'python': identity, 'phases': [dict(phase='preflight', start=invocation_start, end=time.time(), elapsed=time.monotonic()-invocation_clock, exit=0)], 'cleanup': {'status': 'not-started'},
                'bazel': {'path': str(bazel), 'sha256': sha256(bazel)},
                'sdk': str(sdk), 'compilers': {str(path): sha256(path) for path in (cc, cxx)}}
    commands = Commands(run, root, environment, evidence)
    previous = {sig: signal.signal(sig, commands.interrupt) for sig in (signal.SIGINT, signal.SIGTERM)}
    launcher = run / 'bazel'
    first_error = None
    lease = None
    try:
        started = time.monotonic()
        with measured(evidence, 'capture'):
            manifest = local_ci_source.capture(root, source)
            (run / 'source-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        evidence['source'] = {key: value for key, value in manifest.items() if key != 'inputs'}
        evidence['capture_elapsed'] = time.monotonic() - started
        for directory in ('tmp', 'loom-state', 'wheel', 'native-metrics'):
            (run / directory).mkdir()
        disk_cache = None
        if cache is not None:
            with measured(evidence, 'cache-lease'):
                selected_identity = cache_identity(evidence, source)
                lease = CacheLease(cache, selected_identity, run)
                store = lease.acquire()
                if cache_mode == 'read-write':
                    disk_cache = store
                # Both measured modes avoid backend cold overrides and use the
                # same explicit empty compiler stamps in the private rc.
                environment['HRX_BAZEL_CACHE_ENABLED'] = '1'
                evidence['cache'] = dict(directory=str(cache), mode=cache_mode,
                                         selected_inputs=selected_identity,
                                         contract='Same stable developer host/SDK; manual invalidation required')
        evidence['native_configuration'] = {name: sha256(source / name) for name in
                                            ('.bazelrc', '.bazelversion', 'MODULE.bazel', 'MODULE.bazel.lock')
                                            if (source / name).is_file()}
        environment.update(TMPDIR=str(run / 'tmp'), TMP=str(run / 'tmp'), TEMP=str(run / 'tmp'),
                           LOOM_SERVE_DIR=str(run / 'loom-state'), BAZEL=str(launcher),
                           HRX_BAZEL_METRICS_DIR=str(run / 'native-metrics'),
                           IREE_ROCM_PATH=str(sdk), ROCM_PATH=str(sdk), CC=str(cc), CXX=str(cxx),
                           USE_BAZEL_VERSION=(source / '.bazelversion').read_text().strip(),
                           PATH=str(venv / 'bin') + ':/usr/bin:/bin')
        evidence['environment'] = {key: environment[key] for key in
                                   ('TMPDIR', 'LOOM_SERVE_DIR', 'BAZEL', 'IREE_ROCM_PATH', 'CC', 'CXX',
                                    'USE_BAZEL_VERSION', 'PATH', 'PYTHONNOUSERSITE', 'HRX_BAZEL_CACHE_ENABLED', 'HRX_BAZEL_METRICS_DIR')}
        write_launcher(launcher, bazel, run, source, disk_cache)
        commands.cwd = source
        version = commands.command('bazel-version', [launcher, '--version']).strip()
        if version != 'bazel ' + environment['USE_BAZEL_VERSION']:
            raise ValueError(f'Unexpected actual pinned Bazel version: {version}')
        evidence['bazel']['version'] = version
        commands.command('setup', [sys.executable, '-m', 'build_tools.wheel_tools.setup_python',
                                   '--rocm', sdk, '--bazel', launcher])
        output_base = Path(commands.command('bazel-root', [launcher, 'info', 'output_base']).strip()).resolve()
        if output_base != run / 'bazel-output':
            raise ValueError('Bazel did not use the invocation-owned output base')
        # Record the exact server identity before building, for owned shutdown proof.
        pid = int(commands.command('bazel-pid', [launcher, 'info', 'server_pid']).strip())
        server = Path('/proc') / str(pid)
        server_identity = (server / 'stat').read_text().rsplit(')', 1)[1].split()[19]
        evidence['bazel'].update(output_base=str(output_base), server_pid=pid, server_start=server_identity)
        evidence['hrx_module'] = (source / 'MODULE.bazel').read_text()
        commands.command('build', [sys.executable, '-m', 'build', '--wheel', '--no-isolation',
                                   '--config-setting=--global-option=--no-user-cfg', '--outdir', run / 'wheel'])
        wheels = list((run / 'wheel').glob('*.whl'))
        if len(wheels) != 1:
            raise ValueError('Expected exactly one freshly built wheel')
        wheel = wheels[0]
        commands.command('archive-validation', [sys.executable, '-m', 'build_tools.wheel_tools.validate_wheel', wheel])
        with measured(evidence, 'host-tag-validation'):
            validate_host_tags(wheel)
        with measured(evidence, 'source-verification'):
            local_ci_source.verify(source, manifest['inputs'])
        evidence['wheel'] = {'path': str(wheel), 'sha256': sha256(wheel), 'tag': 'py3-none-linux_x86_64'}
        evidence['status'] = 'passed'
    except Exception as error:
        first_error = error
        evidence['error'] = str(error)
    finally:
        try:
            if launcher.exists():
                if 'server_pid' not in evidence['bazel']:
                    pid_file = run / 'bazel-output/server/server.pid.txt'
                    if pid_file.exists():
                        pid = int(pid_file.read_text().strip())
                        stat = Path('/proc') / str(pid) / 'stat'
                        if stat.exists():
                            evidence['bazel'].update(server_pid=pid, server_start=stat.read_text().rsplit(')', 1)[1].split()[19])
                commands.command('bazel-shutdown', [launcher, 'shutdown'], cleanup=True, timeout=45)
                if 'server_pid' in evidence['bazel']:
                    stat = Path('/proc') / str(evidence['bazel']['server_pid']) / 'stat'
                    if stat.exists() and stat.read_text().rsplit(')', 1)[1].split()[19] == evidence['bazel']['server_start']:
                        raise RuntimeError('Owned Bazel server is still running after shutdown')
                if evidence['bazel']['sha256'] != sha256(bazel):
                    raise RuntimeError('Bazel executable changed during build')
                for path, expected in evidence['compilers'].items():
                    if sha256(Path(path)) != expected:
                        raise RuntimeError('Compiler executable changed during build')
            if commands.unresolved:
                raise RuntimeError('Unresolved owned phase cleanup: ' + ', '.join(commands.unresolved))
            evidence['cleanup']['status'] = 'complete'
        except Exception as error:
            evidence['cleanup'].update(status='unresolved', error=str(error))
            first_error = first_error or error
        if lease is not None:
            try:
                with measured(evidence, 'cache-release'):
                    lease.close(cleanup_complete=evidence['cleanup']['status'] == 'complete')
            except Exception as error:
                evidence['cleanup'].update(status='unresolved', error=str(error))
                first_error = first_error or error
        if commands.cancelled:
            evidence['signal'] = commands.cancelled
            first_error = first_error or InterruptedError('Build interrupted')
        if first_error:
            evidence['status'] = 'failed'
        evidence.update(end=time.time(), elapsed=time.monotonic() - invocation_clock)
        (run / 'summary.json').write_text(json.dumps(evidence, indent=2) + '\n')
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    if first_error:
        print(f'Host build failed: {first_error}; evidence: {run / "summary.json"}', file=sys.stderr)
        return 128 + commands.cancelled if commands.cancelled else (first_error.returncode if isinstance(first_error, subprocess.CalledProcessError) else 1)
    print(f'Host wheel: {wheel}\nEvidence: {run / "summary.json"}')
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('rocm', 'venv', 'run-dir', 'bazel'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--cache-dir', help='Private disk cache; opts into same stable host/SDK reuse and manual invalidation')
    parser.add_argument('--cache-mode', choices=('disabled', 'read-write'),
                        help='With --cache-dir: disabled is a matched control; default read-write')
    args = parser.parse_args()
    try:
        environment = clean_environment(os.environ)
        venv = external_path(args.venv, ROOT.resolve())
        python = venv / 'bin' / 'python'
        if not python.is_file():
            raise ValueError('Create the standalone build venv before invoking --fast')
        if Path(sys.prefix).resolve() != venv:
            os.execve(str(python), [str(python), '-E', '-s', str(Path(__file__).resolve()), *sys.argv[1:]], environment)
        # The shell/bootstrap path deliberately imports only the standard library.
        sys.path.insert(0, str(ROOT))
        return run_build(args, environment)
    except (ValueError, RuntimeError, OSError, ImportError) as error:
        print(f'Host build preflight failed: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
