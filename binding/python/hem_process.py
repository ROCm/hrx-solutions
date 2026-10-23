"""Linux managed-process ownership, retained payloads and recoverable empty launch.

No supervision, background lifecycle work, process adoption or numeric-PID kill.
The bootstrap durably identifies itself while sharing the caller's server flock.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, replace
import hashlib
import ipaddress
import os
from pathlib import Path
import select
import shutil
import signal
import socket
import subprocess
import sys
import time
import uuid

from . import hem_display
from ._launcher import native_environment
from .hem_common import Deadline, DeadlineExpired, HemError, InvalidInput, UncertainOutcome
from .hem_http import NativeClient


def identity(pid):
    """Snapshot identity; permission/inspection failures must never mean exited."""
    try:
        proc = Path('/proc') / str(pid)
        fields = (proc / 'stat').read_text().rsplit(')', 1)[1].split()
        executable = (proc / 'exe').stat()
        return dict(pid=pid, boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                    start_ticks=int(fields[19]), state=fields[0],
                    executable=dict(device=executable.st_dev, inode=executable.st_ino))
    except (OSError, ValueError, IndexError) as error:
        raise UncertainOutcome(f'cannot inspect process {pid}: {error}') from error


def _exited(fd):
    poll = select.poll()
    poll.register(fd, select.POLLIN)
    return bool(poll.poll(0))


class _ExecutableMismatch(UncertainOutcome):
    """Identity snapshot rejected while its boot/start identity was pinned."""
    def __init__(self, observed):
        super().__init__('recorded executable differs; refusing control')
        self.observed = observed


class PinnedProcess:
    """One verified Linux pidfd; signal only this handle after re-verification."""
    def __init__(self, process, executable=None, *, emit=hem_display.emit):
        self.expected = process
        self.executable = executable or process['executable']
        self.fd = None
        self.emit = emit

    def __enter__(self):
        if not hasattr(os, 'pidfd_open') or not hasattr(signal, 'pidfd_send_signal'):
            raise UncertainOutcome('managed process control requires Linux pidfd support')
        try:
            self.fd = os.pidfd_open(self.expected['pid'])
        except OSError as error:
            raise UncertainOutcome(f'cannot pin recorded process: {error}') from error
        try:
            self.verify()
        except BaseException:
            self.close()
            raise
        return self

    def close(self):
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None

    def __exit__(self, *_):
        self.close()

    def verify(self, deadline=None):
        if deadline is not None:
            deadline.remaining()
        if self.fd is None or _exited(self.fd):
            raise UncertainOutcome('recorded process exited; reconcile before control')
        current = identity(self.expected['pid'])
        if any(current[key] != self.expected[key] for key in ('pid', 'boot_id', 'start_ticks')):
            raise UncertainOutcome('recorded process identity changed; refusing control')
        if _exited(self.fd):
            raise UncertainOutcome('recorded process exited during identity check')
        if current['executable'] != self.executable:
            raise _ExecutableMismatch(current)
        return current

    def send(self, number, deadline):
        self.verify(deadline)
        # A numeric kill command would falsely imply PID-reuse safety. Print
        # the standalone Python/Linux equivalent with the same pinned checks.
        code = (
            'import os, pathlib, select, signal, sys; '
            'check=lambda ok: ok or sys.exit("process ownership unproven"); '
            f'p={self.expected["pid"]!r}; f=os.pidfd_open(p); '
            'q=select.poll(); q.register(f, select.POLLIN); '
            'check(not q.poll(0)); '
            f'check(pathlib.Path("/proc/sys/kernel/random/boot_id").read_text().strip()=={self.expected["boot_id"]!r}); '
            'r=pathlib.Path("/proc")/str(p); '
            f'check(int((r/"stat").read_text().rsplit(")",1)[1].split()[19])=={self.expected["start_ticks"]!r}); '
            's=(r/"exe").stat(); '
            f'check((s.st_dev,s.st_ino)=={(self.executable["device"], self.executable["inode"])!r}); '
            'check(not q.poll(0)); '
            f'signal.pidfd_send_signal(f,{int(number)}); os.close(f)'
        )
        self.emit(hem_display.command([sys.executable, '-c', code]))
        self.verify(deadline)
        try:
            signal.pidfd_send_signal(self.fd, number)
        except OSError as error:
            raise UncertainOutcome(f'owned process signal failed: {error}') from error

    def wait(self, deadline):
        while not _exited(self.fd):
            time.sleep(min(0.01, deadline.remaining()))

    def connection(self, connected, deadline):
        """Prove this actual accepted socket's inode belongs to the pinned PID.

        Procfs can publish the accepted FD slightly after connect completes. Poll
        only until the invocation deadline. Ambiguity or unreadable procfs fails
        closed; no request bytes have been transmitted by NativeClient yet.
        """
        try:
            return self._connection(connected, deadline)
        except DeadlineExpired as error:
            raise UncertainOutcome('accepted HTTP connection ownership is unproven') from error

    def _connection(self, connected, deadline):
        self.verify(deadline)
        local, peer = connected.getsockname(), connected.getpeername()
        if not ipaddress.ip_address(peer[0]).is_loopback:
            raise UncertainOutcome('managed HTTP ownership requires a loopback peer')
        while True:
            self.verify(deadline)
            try:
                wanted = set()
                family = 'tcp6' if connected.family == socket.AF_INET6 else 'tcp'
                rows = (Path('/proc/net') / family).read_text().splitlines()[1:]
                for row in rows:
                    parts = row.split()
                    if (parts[3] == '01' and _proc_address(parts[1], connected.family) == peer[:2]
                            and _proc_address(parts[2], connected.family) == local[:2]):
                        wanted.add(parts[9])
                owned = set()
                for fd in (Path('/proc') / str(self.expected['pid']) / 'fd').iterdir():
                    try:
                        link = os.readlink(fd)
                    except FileNotFoundError:
                        continue
                    if link.startswith('socket:['):
                        owned.add(link[8:-1])
                if len(wanted) == 1 and wanted <= owned:
                    self.verify(deadline)
                    return
                if len(wanted) > 1 or (wanted and not wanted <= owned):
                    # The table can precede accept(); allow that publication
                    # window, but never send to the other process in the interim.
                    pass
                time.sleep(min(0.005, deadline.remaining()))
            except DeadlineExpired as error:
                raise UncertainOutcome('accepted HTTP connection ownership is unproven') from error
            except OSError as error:
                raise UncertainOutcome(f'cannot inspect accepted socket ownership: {error}') from error


def _proc_address(value, family):
    address, port = value.split(':')
    raw = bytes.fromhex(address)
    # procfs prints native-endian 32-bit words, including IPv6 addresses.
    raw = b''.join(raw[index:index + 4][::-1] if sys.byteorder == 'little' else raw[index:index + 4]
                   for index in range(0, len(raw), 4))
    return socket.inet_ntop(family, raw), int(port, 16)


def _digest_file(path, deadline):
    digest = hashlib.sha256()
    try:
        with path.open('rb') as stream:
            while True:
                deadline.remaining()
                block = stream.read(1024 * 1024)
                if not block:
                    return digest.hexdigest()
                digest.update(block)
    except OSError as error:
        raise HemError(f'cannot verify bundled payload {path}: {error}') from error


def tree_identity(directory, deadline):
    """Deterministic names+content contract, materializing symlinks on retention."""
    directory = Path(directory)
    if not directory.is_dir():
        raise HemError(f'missing bundled payload directory: {directory}')
    digest = hashlib.sha256()
    for path in sorted(directory.rglob('*')):
        deadline.remaining()
        if path.is_symlink() and path.is_dir():
            raise HemError('native/source payload cannot contain directory symlinks')
        if path.is_file():
            name = path.relative_to(directory).as_posix().encode()
            digest.update(len(name).to_bytes(8, 'big') + name)
            digest.update(bytes.fromhex(_digest_file(path, deadline)))
        elif not path.is_dir():
            raise HemError('native/source payload contains a special file')
    return digest.hexdigest()


def package_identity(package_root, deadline):
    root = Path(package_root)
    native, catalog = root / '_native', root / 'loom-pipeline'
    return dict(native_sha256=_digest_file(native / 'bin' / 'loom-serve', deadline),
                payload_sha256=tree_identity(native, deadline),
                catalog_sha256=tree_identity(catalog, deadline))


def retain_payload(registry, server_id, token, package_root, deadline):
    """Copy the complete wheel-native closure and source catalog, then verify.

    A failed/partial copy remains explicitly reserved launch work, never adopted.
    No weights are copied; the deadline is checked between bounded file chunks.
    """
    source = Path(package_root)
    before = package_identity(source, deadline)
    relative = Path('servers') / server_id / token
    destination = registry.root / relative
    if destination.exists():
        raise HemError('launch artifact directory already exists; refusing adoption')
    missing = []
    ancestor = destination
    while not ancestor.exists():
        missing.append(ancestor)
        ancestor = ancestor.parent
    destination.mkdir(parents=True, mode=0o700)
    def copy_file(src, dst):
        with open(src, 'rb') as incoming, open(dst, 'xb') as outgoing:
            while True:
                deadline.remaining()
                chunk = incoming.read(1024 * 1024)
                if not chunk:
                    break
                outgoing.write(chunk)
            outgoing.flush()
            os.fsync(outgoing.fileno())
        shutil.copymode(src, dst)
        # Retained files are immutable inputs for subsequent registrations.
        os.chmod(dst, 0o555 if os.stat(src).st_mode & 0o111 else 0o444)
        return dst
    for name in ('_native', 'loom-pipeline'):
        shutil.copytree(source / name, destination / name, copy_function=copy_file)
    retained = package_identity(destination, deadline)
    if retained != before or package_identity(source, deadline) != before:
        raise HemError('package payload changed during retention; start refused')
    # Flush directory entries as well as file contents before publishing metadata.
    for directory, _, _ in os.walk(destination, topdown=False):
        fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    # mkdir(parents=True) may have created servers/ and the server-ID directory
    # too. Flush each newly created entry's parent before returning metadata for
    # a durable SQLite write, including the namespace root on the first launch.
    for created in reversed(missing):
        fd = os.open(created.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    exe = (destination / '_native/bin/loom-serve').stat()
    return dict(before, directory=str(destination),
                executable=dict(device=exe.st_dev, inode=exe.st_ino))


def require_build(record, package_root, deadline):
    """Registration gate: package and retained live launch must both match."""
    build = record.get('build')
    if not build:
        raise UncertainOutcome('live server build is unknown; refuse registration')
    keys = ('native_sha256', 'payload_sha256', 'catalog_sha256')
    current = package_identity(package_root, deadline)
    retained = package_identity(build['directory'], deadline)
    if any(current[key] != build.get(key) or retained[key] != build.get(key) for key in keys):
        raise HemError('serving build/catalog differs; shut down and launch a new empty server', reason='build_mismatch')
    return Path(build['directory']) / 'loom-pipeline'


def _confirmed_exit(process):
    try:
        boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    except OSError as error:
        raise UncertainOutcome(f'cannot inspect current boot identity: {error}') from error
    if process['boot_id'] != boot:
        return True
    try:
        fd = os.pidfd_open(process['pid'])
    except ProcessLookupError:
        return True
    except (OSError, AttributeError) as error:
        raise UncertainOutcome(f'cannot establish recorded process exit: {error}') from error
    try:
        if _exited(fd):
            return True
        try:
            current = identity(process['pid'])
        except UncertainOutcome:
            # The process can exit after the first poll, removing procfs
            # executable data. Only this same pinned pidfd proves that exit;
            # an inaccessible live process must remain uncertain.
            if _exited(fd):
                return True
            raise
        if current['boot_id'] != process['boot_id'] or current['start_ticks'] != process['start_ticks']:
            return True  # The recorded incarnation is gone; never signal replacement.
        return False
    finally:
        os.close(fd)


def validate_launch_settings(settings):
    """Validate infrastructure choices without interpreting an opaque device.

    Return the loopback origin; retain the supplied settings/selector unchanged.
    CLI-only missing-option and whitespace presentation belongs to the caller.
    """
    if set(settings) != {'device', 'listen'} or not isinstance(settings['device'], str) or not settings['device']:
        raise InvalidInput('server launch requires device and listen settings only')
    try:
        host, port_text = settings['listen'].rsplit(':', 1)
        host = host.strip('[]')
        port = int(port_text)
        if not ipaddress.ip_address(host).is_loopback or not 0 < port < 65536:
            raise ValueError()
    except (ValueError, AttributeError):
        raise InvalidInput('managed server listen must be a numeric loopback host and nonzero port') from None
    return f'http://[{host}]:{port}' if ':' in host else f'http://{host}:{port}'


@dataclass(frozen=True)
class ServerObservation:
    """Scoped evidence for presentation, never authority for a later mutation."""

    record: dict
    process_health: str
    http_health: str
    native: dict | None = None
    error: str | None = None
    observed_at: float = 0.0

    @property
    def state(self):
        if self.process_health == 'unknown':
            return 'unknown'
        if self.record['launch_token'] is None:
            return 'stopped'
        if self.http_health != 'healthy':
            return 'unhealthy'
        return self.record['phase']


def observe_server(record, deadline, *, pin, client, max_response_bytes=None, display=False):
    """Observe one supplied record without state reads, writes or reconciliation.

    Callers own snapshot consistency and any lock. The pin and client retain
    both socket association and post-response identity verification. Passive
    readers can bound the response without constructing a Registry.
    """
    base = dict(record=record, observed_at=time.time())
    if record['launch_token'] is None:
        return ServerObservation(process_health='exited', http_health='not-running', **base)
    if not record.get('process') or not record.get('build') or not record.get('endpoint'):
        return ServerObservation(process_health='unknown', http_health='unknown', **base)
    try:
        with pin(record) as pinned:
            try:
                options = dict(display=display)
                if max_response_bytes is not None:
                    options['max_response_bytes'] = max_response_bytes
                native = client(record, pinned, deadline).status(**options)
            except UncertainOutcome:
                # Socket association uncertainty cannot establish unhealthy HTTP
                # on an otherwise verified process.
                raise
            except HemError as error:
                pinned.verify(deadline)
                return ServerObservation(process_health='verified', http_health='unhealthy',
                                         error=str(error), **base)
        return ServerObservation(process_health='verified', http_health='healthy',
                                 native=native, **base)
    except HemError as error:
        return ServerObservation(process_health='unknown', http_health='unknown',
                                 error=str(error), **base)


class ProcessManager:
    """Lock-owning launch/reconcile operations; server CLI remains a thin caller."""
    def __init__(self, registry, *, package_root=None, emit=hem_display.emit):
        self.registry = registry
        self.package_root = Path(package_root or Path(__file__).resolve().parent)
        self.emit = emit
        self.children = {}  # Only direct children of this invocation, for reaping.

    def _update(self, record, changes):
        return self.registry.update_server(record['id'], changes,
                    expected_token=record['launch_token'], expected_revision=record['revision'])

    def _observe_and_reconcile(self, server_id, deadline=None, *, observe=True, display=False):
        """Caller owns server lock; return the evidence used by these transitions."""
        self.reap()
        record = self.registry.get_server(server_id)
        observed_at = time.time()

        def unknown(error=None):
            return ServerObservation(record, 'unknown', 'unknown', error=error,
                                     observed_at=observed_at)

        if record['launch_token'] is None:
            return ServerObservation(record, 'exited', 'not-running', observed_at=observed_at)
        process = record['process']
        try:
            exited = process is None or _confirmed_exit(process)
        except UncertainOutcome as error:
            record = self._update(record, dict(phase='unknown'))
            return unknown(str(error))
        if exited:
            # Caller owns server lock. A delayed bootstrap still holding the
            # inherited lock cannot race this token invalidation.
            record = self.registry.clear_server(server_id, expected_token=record['launch_token'],
                                                expected_revision=record['revision'])
            return ServerObservation(record, 'exited', 'not-running', observed_at=observed_at)
        expected = record['build']['executable']
        try:
            current = identity(process['pid'])
            if current['executable'] != expected:
                if record['pending'] and record['pending'].get('stage') == 'bootstrap' and current['executable'] == process['executable']:
                    return unknown()  # Live helper, never native readiness.
                record = self._update(record, dict(phase='unknown'))
                return unknown()
            if not observe:
                with self.pin(record):
                    pass
                return unknown()
            observation = observe_server(record, deadline or Deadline.after(),
                                         pin=self.pin, client=self.client, display=display)
            if observation.http_health != 'healthy':
                record = self._update(record, dict(phase='unknown'))
            elif record['pending'] and record['pending']['action'] == 'start':
                record = self._update(record, dict(phase='running', pending=None))
            elif record['phase'] == 'unknown':
                phase = 'stopping' if record['pending'] and record['pending']['action'] == 'shutdown' else 'running'
                record = self._update(record, dict(phase=phase))
            return replace(observation, record=record)
        except HemError as error:
            record = self._update(record, dict(phase='unknown'))
            return unknown(str(error))

    def _reconcile(self, server_id, deadline=None, *, observe=True):
        return self._observe_and_reconcile(server_id, deadline, observe=observe).record

    def reconcile(self, server_id, deadline):
        with self.registry.lock('server', server_id, deadline=deadline):
            return self._reconcile(server_id, deadline)

    def observe_and_reconcile(self, server_id, deadline):
        """Return one locked observation; consumers must revalidate before effects."""
        record = self.registry.get_server(server_id)
        try:
            with self.registry.lock('server', server_id, deadline=deadline):
                return self._observe_and_reconcile(server_id, deadline, display=True)
        except HemError as error:
            return ServerObservation(record, 'unknown', 'unknown', error=str(error),
                                     observed_at=time.time())

    @contextmanager
    def pin(self, record):
        if not record.get('process') or not record.get('build'):
            raise UncertainOutcome('server has no established process/build identity')
        with PinnedProcess(record['process'], record['build']['executable'], emit=self.emit) as process:
            yield process

    def client(self, record, process, deadline):
        return NativeClient(record['endpoint'], deadline, process.connection,
                            ownership_after=process.verify, emit=self.emit)

    def start(self, server_id, settings, deadline):
        """Start or reuse matching healthy server; empty new launches only.

        settings is exactly {'device': str, 'listen': loopback-host:port}. Saved
        definition handling belongs to server CLI; this freezes current settings.
        """
        endpoint = validate_launch_settings(settings)
        with self.registry.lock('server', server_id, deadline=deadline) as lock:
            record = self._reconcile(server_id, deadline)
            if record['launch_token']:
                if record['phase'] == 'unknown':
                    raise UncertainOutcome('prior server ownership/health is unknown; inspect before starting')
                if record['current_settings'] != settings or record['phase'] != 'running' or record['pending']:
                    raise HemError('prior server launch is unresolved/different; inspect before starting')
                with self.pin(record) as pinned:
                    self.client(record, pinned, deadline).status()
                require_build(record, self.package_root, deadline)
                return record
            # Persist all configured connection targets before accepting any
            # native launch. This short publication never spans readiness waits.
            from .hem_pi_endpoints import publish_endpoints
            publish_endpoints(self.registry, deadline)
            token = uuid.uuid4().hex
            record = self.registry.reserve_launch(server_id, launch_token=token,
                            current_settings=settings, endpoint=endpoint)
            build = retain_payload(self.registry, server_id, token, self.package_root, deadline)
            record = self._update(record, dict(build=build))
            executable = str(Path(build['directory']) / '_native/bin/loom-serve')
            argv = [executable, f"--device={settings['device']}", f"--listen={settings['listen']}"]
            # Managed launches use a retained closure, never package-upgrade paths
            # or caller library overrides. HOME/XDG/credentials remain inherited.
            try:
                environment = native_environment(native_root=Path(build['directory']) / '_native')
            except RuntimeError as error:
                raise HemError(f'cannot launch retained native closure: {error}') from error
            environment['LOOM_SERVE_DIR'] = str(self.registry.root)
            environment.pop('LD_PRELOAD', None)
            self.emit(hem_display.command(argv, environment={key: environment[key] for key in
                      ('LOOM_SERVE_DIR', 'LD_LIBRARY_PATH', 'IREE_HAL_AMDGPU_LIBHSA_PATH') if key in environment}) +
                      ' # background via identity bootstrap; bounded process.log capture')
            deadline.remaining()
            child = subprocess.Popen([sys.executable, '-m', f'{__package__}.hem_bootstrap',
                                      str(self.registry.root), server_id, token, str(lock.fileno()),
                                      str(deadline.expires_at)],
                                     pass_fds=(lock.fileno(),), env=environment,
                                     stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                     stderr=subprocess.DEVNULL, start_new_session=True)
            self.children[token] = child
            try:
                return self._wait_start(server_id, child, deadline)
            except DeadlineExpired as error:
                raise DeadlineExpired('accepted server launch remains pending; inspect/reconcile', accepted=True) from error

    def _wait_start(self, server_id, child, deadline):
        while True:
            deadline.remaining()
            record = self.registry.get_server(server_id)
            if child.poll() is not None:
                self._reconcile(server_id, deadline, observe=False)
                raise HemError(f'bootstrap/native exited with status {child.returncode}; inspect process log', accepted=True)
            if record['process']:
                try:
                    with self.pin(record) as pinned:
                        document = self.client(record, pinned, deadline).status()
                        if document['data']:
                            raise UncertainOutcome('new server unexpectedly contains registrations', accepted=True)
                    return self._update(record, dict(phase='running', pending=None))
                except _ExecutableMismatch as error:
                    # Classify the snapshot that failed pinned verification.
                    # A second inspection can already see the legitimate exec,
                    # falsely turning a transient bootstrap into an unknown PID.
                    # Retry only this launch's verified bootstrap snapshot; the
                    # next iteration must still pin native and prove HTTP owner.
                    observed = error.observed
                    expected = record['process']
                    if (record['pending'] != {'action': 'start', 'stage': 'bootstrap'}
                            or any(observed[key] != expected[key] for key in ('pid', 'boot_id', 'start_ticks'))
                            or observed['executable'] != expected['executable']):
                        raise
                except DeadlineExpired:
                    raise
                except HemError as error:
                    # A refused connection during the bind window is expected.
                    # Other malformed/lost observations do not establish health.
                    if error.reason != 'connection_refused':
                        raise UncertainOutcome(f'server readiness is unknown: {error}', accepted=True) from error
            time.sleep(min(0.01, deadline.remaining()))

    def cleanup(self, record, deadline):
        """Explicit lifecycle cleanup only, after proved exit; never reconcile.

        The caller holds the server lock. A saved pre-clear snapshot authorizes
        only its exact retained directory. Unknown/orphan directories are left
        for explicit inspection, never silently adopted or recursively swept.
        """
        deadline.remaining()
        if not record.get('build'):
            return
        if record.get('process') and not _confirmed_exit(record['process']):
            raise HemError('cannot remove artifacts of a live or unresolved server')
        current = self.registry.get_server(record['id'])
        if current['launch_token'] not in (None, record['launch_token']):
            raise HemError('server launch changed; refusing artifact cleanup')
        if not record.get('process') and current['launch_token'] is not None:
            raise UncertainOutcome('bootstrap identity is unresolved; reconcile before artifact cleanup')
        expected = self.registry.root / 'servers' / record['id'] / record['launch_token']
        target = Path(record['build']['directory'])
        if target != expected or target.is_symlink() or target.resolve() != expected:
            raise HemError('retained artifact ownership is unproven')
        # Logger holds no lifecycle lock and may be draining the last bounded
        # pipe contents after native exit. Do not race it with directory removal.
        logger = (record.get('process') or {}).get('logger')
        while logger and not _confirmed_exit(logger):
            time.sleep(min(.01, deadline.remaining()))
        deadline.remaining()
        if target.exists():
            shutil.rmtree(target)

    def reap(self):
        for child in self.children.values():
            child.poll()


def read_logs(record, *, limit=65536):
    if type(limit) is not int or not 0 < limit <= 4 * 1024 * 1024:
        raise InvalidInput('log byte limit must be between 1 and 4194304')
    if not record.get('build'):
        return b''
    directory = Path(record['build']['directory'])
    data = b''
    for name in ('bootstrap-error.txt', 'process.log.1', 'process.log'):
        path = directory / name
        if path.exists():
            with path.open('rb') as stream:
                stream.seek(max(0, path.stat().st_size - limit))
                data = (data + stream.read(limit))[-limit:]
    return data
