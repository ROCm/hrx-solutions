"""One-shot durable identity bootstrap, plus a pipe-only bounded log drainer.

The native executable replaces the bootstrap PID. The drainer has no control
socket, server lock, signals or restart logic; EOF ends its entire lifetime.
"""
from __future__ import annotations

import fcntl
import os
from pathlib import Path
import sys

from .hem_common import Deadline, HemError
from .hem_process import identity
from .hem_state import Registry

LOG_BYTES = 2 * 1024 * 1024


def _checkpoint(stage):
    """No-op seam for subprocess crash-window tests; no environment controls."""


def _drain(descriptor, directory, *, maximum=LOG_BYTES):
    """Retain at most two fixed-size files; continue draining on storage failure."""
    current = Path(directory) / 'process.log'
    previous = Path(directory) / 'process.log.1'
    output = None
    size = 0
    try:
        try:
            output = os.open(current, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
        except OSError:
            # Storage failure must never close the native pipe's sole reader.
            # Initial creation has the same discard-and-drain contract as later
            # writes/rotation; keep reading until the native closes its output.
            pass
        while True:
            block = os.read(descriptor, 65536)
            if not block:
                break
            offset = 0
            while offset < len(block):
                if output is None:
                    break  # Preserve native liveness even if storage has failed.
                try:
                    if size == maximum:
                        os.close(output)
                        output = None
                        os.replace(current, previous)
                        output = os.open(current, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
                        size = 0
                    count = os.write(output, block[offset:offset + maximum - size])
                    if count <= 0:
                        raise OSError('short log write')
                    size += count
                    offset += count
                except OSError:
                    if output is not None:
                        os.close(output)
                    output = None
    finally:
        if output is not None:
            os.close(output)
        os.close(descriptor)


def bootstrap(root, server_id, token, lock_fd, deadline):
    """Caller passes its live flock open-file description, never a pathname lock.

    Check token and intent under this lock, durably save our identity, then exec.
    Every failure closes native output so the log child finishes without cleanup
    signaling; it never owns the server flock. The caller's copy remains locked.
    """
    os.set_inheritable(lock_fd, False)
    # Verify that FD really is the expected lock; inheriting an arbitrary FD does
    # not authorize bootstrap. An independently held flock must block this probe.
    expected_path = Path(root) / 'locks' / f'server-{server_id}.lock'
    if os.fstat(lock_fd) != expected_path.stat():
        raise HemError('bootstrap lock descriptor does not match server lock')
    fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    registry = Registry(root, deadline=deadline)
    _checkpoint('before_identity')
    record = registry.get_server(server_id)
    if record['launch_token'] != token or record['pending'] != {'action': 'start', 'stage': 'reserved'} or record['process']:
        raise HemError('bootstrap launch token/intent changed; native launch refused')
    build = record.get('build')
    if not build:
        raise HemError('bootstrap has no retained native payload')
    executable = Path(build['directory']) / '_native/bin/loom-serve'
    actual = executable.stat()
    if build['executable'] != dict(device=actual.st_dev, inode=actual.st_ino):
        raise HemError('retained native executable identity changed')
    settings = record['current_settings']
    process = identity(os.getpid())
    record = registry.update_server(server_id, dict(process=process,
                 pending={'action': 'start', 'stage': 'bootstrap'}),
                 expected_token=token, expected_revision=record['revision'])
    _checkpoint('after_identity')
    deadline.remaining()
    read_fd, write_fd = os.pipe()
    child = os.fork()
    if child == 0:
        os.close(write_fd)
        os.close(lock_fd)  # Logging cannot retain lifecycle ownership.
        try:
            _drain(read_fd, build['directory'])
        finally:
            os._exit(0)
    os.close(read_fd)
    _checkpoint('after_log_fork')
    try:
        # Current logger identity is bounded ownership metadata, never history.
        process = dict(process, logger=identity(child))
        registry.update_server(server_id, dict(process=process),
                   expected_token=token, expected_revision=record['revision'])
        _checkpoint('after_logger')
        os.dup2(write_fd, 1)
        os.dup2(write_fd, 2)
    finally:
        os.close(write_fd)
    _checkpoint('before_exec')
    deadline.remaining()
    os.execve(str(executable), [str(executable), f"--device={settings['device']}",
                              f"--listen={settings['listen']}"], dict(os.environ))


def main():
    try:
        root, server_id, token, descriptor, expires_at = sys.argv[1:]
        bootstrap(root, server_id, token, int(descriptor), Deadline(float(expires_at)))
    except Exception as error:
        diagnostic = f'loom-hem bootstrap: {error}'[:4096]
        print(diagnostic, file=sys.stderr)
        # Early bootstrap errors precede stdout/stderr pipe capture. Preserve
        # one bounded diagnostic in this caller-selected launch directory.
        try:
            path = Path(root) / 'servers' / server_id / token / 'bootstrap-error.txt'
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
            with os.fdopen(descriptor, 'w') as stream:
                stream.write(diagnostic)
        except (OSError, UnboundLocalError):
            pass
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
