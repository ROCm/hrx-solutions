"""Minimal Pi reservation and passive inventory; no file installer or lifecycle I/O.

Reservation is not proof of installation. Callers commit it before publishing a
pending plan, retain it through recovery, and clear it only after durable cleanup.
The caller holds integration then target locks for changes spanning DB and files.
"""
from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
import sqlite3
import stat
import time
import uuid

from ...hem_common import Deadline, HemError, InvalidInput
from ...hem_state import Registry, ResourceLock, _directory, _validate_schema


def _canonical(path):
    path = Path(path)
    if not path.is_absolute() or str(path.resolve()) != str(path):
        raise InvalidInput("Pi target/state root must be canonical and absolute")
    return path


def _target(value):
    if not isinstance(value, dict) or set(value) != {"target", "integration_id"}:
        raise InvalidInput("Pi target must contain target and integration_id only")
    if not isinstance(value["target"], str):
        raise InvalidInput("Pi target must be a canonical absolute path")
    if str(_canonical(value["target"])) != value["target"]:
        raise InvalidInput("Pi target must use its canonical absolute spelling")
    try:
        identifier = uuid.UUID(value["integration_id"])
        if identifier.version != 4 or str(identifier) != value["integration_id"]:
            raise ValueError()
    except (ValueError, TypeError, AttributeError) as error:
        raise InvalidInput("Pi integration identity must be a canonical UUID4") from error
    return dict(value)


def _read_target(connection):
    rows = connection.execute("SELECT singleton,target,integration_id FROM pi_target").fetchall()
    if not rows:
        return None
    if len(rows) != 1 or rows[0][0] != 1:
        raise HemError("corrupt Pi target reservation; inventory retained")
    try:
        return _target(dict(target=rows[0][1], integration_id=rows[0][2]))
    except InvalidInput as error:
        raise HemError("corrupt Pi target reservation; inventory retained") from error


def get_target(registry):
    with registry._transaction() as connection:
        return _read_target(connection)


def set_target(registry, target):
    """Compare-and-reserve. Never retarget or replace a recovery identity."""
    target = _target(target)
    with registry._transaction() as connection:
        current = _read_target(connection)
        if current is not None and current != target:
            raise InvalidInput(f"Pi target reservation conflicts with {current['target']}; recover that target first")
        if current is None:
            connection.execute("INSERT INTO pi_target VALUES(1,?,?)", (target["target"], target["integration_id"]))


def clear_target(registry, *, integration_id):
    """Compare-and-delete only; caller has already verified durable file cleanup."""
    with registry._transaction() as connection:
        current = _read_target(connection)
        if current is not None:
            if current["integration_id"] != integration_id:
                raise InvalidInput("Pi integration changed; cannot clear another reservation")
            connection.execute("DELETE FROM pi_target WHERE singleton=1 AND integration_id=?", (integration_id,))


def _private_directory(path, *, create):
    """Check every existing component; never chmod or follow target symlinks.

    Ancestors may be searchable by others (normal HOME layout), but must be
    same-user and not group/world writable. Target itself must be private.
    """
    path = _canonical(path)
    for component in reversed((path, *path.parents)):
        if not component.exists():
            if not create:
                raise HemError(f"missing Pi directory: {component}")
            _directory(component)
        info = component.lstat()
        # System-owned immutable ancestors such as / and /home are trusted.
        trusted_sticky = info.st_uid == 0 and info.st_mode & stat.S_ISVTX
        if (not stat.S_ISDIR(info.st_mode) or (info.st_mode & 0o022 and not trusted_sticky) or
                info.st_uid not in (os.getuid(), 0)):
            raise HemError(f"unsafe Pi directory component: {component}")
    info = path.stat()
    if info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise HemError("Pi target must be a private same-user directory")
    return path


@contextmanager
def integration_lock(root, deadline):
    deadline.remaining()
    root = _private_directory(root, create=False)
    # Registry initialization is deliberately separate, never hidden here.
    if not (root / "locks").is_dir() or (root / "locks").is_symlink():
        raise HemError("initialize state before acquiring integration lock")
    with ResourceLock(root, "integration", "unused", deadline,
                      path=root / "locks" / "pi-integration.lock") as lock:
        yield lock


@contextmanager
def target_lock(target, deadline):
    deadline.remaining()
    target = _canonical(target)
    lock = ResourceLock(target, "target", "unused", deadline,
                        path=target / "loom-hem.lock")
    lock._check_order()
    target = _private_directory(target, create=True)
    with lock:
        yield lock


def passive_snapshot(root, deadline):
    """One consistent read transaction; never initialize/migrate/chmod/flock.

    A live SQLite WAL needs its existing shared-memory index. Refuse a WAL
    without sidecars rather than asking SQLite to create/recover them on check.
    """
    root = _canonical(root)
    database = root / "control.sqlite3"
    deadline.remaining()
    if not database.exists() and not database.is_symlink():
        return dict(schema_version=None, pi_target=None, servers=[])
    info = database.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid():
        raise HemError("state database must be a same-user regular file")
    with database.open("rb") as stream:
        header = stream.read(100)
    if len(header) >= 20 and header[18:20] == b"\x02\x02":
        if any(not Path(str(database) + suffix).is_file() for suffix in ("-wal", "-shm")):
            raise HemError("WAL recovery requires an active state writer; passive check made no changes")
    connection = None
    try:
        connection = sqlite3.connect(database.as_uri() + "?mode=ro", timeout=0, isolation_level=None)
        connection.execute("PRAGMA query_only=ON")
        def progress():
            return int(deadline.clock() >= deadline.expires_at)
        connection.set_progress_handler(progress, 1000)
        while True:
            deadline.remaining()
            try:
                connection.execute("BEGIN")
                version = connection.execute("PRAGMA user_version").fetchone()[0]
                _validate_schema(connection, version)
                servers = [Registry._read(connection, "servers", (row[0],))
                           for row in connection.execute("SELECT id FROM servers ORDER BY id").fetchall()]
                target = _read_target(connection) if version == 2 else None
                deadline.remaining()
                return dict(schema_version=version, pi_target=target, servers=servers)
            except sqlite3.OperationalError as error:
                if connection.in_transaction:
                    connection.rollback()
                code = getattr(error, "sqlite_errorcode", 0)
                if code & 0xff not in (5, 6):
                    deadline.remaining()
                    raise
                time.sleep(min(.01, deadline.remaining()))
    except sqlite3.Error as error:
        raise HemError(f"state snapshot unavailable; inventory retained: {error}") from error
    finally:
        if connection is not None:
            connection.close()
