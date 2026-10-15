"""Crash-resilient current inventory and scoped filesystem publication.

Only models, servers and instances are durable records. Callers own validation,
process observation and HTTP; none of those actions run inside transactions.
Lifecycle callers hold model locks before server locks. Updates use record
revisions/current launch tokens to reject stale observations after a wait.
"""

from __future__ import annotations

from contextlib import contextmanager
import ctypes
import errno
import fcntl
import json
import os
from pathlib import Path
import re
import sqlite3
import stat
import threading
import time
import uuid

from .hem_common import Deadline, HemError, InvalidInput

SCHEMA_VERSION = 1
_TABLES = {"models", "servers", "instances"}
_HELD = threading.local()
_SCHEMA = (
    "CREATE TABLE models (id TEXT PRIMARY KEY NOT NULL, revision INTEGER NOT NULL, data TEXT NOT NULL)",
    "CREATE TABLE servers (id TEXT PRIMARY KEY NOT NULL, launch_token TEXT, revision INTEGER NOT NULL, data TEXT NOT NULL, UNIQUE(id, launch_token))",
    "CREATE TABLE instances (server_id TEXT NOT NULL, launch_token TEXT NOT NULL, name TEXT NOT NULL, model_id TEXT, revision INTEGER NOT NULL, data TEXT NOT NULL, PRIMARY KEY(server_id, launch_token, name), FOREIGN KEY(server_id, launch_token) REFERENCES servers(id, launch_token), FOREIGN KEY(model_id) REFERENCES models(id))",
)


def _name(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise InvalidInput("resource IDs must contain letters, digits, '.', '_' or '-'")
    return value


def _json(value):
    try:
        return json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError) as error:
        raise InvalidInput(f"state must be finite JSON: {error}") from error


def _sync_dir(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _directory(path):
    missing = []
    ancestor = path
    while not ancestor.exists() and not ancestor.is_symlink():
        missing.append(ancestor)
        ancestor = ancestor.parent
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    if path.is_symlink() or not path.is_dir():
        raise HemError(f"state directory is not a real directory: {path}")
    for created in reversed(missing):
        _sync_dir(created.parent)


def _flush_tree(path):
    """Flush owned files, never follow aliases into external model storage."""
    for directory, subdirs, files in os.walk(path, followlinks=False):
        for entry in files:
            target = Path(directory) / entry
            if target.is_symlink():
                continue
            descriptor = os.open(target, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
            try:
                if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                    raise HemError(f"publication contains non-regular asset: {target}")
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        for entry in subdirs:
            if (Path(directory) / entry).is_symlink():
                raise HemError("publication cannot contain directory aliases")
    for directory, _, _ in os.walk(path, topdown=False, followlinks=False):
        _sync_dir(directory)


def _rename_new(source, destination):
    """Linux renameat2(RENAME_NOREPLACE); a replacing rename is never safe here."""
    library = ctypes.CDLL(None, use_errno=True)
    try:
        rename = library.renameat2
    except AttributeError:
        raise HemError("safe publication requires Linux renameat2 support") from None
    rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    rename.restype = ctypes.c_int
    if rename(-100, os.fsencode(source), -100, os.fsencode(destination), 1):
        code = ctypes.get_errno()
        if code == errno.EEXIST:
            raise HemError(f"destination already exists; inspect without adopting or replacing: {destination}")
        raise HemError(f"no-overwrite publication failed: {os.strerror(code)}")


class ResourceLock:
    """Crash-released flock. Closing (rather than explicit unlock) preserves an
    inherited open-file description held by a launch bootstrap until exec.
    """

    def __init__(self, root, kind, resource_id, deadline):
        self.path = root / "locks" / f"{kind}-{_name(resource_id)}.lock"
        self.kind = kind
        self.deadline = deadline
        self.descriptor = None

    def __enter__(self):
        held = getattr(_HELD, "locks", [])
        if any(item.path == self.path for item in held):
            raise HemError("resource locks are not reentrant")
        if self.kind == "model" and any(item.kind == "server" for item in held):
            raise HemError("acquire model locks before server locks")
        descriptor = os.open(self.path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            while True:
                self.deadline.remaining()
                try:
                    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    time.sleep(min(0.02, self.deadline.remaining()))
        except BaseException:
            os.close(descriptor)
            raise
        self.descriptor = descriptor
        _HELD.locks = held + [self]
        return self

    def fileno(self):
        if self.descriptor is None:
            raise HemError("lock is not held")
        return self.descriptor

    def __exit__(self, *_):
        os.close(self.descriptor)
        self.descriptor = None
        _HELD.locks.remove(self)


class Registry:
    """Private local inventory. Each method commits one short bounded transaction.

    Pass an invocation's Deadline to bound SQLite contention and lock acquisition.
    No live process truth is inferred here. JSON fields preserve only current
    domain settings/evidence, not histories. Missing rows raise HemError.
    """

    def __init__(self, root=None, *, deadline=None):
        configured = root if root is not None else os.environ.get("LOOM_SERVE_DIR")
        self.root = Path(configured) if configured is not None else Path.home() / ".loom"
        if not self.root.is_absolute():
            raise InvalidInput("LOOM_SERVE_DIR/state root must be absolute")
        self.deadline = deadline
        self.database = self.root / "control.sqlite3"
        for directory in (self.root, self.root / "models", self.root / "locks"):
            _directory(directory)
        if self.root.stat().st_uid != os.getuid():
            raise HemError("state namespace must be owned by the current user")
        os.chmod(self.root, 0o700)
        if self.database.is_symlink():
            raise HemError("state database cannot be a symlink")
        with self._transaction() as connection:
            version = connection.execute("PRAGMA user_version").fetchone()[0]
            tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if version == 0 and not tables:
                for statement in _SCHEMA:
                    connection.execute(statement)
                connection.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
            elif version != SCHEMA_VERSION or tables != _TABLES:
                raise HemError(f"unsupported state schema {version}; inventory was not reset")
            if connection.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                raise HemError("corrupt state database; inventory was not reset")
            if connection.execute("PRAGMA foreign_key_check").fetchone() is not None:
                raise HemError("state database has broken references; inventory was not reset")
            for table, expected in (("models", ["id", "revision", "data"]),
                                    ("servers", ["id", "launch_token", "revision", "data"]),
                                    ("instances", ["server_id", "launch_token", "name", "model_id", "revision", "data"])):
                if [row[1] for row in connection.execute(f"PRAGMA table_info({table})")] != expected:
                    raise HemError("state schema does not match this tool; inventory was not reset")
            actual = {row[0]: row[1] for row in connection.execute("SELECT name,sql FROM sqlite_master WHERE type='table'")}
            for statement in _SCHEMA:
                table = statement.split()[2]
                if actual[table] != statement:
                    raise HemError("state schema constraints do not match; inventory was not reset")
        os.chmod(self.database, 0o600)
        _sync_dir(self.root)

    @contextmanager
    def _transaction(self):
        deadline = self.deadline or Deadline.after()
        connection = None
        def execute(statement):
            while True:
                deadline.remaining()
                try:
                    return connection.execute(statement)
                except sqlite3.OperationalError as error:
                    code = getattr(error, "sqlite_errorcode", None)
                    busy = (code & 0xff) in (5, 6) if code is not None else "locked" in str(error).lower()
                    if not busy:
                        raise
                    time.sleep(min(0.01, deadline.remaining()))
        try:
            deadline.remaining()
            connection = sqlite3.connect(self.database, timeout=0, isolation_level=None)
            version = execute("PRAGMA user_version").fetchone()[0]
            if version not in (0, SCHEMA_VERSION):
                raise HemError(f"unsupported state schema {version}; inventory was not reset")
            execute("PRAGMA foreign_keys=ON")
            execute("PRAGMA synchronous=FULL")
            execute("PRAGMA journal_mode=DELETE")
            execute("BEGIN IMMEDIATE")
            yield connection
            execute("COMMIT")
        except sqlite3.Error as error:
            raise HemError(f"state database unavailable; inventory retained: {error}") from error
        finally:
            if connection is not None:
                connection.close()  # rolls back uncommitted work, including BaseException

    def lock(self, kind, resource_id, *, deadline=None):
        if kind not in ("model", "server"):
            raise InvalidInput("lock kind must be model or server")
        return ResourceLock(self.root, kind, resource_id, deadline or self.deadline or Deadline.after())

    @staticmethod
    def _read(connection, table, key):
        columns = ("server_id", "launch_token", "name") if table == "instances" else ("id",)
        where = " AND ".join(f"{column}=?" for column in columns)
        row = connection.execute(f"SELECT revision,data FROM {table} WHERE {where}", key).fetchone()
        if row is None:
            raise HemError(f"unknown {table[:-1]}: {'/'.join(key)}")
        try:
            data = json.loads(row[1])
            if not isinstance(data, dict):
                raise ValueError("record is not an object")
            _json(data)  # also reject non-finite values accepted by json.loads
            if any(data.get(column) != value for column, value in zip(columns, key)):
                raise ValueError("record identity does not match inventory key")
            required = {"models": {"assets", "provenance", "directory", "pipeline", "validation", "pending"},
                        "servers": {"saved_settings", "launch_token", "phase", "pending"},
                        "instances": {"model_id", "settings", "document", "dependencies", "incarnation", "pending"}}
            if not required[table].issubset(data):
                raise ValueError("record is missing current fields")
        except (ValueError, TypeError, InvalidInput) as error:
            raise HemError("corrupt state record; inventory retained") from error
        return dict(data, revision=row[0])

    @staticmethod
    def _write(connection, table, key, record, expected_revision):
        columns = ("server_id", "launch_token", "name") if table == "instances" else ("id",)
        where = " AND ".join(f"{column}=?" for column in columns)
        data = dict(record)
        data.pop("revision", None)
        extra = ", launch_token=?" if table == "servers" else ""
        args = [_json(data)]
        if table == "servers":
            args.append(data.get("launch_token"))
        args.extend((*key, expected_revision))
        count = connection.execute(f"UPDATE {table} SET data=?,revision=revision+1{extra} WHERE {where} AND revision=?", args).rowcount
        if count != 1:
            raise HemError("state changed; inspect before retrying")
        return dict(data, revision=expected_revision + 1)

    def get_model(self, model_id):
        with self._transaction() as connection:
            return self._read(connection, "models", (_name(model_id),))

    def get_server(self, server_id):
        with self._transaction() as connection:
            return self._read(connection, "servers", (_name(server_id),))

    def get_instance(self, server_id, launch_token, name):
        with self._transaction() as connection:
            return self._read(connection, "instances", (_name(server_id), _name(launch_token), _name(name)))

    def list_models(self):
        return self._list("models")

    def list_servers(self):
        return self._list("servers")

    def list_instances(self, *, server_id=None):
        return self._list("instances", server_id=server_id)

    def _list(self, table, *, server_id=None):
        with self._transaction() as connection:
            columns = "server_id,launch_token,name" if table == "instances" else "id"
            clause, args = (" WHERE server_id=?", (_name(server_id),)) if server_id else ("", ())
            keys = connection.execute(f"SELECT {columns} FROM {table}{clause} ORDER BY {columns}", args).fetchall()
            return [self._read(connection, table, tuple(key)) for key in keys]

    def prepare_model(self, model_id, *, assets, provenance=None):
        """Reserve immutable caller-described mappings and a sibling staging dir.

        Repeating identical mappings returns the existing record; it never
        replaces assets. Caller holds the model lock while populating staging.
        Asset entries are ordered dicts with unique relative 'logical_path',
        absolute 'source_path', and 'ownership' ('owned' or 'reference').
        """
        model_id = _name(model_id)
        if not isinstance(assets, list) or not assets:
            raise InvalidInput("model assets must be a nonempty ordered list")
        names = set()
        for asset in assets:
            if not isinstance(asset, dict):
                raise InvalidInput("model asset mappings must be objects")
            logical = Path(asset.get("logical_path", ""))
            if (not logical.parts or logical.is_absolute() or ".." in logical.parts or
                    str(logical) in names or asset.get("ownership") not in ("owned", "reference") or
                    not Path(asset.get("source_path", "")).is_absolute()):
                raise InvalidInput("ambiguous or unsafe model asset mapping")
            if any(logical in Path(other).parents or Path(other) in logical.parents for other in names):
                raise InvalidInput("model asset mappings overlap")
            names.add(str(logical))
        staging = self.root / "models" / f".{model_id}.staging-{uuid.uuid4().hex}"
        record = dict(id=model_id, assets=assets, provenance=provenance,
                      directory=str(self.root / "models" / model_id), pipeline=None,
                      validation=None, pending=dict(stage="staging", path=str(staging)))
        with self._transaction() as connection:
            row = connection.execute("SELECT id FROM models WHERE id=?", (model_id,)).fetchone()
            if row:
                current = self._read(connection, "models", (model_id,))
                if current["assets"] != assets or current["provenance"] != provenance:
                    raise HemError("model ID already has different mappings; use another ID")
                record = current
                staging = None
                pending = current["pending"]
                if pending and pending["stage"] == "staging" and not pending.get("inode"):
                    staging = Path(pending["path"])
            else:
                if os.path.lexists(record["directory"]):
                    raise HemError("untracked model directory exists; explicit inspection/cleanup required")
                connection.execute("INSERT INTO models VALUES(?,1,?)", (model_id, _json(record)))
        # Reservation is durable before creating a recoverable temporary directory.
        if staging is not None:
            if staging.is_symlink():
                raise HemError("staging directory cannot be an alias")
            staging.mkdir(mode=0o700, exist_ok=True)
            _sync_dir(staging.parent)
        return dict(record, revision=record.get("revision", 1))

    def publish_model(self, model_id):
        """Publish staged files without replacement, retaining unqualified state.

        Requires the model lock. If a crash occurred after rename, this explicit
        retry recognizes only the reserved staging directory's inode, not an
        arbitrary orphan. Qualification is a separate record_validation call.
        """
        current = self.get_model(model_id)
        pending = current["pending"]
        if not pending or pending["stage"] != "staging":
            return current
        staging, destination = Path(pending["path"]), Path(current["directory"])
        if staging.is_symlink():
            raise HemError("staging directory cannot be an alias")
        if staging.exists():
            _flush_tree(staging)
            identity = staging.stat()
            pending = dict(pending, device=identity.st_dev, inode=identity.st_ino)
            current = self.update_model(model_id, {"pending": pending}, expected_revision=current["revision"])
            _rename_new(staging, destination)
            _sync_dir(destination.parent)
        else:
            if destination.is_symlink() or not destination.is_dir():
                raise HemError("staging is missing; preserve record and inspect acquisition")
            identity = destination.stat()
            if (pending.get("device"), pending.get("inode")) != (identity.st_dev, identity.st_ino):
                raise HemError("published directory identity is unproven; refusing orphan adoption")
            _sync_dir(destination.parent)
        return self.update_model(model_id, {"pending": {"stage": "qualification"}}, expected_revision=current["revision"])

    def record_validation(self, model_id, *, valid, pipeline=None, evidence=None):
        """Record the latest caller-performed pipeline validation; no history."""
        if type(valid) is not bool or (valid and not pipeline):
            raise InvalidInput("valid qualification requires a selected pipeline")
        current = self.get_model(model_id)
        if current["pending"] and current["pending"]["stage"] == "staging":
            raise HemError("publish model assets before recording qualification")
        return self.update_model(model_id, {
            "validation": {"valid": valid, "evidence": evidence},
            "pipeline": pipeline if valid else None,
            "pending": None if valid else {"stage": "published-invalid"},
        }, expected_revision=current["revision"])

    def update_model(self, model_id, changes, *, expected_revision):
        if set(changes) - {"pending", "validation", "pipeline", "profile"}:
            raise InvalidInput("model mappings are immutable")
        with self._transaction() as connection:
            current = self._read(connection, "models", (_name(model_id),))
            return self._write(connection, "models", (model_id,), dict(current, **changes), expected_revision)

    def create_server(self, server_id, *, saved_settings):
        server_id = _name(server_id)
        record = dict(id=server_id, saved_settings=saved_settings, current_settings=None,
                      launch_token=None, endpoint=None, process=None, build=None,
                      phase="stopped", pending=None)
        with self._transaction() as connection:
            connection.execute("INSERT INTO servers VALUES(?,NULL,1,?)", (server_id, _json(record)))
        return dict(record, revision=1)

    def update_server(self, server_id, changes, *, expected_token, expected_revision):
        allowed = {"current_settings", "endpoint", "process", "build", "phase", "pending"}
        if set(changes) - allowed:
            raise InvalidInput("server definition is immutable")
        if "phase" in changes and changes["phase"] not in {"starting", "running", "stopping", "stopped", "unknown"}:
            raise InvalidInput("invalid server phase")
        with self._transaction() as connection:
            current = self._read(connection, "servers", (_name(server_id),))
            if current["launch_token"] != expected_token:
                raise HemError("server launch changed; inspect before updating")
            return self._write(connection, "servers", (server_id,), dict(current, **changes), expected_revision)

    def reserve_launch(self, server_id, *, launch_token, current_settings, endpoint):
        with self._transaction() as connection:
            current = self._read(connection, "servers", (_name(server_id),))
            if current["launch_token"] is not None or current["pending"] is not None or current["phase"] != "stopped":
                raise HemError("prior launch is unresolved; reconcile before starting")
            updated = dict(current, launch_token=_name(launch_token), current_settings=current_settings,
                           endpoint=endpoint, phase="starting", pending={"action": "start", "stage": "reserved"})
            return self._write(connection, "servers", (server_id,), updated, current["revision"])

    def clear_server(self, server_id, *, expected_token, expected_revision):
        """Caller has proved exit/reboot. Clear current instances, keep definition."""
        with self._transaction() as connection:
            current = self._read(connection, "servers", (_name(server_id),))
            if current["launch_token"] != expected_token:
                raise HemError("server launch changed; cannot clear another launch")
            connection.execute("DELETE FROM instances WHERE server_id=?", (server_id,))
            updated = dict(current, launch_token=None, current_settings=None, endpoint=None,
                           process=None, build=None, phase="stopped", pending=None)
            return self._write(connection, "servers", (server_id,), updated, expected_revision)

    def create_instance(self, server_id, launch_token, name, *, model_id=None,
                        settings=None, document=None, dependencies=None, incarnation=None,
                        pending=None):
        server_id, launch_token, name = map(_name, (server_id, launch_token, name))
        record = dict(server_id=server_id, launch_token=launch_token, name=name,
                      model_id=model_id, settings=settings, document=document,
                      dependencies=dependencies, incarnation=incarnation, pending=pending)
        with self._transaction() as connection:
            server = self._read(connection, "servers", (server_id,))
            if server["launch_token"] != launch_token or server["phase"] != "running" or server["pending"]:
                raise HemError("server is not accepting new managed registrations")
            if model_id is not None:
                model = self._read(connection, "models", (_name(model_id),))
                if model["pending"] or not model["validation"] or not model["validation"]["valid"]:
                    raise HemError("model is not qualified")
            connection.execute("INSERT INTO instances VALUES(?,?,?,?,1,?)",
                               (server_id, launch_token, name, model_id, _json(record)))
        return dict(record, revision=1)

    def update_instance(self, server_id, launch_token, name, changes, *, expected_revision):
        if set(changes) - {"incarnation", "pending"}:
            raise InvalidInput("submitted instance configuration is immutable")
        key = tuple(map(_name, (server_id, launch_token, name)))
        with self._transaction() as connection:
            current = self._read(connection, "instances", key)
            return self._write(connection, "instances", key, dict(current, **changes), expected_revision)

    def retire_instance(self, server_id, launch_token, name, *, expected_revision):
        """Caller has proved the original incarnation retired; delete current row."""
        key = tuple(map(_name, (server_id, launch_token, name)))
        with self._transaction() as connection:
            count = connection.execute("DELETE FROM instances WHERE server_id=? AND launch_token=? AND name=? AND revision=?", (*key, expected_revision)).rowcount
            if count != 1:
                raise HemError("instance changed; inspect before clearing")

    def references(self, model_id):
        _name(model_id)
        return [row for row in self.list_instances() if row["model_id"] == model_id]

    def write_document(self, relative_path, content):
        """Flush+replace generated JSON/documents within the private namespace.

        Never use this helper to replace model assets; those are immutable.
        Retained live deployment documents should have unique launch/name paths.
        """
        relative = Path(relative_path)
        if (not relative.parts or relative.is_absolute() or ".." in relative.parts or
                relative.parts[0] in {"models", "locks"} or relative.parts[0].startswith("control.sqlite3")):
            raise InvalidInput("generated document must have a safe non-model relative path")
        destination = self.root / relative
        parent = self.root
        for part in relative.parts[:-1]:
            parent = parent / part
            _directory(parent)
        temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(content if isinstance(content, bytes) else content.encode("utf-8"))
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
            _sync_dir(destination.parent)
        finally:
            temporary.unlink(missing_ok=True)
        return destination
