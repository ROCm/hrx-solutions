"""Read-only local inventory observations for the Loom Control TUI."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import os
from pathlib import Path
import sqlite3
import stat
import threading
from typing import Optional
from urllib.parse import quote

from .hem_common import Deadline, DeadlineExpired, HemError
from .hem_http import NativeClient
from .hem_process import PinnedProcess, observe_server
from .runtime_paths import state_root
from .hem_state import _validate_schema, decode_record

MAX_INVENTORY_ROWS = 4096
MAX_INVENTORY_BYTES = 8 * 1024 * 1024
MAX_TUI_STATUS_BYTES = 1024 * 1024


@dataclass(frozen=True)
class _ServerSnapshot:
    record: dict

    @property
    def identity(self):
        return self.record["id"], self.record["launch_token"], self.record["revision"]


@dataclass(frozen=True)
class _InstanceSnapshot:
    record: dict

    @property
    def identity(self):
        row = self.record
        return (row["server_id"], row["launch_token"], row["name"],
                row["incarnation"], row["revision"])


@dataclass(frozen=True)
class _ServerHealth:
    process: str
    http: str
    native: Optional[dict] = None


def _database_uri(path: Path) -> str:
    return "file:" + quote(str(path), safe="/") + "?mode=ro"


def _read_snapshot(root: Path, deadline: Deadline):
    """Read servers and instances together without creating or migrating state."""
    database = root / "control.sqlite3"
    try:
        info = database.lstat()
    except FileNotFoundError:
        return (), ()
    if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid():
        raise HemError("state database must be a same-user regular file")

    connection = None
    try:
        deadline.remaining()
        connection = sqlite3.connect(_database_uri(database), uri=True, timeout=0,
                                     isolation_level=None)
        connection.set_progress_handler(
            lambda: int(deadline.clock() >= deadline.expires_at), 1000)
        connection.execute("PRAGMA query_only=ON")
        connection.execute("BEGIN")
        deadline.remaining()
        version = connection.execute("PRAGMA user_version").fetchone()[0]
        if version != 2:
            raise HemError(f"unsupported state schema {version}; inventory was not reset")
        total_rows = total_bytes = 0
        for table in ("models", "servers", "instances"):
            count, size = connection.execute(
                f"SELECT COUNT(*),COALESCE(SUM(LENGTH(CAST(data AS BLOB))),0) "
                f"FROM {table}").fetchone()
            total_rows += count
            total_bytes += size
            deadline.remaining()
        if total_rows > MAX_INVENTORY_ROWS or total_bytes > MAX_INVENTORY_BYTES:
            raise HemError("local inventory exceeds the TUI observation limit")
        _validate_schema(connection, version, deadline=deadline)
        deadline.remaining()
        server_rows = connection.execute(
            "SELECT id,launch_token,revision,data FROM servers ORDER BY id").fetchall()
        instance_rows = connection.execute(
            "SELECT server_id,launch_token,name,revision,data FROM instances "
            "ORDER BY server_id,launch_token,name").fetchall()
        deadline.remaining()
        connection.execute("COMMIT")
        servers = []
        for server_id, launch_token, revision, data in server_rows:
            deadline.remaining()
            servers.append(_ServerSnapshot(decode_record(
                "servers", data, revision, {"id": server_id, "launch_token": launch_token})))
        instances = []
        for server_id, launch_token, name, revision, data in instance_rows:
            deadline.remaining()
            instances.append(_InstanceSnapshot(decode_record(
                "instances", data, revision,
                {"server_id": server_id, "launch_token": launch_token, "name": name})))
        return servers, instances
    except sqlite3.Error as error:
        raise HemError(f"state database unavailable; inventory retained: {error}") from error
    finally:
        if connection is not None:
            connection.close()


def _observe_server(record, deadline):
    observation = observe_server(
        record, deadline,
        pin=lambda row: PinnedProcess(row['process'], row['build']['executable'],
                                     emit=lambda _message: None),
        client=lambda row, process, budget: NativeClient(
            row['endpoint'], budget, process.connection,
            ownership_after=process.verify, emit=lambda _message: None),
        max_response_bytes=MAX_TUI_STATUS_BYTES,
    )
    return _ServerHealth(
        {'verified': 'running', 'exited': 'stopped', 'unknown': 'unknown'}[observation.process_health],
        {'healthy': 'available', 'unhealthy': 'unavailable', 'not-running': 'not-running',
         'unknown': 'unknown'}[observation.http_health], observation.native)


def _server_state(record, health):
    if record["launch_token"] is None:
        return "stopped"
    if health.process == "unknown":
        return "unknown"
    if health.http != "available":
        return "unavailable"
    return "running" if record["phase"] == "running" else record["phase"]


def _changed_server(server_id):
    from .hem_tui import ServiceObservation
    return ServiceObservation(
        name=server_id, endpoint="", state="unknown",
        summary="State changed during refresh",
        kind="server", server=server_id, recorded_state="unknown",
        process_state="unknown", http_state="unknown",
        identity_verified=False,
    )


def _unknown_model(row):
    from .hem_tui import ServiceObservation
    return ServiceObservation(
        name=row["name"], endpoint="", state="unknown",
        summary="State changed or refresh budget ended",
        kind="model", server=row["server_id"],
        launch_token=row["launch_token"], incarnation=row["incarnation"],
        recorded_state="unknown", process_state="unknown", http_state="unknown",
        identity_verified=False,
    )


def _unknown_tail(servers, grouped_instances):
    rows = []
    for server in servers:
        server_id = server.record["id"]
        rows.append(_changed_server(server_id))
        rows.extend(_unknown_model(item.record)
                    for item in grouped_instances.get(server_id, ()))
    return rows


def _build_observations(servers, instances, health_by_server, current, *, deadline=None):
    from .hem_tui import ServiceObservation

    check = deadline.remaining if deadline is not None else lambda: None
    grouped_instances = {}
    try:
        for item in instances:
            check()
            grouped_instances.setdefault(item.record["server_id"], []).append(item)
        current_servers, current_instances = current
        server_identities = {}
        for item in current_servers:
            check()
            server_identities[item.record["id"]] = item.identity
        instance_identities = {}
        for item in current_instances:
            check()
            instance_identities[item.identity[:3]] = item.identity
    except DeadlineExpired:
        grouped_instances = {}
        for item in instances:  # Hard inventory caps bound this fallback.
            grouped_instances.setdefault(item.record["server_id"], []).append(item)
        return tuple(_unknown_tail(servers, grouped_instances))
    observations = []
    for server_index, server in enumerate(servers):
        record = server.record
        server_id = record["id"]
        try:
            check()
        except DeadlineExpired:
            observations.extend(_unknown_tail(servers[server_index:], grouped_instances))
            return tuple(observations)
        if server_identities.get(server_id) != server.identity:
            observations.append(_changed_server(server_id))
            observations.extend(_unknown_model(item.record)
                                for item in grouped_instances.get(server_id, ()))
            continue
        health = health_by_server[server_id]
        state = _server_state(record, health)
        observations.append(ServiceObservation(
            name=server_id, endpoint=record.get("endpoint") or "", state=state,
            summary=(f"Recorded {record['phase']} · process {health.process} · "
                     f"HTTP {health.http}"),
            kind="server", server=server_id,
            launch_token=record["launch_token"],
            recorded_state=record["phase"], process_state=health.process,
            http_state=health.http,
        ))
        native_by_identity = {}
        try:
            for item in health.native.get("data", ()) if health.native is not None else ():
                check()
                native_by_identity[(item["id"], item["loom"]["incarnation"])] = item
        except DeadlineExpired:
            observations.extend(_unknown_model(item.record)
                                for item in grouped_instances.get(server_id, ()))
            observations.extend(_unknown_tail(servers[server_index + 1:], grouped_instances))
            return tuple(observations)
        server_instances = grouped_instances.get(server_id, ())
        for instance_index, instance in enumerate(server_instances):
            try:
                check()
            except DeadlineExpired:
                observations.extend(_unknown_model(item.record)
                                    for item in server_instances[instance_index:])
                observations.extend(_unknown_tail(
                    servers[server_index + 1:], grouped_instances))
                return tuple(observations)
            row = instance.record
            if instance_identities.get(instance.identity[:3]) != instance.identity:
                observations.append(_unknown_model(row))
                continue
            native = native_by_identity.get((row["name"], row["incarnation"]))
            if health.http != "available":
                instance_state = "unavailable" if health.process == "running" else "unknown"
            elif native is None:
                instance_state = "unavailable"
            else:
                instance_state = native["loom"].get("registration", "running")
            recorded = "pending" if row["pending"] else "registered"
            observations.append(ServiceObservation(
                name=row["name"], endpoint=record.get("endpoint") or "",
                state=instance_state,
                summary=(f"Recorded {recorded} · incarnation "
                         f"{row['incarnation'] if row['incarnation'] is not None else 'unknown'}"),
                kind="model", server=server_id,
                launch_token=row["launch_token"], incarnation=row["incarnation"],
                recorded_state=recorded, process_state=health.process,
                http_state=health.http,
            ))
    return tuple(observations)


class LocalServiceProvider:
    """Observe the existing local namespace without creating or reconciling it."""

    def __init__(self, root=None, *, timeout=2.0):
        self.root = Path(root) if root is not None else state_root()
        if not self.root.is_absolute():
            raise ValueError("TUI state root must be absolute")
        self.timeout = timeout
        self._idle = threading.Event()
        self._idle.set()

    async def list_services(self):
        return await asyncio.to_thread(self._tracked_list_services)

    def _tracked_list_services(self):
        self._idle.clear()
        try:
            return self._list_services()
        finally:
            self._idle.set()

    def _list_services(self):
        deadline = Deadline.after(self.timeout)
        servers, instances = _read_snapshot(self.root, deadline)
        health = {}
        for server in servers:
            try:
                deadline.remaining()
                health[server.record["id"]] = _observe_server(server.record, deadline)
            except DeadlineExpired:
                break
            except (HemError, KeyError, TypeError, ValueError, OSError):
                health[server.record["id"]] = _ServerHealth("unknown", "unknown")
        for server in servers:
            health.setdefault(server.record["id"], _ServerHealth("unknown", "unknown"))
        try:
            current = _read_snapshot(self.root, deadline)
        except (DeadlineExpired, HemError):
            current = ((), ())
        return _build_observations(
            servers, instances, health, current, deadline=deadline)
