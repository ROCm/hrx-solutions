"""Detached, client-side SSH tunnel lifecycle and state reconciliation."""

from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from . import remote_registry, state_store


class ConnectionError(RuntimeError):
    """Raised when a managed SSH connection cannot be operated safely."""


def _connection_dir() -> Path:
    return state_store.state_dir() / "connections"


def _record_path(connection_id: str) -> Path:
    if not connection_id or any(
        ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
        for ch in connection_id
    ):
        raise ValueError(f"Invalid connection ID: {connection_id!r}")
    return _connection_dir() / f"{connection_id}.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _process_info(pid: int) -> Optional[dict[str, Any]]:
    """Read Linux process identity; return None when the process has exited."""
    try:
        raw = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        close = raw.rfind(")")
        fields = raw[close + 2 :].split()
        if fields[0] == "Z":
            return None
        start_time = fields[19]  # /proc stat field 22, process start ticks
        process_group = int(fields[2])
        executable = os.readlink(f"/proc/{pid}/exe")
        command = Path(f"/proc/{pid}/cmdline").read_bytes().rstrip(b"\0").split(b"\0")
        return {
            "start_time": start_time,
            "process_group": process_group,
            "executable": executable,
            "command": [part.decode("utf-8", errors="replace") for part in command],
        }
    except (FileNotFoundError, ProcessLookupError):
        return None
    except (OSError, IndexError, ValueError) as exc:
        raise ConnectionError(f"Cannot verify SSH process identity for PID {pid}: {exc}") from exc


def _matches(record: dict[str, Any]) -> bool:
    pid = record.get("pid")
    if not isinstance(pid, int) or pid <= 0:
        return False
    current = _process_info(pid)
    return bool(
        current
        and current["start_time"] == str(record.get("process_start_time"))
        and current["process_group"] == record.get("process_group_id")
        and os.path.realpath(current["executable"])
        == os.path.realpath(record.get("executable", ""))
        and current["command"] == record.get("command")
    )


def _load_records() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    directory = _connection_dir()
    for path in sorted(directory.glob("*.json")) if directory.is_dir() else []:
        try:
            record = state_store.read_json(path)
            if isinstance(record, dict):
                records.append(record)
        except (OSError, json.JSONDecodeError) as exc:
            raise ConnectionError(f"Cannot read connection record {path}: {exc}") from exc
    return records


def _reconcile(record: dict[str, Any]) -> bool:
    """Mark live records exited or stale if their saved process identity fails."""
    if record.get("state") not in {"connecting", "connected", "unreachable", "disconnecting"}:
        return False
    if record.get("pid") is None:
        # A launch interrupted before its PID was persisted is not safely
        # signalable. Expire the incomplete record after a short recovery grace.
        if record.get("state") == "connecting":
            path = _record_path(record["connection_id"])
            try:
                if time.time() - path.stat().st_mtime > 120:
                    record.update({
                        "state": "failed",
                        "ended_at": _now(),
                        "error": "Connection launch was interrupted before an SSH PID was recorded",
                    })
                    return True
            except FileNotFoundError:
                pass
        return False
    if _matches(record):
        return False
    pid = record.get("pid")
    info = _process_info(pid) if isinstance(pid, int) and pid > 0 else None
    record["state"] = "disconnected" if record.get("state") == "disconnecting" and info is None else (
        "exited" if info is None else "stale"
    )
    record["ended_at"] = _now()
    record["exit_status"] = "unknown"
    if record["state"] == "stale":
        record["error"] = "Recorded PID is alive but no longer matches the managed SSH process"
    return True


def _reconcile_all_locked() -> list[dict[str, Any]]:
    records = _load_records()
    for record in records:
        if _reconcile(record):
            state_store.write_json_atomic(_record_path(record["connection_id"]), record)
    return records


def _probe(endpoint: str, expected_model_id: Optional[str]) -> tuple[bool, str]:
    """Check the forwarded API and, when requested, its exact model identity."""
    try:
        request = urllib.request.Request(endpoint + "/models", headers={"Accept": "application/json"})
        with urllib.request.urlopen(request, timeout=2) as response:
            if response.status != 200:
                return False, f"HTTP endpoint returned status {response.status}"
            data = json.loads(response.read().decode("utf-8"))
        if not isinstance(data, dict) or not isinstance(data.get("data"), list):
            return False, "HTTP endpoint returned an invalid /v1/models response"
        if expected_model_id is not None and not any(
            isinstance(model, dict) and model.get("id") == expected_model_id
            for model in data["data"]
        ):
            advertised = [
                model.get("id") for model in data["data"] if isinstance(model, dict)
            ]
            return False, (
                f"Model mismatch: expected {expected_model_id!r}, "
                f"server advertises {advertised!r}"
            )
        return True, "healthy"
    except (OSError, urllib.error.URLError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        return False, f"Forwarded endpoint is unavailable: {exc}"


def _health(record: dict[str, Any]) -> dict[str, Any]:
    alive = _matches(record)
    if not alive:
        pid = record.get("pid")
        process_health = (
            "exited"
            if not isinstance(pid, int) or _process_info(pid) is None
            else "stale"
        )
        return {"process_health": process_health, "endpoint_health": "unknown"}
    healthy, detail = _probe(record["endpoint"], record.get("expected_model_id"))
    return {
        "process_health": "running",
        "endpoint_health": "healthy" if healthy else "unhealthy",
        "health_detail": detail,
        "last_health_check": _now(),
    }


def connect(
    remote_name: str,
    model_id: Optional[str] = None,
    *,
    ready_timeout: float = 30.0,
) -> dict[str, Any]:
    """Start a detached SSH loopback tunnel and wait until its API is usable."""
    if ready_timeout <= 0:
        raise ValueError("Connection readiness timeout must be greater than zero")
    remote = remote_registry.get_remote(remote_name)
    connection_id = uuid.uuid4().hex[:12]
    local_port = remote.local_port
    endpoint = f"http://127.0.0.1:{local_port}/v1"
    log_path = state_store.state_dir() / "logs" / f"{connection_id}.log"
    argv = [
        "ssh", "-N", "-T",
        "-o", "BatchMode=yes",
        "-o", "ExitOnForwardFailure=yes",
        "-o", "ConnectTimeout=10",
        "-L", f"127.0.0.1:{local_port}:127.0.0.1:{remote.server_port}",
        remote.ssh_target,
    ]
    record: dict[str, Any] = {
        "schema_version": 1,
        "connection_id": connection_id,
        "state": "connecting",
        "created_at": _now(),
        "remote_name": remote.name,
        "ssh_target": remote.ssh_target,
        "pid": None,
        "process_start_time": None,
        "process_group_id": None,
        "executable": None,
        "command": argv,
        "local_bind_address": "127.0.0.1",
        "local_port": local_port,
        "remote_bind_address": "127.0.0.1",
        "remote_port": remote.server_port,
        "endpoint": endpoint,
        "local_url": endpoint,
        "expected_model_id": model_id,
        "log_path": str(log_path),
    }

    with state_store.state_lock():
        existing = _reconcile_all_locked()
        conflicts = [
            item for item in existing
            if item.get("local_port") == local_port
            and item.get("state") in {"connecting", "connected", "unreachable", "disconnecting"}
        ]
        if conflicts:
            owner = conflicts[0]
            raise ConnectionError(
                f"Local tunnel port {local_port} is already assigned to connection "
                f"{owner.get('connection_id')} ({owner.get('remote_name')}, {owner.get('state')})"
            )
        state_store.write_json_atomic(_record_path(connection_id), record)

    # Check port availability before starting ssh. This is an early diagnostic;
    # ExitOnForwardFailure handles the race between this check and bind.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        try:
            probe.bind(("127.0.0.1", local_port))
        except OSError as exc:
            record.update({"state": "failed", "ended_at": _now(), "error": f"Local port {local_port} is unavailable: {exc}"})
            with state_store.state_lock():
                state_store.write_json_atomic(_record_path(connection_id), record)
            raise ConnectionError(record["error"]) from exc

    try:
        with state_store.private_append_log(log_path) as log:
            process = subprocess.Popen(
                argv,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                close_fds=True,
            )
        info = _process_info(process.pid)
        if not info or info["command"] != argv:
            raise ConnectionError(f"Could not verify newly launched SSH process {process.pid}")
        record.update({
            "pid": process.pid,
            "process_start_time": info["start_time"],
            "process_group_id": info["process_group"],
            "executable": info["executable"],
        })
        with state_store.state_lock():
            state_store.write_json_atomic(_record_path(connection_id), record)

        deadline = time.monotonic() + ready_timeout
        last_detail = "SSH tunnel did not become ready"
        while time.monotonic() < deadline:
            if process.poll() is not None:
                tail = ""
                try:
                    tail = log_path.read_text(encoding="utf-8", errors="replace")[-1000:].strip()
                except OSError:
                    pass
                detail = f"SSH exited with status {process.returncode}"
                if tail:
                    detail += f": {tail}"
                raise ConnectionError(f"{detail}; see {log_path}")
            healthy, last_detail = _probe(endpoint, model_id)
            if healthy:
                with state_store.state_lock():
                    current = state_store.read_json(_record_path(connection_id))
                    if isinstance(current, dict) and current.get("state") in {"disconnecting", "disconnected"}:
                        raise ConnectionError("Tunnel disconnect was requested during startup")
                    record.update({
                        "state": "connected",
                        "process_health": "running",
                        "endpoint_health": "healthy",
                        "health_detail": "healthy",
                        "connected_at": _now(),
                        "last_health_check": _now(),
                    })
                    state_store.write_json_atomic(_record_path(connection_id), record)
                return record
            time.sleep(0.5)
        raise ConnectionError(f"SSH tunnel did not become ready within {ready_timeout:g}s: {last_detail}; see {log_path}")
    except Exception as exc:
        cleanup_error: Optional[str] = None
        with state_store.state_lock():
            current = state_store.read_json(_record_path(connection_id))
            disconnect_requested = isinstance(current, dict) and current.get("state") in {
                "disconnecting", "disconnected"
            }
        if "process" in locals() and process.poll() is None and not disconnect_requested:
            try:
                _terminate_verified(record)
            except ConnectionError as stop_exc:
                cleanup_error = str(stop_exc)
        with state_store.state_lock():
            current = state_store.read_json(_record_path(connection_id))
            if isinstance(current, dict) and current.get("state") == "disconnected":
                record = current
            elif isinstance(current, dict) and current.get("state") == "disconnecting":
                if _process_info(current.get("pid")) is None:
                    current.update({"state": "disconnected", "ended_at": _now(), "exit_status": "unknown"})
                    state_store.write_json_atomic(_record_path(connection_id), current)
                record = current
            elif cleanup_error:
                if _matches(record):
                    record.update({"state": "unreachable", "ended_at": None, "error": str(exc)})
                    record["cleanup_error"] = cleanup_error
                else:
                    _reconcile(record)
                    record["error"] = str(exc)
                    record["cleanup_error"] = cleanup_error
                state_store.write_json_atomic(_record_path(connection_id), record)
            else:
                record.update({"state": "failed", "ended_at": _now(), "error": str(exc)})
                state_store.write_json_atomic(_record_path(connection_id), record)
        if isinstance(exc, ConnectionError):
            raise
        raise ConnectionError(f"Could not establish SSH tunnel: {exc}; see {log_path}") from exc


def _terminate_verified(record: dict[str, Any], timeout: float = 8.0) -> None:
    """Terminate only the exact recorded process group, after identity checks."""
    pid = record.get("pid")
    if not isinstance(pid, int) or pid <= 0:
        raise ConnectionError("Cannot disconnect: no SSH process ID was recorded")
    if _process_info(pid) is None:
        return
    if not _matches(record):
        raise ConnectionError(
            f"Refusing to signal PID {record.get('pid')}: it does not match the recorded SSH process identity"
        )
    process_group = record["process_group_id"]
    if process_group != pid:
        raise ConnectionError(f"Refusing to signal unexpected SSH process group {process_group}")
    try:
        os.killpg(process_group, signal.SIGTERM)
    except ProcessLookupError:
        return
    except OSError as exc:
        raise ConnectionError(f"Could not stop SSH tunnel process group {process_group}: {exc}") from exc
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        info = _process_info(pid)
        if info is None:
            return
        if not _matches(record):
            raise ConnectionError(f"SSH process identity changed while stopping PID {pid}; refusing further signals")
        time.sleep(0.1)
    if _matches(record):
        try:
            os.killpg(process_group, signal.SIGKILL)
        except ProcessLookupError:
            return
        except OSError as exc:
            raise ConnectionError(f"Could not force-stop SSH tunnel process group {process_group}: {exc}") from exc
        kill_deadline = time.monotonic() + 2.0
        while time.monotonic() < kill_deadline:
            info = _process_info(pid)
            if info is None:
                return
            if not _matches(record):
                raise ConnectionError(
                    f"SSH process identity changed after SIGKILL for PID {pid}; refusing further action"
                )
            time.sleep(0.05)
        info = _process_info(pid)
        if info is not None:
            if _matches(record):
                raise ConnectionError(f"SSH process {pid} is still alive after SIGKILL")
            raise ConnectionError(
                f"PID {pid} no longer identifies the SSH process after SIGKILL; refusing to claim it stopped"
            )


def list_connections() -> list[dict[str, Any]]:
    """Return all connection records after reconciling process state."""
    with state_store.state_lock():
        records = _reconcile_all_locked()
    for record in records:
        if record.get("state") in {"connecting", "connected", "unreachable"} and record.get("pid"):
            health = _health(record)
            record.update(health)
            if health["process_health"] in {"exited", "stale"}:
                record["state"] = health["process_health"]
                record["ended_at"] = _now()
                record["exit_status"] = "unknown"
            elif health["endpoint_health"] == "healthy":
                record["state"] = "connected"
            else:
                record["state"] = "unreachable"
            with state_store.state_lock():
                path = _record_path(record["connection_id"])
                current = state_store.read_json(path)
                if isinstance(current, dict) and current.get("pid") == record.get("pid") and current.get("state") in {
                    "connecting", "connected", "unreachable"
                }:
                    state_store.write_json_atomic(path, record)
    return records


def get_connection(connection_id: str) -> dict[str, Any]:
    """Return one reconciled connection record with current health information."""
    path = _record_path(connection_id)
    with state_store.state_lock():
        record = state_store.read_json(path)
        if not isinstance(record, dict):
            raise ConnectionError(f"No managed SSH connection with ID {connection_id!r}")
        if _reconcile(record):
            state_store.write_json_atomic(path, record)
    if record.get("state") in {"connecting", "connected", "unreachable"} and record.get("pid"):
        health = _health(record)
        record.update(health)
        if health["process_health"] in {"exited", "stale"}:
            record["state"] = health["process_health"]
            record["ended_at"] = _now()
            record["exit_status"] = "unknown"
        else:
            record["state"] = "connected" if health["endpoint_health"] == "healthy" else "unreachable"
        with state_store.state_lock():
            current = state_store.read_json(path)
            if isinstance(current, dict) and current.get("pid") == record.get("pid") and current.get("state") in {
                "connecting", "connected", "unreachable"
            }:
                state_store.write_json_atomic(path, record)
    return record


def verify_connection(remote_name: str, model_id: str) -> dict[str, Any]:
    """Return a tracked, live tunnel for ``remote_name`` serving ``model_id``."""
    records = list_connections()
    candidates = [
        record for record in records
        if record.get("remote_name") == remote_name
        and record.get("state") == "connected"
        and isinstance(record.get("endpoint"), str)
    ]
    for record in candidates:
        healthy, detail = _probe(record["endpoint"], model_id)
        if healthy:
            record["expected_model_id"] = model_id
            record["health_detail"] = detail
            return record
    if not candidates:
        raise ConnectionError(
            f"No active tracked SSH tunnel for remote {remote_name!r}; run "
            f"`loom-serve-setup remote connect {remote_name} --model {model_id}` first"
        )
    details = "; ".join(
        str(record.get("health_detail") or record.get("endpoint_health") or "unavailable")
        for record in candidates
    )
    raise ConnectionError(
        f"No active tunnel for remote {remote_name!r} advertises model {model_id!r}: {details}"
    )


def disconnect(connection_id: str) -> dict[str, Any]:
    """Stop a verified tunnel process and retain its state record and log."""
    path = _record_path(connection_id)
    with state_store.state_lock():
        record = state_store.read_json(path)
        if not isinstance(record, dict):
            raise ConnectionError(f"No managed SSH connection with ID {connection_id!r}")
        if _reconcile(record):
            state_store.write_json_atomic(path, record)
        if record.get("state") in {"disconnected", "exited", "failed"}:
            return record
        if record.get("state") == "stale":
            raise ConnectionError(
                f"Cannot disconnect {connection_id}: recorded PID identity is stale; refusing to signal it"
            )
        if not isinstance(record.get("pid"), int):
            raise ConnectionError(
                f"Cannot disconnect {connection_id}: SSH launch has not recorded a verifiable PID yet"
            )
        record["state"] = "disconnecting"
        state_store.write_json_atomic(path, record)
    try:
        _terminate_verified(record)
    except ConnectionError as exc:
        healthy = False
        detail = str(exc)
        if _matches(record):
            healthy, detail = _probe(record["endpoint"], record.get("expected_model_id"))
        with state_store.state_lock():
            current = state_store.read_json(path)
            if isinstance(current, dict):
                if current.get("state") == "disconnecting" and _matches(current):
                    current.update({
                        "state": "connected" if healthy else "unreachable",
                        "disconnect_error": str(exc),
                        "endpoint_health": "healthy" if healthy else "unhealthy",
                        "health_detail": detail,
                    })
                    state_store.write_json_atomic(path, current)
                elif _reconcile(current):
                    state_store.write_json_atomic(path, current)
        raise
    record.update({"state": "disconnected", "ended_at": _now(), "exit_status": "unknown"})
    with state_store.state_lock():
        state_store.write_json_atomic(path, record)
    return record
