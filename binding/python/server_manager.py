"""Detached local Loom Serve process lifecycle and state reconciliation."""

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
from typing import Any, Iterable, Mapping, Optional

from . import state_store


def _server_dir() -> Path:
    return state_store.state_dir() / "servers"


def _record_path(server_id: str) -> Path:
    if not server_id or any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for ch in server_id):
        raise ValueError(f"Invalid server ID: {server_id!r}")
    return _server_dir() / f"{server_id}.json"


def _process_info(pid: int) -> Optional[dict[str, Any]]:
    """Read Linux process identity; return None when it has exited."""
    stat_path = Path(f"/proc/{pid}/stat")
    try:
        raw = stat_path.read_text(encoding="utf-8")
        close = raw.rfind(")")
        fields = raw[close + 2:].split()
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
        raise RuntimeError(f"Cannot verify process identity for PID {pid}: {exc}") from exc


def _matches(record: dict[str, Any]) -> bool:
    pid = record.get("pid")
    if not isinstance(pid, int) or pid <= 0:
        return False
    current = _process_info(pid)
    return bool(
        current
        and current["start_time"] == str(record.get("process_start_time"))
        and current["process_group"] == record.get("process_group_id")
        and os.path.realpath(current["executable"]) == os.path.realpath(record.get("executable", ""))
        and current["command"] == record.get("command")
    )


def _load_records() -> list[dict[str, Any]]:
    records = []
    for path in sorted(_server_dir().glob("*.json")):
        try:
            record = state_store.read_json(path)
            if isinstance(record, dict):
                records.append(record)
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Cannot read server record {path}: {exc}") from exc
    return records


def _reconcile(record: dict[str, Any]) -> bool:
    """Update a record if the recorded process has ended or lost identity."""
    if record.get("state") not in {"starting", "running", "stopping"}:
        return False
    if record.get("pid") is None:
        if record.get("state") == "starting" and time.time() - Path(
            _record_path(record["server_id"])
        ).stat().st_mtime > 120:
            record.update({"state": "failed", "ended_at": _now(), "error": "Launch interrupted before a PID was recorded"})
            return True
        return False
    if _matches(record):
        return False
    pid = record.get("pid")
    info = _process_info(pid) if isinstance(pid, int) and pid > 0 else None
    if info is None:
        record["state"] = "stopped" if record.get("state") == "stopping" else "exited"
    else:
        record["state"] = "stale"
    record["ended_at"] = _now()
    record["exit_status"] = "unknown"
    return True


def _now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def _reconcile_all_locked() -> list[dict[str, Any]]:
    records = _load_records()
    for record in records:
        if _reconcile(record):
            state_store.write_json_atomic(_record_path(record["server_id"]), record)
    return records


def _probe(endpoint: str, expected_model_id: str) -> bool:
    try:
        request = urllib.request.Request(endpoint + "/models", headers={"Accept": "application/json"})
        with urllib.request.urlopen(request, timeout=2) as response:
            if response.status != 200:
                return False
            data = json.loads(response.read().decode("utf-8"))
            if not isinstance(data, dict) or not isinstance(data.get("data"), list):
                return False
            return any(
                isinstance(model, dict) and model.get("id") == expected_model_id
                for model in data["data"]
            )
    except (OSError, urllib.error.URLError, UnicodeError, json.JSONDecodeError, ValueError):
        return False


def launch(
    model_id: str,
    pipeline: str,
    executable: Path,
    argv: list[str],
    port: int,
    model_paths: Iterable[Path],
    launch_options: dict[str, Any],
    ready_timeout: float = 180.0,
    environment: Optional[Mapping[str, str]] = None,
) -> dict[str, Any]:
    """Launch a standalone server, wait for its API, and return its record."""
    server_id = uuid.uuid4().hex[:12]
    state_dir = state_store.state_dir()
    log_path = state_dir / "logs" / f"{server_id}.log"
    endpoint = f"http://127.0.0.1:{port}/v1"
    record: dict[str, Any] = {
        "server_id": server_id, "state": "starting", "model_id": model_id,
        "pipeline": pipeline, "created_at": _now(), "pid": None,
        "process_start_time": None, "process_group_id": None,
        "bind_address": "127.0.0.1", "port": port, "endpoint": endpoint,
        "launch_options": launch_options, "model_paths": [str(p) for p in model_paths],
        "executable": str(executable.resolve()), "log_path": str(log_path),
        "command": list(argv),
    }
    existing_server = None
    with state_store.state_lock():
        existing = _reconcile_all_locked()
        active = [
            item for item in existing
            if item.get("port") == port and item.get("state") in {"starting", "running", "stopping"}
        ]
        if active:
            owner = active[0]
            if owner.get("model_id") == model_id and owner.get("state") == "running":
                existing_server = owner
            else:
                raise RuntimeError(
                    f"Port {port} is already assigned to managed server {owner.get('server_id')} "
                    f"({owner.get('model_id')}, {owner.get('state')})"
                )
    if existing_server is not None:
        existing_server["endpoint_health"] = (
            "healthy" if _probe(endpoint, model_id) else "unhealthy"
        )
        if existing_server["endpoint_health"] == "healthy":
            return existing_server
        raise RuntimeError(
            f"Managed server {existing_server.get('server_id')} for {model_id!r} owns port {port} "
            "but its endpoint is unhealthy"
        )
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        if probe.connect_ex(("127.0.0.1", port)) == 0:
            raise RuntimeError(f"Port {port} is already in use by an unmanaged listener")
    with state_store.state_lock():
        existing = _reconcile_all_locked()
        active = [
            item for item in existing
            if item.get("port") == port and item.get("state") in {"starting", "running", "stopping"}
        ]
        if active:
            owner = active[0]
            raise RuntimeError(
                f"Port {port} was claimed by managed server {owner.get('server_id')} "
                f"while a new server was being prepared"
            )
        state_store.write_json_atomic(_record_path(server_id), record)

    try:
        with state_store.private_append_log(log_path) as log:
            process = subprocess.Popen(
                argv, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True, close_fds=True,
                env=dict(environment) if environment is not None else None,
            )
        info = _process_info(process.pid)
        if (
            not info
            or os.path.realpath(info["executable"]) != os.path.realpath(str(executable))
            or info["command"] != list(argv)
        ):
            raise RuntimeError(f"Could not verify newly launched server process {process.pid}")
        record.update({
            "pid": process.pid,
            "process_start_time": info["start_time"],
            "process_group_id": info["process_group"],
        })
        with state_store.state_lock():
            state_store.write_json_atomic(_record_path(server_id), record)

        deadline = time.monotonic() + ready_timeout
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(
                    f"Loom Serve exited during startup (status {process.returncode}); see {log_path}"
                )
            if _probe(endpoint, model_id):
                record.update({"state": "running", "ready_at": _now()})
                with state_store.state_lock():
                    current = state_store.read_json(_record_path(server_id))
                    if isinstance(current, dict) and current.get("state") in {"stopping", "stopped"}:
                        raise RuntimeError("Server stop was requested during startup")
                    state_store.write_json_atomic(_record_path(server_id), record)
                return record
            time.sleep(1)
        raise RuntimeError(f"Loom Serve did not become ready within {ready_timeout:g}s; see {log_path}")
    except Exception as exc:
        if "process" in locals() and process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=5)
            except (OSError, subprocess.TimeoutExpired):
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except OSError:
                    pass
        with state_store.state_lock():
            current = state_store.read_json(_record_path(server_id))
            if isinstance(current, dict) and current.get("state") in {"stopping", "stopped"}:
                record.update({"state": "stopped", "ended_at": _now(), "exit_status": "unknown"})
            else:
                record.update({"state": "failed", "ended_at": _now(), "error": str(exc)})
            state_store.write_json_atomic(_record_path(server_id), record)
        raise


def list_servers() -> list[dict[str, Any]]:
    with state_store.state_lock():
        records = _reconcile_all_locked()
    for record in records:
        if record.get("state") == "running":
            record["endpoint_health"] = (
                "healthy" if _probe(record["endpoint"], record["model_id"]) else "unhealthy"
            )
    return records


def get_server(server_id: str) -> dict[str, Any]:
    path = _record_path(server_id)
    with state_store.state_lock():
        record = state_store.read_json(path)
        if not isinstance(record, dict):
            raise ValueError(f"No managed server with ID {server_id!r}")
        if _reconcile(record):
            state_store.write_json_atomic(path, record)
    if record.get("state") == "running":
        record["endpoint_health"] = (
            "healthy" if _probe(record["endpoint"], record["model_id"]) else "unhealthy"
        )
    return record


def stop_server(server_id: str, timeout: float = 15.0) -> dict[str, Any]:
    path = _record_path(server_id)
    with state_store.state_lock():
        record = state_store.read_json(path)
        if not isinstance(record, dict):
            raise ValueError(f"No managed server with ID {server_id!r}")
        if _reconcile(record):
            state_store.write_json_atomic(path, record)
        if record.get("state") == "starting" and not isinstance(record.get("pid"), int):
            raise RuntimeError(
                f"Server {server_id!r} is still starting and has no recorded process ID; retry shortly"
            )
        if record.get("state") in {"exited", "stopped", "failed", "stale"}:
            return record
        if not _matches(record):
            record.update({"state": "stale", "ended_at": _now()})
            state_store.write_json_atomic(path, record)
            return record
        record.update({"state": "stopping", "stop_requested_at": _now()})
        state_store.write_json_atomic(path, record)

    try:
        os.killpg(record["process_group_id"], signal.SIGTERM)
    except ProcessLookupError:
        record.update({"state": "stopped", "ended_at": _now()})
        with state_store.state_lock():
            state_store.write_json_atomic(path, record)
        return record
    except OSError as exc:
        record["stop_error"] = f"Could not signal managed process group: {exc}"
        if not _matches(record):
            _reconcile(record)
        elif record.get("state") == "stopping":
            record["state"] = "running"
        with state_store.state_lock():
            state_store.write_json_atomic(path, record)
        return record
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _process_info(record["pid"]) is None:
            record.update({"state": "stopped", "ended_at": _now(), "exit_status": "unknown"})
            break
        if not _matches(record):
            record.update({"state": "stale", "ended_at": _now()})
            record["stop_error"] = "PID identity changed while stopping; refusing further signals"
            break
        time.sleep(0.2)
    else:
        if _matches(record):
            record["stop_error"] = f"Process did not exit within {timeout:g}s"
            record["state"] = "running"
        elif _process_info(record["pid"]) is None:
            record.update({"state": "stopped", "ended_at": _now(), "exit_status": "unknown"})
        else:
            record.update({"state": "stale", "ended_at": _now()})
            record["stop_error"] = "PID identity changed while stopping; refusing further signals"
    with state_store.state_lock():
        latest = state_store.read_json(path)
        if isinstance(latest, dict) and latest.get("pid") == record.get("pid"):
            state_store.write_json_atomic(path, record)
    return record
