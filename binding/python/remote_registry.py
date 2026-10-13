"""Client-side SSH remote definitions and model inventory discovery.

Remote definitions contain only a local name and SSH destination. SSH keys,
usernames, proxies, and host-key policy remain in the user's SSH configuration.
"""

from __future__ import annotations

import json
import re
import shlex
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

from . import model_store
from .state_store import read_json, state_dir, state_lock, write_json_atomic


_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_TARGET_RE = re.compile(r"^[A-Za-z0-9_.@:-]+$")
_MODEL_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_DEFAULT_SERVER_PORT = 8080
_DEFAULT_LOCAL_PORT = 18080
_SERVER_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")


class RemoteRegistryError(ValueError):
    """Raised when a remote definition or inventory operation is invalid."""


@dataclass(frozen=True)
class Remote:
    """A named SSH destination and the ports used by Loom Serve."""

    name: str
    ssh_target: str
    server_port: int = _DEFAULT_SERVER_PORT
    local_port: int = _DEFAULT_LOCAL_PORT
    schema_version: int = 1


def _validate_name(name: str) -> None:
    if not isinstance(name, str) or not _NAME_RE.fullmatch(name):
        raise RemoteRegistryError(
            "Remote name must start with a letter or digit and contain only letters, digits, '.', '_' or '-'"
        )


def _validate_target(target: str) -> None:
    # Keep the SSH destination to a host or user@host/config alias. In
    # particular, reject leading dashes and whitespace so it cannot become an
    # ssh option or a remote shell fragment.
    if not isinstance(target, str) or not _TARGET_RE.fullmatch(target) or target.startswith("-"):
        raise RemoteRegistryError(
            "SSH target must be a host, SSH config alias, or user@host without whitespace"
        )


def _validate_port(value: int, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 65535:
        raise RemoteRegistryError(f"{label} must be an integer between 1 and 65535")


def _remote_dir() -> Path:
    return state_dir() / "remotes"


def _remote_path(name: str) -> Path:
    _validate_name(name)
    return _remote_dir() / f"{name}.json"


def _decode_remote(path: Path, value: Any) -> Remote:
    if not isinstance(value, dict):
        raise RemoteRegistryError(f"Remote record {path} must contain a JSON object")
    try:
        remote = Remote(
            name=value["name"],
            ssh_target=value["ssh_target"],
            server_port=value.get("server_port", _DEFAULT_SERVER_PORT),
            local_port=value.get("local_port", _DEFAULT_LOCAL_PORT),
            schema_version=value.get("schema_version", 1),
        )
    except (KeyError, TypeError) as exc:
        raise RemoteRegistryError(f"Remote record {path} is missing required fields") from exc
    _validate_name(remote.name)
    _validate_target(remote.ssh_target)
    _validate_port(remote.server_port, "Remote server port")
    _validate_port(remote.local_port, "Local tunnel port")
    if remote.name != path.stem:
        raise RemoteRegistryError(f"Remote record name does not match filename: {path}")
    if remote.schema_version != 1:
        raise RemoteRegistryError(f"Unsupported remote record version in {path}")
    return remote


def register_remote(
    name: str,
    ssh_target: str,
    *,
    server_port: int = _DEFAULT_SERVER_PORT,
    local_port: int = _DEFAULT_LOCAL_PORT,
    replace: bool = False,
) -> Remote:
    """Create or replace a named remote definition atomically."""
    _validate_name(name)
    _validate_target(ssh_target)
    _validate_port(server_port, "Remote server port")
    _validate_port(local_port, "Local tunnel port")
    remote = Remote(name, ssh_target, server_port, local_port)
    path = _remote_path(name)
    with state_lock():
        if path.exists() and not replace:
            raise RemoteRegistryError(f"Remote {name!r} is already registered")
        if path.exists() and replace:
            references = _referencing_records(name)
            if references:
                names = ", ".join(str(item) for item in references)
                raise RemoteRegistryError(
                    f"Cannot replace remote {name!r}; tracked records still reference it: {names}"
                )
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json_atomic(path, asdict(remote))
    return remote


def get_remote(name: str) -> Remote:
    """Read a registered remote definition."""
    path = _remote_path(name)
    with state_lock():
        value = read_json(path)
    if value is None:
        raise RemoteRegistryError(f"Remote {name!r} is not registered")
    return _decode_remote(path, value)


def list_remotes() -> list[Remote]:
    """Return registered remotes sorted by name."""
    directory = _remote_dir()
    with state_lock():
        paths = sorted(directory.glob("*.json")) if directory.is_dir() else []
        records = [(path, read_json(path)) for path in paths]
    return [_decode_remote(path, value) for path, value in records]


def _referencing_records(name: str) -> list[Path]:
    """Find client-side connection/server records that still reference a remote."""
    root = state_dir()
    matches: list[Path] = []
    for directory_name in ("connections", "servers"):
        directory = root / directory_name
        if not directory.is_dir():
            continue
        for path in directory.glob("*.json"):
            record = read_json(path)
            if isinstance(record, dict) and record.get("remote_name") == name:
                matches.append(path)
    return matches


def unregister_remote(name: str) -> None:
    """Remove a remote definition unless tracked records still refer to it."""
    path = _remote_path(name)
    with state_lock():
        if not path.exists():
            raise RemoteRegistryError(f"Remote {name!r} is not registered")
        references = _referencing_records(name)
        if references:
            names = ", ".join(str(item) for item in references)
            raise RemoteRegistryError(
                f"Cannot unregister remote {name!r}; tracked records still reference it: {names}"
            )
        path.unlink()


def local_model_inventory(model_root: Optional[Path] = None) -> dict[str, Any]:
    """Return machine-readable installed model data for local or SSH queries.

    The result is stable JSON data rather than the human table printed by the
    CLI. It intentionally reports validation and file availability separately.
    """
    root = Path(model_root).expanduser() if model_root else model_store.DEFAULT_MODEL_ROOT
    models: list[dict[str, Any]] = []
    if root.is_dir():
        for directory in sorted(root.iterdir(), key=lambda item: item.name):
            if not directory.is_dir() or directory.name.startswith("."):
                continue
            item: dict[str, Any] = {"id": directory.name}
            try:
                manifest = model_store.read_manifest(directory.name, root)
                files = manifest.get("files", {})
                broken_aliases = [
                    key
                    for key, entry in files.items()
                    if manifest.get("install_mode") == "alias"
                    and (
                        not (directory / entry.get("path", "")).is_symlink()
                        or not (directory / entry.get("path", "")).is_file()
                    )
                ]
                if manifest.get("install_mode") == "alias" and broken_aliases:
                    availability = "broken alias (" + ", ".join(broken_aliases) + ")"
                elif (directory / "model.gguf").is_file() and (directory / "tokenizer.json").is_file():
                    availability = "ready"
                else:
                    availability = "incomplete"
                item.update(
                    {
                        "pipeline": manifest.get("pipeline"),
                        "validated": bool(manifest.get("pipeline")),
                        "install_mode": manifest.get("install_mode", "unknown"),
                        "availability": availability,
                    }
                )
            except (model_store.ModelImportError, AttributeError, TypeError) as exc:
                item.update(
                    {
                        "pipeline": None,
                        "validated": False,
                        "install_mode": "unknown",
                        "availability": "invalid",
                        "error": str(exc),
                    }
                )
            models.append(item)
    return {"schema_version": 1, "models": models}


def inventory_json(model_root: Optional[Path] = None) -> str:
    """Serialize :func:`local_model_inventory` as one JSON line."""
    return json.dumps(local_model_inventory(model_root), sort_keys=True, separators=(",", ":"))


def query_remote_models(name: str, *, timeout: float = 30.0) -> dict[str, Any]:
    """Fetch and validate the remote host's machine-readable model inventory."""
    if timeout <= 0:
        raise RemoteRegistryError("SSH timeout must be greater than zero")
    remote = get_remote(name)
    command = [
        "ssh",
        "-T",
        "-o",
        "BatchMode=yes",
        "-o",
        f"ConnectTimeout={max(1, int(timeout))}",
        remote.ssh_target,
        "loom-serve-setup models list --json",
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError as exc:
        raise RemoteRegistryError("ssh executable was not found") from exc
    except subprocess.TimeoutExpired as exc:
        raise RemoteRegistryError(f"Timed out querying remote {name!r} after {timeout:g}s") from exc
    except subprocess.CalledProcessError as exc:
        details = (exc.stderr or exc.stdout or str(exc)).strip()
        raise RemoteRegistryError(f"Could not query remote {name!r}: {details}") from exc
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RemoteRegistryError(
            f"Remote {name!r} returned invalid model inventory JSON: {result.stdout[:300]!r}"
        ) from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise RemoteRegistryError(f"Remote {name!r} returned an unsupported inventory format")
    if not isinstance(payload.get("models"), list):
        raise RemoteRegistryError(f"Remote {name!r} returned malformed inventory: 'models' must be an array")
    for model in payload["models"]:
        if not isinstance(model, dict) or not isinstance(model.get("id"), str):
            raise RemoteRegistryError(f"Remote {name!r} returned a model entry without a string ID")
    return payload


def _remote_cli(name: str, arguments: list[str], *, timeout: float) -> subprocess.CompletedProcess[str]:
    """Run an HRX Solutions CLI command on a registered host and capture output.

    SSH passes the final argument through the remote user's shell. Quote every
    CLI token so model IDs and other future values cannot become shell syntax.
    Host-key verification is left at the user's OpenSSH defaults.
    """
    if timeout <= 0:
        raise RemoteRegistryError("SSH timeout must be greater than zero")
    remote = get_remote(name)
    remote_command = " ".join(shlex.quote(value) for value in ["loom-serve-setup", *arguments])
    command = [
        "ssh",
        "-T",
        "-o",
        "BatchMode=yes",
        "-o",
        f"ConnectTimeout={max(1, int(timeout))}",
        remote.ssh_target,
        remote_command,
    ]
    try:
        return subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
    except FileNotFoundError as exc:
        raise RemoteRegistryError("ssh executable was not found") from exc
    except subprocess.TimeoutExpired as exc:
        raise RemoteRegistryError(f"Timed out running a command on remote {name!r} after {timeout:g}s") from exc


def _decode_remote_server_result(
    name: str, result: subprocess.CompletedProcess[str], *, allow_nonzero: bool = False
) -> tuple[Any, int]:
    if result.returncode and not allow_nonzero:
        details = (result.stderr or result.stdout or f"ssh exited with status {result.returncode}").strip()
        raise RemoteRegistryError(f"Remote command on {name!r} failed: {details}")
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        details = result.stdout[:300]
        remote_error = (result.stderr or "").strip()
        raise RemoteRegistryError(
            f"Remote {name!r} returned invalid server JSON: {details!r}"
            + (f"; stderr: {remote_error}" if remote_error else "")
        ) from exc
    return payload, result.returncode


def _validate_server_record(name: str, value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or not isinstance(value.get("server_id"), str):
        raise RemoteRegistryError(f"Remote {name!r} returned a malformed server record")
    if not _SERVER_ID_RE.fullmatch(value["server_id"]):
        raise RemoteRegistryError(f"Remote {name!r} returned an invalid server ID")
    return value


def start_remote_server(name: str, model_id: str, *, timeout: float = 240.0) -> dict[str, Any]:
    """Start a validated installed model through the remote managed lifecycle."""
    if not isinstance(model_id, str) or not _MODEL_ID_RE.fullmatch(model_id):
        raise RemoteRegistryError(
            "Model ID must start with a letter or digit and contain only letters, digits, '.', '_' or '-'."
        )
    remote = get_remote(name)
    result = _remote_cli(
        name,
        ["serve", model_id, "--port", str(remote.server_port), "--json"],
        timeout=timeout,
    )
    payload, _ = _decode_remote_server_result(name, result)
    record = _validate_server_record(name, payload)
    if record.get("model_id") != model_id:
        raise RemoteRegistryError(
            f"Remote {name!r} returned server {record['server_id']} for unexpected model "
            f"{record.get('model_id')!r}"
        )
    return record


def list_remote_servers(name: str, *, timeout: float = 30.0) -> list[dict[str, Any]]:
    result = _remote_cli(name, ["servers", "list", "--json"], timeout=timeout)
    payload, _ = _decode_remote_server_result(name, result)
    if not isinstance(payload, list):
        raise RemoteRegistryError(f"Remote {name!r} returned malformed server list JSON")
    return [_validate_server_record(name, item) for item in payload]


def remote_server_action(
    name: str,
    server_id: str,
    action: str,
    *,
    timeout: float = 30.0,
) -> tuple[dict[str, Any], int]:
    """Query or stop one server by validated managed ID only."""
    if action not in {"status", "stop"}:
        raise RemoteRegistryError(f"Unsupported remote server action: {action}")
    if not isinstance(server_id, str) or not _SERVER_ID_RE.fullmatch(server_id):
        raise RemoteRegistryError("Server ID must contain only letters, digits, '-' or '_'")
    result = _remote_cli(
        name,
        ["servers", action, server_id, "--json"],
        timeout=timeout,
    )
    payload, returncode = _decode_remote_server_result(
        name, result, allow_nonzero=(action == "stop")
    )
    record = _validate_server_record(name, payload)
    if record["server_id"] != server_id:
        raise RemoteRegistryError(f"Remote {name!r} returned a different server ID than requested")
    return record, returncode
