"""Dedicated OpenCode drop-in ownership and recoverable file replacement.

Only current installation bytes live here. No model roster, client registry or
history is stored. Readers refuse the pending marker until the full generation
has committed. Shared projection publication happens outside this target lock.
"""
from __future__ import annotations

import base64
import hashlib
from importlib.resources import files
import os
from pathlib import Path
import uuid
import re
import shlex
import shutil
import stat
import subprocess

from . import hem_display
from .hem_client_documents import MAX_BYTES, decode_document, canonical_uuid as _uuid
from .hem_clients_endpoints import ensure_endpoints, project_endpoints, read_endpoints
from .hem_pi_state import passive_snapshot

from .hem_common import HemError, InvalidInput, UncertainOutcome, DeadlineExpired, result
from .hem_pi_files import (atomic_replace, encoded, private_path, read_owned,
                           remove_owned, staged_path, sync_dir, validate_stage)
from .hem_state import Registry, ResourceLock

ASSETS = ("index.ts", "discovery.ts", "native_protocol.ts", "installation.ts", "profile.json", "package.json")
ARTIFACTS = (*ASSETS, "config.json")
FILES = (*ARTIFACTS, "manifest.json")
PLUGIN = "plugins/loom-control-opencode"
REPAIR = "inspect/export the retained files before repair; rerun loom-control opencode configure --check"


VERSION = "2.0.31"
# Genuine qualified standalone/native npm binary, linux-x64, no Node requirement.
QUALIFIED_SHA256 = "2df936a20cf74af5706020342164628f9a31978c50c2b0ad306b42b614fe245d"


def resolve_target(override=None):
    value = override if override is not None else os.environ.get("OPENCODE_CONFIG_DIR")
    if value is None:
        xdg = Path(os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config")))
        value = xdg / "opencode"
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise InvalidInput("OpenCode configuration directory must be absolute")
    # Reject aliases before canonicalization; resolving first would hide symlinks.
    private_path(path)
    return path.resolve()


def inspect_runtime(deadline, *, passive=False):
    selected = shutil.which("opencode")
    if selected is None:
        raise InvalidInput("OpenCode is absent from PATH; select qualified OpenCode 2.0.31 (no automatic install/upgrade)")
    executable = Path(selected).resolve()
    info = executable.stat()
    if not stat.S_ISREG(info.st_mode): raise InvalidInput("selected opencode is not a regular executable")
    if passive:
        # No process, npm, plugins, daemon or config is executed by --check.
        hasher = hashlib.sha256()
        if info.st_size > 512 * 1024 * 1024:
            raise InvalidInput("selected OpenCode binary is too large for passive identity inspection")
        with executable.open("rb") as stream:
            while True:
                deadline.remaining()
                chunk = stream.read(1024 * 1024)
                if not chunk: break
                hasher.update(chunk)
        if hasher.hexdigest() != QUALIFIED_SHA256:
            raise InvalidInput("selected OpenCode binary is not the passively qualified linux-x64 2.0.31 identity; version was not executed")
    else:
        try:
            observed = subprocess.run([str(executable), "--version"], stdin=subprocess.DEVNULL,
                capture_output=True, text=True, timeout=deadline.remaining(), check=False)
        except subprocess.TimeoutExpired as error:
            raise HemError("OpenCode version check timed out before configuration writes", accepted=False) from error
        except OSError as error:
            raise InvalidInput("cannot inspect selected OpenCode executable") from error
        if observed.returncode or re.fullmatch(r"opencode v2\.0\.31\s*", observed.stdout) is None:
            raise InvalidInput("unsupported OpenCode executable; require OpenCode 2.0.31; existing client unchanged")
    return dict(executable=str(executable), package="@opencode/cli", version=VERSION,
                qualification="passive qualified binary identity" if passive else "selected executable version")


class OpenCodeResult(dict):
    def __init__(self, *args, exit_code=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.exit_code = exit_code


def bundled_assets():
    resource = files(__package__).joinpath("opencode_plugin")
    return {name: (files(__package__).joinpath(name) if name == "native_protocol.ts" else resource.joinpath(name)).read_bytes()
            for name in ASSETS}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def document(data):
    try:
        return decode_document(data)
    except (ValueError, TypeError, UnicodeError, RecursionError) as error:
        raise InvalidInput("malformed OpenCode ownership data; " + REPAIR) from error


def validate_set(values, target, root):
    """Prove one complete owned generation, including root and exact whitelist."""
    if not values:
        return None
    # The sole prior format predates the shared companion; no partial set is owned.
    artifacts = ARTIFACTS if "native_protocol.ts" in values else tuple(name for name in ARTIFACTS if name != "native_protocol.ts")
    if set(values) != {*artifacts, "manifest.json"} or any(not isinstance(data, bytes) or len(data) > MAX_BYTES for data in values.values()):
        raise InvalidInput("unsupported OpenCode owned file set; " + REPAIR)
    manifest, config = document(values["manifest.json"]), document(values["config.json"])
    if (set(manifest) != {"schema_version", "integration_version", "enabled", "generation", "integration_id", "target", "artifacts"} or
        type(manifest["schema_version"]) is not int or manifest["schema_version"] != 1 or
        type(manifest["integration_version"]) is not int or manifest["integration_version"] != 1 or
        manifest["enabled"] is not True or not _uuid(manifest["generation"]) or not _uuid(manifest["integration_id"]) or
        manifest["target"] != str(target) or not isinstance(manifest["artifacts"], dict) or
        set(manifest["artifacts"]) != set(artifacts) or
        any(manifest["artifacts"][name] != digest(values[name]) for name in artifacts)):
        raise InvalidInput("foreign, edited or unsupported OpenCode manifest; " + REPAIR)
    expected = dict(schema_version=1, generation=manifest["generation"], integration_id=manifest["integration_id"],
                    target=str(target), hem_root=str(root), endpoint_file=str(root / "clients/endpoints.json"))
    if config != expected or type(config.get("schema_version")) is not int:
        raise InvalidInput("OpenCode installation belongs to another root/target or generation; " + REPAIR)
    return manifest


def _read(path):
    data = read_owned(path)
    if data is not None and len(data) > MAX_BYTES:
        raise InvalidInput("oversized OpenCode owned file; " + REPAIR)
    return data


def _inspect(target):
    directory = target / PLUGIN
    private_path(directory)
    allowed = set(FILES) | {"pending.json"}
    # Fixed recovery temporaries are inspected even before a pending commit.
    allowed |= {staged_path(Path(name)).name for name in (*FILES, "pending.json")}
    if directory.exists():
        unknown = [path.name for path in directory.iterdir() if path.name not in allowed]
        if unknown:
            raise InvalidInput("unknown OpenCode plugin content retained: " + ", ".join(sorted(unknown)))
    values = {name: data for name in FILES if (data := _read(directory / name)) is not None}
    pending = _read(directory / "pending.json")
    stages = {name: _read(staged_path(directory / name)) for name in (*FILES, "pending.json")}
    return directory, values, pending, {name: data for name, data in stages.items() if data is not None}


def _plan(previous, desired, target, root):
    old = validate_set(previous, target, root)
    new = validate_set(desired, target, root)
    if not old and not new:
        raise InvalidInput("empty OpenCode recovery plan")
    if old and new and old["integration_id"] != new["integration_id"]:
        raise InvalidInput("OpenCode recovery identity changed")
    return encoded(dict(schema_version=1, target=str(target), hem_root=str(root),
        previous={name: base64.b64encode(data).decode("ascii") for name, data in previous.items()},
        desired={name: base64.b64encode(data).decode("ascii") for name, data in desired.items()}))


def _decode_plan(data, target, root):
    value = document(data)
    if (set(value) != {"schema_version", "target", "hem_root", "previous", "desired"} or
        type(value["schema_version"]) is not int or value["schema_version"] != 1 or
        value["target"] != str(target) or value["hem_root"] != str(root)):
        raise InvalidInput("foreign OpenCode pending plan; " + REPAIR)
    try:
        sets = [{name: base64.b64decode(item, validate=True) for name, item in value[key].items()}
                for key in ("previous", "desired")]
    except (ValueError, TypeError, AttributeError) as error:
        raise InvalidInput("malformed OpenCode pending bytes; " + REPAIR) from error
    # Re-encoding provides an exact bounded whitelist/identity/hash validation.
    if _plan(*sets, target, root) != data:
        raise InvalidInput("noncanonical OpenCode pending plan; " + REPAIR)
    return sets


def _validate_recovery(directory, pending, target, root):
    """Passively prove the complete plan, current bytes and staged prefixes."""
    previous, desired = _decode_plan(pending, target, root)
    current = {name: _read(directory / name) for name in FILES}
    # Validate the ENTIRE recovery set before changing any file.
    for name in FILES:
        if current[name] not in (previous.get(name), desired.get(name)):
            raise InvalidInput("edited OpenCode recovery file retained: " + name + "; " + REPAIR)
        if name in desired:
            validate_stage(directory / name, desired[name], allow_prefix=True)
        elif _read(staged_path(directory / name)) is not None:
            raise InvalidInput("unexpected OpenCode removal temporary retained; " + REPAIR)
    if _read(staged_path(directory / "pending.json")) is not None:
        validate_stage(directory / "pending.json", pending)
    return previous, desired, current


def _resume(directory, pending, target, root, deadline):
    _, desired, current = _validate_recovery(directory, pending, target, root)
    if _read(staged_path(directory / "pending.json")) is not None:
        remove_owned(staged_path(directory / "pending.json"), pending, deadline)
    # Entry point last when installing; pending keeps all readers fail-closed.
    order = [name for name in FILES if name != "index.ts"] + ["index.ts"]
    for name in order:
        path = directory / name
        if name in desired:
            if current[name] != desired[name] or _read(staged_path(path)) is not None:
                atomic_replace(path, desired[name], expected=current[name], deadline=deadline, allow_prefix=True)
        else:
            remove_owned(path, current[name], deadline)
    remove_owned(directory / "pending.json", pending, deadline)


def _desired(target, root, assets, previous):
    old = validate_set(previous, target, root)
    if old and all(previous.get(name) == data for name, data in assets.items()):
        return previous
    identity = old["integration_id"] if old else str(uuid.uuid4())
    generation = str(uuid.uuid4())
    config = encoded(dict(schema_version=1, generation=generation, integration_id=identity,
        target=str(target), hem_root=str(root), endpoint_file=str(root / "clients/endpoints.json")))
    values = dict(assets, **{"config.json": config})
    values["manifest.json"] = encoded(dict(schema_version=1, integration_version=1, enabled=True,
        generation=generation, integration_id=identity, target=str(target),
        artifacts={name: digest(data) for name, data in values.items()}))
    validate_set(values, target, root)
    return values


def _check_owned(root, target, assets):
    """Inspect one complete generation without recovering or publishing."""
    try:
        directory, values, pending, stages = _inspect(target)
        manifest = None
        if pending is not None or stages:
            state = "pending"
        else:
            manifest = validate_set(values, target, root)
            state = "configured" if manifest else "absent"
        configured = state == "configured" and all(values.get(name) == data for name, data in assets.items())
        if state == "configured" and not configured: state = "outdated"
        return OpenCodeResult(result(state, completed=True, configured=configured, target=str(target),
            pending=bool(pending is not None or stages), findings=[] if configured else [state],
            integration_id=manifest["integration_id"] if manifest else None), exit_code=0 if configured else 1)
    except InvalidInput as error:
        return OpenCodeResult(result("conflict", completed=True, configured=False, target=str(target),
            findings=[str(error)], next_action=REPAIR), exit_code=2)


def _preflight(root, target):
    """Refuse known conflicts before root/endpoint publication."""
    directory, initial, pending, stages = _inspect(target)
    if pending is None and "pending.json" in stages:
        pending = stages["pending.json"]
    elif pending is None and stages:
        raise InvalidInput("orphan OpenCode staging files retained; " + REPAIR)
    if pending is not None:
        _validate_recovery(directory, pending, target, root)
    else:
        validate_set(initial, target, root)


def _apply_owned(root, target, deadline, assets, *, disable=False):
    """Reinspect under the target lock before any generation changes."""
    # A target-local lock serializes different Loom roots targeting the same drop-in.
    if disable and not target.exists():
        return OpenCodeResult(result("disabled", completed=True, changed=False, configured=False, target=str(target), affected_paths=[]))
    private_path(target, create=True)
    with ResourceLock(target, "target", "unused", deadline, path=target / ".loom-control-opencode.lock"):
        directory, values, pending, stages = _inspect(target)
        if pending is None and "pending.json" in stages:
            # A fully flushed plan interrupted before rename is recoverable. An
            # incomplete/foreign prefix is retained rather than guessed/deleted.
            pending = stages["pending.json"]
            _validate_recovery(directory, pending, target, root)
            atomic_replace(directory / "pending.json", pending, expected=None, deadline=deadline)
        elif pending is None and stages:
            raise InvalidInput("orphan OpenCode staging files retained; " + REPAIR)
        if pending is not None:
            _resume(directory, pending, target, root, deadline)
            _, values, _, _ = _inspect(target)
        validate_set(values, target, root)
        desired = {} if disable else _desired(target, root, assets, values)
        changed = values != desired
        if changed:
            plan = _plan(values, desired, target, root)
            if len(plan) > MAX_BYTES: raise InvalidInput("OpenCode recovery plan exceeds 1 MiB")
            atomic_replace(directory / "pending.json", plan, expected=None, deadline=deadline)
            _resume(directory, plan, target, root, deadline)
        if disable and directory.exists():
            # Never recursively delete; unknown content is an earlier refusal.
            directory.rmdir()
            sync_dir(directory.parent)
        return OpenCodeResult(result("disabled" if disable else "configured", completed=True,
            changed=changed, configured=not disable, target=str(target),
            affected_paths=[str(directory / name) for name in FILES] if changed else []))


def configure_opencode(root, target, deadline, *, check=False, disable=False):
    """Own complete qualified OpenCode setup for CLI and direct callers."""
    if check and disable:
        raise InvalidInput("OpenCode check and disable are mutually exclusive")
    root_value = Path(root if root is not None else os.environ.get("LOOM_SERVE_DIR", str(Path.home() / ".loom")))
    if not root_value.is_absolute(): raise InvalidInput("LOOM_SERVE_DIR/state root must be absolute")
    private_path(root_value)
    root, target = root_value.resolve(), resolve_target(target)
    deadline.remaining()
    runtime = None
    compatibility = dict(state="not_checked", reason="client unnecessary for owned-file removal")
    if not disable:
        try:
            runtime = inspect_runtime(deadline, passive=check)
            compatibility = dict(state="compatible", reason=runtime["qualification"])
        except (HemError, OSError) as error:
            if not check: raise
            compatibility = dict(state="unqualified", reason=str(error))
    assets = bundled_assets()  # Missing wheel assets fail before state/target writes.
    projection = None
    if not disable:
        snapshot = passive_snapshot(root, deadline)
        projection = project_endpoints(root, snapshot["servers"])
        if len(projection["servers"]) > 256:
            raise InvalidInput("OpenCode supports at most 256 candidate endpoints")
    hem_display.emit("No native Loom equivalent: owned OpenCode integration file configuration")
    try:
        if check:
            response = _check_owned(root, target, assets)
        else:
            _preflight(root, target)
            if not disable:
                ensure_endpoints(Registry(root, deadline=deadline), deadline)
            response = _apply_owned(root, target, deadline, assets, disable=disable)
    except (OSError, DeadlineExpired) as error:
        if check: raise HemError("OpenCode inspection failed without writes", accepted=False) from error
        raise UncertainOutcome("OpenCode update interrupted; " + REPAIR) from error
    response.update(runtime=runtime, runtime_compatibility=compatibility,
                    availability=dict(state="not_observed", reason="models are discovered by the client on startup or explicit reload"))
    if check:
        try:
            current = read_endpoints(root)
            valid = current == projection and current is not None
            response["endpoint_projection"] = "current" if valid else "absent_or_outdated"
            if not valid:
                response.setdefault("findings", []).append("root endpoint projection absent or outdated; rerun configure")
                response.exit_code = response.exit_code or 1
        except (HemError, OSError) as error:
            response["endpoint_projection"] = "conflict"
            response.setdefault("findings", []).append(str(error))
            response.exit_code = response.exit_code or 2
        if compatibility["state"] != "compatible":
            response.exit_code = response.exit_code or 1
    if not disable:
        recipe = "OPENCODE_CONFIG_DIR=" + shlex.quote(str(target)) + " opencode"
        response.update(launch=recipe, refresh=recipe + " reload --server http://127.0.0.1:PORT",
            guidance="Startup or explicit reload of the correct OpenCode server refreshes models; restart is the fallback. This directory does not retarget an already-running server. Credentials, settings/defaults and other plugins are unchanged. Effective host policy is unassessed; inspect actual provider visibility after launch. Export/backups are your responsibility.")
        hem_display.emit("Launch: " + recipe)
        hem_display.emit("Refresh the correct OpenCode server using its public reload command, or restart. No background polling.")
    return response
