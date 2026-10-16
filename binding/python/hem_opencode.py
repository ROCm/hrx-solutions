"""Dedicated OpenCode drop-in ownership and recoverable file replacement.

Only current installation bytes live here. No model roster, client registry or
history is stored. Readers refuse the pending marker until the full generation
has committed. Shared projection publication happens outside this target lock.
"""
from __future__ import annotations

import base64
import hashlib
from importlib.resources import files
import json
import os
from pathlib import Path
import uuid

from .hem_common import InvalidInput, result
from .hem_pi_files import (atomic_replace, encoded, private_path, read_owned,
                           remove_owned, staged_path, sync_dir, validate_stage)
from .hem_state import ResourceLock

ASSETS = ("index.ts", "discovery.ts", "installation.ts", "profile.json", "package.json")
ARTIFACTS = (*ASSETS, "config.json")
FILES = (*ARTIFACTS, "manifest.json")
PLUGIN = "plugins/loom-control-opencode"
MAX_BYTES = 1024 * 1024
REPAIR = "inspect/export the retained files before repair; rerun loom-control opencode configure --check"


class OpenCodeResult(dict):
    def __init__(self, *args, exit_code=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.exit_code = exit_code


def bundled_assets():
    resource = files(__package__).joinpath("opencode_plugin")
    return {name: resource.joinpath(name).read_bytes() for name in ASSETS}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def _unique(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate field")
        value[key] = item
    return value


def document(data):
    try:
        if len(data) > MAX_BYTES:
            raise ValueError()
        value = json.loads(data, object_pairs_hook=_unique)
        def depth(item, level=0):
            if level > 64:
                raise ValueError()
            if isinstance(item, dict):
                for child in item.values(): depth(child, level + 1)
            elif isinstance(item, list):
                for child in item: depth(child, level + 1)
        depth(value)
        if not isinstance(value, dict): raise ValueError()
        return value
    except (ValueError, TypeError, UnicodeError, RecursionError) as error:
        raise InvalidInput("malformed OpenCode ownership data; " + REPAIR) from error


def _uuid(value):
    try:
        identifier = uuid.UUID(value)
        return identifier.version == 4 and str(identifier) == value
    except (ValueError, TypeError, AttributeError):
        return False


def _canonical(path):
    path = Path(path)
    if not path.is_absolute() or str(path.resolve()) != str(path):
        raise InvalidInput("OpenCode target/state root must be canonical and absolute")
    private_path(path)
    return path


def validate_set(values, target, root):
    """Prove one complete owned generation, including root and exact whitelist."""
    if not values:
        return None
    if set(values) != set(FILES) or any(not isinstance(data, bytes) or len(data) > MAX_BYTES for data in values.values()):
        raise InvalidInput("unsupported OpenCode owned file set; " + REPAIR)
    manifest, config = document(values["manifest.json"]), document(values["config.json"])
    if (set(manifest) != {"schema_version", "integration_version", "enabled", "generation", "integration_id", "target", "artifacts"} or
        type(manifest["schema_version"]) is not int or manifest["schema_version"] != 1 or
        type(manifest["integration_version"]) is not int or manifest["integration_version"] != 1 or
        manifest["enabled"] is not True or not _uuid(manifest["generation"]) or not _uuid(manifest["integration_id"]) or
        manifest["target"] != str(target) or not isinstance(manifest["artifacts"], dict) or
        set(manifest["artifacts"]) != set(ARTIFACTS) or
        any(manifest["artifacts"][name] != digest(values[name]) for name in ARTIFACTS)):
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


def _resume(directory, pending, target, root, deadline):
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
    if old and all(previous[name] == data for name, data in assets.items()):
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


def configure_opencode(root, target, deadline, *, check=False, disable=False):
    root, target = _canonical(root), _canonical(target)
    assets = bundled_assets()
    if check:
        try:
            directory, values, pending, stages = _inspect(target)
            manifest = None
            if pending is not None or stages:
                state = "pending"
            else:
                manifest = validate_set(values, target, root)
                state = "configured" if manifest else "absent"
            configured = state == "configured" and all(values[name] == data for name, data in assets.items())
            if state == "configured" and not configured: state = "outdated"
            return OpenCodeResult(result(state, completed=True, configured=configured, target=str(target),
                pending=bool(pending is not None or stages), findings=[] if configured else [state],
                integration_id=manifest["integration_id"] if manifest else None), exit_code=0 if configured else 1)
        except InvalidInput as error:
            return OpenCodeResult(result("conflict", completed=True, configured=False, target=str(target),
                findings=[str(error)], next_action=REPAIR), exit_code=2)
    # Refuse known conflicts before creating even a target lock file. Recheck
    # after locking to catch a concurrent writer.
    _, initial, pending, stages = _inspect(target)
    if pending is None and not stages:
        validate_set(initial, target, root)
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
            _decode_plan(pending, target, root)
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
