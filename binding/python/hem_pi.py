"""Owned Pi install/recovery. Public CLI checks Pi/Node compatibility first.

No Pi process, settings file, package installer, HTTP or lifecycle operation is
invoked here. One root reservation plus one bounded pending plan bridge SQLite
and atomic files; a reservation alone never means configured.
"""
from __future__ import annotations

import base64
import hashlib
import os
from pathlib import Path
import stat
import uuid

from .hem_common import InvalidInput, UncertainOutcome
from .hem_pi_assets import bundled_assets, asset_digests
from .hem_pi_contracts import (PiResult, MANIFEST_PATH, ENDPOINT_PATH,
    MANIFEST_SCHEMA_VERSION, INTEGRATION_VERSION, EXTENSION_VERSION,
    NATIVE_PROFILE, NATIVE_COMMIT, PI_PACKAGE, PI_VERSIONS)
from .hem_pi_state import (passive_snapshot, integration_lock, target_lock,
    get_target, set_target, clear_target)
from .hem_state import Registry
from . import hem_pi_files as files


def resolve_target(target=None):
    chosen = target if target is not None else os.environ.get("PI_CODING_AGENT_DIR", str(Path.home() / ".pi/agent"))
    if not isinstance(chosen, (str, Path)) or not str(chosen):
        raise InvalidInput("Pi directory must be a nonempty path")
    return Path(chosen).expanduser().resolve()


def _manifest(root, identity, assets, *, enabled=True):
    return dict(schema_version=MANIFEST_SCHEMA_VERSION, integration_version=INTEGRATION_VERSION,
        integration_id=identity["integration_id"], enabled=enabled, hem_root=str(root),
        target=identity["target"], extension_version=EXTENSION_VERSION,
        native_profile=NATIVE_PROFILE, native_commit=NATIVE_COMMIT, pi_package=PI_PACKAGE,
        pi_versions=list(PI_VERSIONS), endpoint_file=str(root / ENDPOINT_PATH), artifacts=asset_digests(assets))


def _validate_assets(assets):
    # Refuse an incomplete or incompatible packaged profile before state writes.
    from .hem_pi_assets import ASSET_NAMES
    from .hem_pi_contracts import EXTENSION_PATH, NODE_MIN_VERSION, OUTPUT_CAP
    if set(assets) != {EXTENSION_PATH + "/" + name for name in ASSET_NAMES}:
        raise InvalidInput("unsupported bundled Pi resource set")
    if any(not isinstance(data, bytes) or len(data) > files.MAX_BYTES for data in assets.values()):
        raise InvalidInput("invalid bundled Pi resources")
    profile = files.decode(assets[EXTENSION_PATH + "/profile.json"], "bundled Pi profile")
    expected = dict(integration_version=INTEGRATION_VERSION, manifest_schema_version=MANIFEST_SCHEMA_VERSION,
        endpoint_schema_version=1, extension_version=EXTENSION_VERSION, pi_package=PI_PACKAGE,
        pi_versions=list(PI_VERSIONS), node_min_version=list(NODE_MIN_VERSION),
        native_profile=NATIVE_PROFILE, native_commit=NATIVE_COMMIT, output_cap=OUTPUT_CAP)
    if any(profile.get(key) != value for key, value in expected.items()):
        raise InvalidInput("incompatible bundled Pi profile")


def _identity_check(reserved, target):
    if reserved is not None and reserved["target"] != str(target):
        raise InvalidInput(f"Pi target conflicts with reserved {reserved['target']}; recover that target first")


def _endpoint(data, root, identity):
    if data is not None:
        value = files.decode(data, "Pi endpoint document")
        if (value.get("schema_version") != 1 or value.get("hem_root") != str(root) or
                value.get("integration_id") != identity["integration_id"] or not isinstance(value.get("servers"), list)):
            raise InvalidInput("foreign or incompatible Pi endpoint document")


def _path(root, target, name):
    return root / ENDPOINT_PATH if name == "@endpoint" else target / name


def _pack(data):
    return None if data is None else base64.b64encode(data).decode("ascii")


def _unpack(value):
    if value is None:
        return None
    try:
        data = base64.b64decode(value, validate=True)
        if len(data) > files.MAX_BYTES:
            raise ValueError()
        return data
    except (ValueError, TypeError) as error:
        raise InvalidInput("invalid bounded Pi recovery bytes") from error


def _mode_marker(path):
    files.private_path(path)
    info = path.lstat()
    if info.st_uid != os.getuid() or info.st_mode & 0o077 or not stat.S_ISDIR(info.st_mode):
        raise InvalidInput("unexpected Pi preparation marker retained")
    entries = list(path.iterdir())
    if len(entries) > 1:
        raise InvalidInput("unexpected Pi preparation fingerprint retained")
    if not entries:
        return None
    fingerprint = entries[0]
    if len(fingerprint.name) != 64 or any(c not in "0123456789abcdef" for c in fingerprint.name):
        raise InvalidInput("unexpected Pi preparation fingerprint retained")
    files.private_path(fingerprint)
    info = fingerprint.lstat()
    if info.st_uid != os.getuid() or info.st_mode & 0o077 or list(fingerprint.iterdir()):
        raise InvalidInput("unexpected Pi preparation fingerprint retained")
    return fingerprint.name


def _make_plan(root, target, reserved, assets, inspected, endpoint, servers, mode):
    manifest = inspected["manifest"]
    if mode == "disable":
        if manifest is None:
            raise InvalidInput("cannot reconstruct disable preparation without ownership manifest")
        disabled = dict(manifest, enabled=False)
        return dict(schema_version=1, mode=mode, identity=reserved, hem_root=str(root),
            old={MANIFEST_PATH:_pack(inspected["manifest_bytes"]), "@endpoint":_pack(endpoint)},
            next={MANIFEST_PATH:_pack(files.encoded(disabled)), "@endpoint":None}, digests=manifest["artifacts"])
    if inspected["drift"]:
        raise InvalidInput("edited owned Pi artifacts retained: " + ", ".join(inspected["drift"]))
    from .hem_pi_endpoints import project_endpoints
    projection = files.encoded(project_endpoints(root, reserved, servers))
    desired_manifest = files.encoded(_manifest(root, reserved, assets))
    old = {name:_pack(files.read_owned(target / name)) for name in assets}
    old.update({MANIFEST_PATH:_pack(inspected["manifest_bytes"]), "@endpoint":_pack(endpoint)})
    nxt = {name:_pack(data) for name,data in assets.items()}
    nxt.update({MANIFEST_PATH:_pack(desired_manifest), "@endpoint":_pack(projection)})
    return dict(schema_version=1, mode=mode, identity=reserved, hem_root=str(root), old=old, next=nxt, digests={})


def _load_plan(directory, root, identity, assets, *, filename="plan.json"):
    files.private_path(directory)
    if not directory.exists():
        return None
    entries = {p.name for p in directory.iterdir()}
    modes = entries & {"configure", "disable"}
    if len(modes) > 1 or entries != {filename} | modes:
        raise InvalidInput(f"unexpected Pi recovery content retained: {directory}")
    for mode in modes:
        _mode_marker(directory / mode)
    plan = files.decode(files.read_owned(directory / filename), "Pi recovery plan")
    if (set(plan) != {"schema_version", "mode", "identity", "hem_root", "old", "next", "digests"} or
            plan["schema_version"] != 1 or plan["identity"] != identity or plan["hem_root"] != str(root) or
            plan["mode"] not in ("configure", "disable")):
        raise InvalidInput("foreign or unsupported Pi recovery plan")
    if modes and modes != {plan["mode"]}:
        raise InvalidInput("Pi recovery intent conflicts with plan")
    for mode in modes:
        fingerprint = _mode_marker(directory / mode)
        if fingerprint is not None and fingerprint != hashlib.sha256(files.encoded(plan)).hexdigest():
            raise InvalidInput("Pi recovery fingerprint conflicts with plan")
    keys = {*assets, MANIFEST_PATH, "@endpoint"} if plan["mode"] == "configure" else {MANIFEST_PATH, "@endpoint"}
    if not isinstance(plan["old"], dict) or not isinstance(plan["next"], dict) or set(plan["old"]) != keys or set(plan["next"]) != keys:
        raise InvalidInput("unsupported Pi recovery paths")
    old = {key: _unpack(value) for key, value in plan["old"].items()}
    nxt = {key: _unpack(value) for key, value in plan["next"].items()}
    for values in (old, nxt):
        manifest = values[MANIFEST_PATH]
        if manifest is not None:
            manifest = files.validate_manifest(files.decode(manifest, "recovery manifest"), identity["target"], root, assets)
            if manifest["integration_id"] != identity["integration_id"]:
                raise InvalidInput("recovery manifest identity mismatch")
        _endpoint(values["@endpoint"], root, identity)
        if plan["mode"] == "configure":
            for name in assets:
                data = values[name]
                if manifest is None:
                    if data is not None:
                        raise InvalidInput("unowned recovery artifact")
                elif data is None or hashlib.sha256(data).hexdigest() != manifest["artifacts"][name]:
                    raise InvalidInput("recovery artifact digest mismatch")
    if nxt[MANIFEST_PATH] is None:
        raise InvalidInput("missing next recovery manifest")
    next_manifest = files.decode(nxt[MANIFEST_PATH], "next manifest")
    if next_manifest["enabled"] != (plan["mode"] == "configure"):
        raise InvalidInput("recovery mode mismatch")
    if plan["mode"] == "disable":
        if plan["digests"] != next_manifest["artifacts"] or nxt["@endpoint"] is not None:
            raise InvalidInput("invalid disable recovery digests")
    elif plan["digests"] != {}:
        raise InvalidInput("unexpected recovery digest set")
    return plan


def _cleanup_plan(directory, plan, deadline):
    # Compare content and exact fixed directory entries before deletion.
    entries = {p.name for p in directory.iterdir()}
    if entries not in ({"plan.json"}, {"plan.json",plan["mode"]}):
        raise InvalidInput("unexpected recovery cleanup content retained")
    marker = directory / plan["mode"]
    if marker.exists():
        fingerprint = _mode_marker(marker)
        if fingerprint is not None:
            (marker / fingerprint).rmdir()
            files.sync_dir(marker)
        marker.rmdir()
        files.sync_dir(directory)
    files.remove_owned(directory / "plan.json", files.encoded(plan), deadline)
    directory.rmdir()
    files.sync_dir(directory.parent)


def _prepare(root, target, identity, plan, assets, deadline):
    preparing, pending = target / files.PREPARING_PATH, target / files.PENDING_PATH
    files.private_path(preparing.parent, create=True)
    if preparing.exists() or pending.exists():
        raise InvalidInput("another Pi recovery operation is pending")
    preparing.mkdir(mode=0o700)
    files.sync_dir(preparing.parent)
    (preparing / plan["mode"]).mkdir(mode=0o700)
    files.sync_dir(preparing)
    fingerprint = hashlib.sha256(files.encoded(plan)).hexdigest()
    (preparing / plan["mode"] / fingerprint).mkdir(mode=0o700)
    files.sync_dir(preparing / plan["mode"])
    files.atomic_replace(preparing / "plan.json", files.encoded(plan), expected=None, deadline=deadline)
    _load_plan(preparing, root, identity, assets)
    deadline.remaining()
    os.rename(preparing, pending)
    files.sync_dir(pending.parent)


def _resume(root, target, identity, plan, assets, deadline):
    old = {k: _unpack(v) for k, v in plan["old"].items()}
    nxt = {k: _unpack(v) for k, v in plan["next"].items()}
    pending = target / files.PENDING_PATH
    if plan["mode"] == "configure":
        # Inspect every affected file before changing any; accept only exact old/next.
        current = {name: files.read_owned(_path(root, target, name)) for name in old}
        for name, data in current.items():
            if data != old[name] and data != nxt[name]:
                raise InvalidInput(f"edited recovery file retained: {_path(root, target, name)}")
            files.validate_stage(_path(root, target, name), nxt[name], allow_prefix=True)
        for name in ["@endpoint", *sorted(assets), MANIFEST_PATH]:
            if current[name] != nxt[name] or files.read_owned(files.staged_path(_path(root, target, name))) is not None:
                files.atomic_replace(_path(root, target, name), nxt[name], expected=current[name], deadline=deadline, allow_prefix=True)
        _cleanup_plan(pending, plan, deadline)
        return []
    # Disabled manifest first; after this cooperating publishers cannot write.
    current_manifest = files.read_owned(target / MANIFEST_PATH)
    if current_manifest not in (old[MANIFEST_PATH], nxt[MANIFEST_PATH], None):
        raise InvalidInput("edited disable manifest retained")
    manifest_stage = files.validate_stage(target / MANIFEST_PATH, nxt[MANIFEST_PATH], allow_prefix=True)
    if current_manifest == old[MANIFEST_PATH] or current_manifest is not None and manifest_stage is not None:
        files.atomic_replace(target / MANIFEST_PATH, nxt[MANIFEST_PATH], expected=current_manifest, deadline=deadline, allow_prefix=True)
    retained = []
    for relative, digest in plan["digests"].items():
        path = target / relative
        data = files.read_owned(path)
        if data is not None:
            if hashlib.sha256(data).hexdigest() != digest:
                retained.append(str(path))
            else:
                files.remove_owned(path, data, deadline)
    endpoint = files.read_owned(root / ENDPOINT_PATH)
    if endpoint != old["@endpoint"] and endpoint is not None:
        raise InvalidInput("changed endpoint retained during disable")
    files.remove_owned(root / ENDPOINT_PATH, old["@endpoint"], deadline)
    if retained:
        return retained
    files.remove_owned(target / MANIFEST_PATH, nxt[MANIFEST_PATH], deadline)
    _cleanup_plan(pending, plan, deadline)
    return []


def _result(target, *, state, configured=False, changed=False, completed=True, exit_code=0, **fields):
    return PiResult(state, configured=configured, changed=changed, completed=completed,
        exit_code=exit_code, target=str(target), integration_version=INTEGRATION_VERSION,
        native_profile=NATIVE_PROFILE, **fields)


def configure_pi(root, target, deadline, *, check=False, disable=False):
    """Configure/check/disable owned resources; CLI supplies qualified Pi runtime.

    check is strictly passive, including an absent state namespace. Mutation
    errors preserve reservation/recovery evidence rather than guessing rollback.
    """
    if check and disable:
        raise InvalidInput("Pi check and disable are mutually exclusive")
    root, target = Path(root).resolve(), resolve_target(target)
    assets = bundled_assets()
    _validate_assets(assets)
    snapshot = passive_snapshot(root, deadline)
    _identity_check(snapshot["pi_target"], target)
    inspected = files.inspect_target(target, root, assets=assets)
    if snapshot["pi_target"] and inspected["manifest"] and inspected["integration_id"] != snapshot["pi_target"]["integration_id"]:
        raise InvalidInput("Pi manifest and root reservation identities conflict")
    if check:
        configured = bool(snapshot["pi_target"] and inspected["configured"])
        changed = inspected["changed"] or not configured
        if snapshot["pi_target"]:
            endpoint = files.read_owned(root / ENDPOINT_PATH)
            _endpoint(endpoint, root, snapshot["pi_target"])
            from .hem_pi_endpoints import project_endpoints
            projection = files.encoded(project_endpoints(root, snapshot["pi_target"], snapshot["servers"]))
            changed = changed or endpoint != projection
            configured = configured and endpoint is not None
        return _result(target, state="changes_required" if changed else "configured", configured=configured,
            exit_code=int(changed), affected_paths=inspected["affected_paths"], available=[], servers=[],
            next_action="loom-hem pi configure" if changed else None)
    if disable and snapshot["pi_target"] is None:
        if inspected["manifest"] or inspected["pending"] or inspected["drift"]:
            raise InvalidInput("Pi files have no matching root reservation; retained")
        return _result(target, state="disabled")
    registry = Registry(root, deadline=deadline)
    with integration_lock(root, deadline):
        reserved = get_target(registry)
        _identity_check(reserved, target)
        with target_lock(target, deadline):
            inspected = files.inspect_target(target, root, assets=assets)
            manifest = inspected["manifest"]
            if manifest and (reserved is None or inspected["integration_id"] != reserved["integration_id"]):
                raise InvalidInput("Pi target has no matching ownership reservation")
            if reserved is None:
                if inspected["drift"] or inspected["pending"]:
                    raise InvalidInput("unowned Pi integration files retained")
                reserved = dict(target=str(target), integration_id=str(uuid.uuid4()))
                set_target(registry, reserved)  # Durable BEFORE any preparation.
            # Complete a previously staged operation under its original identity.
            preparing, pending = target / files.PREPARING_PATH, target / files.PENDING_PATH
            if preparing.exists():
                if pending.exists():
                    raise InvalidInput("multiple Pi recovery directories retained")
                if not list(preparing.iterdir()):
                    preparing.rmdir()
                    files.sync_dir(preparing.parent)
                else:
                    temporary = preparing / ".plan.json.loom-hem-tmp"
                    entries = {p.name for p in preparing.iterdir()}
                    modes = entries & {"configure", "disable"}
                    if len(modes) == 1 and entries in (modes, modes | {temporary.name}):
                        mode = next(iter(modes))
                        fingerprint = _mode_marker(preparing / mode)
                        endpoint = files.read_owned(root / ENDPOINT_PATH)
                        _endpoint(endpoint, root, reserved)
                        plan = _make_plan(root, target, reserved, assets, inspected, endpoint, registry.list_servers(), mode)
                        expected_fingerprint = hashlib.sha256(files.encoded(plan)).hexdigest()
                        if fingerprint is None:
                            if temporary.exists():
                                raise InvalidInput("temporary precedes durable Pi intent fingerprint; retained")
                            (preparing / mode / expected_fingerprint).mkdir(mode=0o700)
                            files.sync_dir(preparing / mode)
                        elif fingerprint != expected_fingerprint:
                            raise InvalidInput("Pi preparation inputs changed; recovery content retained")
                        # No target/endpoint replacement precedes durable pending.
                        # Reconstruct solely from unchanged old state and the
                        # reserved identity; refuse any nonprefix temporary.
                        files.atomic_replace(preparing / "plan.json", files.encoded(plan), expected=None,
                            deadline=deadline, allow_prefix=True)
                    elif entries == {temporary.name}:
                        _load_plan(preparing, root, reserved, assets, filename=temporary.name)
                        os.replace(temporary, preparing / "plan.json")
                        files.sync_dir(preparing)
                    _load_plan(preparing, root, reserved, assets)
                    os.rename(preparing, pending)
                    files.sync_dir(pending.parent)
            recovered = False
            if pending.exists() and not list(pending.iterdir()):
                # A crash after deleting the plan but before rmdir is bounded.
                current_endpoint = files.read_owned(root / ENDPOINT_PATH)
                _endpoint(current_endpoint, root, reserved)
                complete = inspected["manifest"] and inspected["hashes_valid"]
                empty = not inspected["manifest"] and not inspected["drift"] and current_endpoint is None
                if not (complete or empty):
                    raise InvalidInput("empty Pi recovery directory with incomplete install retained")
                pending.rmdir()
                files.sync_dir(pending.parent)
                inspected = files.inspect_target(target, root, assets=assets)
                manifest = inspected["manifest"]
            if pending.exists():
                plan = _load_plan(pending, root, reserved, assets)
                retained = _resume(root, target, reserved, plan, assets, deadline)
                if retained:
                    return _result(target, state="cleanup_required", changed=True, completed=False,
                        exit_code=1, accepted=True, retained_paths=retained, next_action="restore retained owned files, then rerun loom-hem pi configure --disable")
                recovered = True
                if plan["mode"] == "disable":
                    clear_target(registry, integration_id=reserved["integration_id"])
                    if not disable:
                        # Complete recovery first; a fresh install is a separate invocation.
                        return _result(target, state="disabled", changed=True, next_action="loom-hem pi configure")
                    return _result(target, state="disabled", changed=True)
                inspected = files.inspect_target(target, root, assets=assets)
                manifest = inspected["manifest"]
            endpoint = files.read_owned(root / ENDPOINT_PATH)
            _endpoint(endpoint, root, reserved)
            if manifest is None:
                if inspected["drift"]:
                    raise InvalidInput("reservation has unowned installed files; retained")
                if disable:
                    if endpoint is not None:
                        raise InvalidInput("reservation has endpoint without recovery plan; retained")
                    clear_target(registry, integration_id=reserved["integration_id"])
                    return _result(target, state="disabled", changed=True)
            if disable:
                plan = _make_plan(root, target, reserved, assets, inspected, endpoint, registry.list_servers(), "disable")
            else:
                if inspected["drift"]:
                    raise InvalidInput("edited owned Pi artifacts retained: " + ", ".join(inspected["drift"]))
                from .hem_pi_endpoints import project_endpoints
                projection = files.encoded(project_endpoints(root, reserved, registry.list_servers()))
                desired_manifest = files.encoded(_manifest(root, reserved, assets))
                if inspected["configured"] and not inspected["changed"] and endpoint == projection:
                    return _result(target, state="configured", configured=True, changed=recovered,
                        affected_paths=[], available=[], servers=[])
                plan = _make_plan(root, target, reserved, assets, inspected, endpoint, registry.list_servers(), "configure")
            _prepare(root, target, reserved, plan, assets, deadline)
            retained = _resume(root, target, reserved, plan, assets, deadline)
            if retained:
                return _result(target, state="cleanup_required", changed=True, completed=False, exit_code=1,
                    accepted=True, retained_paths=retained, next_action="restore retained owned files, then rerun loom-hem pi configure --disable")
            if disable:
                clear_target(registry, integration_id=reserved["integration_id"])
            else:
                verified = files.inspect_target(target, root, assets=assets)
                if not verified["configured"] or get_target(registry) != reserved:
                    raise UncertainOutcome("Pi installation verification failed; rerun configure")
            return _result(target, state="disabled" if disable else "configured", configured=not disable,
                changed=True, accepted=True, affected_paths=inspected["affected_paths"], available=[], servers=[])
