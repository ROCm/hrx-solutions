"""Complete qualified Pi setup, owned installation and recovery.

No Pi process, settings file, package installer, HTTP or lifecycle operation is
invoked here. One root reservation plus one bounded pending plan bridge SQLite
and atomic files; a reservation alone never means configured.
"""
from __future__ import annotations

import base64
import hashlib
import os
import json
import re
import shlex
import shutil
import subprocess
from urllib.parse import urlsplit
from pathlib import Path
import stat
import uuid

from .hem_common import DeadlineExpired, HemError, InvalidInput, UncertainOutcome
from . import hem_display
from .hem_pi_contracts import NODE_MIN_VERSION
from .hem_pi_endpoints import project_endpoints, _origin
from .hem_pi_assets import bundled_assets, asset_digests
from .hem_pi_contracts import (PiResult, MANIFEST_PATH, ENDPOINT_PATH,
    MANIFEST_SCHEMA_VERSION, INTEGRATION_VERSION, EXTENSION_VERSION,
    NATIVE_PROFILE, NATIVE_COMMIT, PI_PACKAGE, PI_VERSIONS)
from .hem_pi_state import (passive_snapshot, integration_lock, target_lock,
    get_target, set_target, clear_target)
from .hem_state import Registry
from . import hem_pi_files as files


MAX_BYTES = files.MAX_BYTES


def _read(path):
    """Bounded regular metadata only; never follow user auth/settings aliases."""
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_BYTES:
            raise InvalidInput("unsupported or oversized Pi metadata")
        with os.fdopen(fd, "rb", closefd=False) as stream:
            value = stream.read(MAX_BYTES + 1)
        if len(value) > MAX_BYTES:
            raise InvalidInput("oversized Pi metadata")
        return value
    finally:
        os.close(fd)


def _metadata(path):
    try:
        data = json.loads(_read(path))
        if not isinstance(data, dict):
            raise ValueError()
        return data
    except (OSError, ValueError, UnicodeError) as error:
        raise InvalidInput("cannot inspect installed Pi package metadata; select a qualified Pi on PATH") from error


def _entry(package, relative):
    if not isinstance(relative, str) or not relative:
        raise InvalidInput("installed Pi lacks the qualified public package entry point")
    try:
        entry = (package / relative).resolve()
        if entry.is_relative_to(package) and entry.is_file():
            return entry
    except (OSError, ValueError, RuntimeError):
        pass
    raise InvalidInput("installed Pi public entry point is missing or escapes its package")


def _public_entry(package, metadata):
    exports = metadata.get("exports")
    exported = exports.get(".") if isinstance(exports, dict) else None
    relative = exported.get("import") if isinstance(exported, dict) else exported
    return _entry(package, relative)


def inspect_runtime(deadline):
    """Identify the selected npm bin without executing Pi, extensions or npm."""
    selected = shutil.which("pi")
    if selected is None:
        raise InvalidInput("Pi is not on PATH; select qualified @earendil-works/pi-coding-agent 1.3.0 (no automatic upgrade)")
    executable = Path(selected).resolve()
    package = None
    for ancestor in executable.parents:
        manifest = ancestor / "package.json"
        if manifest.is_file():
            metadata = _metadata(manifest)
            if metadata.get("name") == PI_PACKAGE:
                package = ancestor
                break
    if package is None:
        raise InvalidInput("selected pi launcher has no identifiable qualified npm package; select its unambiguous npm bin on PATH")
    version = metadata.get("version")
    if version not in PI_VERSIONS:
        raise InvalidInput("unsupported selected Pi version; require @earendil-works/pi-coding-agent 1.3.0; existing Pi is unchanged")
    bins = metadata.get("bin")
    if not isinstance(bins, dict) or _entry(package, bins.get("pi")) != executable:
        raise InvalidInput("selected pi does not match its package bin; resolve the ambiguous launcher on PATH")
    _public_entry(package, metadata)
    dependencies = metadata.get("dependencies")
    if not isinstance(dependencies, dict) or dependencies.get("@earendil-works/pi-ai") != version:
        raise InvalidInput("selected Pi package has an unsupported pi-ai dependency")
    # Follow ordinary Node module lookup, stopping at the first installed match.
    ai = next((parent / "node_modules/@earendil-works/pi-ai" for parent in (package, *package.parents)
               if (parent / "node_modules/@earendil-works/pi-ai/package.json").is_file()), None)
    if ai is None:
        raise InvalidInput("selected Pi is missing its qualified pi-ai package")
    ai = ai.resolve()
    ai_metadata = _metadata(ai / "package.json")
    if ai_metadata.get("name") != "@earendil-works/pi-ai" or ai_metadata.get("version") != version:
        raise InvalidInput("selected Pi resolves an unsupported pi-ai package")
    _public_entry(ai, ai_metadata)
    try:
        with executable.open("rb") as stream:
            shebang = stream.readline(512).decode("ascii").strip()
        launcher = shlex.split(shebang[2:]) if shebang.startswith("#!") else []
    except (OSError, ValueError, UnicodeError) as error:
        raise InvalidInput("cannot identify the selected Pi Node launcher") from error
    if launcher == ["/usr/bin/env", "node"]:
        node = shutil.which("node")
    elif len(launcher) == 1 and Path(launcher[0]).is_absolute() and Path(launcher[0]).name == "node":
        node = launcher[0]
    else:
        raise InvalidInput("unsupported or ambiguous Pi launcher; require the qualified npm Node bin")
    if node is None:
        raise InvalidInput("Pi requires Node >=22.19.0; node is not on PATH")
    node = str(Path(node).resolve())
    environment = {key: value for key, value in os.environ.items() if key not in ("NODE_OPTIONS", "NODE_PATH")}
    try:
        observed = subprocess.run([node, "--version"], env=environment, stdin=subprocess.DEVNULL,
            capture_output=True, text=True, timeout=deadline.remaining(), check=False)
    except subprocess.TimeoutExpired as error:
        raise HemError("Node identity check timed out before any Pi configuration writes", accepted=False) from error
    except OSError as error:
        raise InvalidInput("cannot inspect selected Node runtime") from error
    match = re.fullmatch(r"v(\d+)\.(\d+)\.(\d+)\s*", observed.stdout)
    if observed.returncode or match is None or tuple(map(int, match.groups())) < NODE_MIN_VERSION:
        raise InvalidInput("selected Pi requires Node >=22.19.0; no runtime or package was changed")
    deadline.remaining()
    return dict(package=PI_PACKAGE, version=version, package_directory=str(package),
                executable=str(executable), node=node, node_version=observed.stdout.strip())


def _legacy(target, managed_providers=(), *, report=True):
    """Check static provider IDs and report only allowlisted legacy metadata."""
    path = target / "models.json"
    try:
        value = json.loads(_read(path))
    except FileNotFoundError:
        return []
    except (OSError, ValueError, UnicodeError, InvalidInput):
        if report:
            hem_display.emit("Pi models.json could not be safely inspected; preserved unchanged. Inspect it manually for unmanaged local listings.")
        return []
    providers = value.get("providers", {}) if isinstance(value, dict) else {}
    providers = providers if isinstance(providers, dict) else {}
    collisions = set(managed_providers).intersection(providers)
    if collisions:
        raise InvalidInput("Pi models.json has static provider IDs that conflict with configured Hem servers: " +
            ", ".join(sorted(collisions)) + "; export/back up models.json and rename or remove those static providers before rerunning loom-control pi configure; no configuration was changed")
    provider = providers.get("hrx-loom")
    if not isinstance(provider, dict):
        return []
    endpoint = provider.get("baseUrl")
    try:
        parsed = urlsplit(endpoint)
        endpoint = _origin(f"{parsed.scheme}://{parsed.netloc}") + parsed.path
        if parsed.username or parsed.password or parsed.query or parsed.fragment or parsed.path not in ("", "/v1"):
            raise ValueError()
    except (ValueError, TypeError, AttributeError, InvalidInput):
        endpoint = "<unverified endpoint>"
    models = provider.get("models", [])
    identifiers = [item["id"] for item in models if isinstance(item, dict) and
                   isinstance(item.get("id"), str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", item["id"])] if isinstance(models, list) else []
    if report:
        hem_display.emit("Unmanaged legacy Pi provider hrx-loom is preserved. Export/back up models.json and check default-model references yourself before manually removing only obsolete hrx-loom entries; Hem does not change settings or credentials.")
    return [dict(provider="hrx-loom", ids=identifiers, endpoint=endpoint,
                 state="unmanaged", reason="availability unknown; manual export and cleanup are the user's responsibility")]


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
        # Reuse the shared row contract, accepting only current or one known
        # prior owned profile while preserving this Pi reservation envelope.
        from .hem_clients_endpoints import validate_endpoints, OWNER
        validate_endpoints(dict(schema_version=1, owner=OWNER, hem_root=str(root),
                                servers=value["servers"]), root)


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


def _make_plan(root, target, reserved, assets, inspected, endpoint, projection, mode):
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
    projection = files.encoded(projection)
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


def _configure_owned(root, target, deadline, assets, snapshot, identity, *, check=False, disable=False):
    """Apply qualified intent, preserving reservation and bounded recovery evidence."""
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
            projection = files.encoded(project_endpoints(root, snapshot["pi_target"], snapshot["servers"]))
            changed = changed or endpoint != projection
            configured = configured and endpoint is not None
        return _result(target, state="changes_required" if changed else "configured", configured=configured,
            exit_code=int(changed), affected_paths=inspected["affected_paths"],
            next_action="loom-control pi configure" if changed else None)
    if disable and snapshot["pi_target"] is None:
        if inspected["manifest"] or inspected["pending"] or inspected["drift"]:
            raise InvalidInput("Pi files have no matching root reservation; retained")
        return _result(target, state="disabled")
    registry = Registry(root, deadline=deadline)
    with integration_lock(root, deadline):
        reserved = get_target(registry)
        _identity_check(reserved, target)
        if reserved is None and snapshot["pi_target"] is not None:
            # A concurrent disable retired the preflight generation. Never
            # recreate its UUID; a fresh invocation must read current state.
            raise InvalidInput("Pi reservation changed during setup; inspect with loom-control pi configure --check before retrying")
        with target_lock(target, deadline):
            inspected = files.inspect_target(target, root, assets=assets)
            manifest = inspected["manifest"]
            if manifest and (reserved is None or inspected["integration_id"] != reserved["integration_id"]):
                raise InvalidInput("Pi target has no matching ownership reservation")
            if not disable:
                projection = project_endpoints(root, reserved or identity, registry.list_servers())
                _legacy(target, [row["provider"] for row in projection["servers"]], report=False)
            if reserved is None:
                if inspected["drift"] or inspected["pending"]:
                    raise InvalidInput("unowned Pi integration files retained")
                reserved = identity
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
                        recovery_projection = project_endpoints(root, reserved, registry.list_servers()) if mode == "configure" else None
                        plan = _make_plan(root, target, reserved, assets, inspected, endpoint, recovery_projection, mode)
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
                        exit_code=1, accepted=True, retained_paths=retained, next_action="restore retained owned files, then rerun loom-control pi configure --disable")
                recovered = True
                if plan["mode"] == "disable":
                    clear_target(registry, integration_id=reserved["integration_id"])
                    if not disable:
                        # Complete recovery first; a fresh install is a separate invocation.
                        return _result(target, state="disabled", changed=True, next_action="loom-control pi configure")
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
                plan = _make_plan(root, target, reserved, assets, inspected, endpoint, None, "disable")
            else:
                if inspected["drift"]:
                    raise InvalidInput("edited owned Pi artifacts retained: " + ", ".join(inspected["drift"]))
                plan = _make_plan(root, target, reserved, assets, inspected, endpoint, projection, "configure")
                if inspected["configured"] and not inspected["changed"] and endpoint == _unpack(plan["next"]["@endpoint"]):
                    return _result(target, state="configured", configured=True, changed=recovered, affected_paths=[])
            _prepare(root, target, reserved, plan, assets, deadline)
            retained = _resume(root, target, reserved, plan, assets, deadline)
            if retained:
                return _result(target, state="cleanup_required", changed=True, completed=False, exit_code=1,
                    accepted=True, retained_paths=retained, next_action="restore retained owned files, then rerun loom-control pi configure --disable")
            if disable:
                clear_target(registry, integration_id=reserved["integration_id"])
            else:
                verified = files.inspect_target(target, root, assets=assets)
                if not verified["configured"] or get_target(registry) != reserved:
                    raise UncertainOutcome("Pi installation verification failed; rerun configure")
            return _result(target, state="disabled" if disable else "configured", configured=not disable,
                changed=True, accepted=True, affected_paths=inspected["affected_paths"])


def configure_pi(root, target, deadline, *, check=False, disable=False):
    """Qualify and configure Pi; direct callers receive the CLI safeguards."""
    if check and disable:
        raise InvalidInput("Pi check and disable are mutually exclusive")
    try:
        deadline.remaining()
    except DeadlineExpired as error:
        raise HemError("Pi configuration deadline expired before writes", accepted=False) from error
    if root is None:
        root = os.environ.get("LOOM_SERVE_DIR", str(Path.home() / ".loom"))
    root = Path(root)
    if not root.is_absolute():
        raise InvalidInput("LOOM_SERVE_DIR/state root must be absolute")
    root, target = root.resolve(), resolve_target(target)
    managed_providers = []
    runtime = None
    compatibility = dict(state="not_checked", reason="runtime qualification is unnecessary for safe owned-file removal",
                         next_action=None)
    try:
        if not disable:
            try:
                runtime = inspect_runtime(deadline)
                compatibility = dict(state="compatible", reason="selected Pi and Node match the qualified package profile",
                                     next_action=None)
            except (HemError, OSError) as error:
                if not check:
                    raise
                compatibility = dict(state="unqualified", reason=str(error),
                                     next_action="select qualified Pi 1.3.0 and Node >=22.19.0 on PATH, then rerun loom-control pi configure --check")
                hem_display.emit("Pi runtime compatibility: " + compatibility["reason"])
                hem_display.emit("Next: " + compatibility["next_action"])
        assets = bundled_assets()
        _validate_assets(assets)
        snapshot = passive_snapshot(root, deadline)
        identity = snapshot["pi_target"] or dict(target=str(target), integration_id=str(uuid.uuid4()))
        if not disable:
            # Validate using the actual proposed reservation identity, before writes.
            projection = project_endpoints(root, identity, snapshot["servers"])
            managed_providers = [row["provider"] for row in projection["servers"]]
    except DeadlineExpired as error:
        raise HemError("Pi compatibility inspection deadline expired before writes", accepted=False) from error
    except OSError as error:
        raise HemError("bundled Pi resources are unavailable; reinstall this wheel before configuring Pi", accepted=False) from error
    legacy = _legacy(target, managed_providers)
    hem_display.emit("No native Loom equivalent: owned Pi integration file configuration")
    try:
        response = _configure_owned(root, target, deadline, assets, snapshot, identity, check=check, disable=disable)
    except DeadlineExpired as error:
        if error.accepted is True:
            raise
        if check:
            raise HemError("Pi inspection deadline expired before any writes", accepted=False) from error
        raise UncertainOutcome("Pi update deadline expired; inspect with loom-control pi configure --check before retrying") from error
    except OSError as error:
        if check:
            raise HemError("Pi inspection failed without writes", accepted=False) from error
        raise UncertainOutcome("Pi update interrupted; inspect with loom-control pi configure --check before retrying") from error
    response.update(runtime=runtime, runtime_compatibility=compatibility, legacy=legacy)
    if check and compatibility["state"] == "unqualified":
        # Preserve the configuration findings and existing stronger failure codes.
        response.exit_code = response.exit_code or 1
    if response.get("affected_paths"):
        response["affected_paths"] = sorted(set(response["affected_paths"]) | {str(root / ENDPOINT_PATH), str(root / "control.sqlite3")})
    for path in response.get("affected_paths", []):
        hem_display.emit("Pi integration path: " + str(path))
    if not disable:
        response["availability"] = "not_observed"
        hem_display.emit("Pi settings/defaults/auth are unchanged. Launch Pi normally; use /reload or restart for new server names.")
    if response.get("next_action"):
        hem_display.emit("Next: " + response["next_action"])
    return response
