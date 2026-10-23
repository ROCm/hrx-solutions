"""Bounded owned-file primitives. Callers serialize writers with Pi locks."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat

from ...hem_common import InvalidInput, UncertainOutcome
from .hem_pi_assets import bundled_assets, asset_digests
from .hem_pi_contracts import (PREVIOUS_NATIVE_PROFILE, PREVIOUS_NATIVE_COMMIT, MANIFEST_PATH, ENDPOINT_PATH, INTEGRATION_VERSION,
    MANIFEST_SCHEMA_VERSION, EXTENSION_VERSION, NATIVE_PROFILE, NATIVE_COMMIT,
    PI_PACKAGE, PI_VERSIONS)

MAX_BYTES = 2 * 1024 * 1024
PENDING_PATH = "loom-hem/.pending"
PREPARING_PATH = "loom-hem/.preparing"


def encoded(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def private_path(path, *, create=False):
    """No managed descendant aliases or writable/unowned directory components."""
    path = Path(path)
    if not path.is_absolute():
        raise InvalidInput("managed Pi paths must be absolute")
    for component in reversed((path, *path.parents)):
        try:
            info = component.lstat()
        except FileNotFoundError:
            if create:
                component.mkdir(mode=0o700)
                sync_dir(component.parent)
            continue
        sticky_root = info.st_uid == 0 and info.st_mode & stat.S_ISVTX
        if (not stat.S_ISDIR(info.st_mode) or info.st_uid not in (0, os.getuid()) or
                info.st_mode & 0o022 and not sticky_root):
            raise InvalidInput(f"unsafe managed directory: {component}")
    return path


def read_owned(path):
    path = Path(path)
    private_path(path.parent)
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except FileNotFoundError:
        return None
    except OSError as error:
        raise InvalidInput(f"cannot safely read managed file: {path}") from error
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o022 or info.st_size > MAX_BYTES:
            raise InvalidInput(f"unsafe or oversized managed file: {path}")
        with os.fdopen(fd, "rb", closefd=False) as stream:
            data = stream.read(MAX_BYTES + 1)
        if len(data) > MAX_BYTES:
            raise InvalidInput(f"oversized managed file: {path}")
        return data
    finally:
        os.close(fd)


def sync_dir(path):
    fd = os.open(path, os.O_DIRECTORY | os.O_RDONLY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def staged_path(path):
    path = Path(path)
    return path.with_name("." + path.name + ".loom-hem-tmp")


def validate_stage(path, data, *, allow_prefix=False):
    temporary = staged_path(path)
    staged = read_owned(temporary)
    if staged is not None:
        info = temporary.lstat()
        if info.st_mode & 0o077:
            raise InvalidInput(f"recovery temporary must be private: {temporary}")
    if staged is not None and staged != data and not (allow_prefix and data.startswith(staged)):
        raise InvalidInput(f"unexpected recovery temporary retained: {temporary}")
    return staged


def atomic_replace(path, data, *, expected, deadline, allow_prefix=False):
    """Compare immediately before replacement; one fixed recovery temp per file."""
    path = Path(path)
    if not isinstance(data, bytes) or len(data) > MAX_BYTES:
        raise InvalidInput("invalid or oversized managed replacement")
    deadline.remaining()
    private_path(path.parent, create=True)
    if read_owned(path) != expected:
        raise InvalidInput(f"managed file changed: {path}")
    temporary = staged_path(path)
    staged = validate_stage(path, data, allow_prefix=allow_prefix)
    if staged is None:
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    elif staged != data:
        # The durable plan or preparation intent authorizes only this exact
        # staged prefix at this fixed owned path; arbitrary third bytes refuse.
        fd = os.open(temporary, os.O_WRONLY | os.O_NOFOLLOW)
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
            os.close(fd)
            raise InvalidInput(f"unsafe recovery temporary retained: {temporary}")
        os.ftruncate(fd, 0)
    else:
        fd = None
    if fd is not None:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    deadline.remaining()
    if read_owned(path) != expected or read_owned(temporary) != data:
        raise InvalidInput(f"managed replacement conflict: {path}")
    os.replace(temporary, path)
    try:
        sync_dir(path.parent)
    except OSError as error:
        raise UncertainOutcome(f"replacement durability uncertain: {path}") from error


def remove_owned(path, expected, deadline):
    deadline.remaining()
    path = Path(path)
    current = read_owned(path)
    if current is None:
        return
    if current != expected:
        raise InvalidInput(f"edited managed file retained: {path}")
    path.unlink()
    try:
        sync_dir(path.parent)
    except OSError as error:
        raise UncertainOutcome(f"removal durability uncertain: {path}") from error


def decode(data, label):
    try:
        value = json.loads(data)
        if not isinstance(value, dict):
            raise ValueError()
        return value
    except (ValueError, TypeError, UnicodeError) as error:
        raise InvalidInput(f"malformed {label}; retained for inspection") from error


def validate_manifest(manifest, target, root, assets):
    from uuid import UUID
    expected = dict(schema_version=MANIFEST_SCHEMA_VERSION, integration_version=INTEGRATION_VERSION,
        extension_version=EXTENSION_VERSION, native_profile=NATIVE_PROFILE,
        native_commit=NATIVE_COMMIT, pi_package=PI_PACKAGE, pi_versions=list(PI_VERSIONS),
        hem_root=str(root), target=str(target), endpoint_file=str(Path(root) / ENDPOINT_PATH))
    native_identity = (manifest.get("native_profile"), manifest.get("native_commit"))
    if native_identity == (PREVIOUS_NATIVE_PROFILE, PREVIOUS_NATIVE_COMMIT):
        expected.update(native_profile=PREVIOUS_NATIVE_PROFILE, native_commit=PREVIOUS_NATIVE_COMMIT)
    if any(manifest.get(k) != v for k, v in expected.items()) or type(manifest.get("enabled")) is not bool:
        raise InvalidInput("foreign or unsupported Pi ownership manifest")
    try:
        identity = UUID(manifest["integration_id"])
        if identity.version != 4 or str(identity) != manifest["integration_id"]:
            raise ValueError()
    except (KeyError, ValueError, TypeError, AttributeError) as error:
        raise InvalidInput("invalid Pi manifest identity") from error
    hashes = manifest.get("artifacts")
    if not isinstance(hashes, dict) or set(hashes) != set(assets):
        raise InvalidInput("unsupported Pi managed artifact set")
    for digest in hashes.values():
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise InvalidInput("invalid managed artifact digest")
    return manifest


def inspect_target(target, root, *, assets=None):
    """Passive inspection, including partial/drifted installs; never create paths."""
    target, root = Path(target), Path(root)
    private_path(target)
    assets = bundled_assets() if assets is None else assets
    pending = any((target / p).exists() for p in (PENDING_PATH, PREPARING_PATH))
    # Only ancestors outside the managed target may use searchable HOME modes.
    for directory in {target, *( (target / name).parent for name in (*assets, MANIFEST_PATH))}:
        private_path(directory)
        current = directory
        while current == target or target in current.parents:
            if current.exists():
                info = current.lstat()
                if info.st_uid != os.getuid() or info.st_mode & 0o077:
                    raise InvalidInput(f"managed Pi directory must be private: {current}")
            if current == target:
                break
            current = current.parent
    # A dedicated package cannot silently adopt extra user code. Other Pi
    # extensions and settings are outside this package and remain untouched.
    package = target / "extensions/loom-hem"
    if package.exists():
        allowed = {Path(name).name for name in assets}
        if pending:
            allowed |= {staged_path(Path(name)).name for name in assets}
        unknown = [entry for entry in package.iterdir() if entry.name not in allowed]
        if unknown:
            raise InvalidInput("unknown Pi extension content retained: " + ", ".join(str(path) for path in sorted(unknown)))
    manifest_bytes = read_owned(target / MANIFEST_PATH)
    manifest = None if manifest_bytes is None else validate_manifest(decode(manifest_bytes, "Pi manifest"), target, root, assets)
    drift = []
    for relative in assets:
        data = read_owned(target / relative)
        if manifest is None:
            if data is not None:
                drift.append(relative)
        elif data is None or hashlib.sha256(data).hexdigest() != manifest["artifacts"][relative]:
            drift.append(relative)
    for relative in (PENDING_PATH, PREPARING_PATH):
        private_path(target / relative)
    configured = bool(manifest and manifest["enabled"] and not drift and not pending
                      and manifest["native_profile"] == NATIVE_PROFILE and manifest["native_commit"] == NATIVE_COMMIT)
    changed = not configured or manifest["artifacts"] != asset_digests(assets)
    return dict(manifest=manifest, manifest_bytes=manifest_bytes, configured=configured,
        hashes_valid=manifest is not None and not drift, pending=pending, drift=drift,
        changed=changed, integration_id=manifest["integration_id"] if manifest else None,
        affected_paths=[str(target / path) for path in sorted((*assets, MANIFEST_PATH))])
