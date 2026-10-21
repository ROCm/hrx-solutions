"""Pure WebUI ownership/merge planning; no file, network or registry access.

Only ``remote_config`` may contain foreign credentials. Never serialize a plan
or persist that field. Callers own private-path checks, expected-replace writes,
locking, remote rechecks and readback. Pending records describe affected owned
connections only; they cannot reconstruct historical foreign configuration.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import math
import json
from pathlib import Path

from .hem_common import InvalidInput
from .hem_openwebui_http import canonical_target
from .hem_pi_contracts import NATIVE_PROFILE, READABLE_NATIVE_PROFILES, provider_id

OWNER = "loom-control-openwebui"
SCHEMA_VERSION = 1
MAX_BYTES = 2 * 1024 * 1024
_DOCUMENT_KEYS = {"schema_version", "owner", "hem_root", "target_url", "deployment_id", "connections", "pending"}


def _fail():
    raise InvalidInput("WebUI ownership/schema conflict; retain evidence for inspection")


def _origin(value):
    # Shared pure canonicalizer: ownership and HTTP must identify one target.
    return canonical_target(value)


def _root(root):
    # Lexical only: callers check filesystem aliases with private_path/read_owned.
    value = str(root)
    path = Path(value)
    if not path.is_absolute() or str(path) != value or ".." in path.parts or value.startswith("//"):
        _fail()
    return value


def ownership_path(root: Path, target: str) -> Path:
    target = _origin(target)
    return Path(_root(root)) / "clients/openwebui" / (hashlib.sha256(target.encode()).hexdigest() + ".json")


def _encoded(value):
    def check(item):
        if item is None or type(item) in (str, bool, int):
            return
        if type(item) is float and math.isfinite(item):
            return
        if type(item) is list:
            for child in item:
                check(child)
            return
        if type(item) is dict and all(type(key) is str for key in item):
            for child in item.values():
                check(child)
            return
        _fail()
    try:
        check(value)
        data = json.dumps(value, allow_nan=False, separators=(",", ":")).encode()
        if len(data) > MAX_BYTES:
            _fail()
    except (ValueError, TypeError, RecursionError, UnicodeError, OverflowError):
        _fail()
    return data


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            _fail()
        result[key] = value
    return result


def _managed(provider):
    return dict(enable=True, auth_type="none", prefix_id=provider, model_ids=[], connection_type="local")


def _strict(left, right):
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(_strict(left[k], right[k]) for k in left)
    if isinstance(left, list):
        return len(left) == len(right) and all(_strict(a, b) for a, b in zip(left, right))
    return left == right


def _records(rows):
    if not isinstance(rows, list):
        _fail()
    providers = []
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"server", "provider", "base_url", "config"}:
            _fail()
        try:
            provider = provider_id(row["server"])
            base = row["base_url"]
            if (row["provider"] != provider or not isinstance(base, str) or not base.endswith("/v1") or
                    _origin(base[:-3]) + "/v1" != base or not _strict(row["config"], _managed(provider))):
                _fail()
        except (TypeError, AttributeError):
            _fail()
        providers.append(provider)
    if providers != sorted(set(providers)):
        _fail()
    return rows


def _validate_document(document, root, target):
    if (not isinstance(document, dict) or set(document) != _DOCUMENT_KEYS or
            type(document["schema_version"]) is not int or document["schema_version"] != SCHEMA_VERSION or
            document["owner"] != OWNER or document["hem_root"] != _root(root) or
            document["target_url"] != _origin(target) or not isinstance(document["deployment_id"], str)):
        _fail()
    owned = _records(document["connections"])
    pending = document["pending"]
    if pending is not None:
        if (not isinstance(pending, dict) or set(pending) != {"operation", "before", "after", "deployment_id"} or
                pending["operation"] not in ("configure", "disable") or not isinstance(pending["deployment_id"], str)):
            _fail()
        before, after = _records(pending["before"]), _records(pending["after"])
        current = {r["provider"]: r for r in owned}
        old, new = {r["provider"]: r for r in before}, {r["provider"]: r for r in after}
        affected = old.keys() | new.keys()
        if (not affected or any(_strict(old.get(p), new.get(p)) for p in affected) or
                any(p not in current or not _strict(current[p], row) for p, row in old.items()) or
                pending["operation"] == "disable" and after or
                document["deployment_id"] and pending["deployment_id"] != document["deployment_id"]):
            _fail()
    _encoded(document)
    return document


def decode_document(data: bytes | None, root: Path, target: str) -> dict:
    root, target = _root(root), _origin(target)
    if data is None:
        return dict(schema_version=SCHEMA_VERSION, owner=OWNER, hem_root=root, target_url=target,
                    deployment_id="", connections=[], pending=None)
    if not isinstance(data, bytes) or len(data) > MAX_BYTES:
        _fail()
    try:
        document = json.loads(data, object_pairs_hook=_unique,
                              parse_constant=lambda _: _fail())
    except (ValueError, UnicodeError, RecursionError):
        _fail()
    return _validate_document(document, root, target)


def desired_connections(endpoints: dict) -> list[dict]:
    if not isinstance(endpoints, dict):
        _fail()
    if (set(endpoints) != {"schema_version", "owner", "hem_root", "servers"} or
            type(endpoints["schema_version"]) is not int or endpoints["schema_version"] != 1 or
            endpoints["owner"] != "loom-control-clients" or not isinstance(endpoints["servers"], list)):
        _fail()
    _root(endpoints["hem_root"])
    servers, profiles = [], set()
    for row in endpoints["servers"]:
        if (not isinstance(row, dict) or set(row) != {"server", "provider", "base_url", "native_profile"} or
                row["native_profile"] not in READABLE_NATIVE_PROFILES):
            _fail()
        servers.append(row["server"])
        profiles.add(row["native_profile"])
    if any(not isinstance(s, str) for s in servers) or servers != sorted(set(servers)) or len(profiles) > 1:
        _fail()
    rows = [dict(server=r["server"], provider=r["provider"], base_url=r["base_url"],
                 config=_managed(r["provider"])) for r in endpoints["servers"]]
    return _records(sorted(rows, key=lambda row: row["provider"]))


@dataclass
class MergePlan:
    configuration: str
    findings: list
    changed: bool = False
    partial: bool = False
    recovery: str = "none"
    remote_config: dict | None = field(default=None, repr=False)
    pending_document: dict | None = None
    next_document: dict | None = None


def _remote(remote):
    # Transport validates the same envelope, but pure entry points must not trust callers.
    _encoded(remote)
    if (not isinstance(remote, dict) or set(remote) != {"ENABLE_OPENAI_API", "OPENAI_API_BASE_URLS", "OPENAI_API_KEYS", "OPENAI_API_CONFIGS"} or
            type(remote["ENABLE_OPENAI_API"]) is not bool or
            not isinstance(remote["OPENAI_API_BASE_URLS"], list) or not isinstance(remote["OPENAI_API_KEYS"], list) or
            not all(isinstance(v, str) for v in remote["OPENAI_API_BASE_URLS"] + remote["OPENAI_API_KEYS"]) or
            len(remote["OPENAI_API_BASE_URLS"]) != len(remote["OPENAI_API_KEYS"]) or
            not isinstance(remote["OPENAI_API_CONFIGS"], dict)):
        _fail()
    count = len(remote["OPENAI_API_BASE_URLS"])
    for index, config in remote["OPENAI_API_CONFIGS"].items():
        if (not isinstance(index, str) or not index.isascii() or not index.isdecimal() or
                len(index) > 10 or str(int(index)) != index or int(index) >= count or not isinstance(config, dict)):
            _fail()


def _indices(remote, provider):
    return [i for i in range(len(remote["OPENAI_API_BASE_URLS"]))
            if remote["OPENAI_API_CONFIGS"].get(str(i), {}).get("prefix_id") == provider]


def _matches(remote, row):
    indices = _indices(remote, row["provider"])
    return (len(indices) == 1 and remote["OPENAI_API_BASE_URLS"][indices[0]] == row["base_url"] and
            remote["OPENAI_API_KEYS"][indices[0]] == "" and
            _strict(remote["OPENAI_API_CONFIGS"].get(str(indices[0])), row["config"]))


def _set_matches(remote, rows, providers):
    expected = {r["provider"]: r for r in rows}
    return all(_matches(remote, expected[p]) if p in expected else not _indices(remote, p) for p in providers)


def _with_connections(document, rows, identity):
    value = deepcopy(document)
    value.update(connections=sorted(deepcopy(rows), key=lambda r: r["provider"]), pending=None,
                 deployment_id=identity or document["deployment_id"])
    _validate_document(value, value["hem_root"], value["target_url"])
    return value


def plan_change(remote: dict, document: dict, desired: list[dict], *, mode: str, deployment_id: str) -> MergePlan:
    """Plan one mutation, or assess passively. All returned values are detached.

    ``changed`` includes local bookkeeping; ``remote_config is not None`` alone
    authorizes a proposed POST. A conflict never authorizes POST. ``check`` never
    proposes local writes, including finalization of an observed pending result.
    """
    _remote(remote)
    if not isinstance(document, dict) or mode not in ("configure", "check", "disable") or not isinstance(deployment_id, str):
        _fail()
    _validate_document(document, document.get("hem_root"), document.get("target_url"))
    _records(desired)
    original = deepcopy(document)
    current = deepcopy(document)
    findings = []
    recovery = "none"
    identities = [current["deployment_id"]]
    if current["pending"]:
        identities.append(current["pending"]["deployment_id"])
    if any(identity and identity != deployment_id for identity in identities):
        return MergePlan("drift", [dict(status="target_identity_conflict")], recovery="conflict")
    pending = current["pending"]
    if pending:
        affected = {r["provider"] for r in pending["before"] + pending["after"]}
        if _set_matches(remote, pending["after"], affected):
            recovery = "completed"
            rows = [r for r in current["connections"] if r["provider"] not in affected] + pending["after"]
            current = _with_connections(current, rows, pending["deployment_id"])
        elif _set_matches(remote, pending["before"], affected):
            recovery = "unchanged"
            current["pending"] = None
        else:
            return MergePlan("pending", [dict(status="pending_conflict")], recovery="conflict")
    owned = {r["provider"]: r for r in current["connections"]}
    wanted = {r["provider"]: r for r in desired}
    conflicts = False
    eligible = {}
    for provider in sorted(owned.keys() | (wanted.keys() if mode != "disable" else set())):
        row = owned.get(provider)
        indices = _indices(remote, provider)
        status = "configured"
        if row is not None:
            if not indices:
                status = "missing"
                eligible[provider] = row
            elif not _matches(remote, row):
                status = "drift"
                conflicts = True
            else:
                eligible[provider] = row
        elif indices:
            status = "collision"
            conflicts = True
        else:
            status = "missing"
        findings.append(dict(provider=provider, server=(row or wanted[provider])["server"], status=status))
    if mode != "disable" and not remote["ENABLE_OPENAI_API"]:
        conflicts = True
        findings.append(dict(status="global_disabled"))
    if mode == "check":
        if conflicts:
            configuration = "drift"
        elif pending is not None:
            configuration = "pending"
        elif not _strict(owned, wanted) or any(f["status"] != "configured" for f in findings):
            configuration = "absent"
        else:
            configuration = "configured" if owned else "absent"
        return MergePlan(configuration, findings, recovery=recovery)
    if conflicts and mode == "configure":
        return MergePlan("drift", findings, recovery=recovery)
    after = dict(owned)
    if mode == "disable":
        for provider in eligible:
            after.pop(provider, None)
    else:
        after = wanted
    changed_providers = {p for p in owned.keys() | after.keys()
                         if not _strict(owned.get(p), after.get(p)) or
                         mode == "configure" and p in after and not _matches(remote, after[p])}
    merged = deepcopy(remote)
    # Replace existing proven-owned slots in place; preserve foreign order and
    # sparse configuration absence when removed slots cause index changes.
    urls, keys, configs = [], [], {}
    consumed = set()
    for index, url in enumerate(remote["OPENAI_API_BASE_URLS"]):
        old_config = remote["OPENAI_API_CONFIGS"].get(str(index))
        provider = old_config.get("prefix_id") if old_config is not None else None
        if isinstance(provider, str) and provider in eligible and provider in changed_providers:
            if provider not in after:
                continue
            row = after[provider]
            url, key, config = row["base_url"], "", row["config"]
            consumed.add(provider)
        else:
            key, config = remote["OPENAI_API_KEYS"][index], old_config
        slot = str(len(urls))
        urls.append(url)
        keys.append(key)
        if config is not None:
            configs[slot] = deepcopy(config)
    for provider in sorted(changed_providers):
        if provider in after and provider not in consumed:
            row = after[provider]
            configs[str(len(urls))] = deepcopy(row["config"])
            urls.append(row["base_url"])
            keys.append("")
    merged.update(OPENAI_API_BASE_URLS=urls, OPENAI_API_KEYS=keys, OPENAI_API_CONFIGS=configs)
    _remote(merged)
    next_document = _with_connections(current, list(after.values()), deployment_id)
    remote_changed = not _strict(remote, merged)
    pending_document = None
    if remote_changed:
        pending_document = deepcopy(current)
        pending_document["pending"] = dict(operation=mode,
            before=[deepcopy(owned[p]) for p in sorted(changed_providers) if p in owned and _matches(remote, owned[p])],
            after=[deepcopy(after[p]) for p in sorted(changed_providers) if p in after], deployment_id=deployment_id)
        _validate_document(pending_document, current["hem_root"], current["target_url"])
    configuration = "partial" if conflicts else ("configured" if mode == "configure" and after else "absent")
    return MergePlan(configuration, findings, changed=remote_changed or not _strict(original, next_document),
                     partial=conflicts, recovery=recovery, remote_config=merged if remote_changed else None,
                     pending_document=pending_document, next_document=next_document)
