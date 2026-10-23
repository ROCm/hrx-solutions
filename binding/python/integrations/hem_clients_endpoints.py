"""Root-owned connection projection for optional clients, never live inventory.

Writers acquire the existing integration lock themselves. Callers may hold a
model/server lock, but MUST NOT hold integration or target locks. This shares
serialization with Pi without nesting its publisher or changing its wire format.
"""
from __future__ import annotations

import ipaddress
import json
from pathlib import Path
from urllib.parse import urlsplit

from .. import hem_display
from ..hem_common import HemError, InvalidInput
from .pi.hem_pi_contracts import NATIVE_PROFILE, READABLE_NATIVE_PROFILES, provider_id
from .pi.hem_pi_files import atomic_replace, encoded, read_owned
from .pi.hem_pi_state import integration_lock

ENDPOINT_PATH = "clients/endpoints.json"
SCHEMA_VERSION = 1
OWNER = "loom-control-clients"
MAX_BYTES = 1024 * 1024
REPAIR = "run loom-control omp configure or loom-control opencode configure; retain and inspect conflicting files"


def _root(root):
    root = Path(root)
    if not root.is_absolute() or str(root) != str(root.resolve()):
        raise InvalidInput("client projection root must be canonical and absolute")
    return str(root)


def _origin(value):
    try:
        parsed = urlsplit(value)
        host = ipaddress.ip_address(parsed.hostname)
        port = parsed.port
        if (parsed.scheme != "http" or not host.is_loopback or port is None or
                not 0 < port < 65536 or parsed.username is not None or
                parsed.password is not None or parsed.path not in ("", "/") or
                parsed.query or parsed.fragment):
            raise ValueError()
    except (ValueError, TypeError, AttributeError):
        raise InvalidInput("client endpoint requires numeric loopback HTTP origin and nonzero port") from None
    return f"http://[{host}]:{port}" if host.version == 6 else f"http://{host}:{port}"


def _encoded(document):
    data = encoded(document)
    if len(data) > MAX_BYTES:
        raise InvalidInput("client endpoint projection exceeds 1 MiB")
    return data


def project_endpoints(root, servers):
    """Pure projection of saved definitions, including stopped servers."""
    rows = []
    for server in sorted(servers, key=lambda item: item["id"]):
        origin = _origin("http://" + server["saved_settings"]["listen"])
        if server.get("launch_token") and server.get("endpoint"):
            origin = _origin(server["endpoint"])
        rows.append(dict(server=server["id"], provider=provider_id(server["id"]),
                         base_url=origin + "/v1", native_profile=NATIVE_PROFILE))
    document = dict(schema_version=SCHEMA_VERSION, owner=OWNER,
                    hem_root=_root(root), servers=rows)
    validate_endpoints(document, root)
    return document


def validate_endpoints(document, root):
    """Validate complete owned envelope; never accept/adopt unknown fields."""
    try:
        if (not isinstance(document, dict) or
                set(document) != {"schema_version", "owner", "hem_root", "servers"} or
                type(document["schema_version"]) is not int or document["schema_version"] != SCHEMA_VERSION or
                document["owner"] != OWNER or document["hem_root"] != _root(root) or
                not isinstance(document["servers"], list)):
            raise ValueError()
        names, profiles = [], set()
        for row in document["servers"]:
            if (not isinstance(row, dict) or set(row) != {"server", "provider", "base_url", "native_profile"} or
                    row["provider"] != provider_id(row["server"]) or row["native_profile"] not in READABLE_NATIVE_PROFILES or
                    not row["base_url"].endswith("/v1") or
                    _origin(row["base_url"][:-3]) + "/v1" != row["base_url"]):
                raise ValueError()
            names.append(row["server"])
            profiles.add(row["native_profile"])
        if names != sorted(set(names)) or len(profiles) > 1:
            raise ValueError()
        _encoded(document)
    except (ValueError, TypeError, KeyError, AttributeError, InvalidInput) as error:
        raise InvalidInput(f"client endpoint projection ownership/schema invalid; {REPAIR}") from error
    return document


def _unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate JSON field")
        value[key] = item
    return value


def read_endpoints(root):
    """Passive bounded read. Absent returns None; foreign/corrupt never adopted."""
    root = Path(_root(root))
    previous = read_owned(root / ENDPOINT_PATH)
    if previous is None:
        return None
    if len(previous) > MAX_BYTES:
        raise InvalidInput(f"client endpoint projection exceeds 1 MiB; {REPAIR}")
    try:
        document = json.loads(previous, object_pairs_hook=_unique_object)
    except (ValueError, UnicodeError, RecursionError) as error:
        raise InvalidInput(f"malformed client endpoint projection; {REPAIR}") from error
    return validate_endpoints(document, root)


def _publish(registry, deadline, *, create):
    path = registry.root / ENDPOINT_PATH
    # Passive fast path: no clients directory or root lock for unconfigured roots.
    if not create and not path.exists() and not path.is_symlink():
        return dict(changed=False, clients_update="unchanged")
    with integration_lock(registry.root, deadline):
        previous = read_owned(path)
        if previous is None and not create:
            return dict(changed=False, clients_update="unchanged")
        if previous is not None:
            prior = read_endpoints(registry.root)
            if not create and any(row["native_profile"] != NATIVE_PROFILE for row in prior["servers"]):
                raise InvalidInput("client endpoint projection needs explicit native-profile refresh; " + REPAIR)
        data = _encoded(project_endpoints(registry.root, registry.list_servers()))
        changed = data != previous
        if changed:
            atomic_replace(path, data, expected=previous, deadline=deadline)
        return dict(changed=changed, clients_update="updated" if changed else "unchanged")


def ensure_endpoints(registry, deadline):
    """Configure-time opt-in; caller must hold no integration/target lock.

    Foreign/corrupt files are retained and refused, including on configure.
    Repair requires the user to inspect/move a conflict before retry; never
    delete a projection at disable because another client may consume it.
    """
    return _publish(registry, deadline, create=True)


def publish_endpoints(registry, deadline):
    """Refresh only an existing owned projection, with no HTTP/client I/O."""
    return _publish(registry, deadline, create=False)


def update_result(registry, deadline, payload):
    """Secondary failures preserve the completed native result and exit code."""
    try:
        update = publish_endpoints(registry, deadline)
        if update["changed"] or (registry.root / ENDPOINT_PATH).exists():
            payload["clients_update"] = update["clients_update"]
    except (HemError, OSError) as error:
        payload.update(clients_update="failed", clients_error=str(error), clients_repair=REPAIR)
        hem_display.emit(f"Client endpoint update failed: {error}; native result unchanged. Repair: {REPAIR}")
    return payload
