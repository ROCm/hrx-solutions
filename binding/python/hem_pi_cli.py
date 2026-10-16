"""One Pi setup command: metadata checks, owned installation and observations."""
from __future__ import annotations

import http.client
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import socket
import stat
import subprocess
import threading
from urllib.parse import urlsplit

from . import hem_arguments as arguments, hem_display
from .hem_common import Deadline, DeadlineExpired, HemError, InvalidInput, UncertainOutcome
from .hem_pi import configure_pi, resolve_target
from .hem_pi_contracts import PI_PACKAGE, PI_VERSIONS, NODE_MIN_VERSION, REFRESH_TIMEOUT, ENDPOINT_PATH
from .hem_pi_assets import bundled_assets
from .hem_pi_state import passive_snapshot
from .hem_pi_endpoints import project_endpoints, _origin

MAX_BYTES = 1024 * 1024
REPAIR = "loom-hem pi configure"


def configure(parser):
    commands = parser.add_subparsers(dest="command", required=True)
    child = commands.add_parser("configure", help="configure or inspect Pi integration")
    arguments.output(child)
    arguments.lifecycle(child)
    mode = child.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="inspect without writing")
    mode.add_argument("--disable", action="store_true", help="remove unchanged owned files")
    child.add_argument("--pi-dir", type=Path, help="alternate Pi agent directory")


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


def _legacy(target, managed_providers=()):
    """Check static provider IDs and report only allowlisted legacy metadata."""
    path = target / "models.json"
    try:
        value = json.loads(_read(path))
    except FileNotFoundError:
        return []
    except (OSError, ValueError, UnicodeError, InvalidInput):
        hem_display.emit("Pi models.json could not be safely inspected; preserved unchanged. Inspect it manually for unmanaged local listings.")
        return []
    providers = value.get("providers", {}) if isinstance(value, dict) else {}
    providers = providers if isinstance(providers, dict) else {}
    collisions = set(managed_providers).intersection(providers)
    if collisions:
        raise InvalidInput("Pi models.json has static provider IDs that conflict with configured Hem servers: " +
            ", ".join(sorted(collisions)) + "; export/back up models.json and rename or remove those static providers before rerunning loom-hem pi configure; no configuration was changed")
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
    hem_display.emit("Unmanaged legacy Pi provider hrx-loom is preserved. Export/back up models.json and check default-model references yourself before manually removing only obsolete hrx-loom entries; Hem does not change settings or credentials.")
    return [dict(provider="hrx-loom", ids=identifiers, endpoint=endpoint,
                 state="unmanaged", reason="availability unknown; manual export and cleanup are the user's responsibility")]


def _document(base, suffix, deadline):
    """Read-only numeric-loopback discovery, bounded by bytes and total time."""
    deadline.remaining()
    parsed = urlsplit(base)
    hem_display.emit(hem_display.http("GET", base + suffix))
    connection = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=deadline.remaining())
    timer = None
    try:
        connection.connect()
        connected = connection.sock
        def interrupt():
            try:
                connected.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        timer = threading.Timer(deadline.remaining(), interrupt)
        timer.daemon = True
        timer.start()
        connection.request("GET", parsed.path + suffix)
        response = connection.getresponse()
        if response.status != 200:
            raise ValueError()
        value = response.read(MAX_BYTES + 1)
        if len(value) > MAX_BYTES:
            raise ValueError()
        deadline.remaining()
        return json.loads(value)
    finally:
        if timer:
            timer.cancel()
        connection.close()


def _rows(document):
    if not isinstance(document, dict) or document.get("object") != "list" or not isinstance(document.get("data"), list):
        raise ValueError()
    result = {}
    for row in document["data"]:
        if not isinstance(row, dict) or row.get("object") != "model" or not isinstance(row.get("id"), str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", row["id"]) or row["id"] in result:
            raise ValueError()
        result[row["id"]] = row
    return result


def _available(catalog, status):
    models, states = _rows(catalog), _rows(status)
    if models.keys() != states.keys():
        raise ValueError()
    available = []
    for name, model in models.items():
        state = states[name]
        fields = lambda row: (row.get("context_length"), row.get("limits"), row.get("input_modalities"), row.get("output_modalities"),
                              {key: row.get("loom", {}).get(key) for key in ("incarnation", "registration", "kind", "endpoint", "tokenize")})
        if not isinstance(model.get("loom"), dict) or not isinstance(state.get("loom"), dict) or json.dumps(fields(model), sort_keys=True) != json.dumps(fields(state), sort_keys=True):
            raise ValueError()
        loom = state["loom"]
        native_status = loom.get("status") or {}
        limits = model.get("limits") or {}
        catalog_status = model["loom"].get("status") or {}
        if not isinstance(catalog_status, dict) or not isinstance(native_status, dict) or not isinstance(limits, dict) or not isinstance(loom.get("parameters", {}), dict):
            raise ValueError()
        positive = lambda value: type(value) is int and 0 < value <= 2**53-1
        if (positive(model.get("context_length")) and positive(limits.get("max_output_tokens")) and
            model.get("input_modalities") == ["text"] and model.get("output_modalities") == ["text"] and
            positive(loom.get("incarnation")) and loom.get("kind") == "text" and loom.get("endpoint") == "/v1/chat/completions" and
            loom.get("tokenize") is True and loom.get("registration") == "ready" and loom.get("parameter_operation") == "idle" and
            loom.get("parameters", {}).get("active") is True and native_status.get("admission") == "open" and
            native_status.get("phase") in ("idle", "execute") and not loom.get("error") and not native_status.get("error") and
            not model["loom"].get("error") and not catalog_status.get("error")):
            available.append(name)
    return available, len(available) != len(models)


def _observations(root, target, deadline):
    snapshot = passive_snapshot(root, deadline)
    identity = snapshot["pi_target"] or dict(target=str(target), integration_id="preview")
    endpoints = project_endpoints(root, identity, snapshot["servers"])["servers"]
    budget = Deadline(min(deadline.expires_at, deadline.clock() + REFRESH_TIMEOUT), clock=deadline.clock)
    observations, available = [], []
    for endpoint in endpoints:
        row = dict(endpoint, state="unavailable", reason="offline or unsupported/inconsistent native inventory; availability unknown, refresh when available")
        try:
            names, withheld = _available(_document(endpoint["base_url"], "/models", budget), _document(endpoint["base_url"], "/status", budget))
            row.update(state="ready" if names else "unavailable" if withheld else "empty", available=names,
                       reason="some entries unavailable or unsupported" if withheld else "native-ready; Pi budgets are checked before inference")
            available.extend(dict(provider=endpoint["provider"], id=name) for name in names)
        except DeadlineExpired:
            row.update(state="unknown", reason="discovery budget exhausted; availability unknown")
        except (HemError, OSError, ValueError, TypeError, http.client.HTTPException):
            pass  # Availability is advisory; never undo a durable installation.
        observations.append(row)
    return observations, available


def run(args, deadline):
    try:
        deadline.remaining()
    except DeadlineExpired as error:
        raise HemError("Pi configuration deadline expired before writes", accepted=False) from error
    configured_root = os.environ.get("LOOM_SERVE_DIR")
    root = Path(configured_root) if configured_root is not None else Path.home() / ".loom"
    if not root.is_absolute():
        raise InvalidInput("LOOM_SERVE_DIR/state root must be absolute")
    root, target = root.resolve(), resolve_target(args.pi_dir)
    managed_providers = []
    try:
        runtime = inspect_runtime(deadline)
        bundled_assets()  # Missing packaged resources are a known pre-write failure.
        if not args.disable:
            snapshot = passive_snapshot(root, deadline)
            # Same encoded-size guard as lifecycle publication, before reservation.
            identity = snapshot["pi_target"] or dict(target=str(target), integration_id="00000000-0000-4000-8000-000000000000")
            projection = project_endpoints(root, identity, snapshot["servers"])
            managed_providers = [row["provider"] for row in projection["servers"]]
    except DeadlineExpired as error:
        raise HemError("Pi compatibility inspection deadline expired before writes", accepted=False) from error
    except OSError as error:
        raise HemError("bundled Pi resources are unavailable; reinstall this wheel before configuring Pi", accepted=False) from error
    legacy = _legacy(target, managed_providers)
    hem_display.emit("No native Loom equivalent: owned Pi integration file configuration")
    try:
        response = configure_pi(root, target, deadline, check=args.check, disable=args.disable)
    except DeadlineExpired as error:
        if error.accepted is True:
            raise
        if args.check:
            raise HemError("Pi inspection deadline expired before any writes", accepted=False) from error
        raise UncertainOutcome("Pi update deadline expired; inspect with loom-hem pi configure --check before retrying") from error
    except OSError as error:
        if args.check:
            raise HemError("Pi inspection failed without writes", accepted=False) from error
        raise UncertainOutcome("Pi update interrupted; inspect with loom-hem pi configure --check before retrying") from error
    response.update(runtime=runtime, legacy=legacy)
    if response.get("affected_paths"):
        response["affected_paths"] = sorted(set(response["affected_paths"]) | {str(root / ENDPOINT_PATH), str(root / "control.sqlite3")})
    for path in response.get("affected_paths", []):
        hem_display.emit("Pi integration path: " + str(path))
    if not args.disable:
        try:
            response["servers"], response["available"] = _observations(root, target, deadline)
        except (HemError, OSError):
            response.update(servers=[], available=[], availability_reason="observation unavailable; refresh Pi when ready")
        hem_display.emit("Pi settings/defaults/auth are unchanged. Launch Pi normally; use /reload or restart for new server names.")
    if response.get("next_action"):
        hem_display.emit("Next: " + response["next_action"])
    return response
