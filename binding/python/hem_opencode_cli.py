"""One optional OpenCode 2 setup command; no client lifecycle ownership."""
from __future__ import annotations

import fnmatch
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import stat
import subprocess

from . import hem_arguments as arguments, hem_display
from .hem_clients_endpoints import ensure_endpoints, project_endpoints, read_endpoints
from .hem_common import HemError, InvalidInput, UncertainOutcome, DeadlineExpired
from .hem_opencode import (OpenCodeResult, bundled_assets, configure_opencode, document,
                           MAX_BYTES, REPAIR)
from .hem_pi_files import private_path
from .hem_pi_state import passive_snapshot
from .hem_state import Registry

VERSION = "2.0.31"
# Genuine qualified standalone/native npm binary, linux-x64, no Node requirement.
QUALIFIED_SHA256 = "2df936a20cf74af5706020342164628f9a31978c50c2b0ad306b42b614fe245d"


def configure(parser):
    commands = parser.add_subparsers(dest="command", required=True)
    child = commands.add_parser("configure", help="configure or inspect OpenCode 2 integration")
    arguments.output(child)
    arguments.lifecycle(child)
    mode = child.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="inspect without writes or client execution")
    mode.add_argument("--disable", action="store_true", help="remove unchanged owned files without client")
    child.add_argument("--opencode-dir", type=Path, help="alternate OpenCode configuration directory")


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


def _jsonc(text):
    """Read-only JSONC subset: comments and trailing commas, preserving strings."""
    out, index, quoted = [], 0, False
    while index < len(text):
        char = text[index]
        if quoted:
            out.append(char)
            if char == "\\" and index + 1 < len(text):
                index += 1; out.append(text[index])
            elif char == '"': quoted = False
        elif char == '"': quoted = True; out.append(char)
        elif text[index:index + 2] == "//":
            end = text.find("\n", index + 2)
            index = len(text) if end < 0 else end
            continue
        elif text[index:index + 2] == "/*":
            end = text.find("*/", index + 2)
            if end < 0: raise ValueError("unterminated comment")
            index = end + 2; out.append(" "); continue
        elif char == ",":
            tail = index + 1
            while tail < len(text) and text[tail].isspace(): tail += 1
            if tail == len(text) or text[tail] not in "}]": out.append(char)
        else: out.append(char)
        index += 1
    return document("".join(out).encode())


def _config_read(path):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_BYTES:
            raise ValueError("nonregular/oversized config")
        with os.fdopen(fd, "rb", closefd=False) as stream: data = stream.read(MAX_BYTES + 1)
        if len(data) > MAX_BYTES: raise ValueError("oversized config")
        return _jsonc(data.decode("utf8"))
    finally: os.close(fd)


def policy_findings(target, providers):
    """Detect explicit suppressions only; never reproduce the host policy engine."""
    paths = [target / name for name in ("opencode.json", "opencode.jsonc")]
    if os.environ.get("OPENCODE_CONFIG_PROJECT_DISABLE") != "1":
        cwd = Path.cwd()
        for parent in (cwd, *cwd.parents):
            paths.extend(parent / directory / name for directory in ("", ".opencode")
                         for name in ("opencode.json", "opencode.jsonc"))
    if os.environ.get("OPENCODE_CONFIG"):
        paths.append(Path(os.environ["OPENCODE_CONFIG"]).expanduser())
    values, warnings, conflicts = [], [], []
    for path in dict.fromkeys(paths):
        try:
            values.append((str(path), _config_read(path)))
        except FileNotFoundError: pass
        except (OSError, ValueError, UnicodeError, InvalidInput):
            warnings.append("Configuration could not be safely inspected: " + str(path))
    inline = os.environ.get("OPENCODE_CONFIG_CONTENT")
    if inline:
        try: values.append(("OPENCODE_CONFIG_CONTENT", _jsonc(inline)))
        except (ValueError, InvalidInput): warnings.append("Inline OpenCode policy could not be inspected")
    for source, value in values:
        enabled, disabled = value.get("enabled_providers"), value.get("disabled_providers")
        if isinstance(enabled, list) and any(provider not in enabled for provider in providers):
            conflicts.append("Provider allowlist may suppress Loom providers: " + source)
        if isinstance(disabled, list) and any(provider in disabled for provider in providers):
            conflicts.append("Provider denylist suppresses Loom providers: " + source)
        policies = value.get("policies", [])
        experimental = value.get("experimental", {})
        if isinstance(experimental, dict) and isinstance(experimental.get("policies"), list):
            policies = (policies if isinstance(policies, list) else []) + experimental["policies"]
        if isinstance(policies, list):
            for rule in policies:
                if (isinstance(rule, dict) and rule.get("action") == "provider.use" and
                    rule.get("effect") == "deny" and isinstance(rule.get("resource"), str) and
                    any(fnmatch.fnmatchcase(provider, rule["resource"]) for provider in providers)):
                    conflicts.append("Explicit policy may suppress Loom providers: " + source)
            if policies: warnings.append("Effective policy precedence is evaluated by OpenCode; inspect visibility after launch: " + source)
    return sorted(set(conflicts)), sorted(set(warnings))


def run(args, deadline):
    root_value = Path(os.environ.get("LOOM_SERVE_DIR", str(Path.home() / ".loom")))
    if not root_value.is_absolute(): raise InvalidInput("LOOM_SERVE_DIR/state root must be absolute")
    private_path(root_value)
    root, target = root_value.resolve(), resolve_target(args.opencode_dir)
    deadline.remaining()
    runtime = None
    compatibility = dict(state="not_checked", reason="client unnecessary for owned-file removal")
    if not args.disable:
        try:
            runtime = inspect_runtime(deadline, passive=args.check)
            compatibility = dict(state="compatible", reason=runtime["qualification"])
        except (HemError, OSError) as error:
            if not args.check: raise
            compatibility = dict(state="unqualified", reason=str(error))
    bundled_assets()  # Missing wheel assets fail before state/target writes.
    conflicts, warnings = [], []
    projection = None
    if not args.disable:
        snapshot = passive_snapshot(root, deadline)
        projection = project_endpoints(root, snapshot["servers"])
        if len(projection["servers"]) > 256:
            raise InvalidInput("OpenCode supports at most 256 candidate endpoints")
        conflicts, warnings = policy_findings(target, [row["provider"] for row in projection["servers"]])
        if conflicts and not args.check:
            raise InvalidInput("; ".join(conflicts) + "; export/back up configuration and resolve policy yourself; no configuration changed")
    hem_display.emit("No native Loom equivalent: owned OpenCode integration file configuration")
    try:
        if not args.check and not args.disable:
            preview = configure_opencode(root, target, deadline, check=True)
            if preview.get("state") == "conflict":
                raise InvalidInput("; ".join(preview["findings"]))
            registry = Registry(root, deadline=deadline)
            ensure_endpoints(registry, deadline)
        response = configure_opencode(root, target, deadline, check=args.check, disable=args.disable)
    except (OSError, DeadlineExpired) as error:
        if args.check: raise HemError("OpenCode inspection failed without writes", accepted=False) from error
        raise UncertainOutcome("OpenCode update interrupted; " + REPAIR) from error
    response.update(runtime=runtime, runtime_compatibility=compatibility, policy_conflicts=conflicts,
                    policy_warnings=warnings, availability=dict(state="not_observed", reason="models are discovered by the client on startup or explicit reload"))
    if args.check:
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
        if compatibility["state"] != "compatible" or conflicts:
            response.exit_code = response.exit_code or 1
    if not args.disable:
        recipe = "OPENCODE_CONFIG_DIR=" + shlex.quote(str(target)) + " opencode"
        response.update(launch=recipe, refresh=recipe + " reload --server http://127.0.0.1:PORT",
            guidance="Startup or explicit reload of the correct OpenCode server refreshes models; restart is the fallback. This directory does not retarget an already-running server. Credentials, settings/defaults and other plugins are unchanged. Export/backups are your responsibility.")
        hem_display.emit("Launch: " + recipe)
        hem_display.emit("Refresh the correct OpenCode server using its public reload command, or restart. No background polling.")
        for warning in warnings: hem_display.emit(warning)
    return response
