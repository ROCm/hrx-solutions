"""Coordinate owned WebUI updates; remote secrets never leave process memory.

The local integration lock serializes our writers only. WebUI has no CAS: an
administrator must serialize connection edits. Pending stores owned intent,
never a replayable whole-configuration request or foreign credentials.
"""
from __future__ import annotations

from pathlib import Path

from .hem_common import DeadlineExpired, HemError, InvalidInput, UncertainOutcome
from .hem_openwebui_http import QUALIFIED_VERSIONS, WebUIClient, canonical_target
from .hem_openwebui_state import decode_document, desired_connections, ownership_path, plan_change
from .hem_clients_endpoints import ensure_endpoints, project_endpoints, read_endpoints
from .hem_pi_state import integration_lock, passive_snapshot
from .hem_state import Registry
from . import hem_pi_files as files


class OpenWebUIResult(dict):
    def __init__(self, *args, exit_code=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.exit_code = exit_code


def _result(target, mode, *, configuration="unknown", service="reachable",
            compatibility="unknown", changed=False, pending=False, findings=None,
            completed=True, exit_code=0, **extra):
    return OpenWebUIResult(state=configuration, completed=completed,
        target_url=target, mode=mode, changed=changed, configuration=configuration,
        service=service, compatibility=compatibility, model_availability="unprobed",
        prerequisites={"persistence": "operator-required/unverified",
                       "user_info_forwarding_disabled": "operator-required/unverified"},
        pending=pending, findings=findings or [], exit_code=exit_code, **extra)


def _same(left, right):
    # Canonical strict JSON comparison also distinguishes bool from numeric 1.
    return files.encoded(left) == files.encoded(right)


def _replace_before_submission(path, data, *, expected, deadline):
    try:
        files.atomic_replace(path, data, expected=expected, deadline=deadline, allow_prefix=True)
    except OSError:
        # Disk/permission exceptions may contain arbitrary filenames or bytes.
        # No remote mutation was submitted by this invocation.
        raise HemError("WebUI local ownership update failed; retain evidence and inspect before retrying",
                       accepted=False) from None


def configure_openwebui(root: Path, target: str, token: str, deadline, *,
                        check=False, disable=False, client_factory=WebUIClient,
                        display=None) -> OpenWebUIResult:
    """Inspect passively or execute one bounded owned merge under a local lock.

    ``display(method, path, intent=None)`` receives only routes and nonsecret
    affected-owned records. It must never be passed a whole remote config.
    """
    if check and disable:
        raise InvalidInput("check and disable are mutually exclusive")
    root = Path(root)
    if not root.is_absolute() or str(root.resolve()) != str(root):
        raise InvalidInput("WebUI state root must be canonical and absolute")
    files.private_path(root)
    target = canonical_target(target)
    mode = "check" if check else "disable" if disable else "configure"
    path = ownership_path(root, target)
    client = client_factory(target, token, deadline)

    def request(method, route, operation, intent=None):
        try:
            deadline.remaining()
        except DeadlineExpired:
            if method == "POST":
                raise DeadlineExpired("WebUI management deadline expired before submission", accepted=False) from None
            raise
        if display is not None:
            display(method, route, intent)
        return operation()

    def inspect(document):
        compatibility = "unknown"
        try:
            version = request("GET", "/api/version", client.version)
            compatibility = "qualified" if version["version"] in QUALIFIED_VERSIONS else "unsupported"
            remote = request("GET", "/openai/config", client.config)
            return version, remote, compatibility, None
        except HemError as error:
            if error.exit_code != 1:
                raise
            service = "unauthorized" if "HTTP 401" in str(error) or "HTTP 403" in str(error) else "offline"
            return None, None, compatibility, _result(target, mode, service=service,
                compatibility=compatibility, pending=document["pending"] is not None,
                completed=False, exit_code=1, findings=[dict(status="management_unavailable")])

    def inputs():
        snapshot = passive_snapshot(root, deadline)
        expected = project_endpoints(root, snapshot["servers"])
        projection = read_endpoints(root)
        previous = files.read_owned(path)
        document = decode_document(previous, root, target)
        return expected, projection, previous, document

    if check:
        expected, projection, previous, document = inputs()
        version, remote, compatibility, unavailable = inspect(document)
        if unavailable is not None:
            return unavailable
        plan = plan_change(remote, document, desired_connections(expected), mode=mode,
                           deployment_id=version["deployment_id"])
        findings = list(plan.findings)
        configuration = plan.configuration
        if not _same(projection, expected):
            findings.append(dict(status="endpoint_projection_missing_or_changed"))
            if configuration == "configured":
                configuration = "absent"
        conflict = configuration == "drift" or plan.recovery == "conflict"
        code = 2 if conflict else int(configuration != "configured" or compatibility != "qualified")
        return _result(target, mode, configuration=configuration, compatibility=compatibility,
            pending=document["pending"] is not None, findings=findings,
            completed=code == 0, exit_code=code)

    # This publisher acquires its own integration lock. Never nest acquisition.
    registry = Registry(root, deadline=deadline)
    ensure_endpoints(registry, deadline)
    with integration_lock(root, deadline):
        expected, projection, previous, document = inputs()
        if not _same(projection, expected):
            raise InvalidInput("WebUI endpoint projection changed; rerun configure")
        version, remote, compatibility, unavailable = inspect(document)
        if unavailable is not None:
            return unavailable
        if compatibility != "qualified":
            raise InvalidInput("WebUI version is not qualified for mutation; inspect with --check")
        plan = plan_change(remote, document, desired_connections(expected), mode=mode,
                           deployment_id=version["deployment_id"])
        if plan.recovery == "conflict" or plan.configuration == "drift":
            raise InvalidInput("WebUI ownership or connection conflict; retain evidence and inspect with --check")
        if plan.remote_config is None:
            # changed includes identity adoption and resolved pending bookkeeping.
            if plan.next_document is not None and not _same(document, plan.next_document):
                try:
                    _replace_before_submission(path, files.encoded(plan.next_document), expected=previous, deadline=deadline)
                except (HemError, OSError):
                    if document["pending"] is not None:
                        raise UncertainOutcome("WebUI recovery local finalization incomplete; inspect before retrying",
                                               accepted=True if plan.recovery == "completed" else None) from None
                    raise
            return _result(target, mode, configuration=plan.configuration, compatibility=compatibility,
                changed=plan.changed, findings=plan.findings, completed=not plan.partial,
                exit_code=1 if plan.partial else 0)

        # Durable, bounded, nonsecret intent precedes all potentially submitted POSTs.
        staged = files.encoded(plan.pending_document)
        _replace_before_submission(path, staged, expected=previous, deadline=deadline)
        immediate = request("GET", "/openai/config", client.config)
        if not _same(remote, immediate):
            raise InvalidInput("WebUI connections changed before submission; pending intent retained for inspection")
        submitted_error = None
        acknowledged = False
        try:
            request("POST", "/openai/config/update", lambda: client.update(plan.remote_config),
                    plan.pending_document["pending"])
            acknowledged = True
        except HemError as error:
            if error.exit_code != 4:
                # Explicit rejection/pre-submission failure is safe; durable intent
                # remains assessable against current truth on the next invocation.
                raise
            submitted_error = error
            acknowledged = error.accepted is True
        try:
            actual = request("GET", "/openai/config", client.config)
        except HemError:
            raise UncertainOutcome("WebUI update readback incomplete; pending intent retained; inspect before retrying",
                                   accepted=True if acknowledged else None) from None
        if not _same(actual, plan.remote_config):
            if submitted_error is not None and _same(actual, remote):
                # Readback establishes current owned before-state, not that a
                # delayed remote handler cannot still complete. Keep the intent.
                raise UncertainOutcome("WebUI update not confirmed; pending intent retained; inspect before retrying",
                                       accepted=True if acknowledged else None) from None
            raise UncertainOutcome("WebUI update verification conflicts; pending intent retained; inspect before retrying",
                                   accepted=True if acknowledged else None) from None
        try:
            files.atomic_replace(path, files.encoded(plan.next_document), expected=staged, deadline=deadline, allow_prefix=True)
        except (HemError, OSError):
            raise UncertainOutcome("WebUI update verified but local finalization incomplete; inspect before retrying",
                                   accepted=True) from None
        return _result(target, mode, configuration=plan.configuration, compatibility=compatibility,
            changed=True, findings=plan.findings, completed=not plan.partial,
            exit_code=1 if plan.partial else 0, accepted=True)
