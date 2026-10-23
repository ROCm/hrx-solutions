"""Coordinate owned WebUI updates; remote secrets never leave process memory.

The local integration lock serializes our writers only. WebUI has no CAS: an
administrator must serialize connection edits. Pending stores owned intent,
never a replayable whole-configuration request or foreign credentials.
"""
from __future__ import annotations

from pathlib import Path
from copy import deepcopy
import json
import os
import re
import shlex
import uuid
from urllib.parse import quote
from contextlib import nullcontext

from ...hem_common import DeadlineExpired, HemError, InvalidInput, UncertainOutcome
from .hem_openwebui_http import QUALIFIED_VERSIONS, WebUIClient, canonical_target
from .hem_openwebui_state import decode_document, desired_connections, ownership_path, plan_change
from ...hem_clients_endpoints import ensure_endpoints, project_endpoints, read_endpoints
from ..pi.hem_pi_state import integration_lock, passive_snapshot
from ...hem_state import Registry, _name
from ..pi.hem_pi_contracts import provider_id
from ... import hem_display
from ... import hem_operation_targets as operation_targets
from . import hem_openwebui_stt as stt_policy
from . import hem_openwebui_state as state
from .hem_openwebui_wire import validate_audio, strict_equal as same
from .hem_openwebui_receipts import (CLEAR, fail, projection, desired, create_body, comparable, created_row)
from ..pi import hem_pi_files as files


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


GUIDANCE = ('Refresh model list in WebUI Admin Settings → Connections after model changes. '
            'Adding a saved server requires configure, then refresh. '
            'Persistent configuration must be enabled without startup resets, and global '
            'user-info forwarding must be disabled; these prerequisites are unverified. '
            'Disable requires reachable WebUI and an administrator credential. '
            'Export/backups are your responsibility.')


def resolve_target(override=None):
    value = override if override is not None else os.environ.get('OPENWEBUI_URL')
    if value is None:
        raise InvalidInput('Supply --url or OPENWEBUI_URL for the existing local WebUI management origin')
    return canonical_target(value)


def display_request(target):
    """Display actual routes without expanding credentials or foreign configuration."""
    def display(method, path, intent=None):
        # Routes are fixed adapter operations, never remote-controlled response fields.
        if (method, path) not in {('GET', '/api/version'), ('GET', '/openai/config'),
                                  ('POST', '/openai/config/update'), ('GET', '/api/v1/audio/config'),
                                  ('POST', '/api/v1/audio/config/update'),
                                  ('POST', '/api/v1/models/create'), ('POST', '/api/v1/models/model/delete')} and not (
                method == 'GET' and path.startswith('/api/v1/models/model?id=')):
            raise InvalidInput('Unsupported WebUI management display operation')
        command = ('curl --fail --silent --show-error --request ' + method + ' '
                   + shlex.quote(target + path)
                   + ' --header "Authorization: Bearer ${OPENWEBUI_TOKEN}"')
        if method == 'POST':
            hem_display.emit('# WebUI API: REDACTED, NON-REPLAYABLE curl template')
            hem_display.emit(command + " --header 'Content-Type: application/json'"
                             + " --data-binary '<redacted whole configuration>'")
            hem_display.emit('# Read/merge/recheck before update; preserved provider keys omitted. '
                             'Do not submit the owned-only intent as a whole configuration.')
            hem_display.emit('Owned connection intent: ' + json.dumps(intent, ensure_ascii=True, sort_keys=True))
        else:
            hem_display.emit('# WebUI API request; credential supplied from environment')
            hem_display.emit(command)
    return display


def _configure_chat(root, target, deadline, *, check, disable, client, request,
                    expected, projection, previous, document, version=None):
    """Participate in the entry-owned scope; no client, publication or lock owner."""
    mode = "check" if check else "disable" if disable else "configure"
    path = ownership_path(root, target)
    def inspect(document):
        compatibility = "unknown"
        try:
            observed_version = version if version is not None else request("GET", "/api/version", client.version)
            compatibility = "qualified" if observed_version["version"] in QUALIFIED_VERSIONS else "unsupported"
            remote = request("GET", "/openai/config", client.config)
            return observed_version, remote, compatibility, None
        except HemError as error:
            if error.exit_code != 1:
                raise
            service = "unauthorized" if "HTTP 401" in str(error) or "HTTP 403" in str(error) else "offline"
            return None, None, compatibility, _result(target, mode, service=service,
                compatibility=compatibility, pending=document["pending"] is not None,
                completed=False, exit_code=1, findings=[dict(status="management_unavailable")])

    if check:
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


def _configure_audio(root, target, deadline, *, check, disable, client, request,
                     stt_server, stt_instance, previous, document, endpoints, chat_operation,
                     registry, operation_target, qualification_error):
    """Sole audio/picker effect and recovery participant under the entry's lock.

    STT policy supplies desired/CLEAR projections and picker ownership rules.
    Future reviewed TTS policy belongs here in this same merge/write/readback
    sequence; no second audio writer or new operation is introduced now.
    """
    path = ownership_path(root, target)
    record = deepcopy(document.get('stt'))
    saved_record = deepcopy(record)
    selection = state.stt_selection(document)
    saved_selection = deepcopy(selection)
    stt_changed = False
    confirmed_effects = []

    def partial_failure(error):
        if confirmed_effects:
            error.accepted = True
            error.args = (str(error) + '; WebUI ' + ' and '.join(confirmed_effects) +
                          ' confirmed; new audio routing was not submitted',)
        return error

    def pending_inspection_failure(error):
        if (qualification_error is not None and (document['pending'] is not None
                or record is not None and record['pending'] is not None)):
            return partial_failure(UncertainOutcome('WebUI pending operation inspection incomplete; retain intent'))
        return partial_failure(error)

    def qualify():
        if operation_target is not None:
            try:
                operation_targets.revalidate_operation_target(registry, operation_target, deadline)
            except HemError as error:
                if error.reason == 'unavailable' and record['pending'] is None:
                    return deferred(error)
                raise partial_failure(error) from None

    if stt_server is not None and not check:
        selectors = dict(server=stt_server, instance=stt_instance)
        if record is not None and any(record['target'][key] != value for key, value in selectors.items()):
            fail('Disable the existing STT target before selecting another')
        selection = selectors
        if operation_target is not None:
            # UI alone projects native identity into its existing receipt.
            selected = dict(server=operation_target.server_id, instance=operation_target.instance_name,
                base_url=operation_target.origin + '/v1',
                prefixed_id=provider_id(operation_target.server_id) + '.' + operation_target.instance_name)
            if record is not None and not same(selected, record['target']):
                fail('Disable the existing STT target before selecting another')
            if record is None:
                record = dict(target=selected, routing=None, overlay=None, pending=None)
            selection = dict(server=stt_server, instance=stt_instance)
    try:
        version = request('GET', '/api/version', client.version)
        if version['version'] not in QUALIFIED_VERSIONS and not check:
            fail('WebUI version is not qualified for mutation; inspect with --check')
        if document['deployment_id'] and document['deployment_id'] != version['deployment_id']:
            fail('WebUI deployment identity conflict')
    except HemError as error:
        raise pending_inspection_failure(error) from None
    if not check and stt_server is not None and document['pending'] is not None:
        try:
            remote = request('GET', '/openai/config', client.config)
            state._remote(remote)
            recovered, recovery = state._recover_pending(remote, document, version['deployment_id'])
        except HemError as error:
            raise pending_inspection_failure(error) from None
        if recovery != 'completed':
            raise UncertainOutcome('WebUI chat operation unconfirmed; retain intent without replay')
        try:
            _replace_before_submission(path, files.encoded(recovered), expected=previous, deadline=deadline)
        except (HemError, OSError):
            raise UncertainOutcome('WebUI prior chat update confirmed but local finalization incomplete; retain intent',
                                   accepted=True) from None
        document = recovered
        previous = files.read_owned(path)
        confirmed_effects.append('prior chat update')
    if stt_server is not None and check:
        selectors = dict(server=stt_server, instance=stt_instance)
        if not same(selectors, selection):
            return _result(target, 'check', configuration='absent', completed=False, exit_code=1,
                           findings=[dict(status='stt_intent_absent')])
    def audio_get():
        return validate_audio(request('GET', '/api/v1/audio/config', client.audio_config))

    def model_get(model_id):
        value = request('GET', '/api/v1/models/model?id=' + quote(model_id, safe=''), lambda: client.model(model_id))
        if value is not None and (type(value) is not dict or value.get('id') != model_id or value.get('write_access') is not True):
            fail('WebUI model lookup lacks matching identity/write access')
        return value

    def persist(receipt, *, confirmed=False):
        nonlocal previous, document, saved_record, saved_selection, stt_changed
        try:
            current = files.read_owned(path)
            document = state.decode_document(current, root, target)
            if (not same(document.get('stt'), saved_record)
                    or not same(state.stt_selection(document), saved_selection)):
                fail('WebUI local STT ownership changed before update')
            # The existing chat coordinator may just have finalized its own receipt.
            document = state.with_stt_selection(document, selection)
            document.update(stt=deepcopy(receipt), deployment_id=version['deployment_id'])
            state._validate_document(document, root, target)
        except HemError as error:
            raise partial_failure(error)
        try:
            _replace_before_submission(path, files.encoded(document), expected=current, deadline=deadline)
        except (HemError, OSError) as error:
            if confirmed:
                raise UncertainOutcome('WebUI STT remote outcome observed but local finalization incomplete; retain intent', accepted=True) from None
            if isinstance(error, HemError):
                raise partial_failure(error)
            raise
        try:
            previous = files.read_owned(path)
        except HemError as error:
            raise partial_failure(error)
        saved_record = deepcopy(receipt)
        saved_selection = deepcopy(selection)
        stt_changed = True

    def chat_preflight():
        try:
            remote_chat = request('GET', '/openai/config', client.config)
            plan = state.plan_change(remote_chat, document, state.desired_connections(endpoints),
                                    mode='disable' if disable else 'configure', deployment_id=version['deployment_id'])
            if plan.configuration == 'drift' or plan.recovery == 'conflict':
                fail('WebUI chat connection conflict; inspect before configuring STT')
        except HemError as error:
            raise partial_failure(error)

    def save_selection():
        # Names confer no live authority. Preserve the prior receipt until an
        # actual effect's own intent/readback changes it.
        if document['schema_version'] != 3 or not same(saved_selection, selection):
            persist(saved_record)

    def deferred(error):
        if error.reason not in ('missing', 'unavailable'):
            raise partial_failure(error)
        reason = ('missing_saved_server' if selection['server'] not in {r['server'] for r in endpoints['servers']}
                  else 'missing_instance' if error.reason == 'missing' else 'unavailable')
        try:
            chat = chat_operation(check=False, disable=False, version=None)
        except HemError as error:
            raise partial_failure(error)
        chat['stt'] = dict(selection=deepcopy(selection),
            configuration='deferred' if chat.exit_code == 0 else 'unknown',
            reason=reason if chat.exit_code == 0 else 'configuration_failed')
        if saved_record is not None:
            chat['stt']['target'] = deepcopy(saved_record['target'])
        chat['changed'] = chat['changed'] or stt_changed
        if confirmed_effects:
            chat['accepted'] = True
        return chat

    if record is None:
        if check:
            chat = chat_operation(check=True, disable=False, version=version)
            if selection is not None:
                chat['stt'] = dict(selection=deepcopy(selection), configuration='deferred', reason='routing_not_applied')
                chat.update(configuration='partial', state='partial', completed=False)
                chat.exit_code = max(chat.exit_code, 1)
            return chat
        if disable:
            return chat_operation(check=False, disable=True, version=version)
        chat_preflight()
        save_selection()
        if qualification_error is not None:
            return deferred(qualification_error)
        raise UncertainOutcome('WebUI selected STT target has no fresh operation proof')

    try:
        audio = audio_get()
        model_id = record['target']['prefixed_id']
        row = model_get(model_id)
    except HemError as error:
        raise pending_inspection_failure(error) from None

    def stage(operation, intent):
        record['pending'] = dict(operation=operation, intent=deepcopy(intent), outcome='unconfirmed')
        persist(record)

    def observed_pending():
        pending = record['pending']
        if pending is None:
            return True
        op = pending['operation']
        if op == 'create_overlay' and row is not None:
            nonce = pending['intent']['meta']['loom_control_stt']['token']
            receipt = created_row(row, record['target'], nonce)
            record['overlay'] = dict(ownership='created', nonce=nonce, row=receipt)
        elif op in ('configure_audio', 'clear_audio') and same(projection(audio), pending['intent']):
            record['routing'] = deepcopy(pending['intent']) if op == 'configure_audio' else None
        elif op == 'delete_overlay' and row is None:
            record['overlay'] = None
        else:
            return False
        record['pending'] = None
        return True

    recovered_operation = record['pending']['operation'] if record['pending'] is not None else None
    had_pending = recovered_operation is not None
    if not observed_pending():
        if check:
            return _result(target, 'check', configuration='pending', pending=True, completed=False,
                exit_code=1, findings=[dict(status='stt_unconfirmed_no_replay')])
        raise partial_failure(UncertainOutcome('WebUI STT operation unconfirmed; inspect/operator reconcile before retrying'))
    if check and had_pending:
        return _result(target, 'check', configuration='pending', pending=True, completed=False,
            exit_code=1, findings=[dict(status='stt_observed_completed_not_finalized')])
    if had_pending:
        persist(record, confirmed=True)
        confirmed_effects.append({'create_overlay': 'prior overlay creation',
            'configure_audio': 'prior audio routing update', 'clear_audio': 'prior audio clearing',
            'delete_overlay': 'prior overlay deletion'}[recovered_operation])

    def overlay_matches():
        overlay = record['overlay']
        if overlay is None:
            return True
        if overlay['ownership'] == 'created':
            return row is not None and same(comparable(row), overlay['row'])
        return row is not None and row.get('base_model_id') is None and row.get('is_active') is False

    routing_matches = record['routing'] is not None and same(projection(audio), record['routing'])
    if not overlay_matches() or record['routing'] is not None and not routing_matches:
        raise partial_failure(InvalidInput('WebUI owned STT routing/model row drift; preserve evidence'))
    if check:
        chat = chat_operation(check=True, disable=False, version=version)
        configured = routing_matches and record['overlay'] is not None and record['pending'] is None
        chat['stt'] = dict(configuration='configured' if configured else 'partial',
            selection=deepcopy(selection), reason=None,
            target=deepcopy(record['target']), overlay_ownership=(record['overlay'] or {}).get('ownership'))
        if not configured:
            chat.update(configuration='partial', state='partial', completed=False)
            chat.exit_code = max(chat.exit_code, 1)
        return chat

    # All chat/audio/model conflicts are checked before creating remote state.
    chat_preflight()
    if not disable:
        try:
            if record['routing'] is None and not stt_policy.pristine(audio):
                fail('Foreign STT provider/configuration conflicts; no automatic replacement')
            stt_policy.upload_prerequisites(audio)
            if record['overlay'] is None and row is not None and (
                    row.get('write_access', True) is not True or row.get('base_model_id') is not None
                    or row.get('is_active') is not False):
                fail('WebUI selected model has an active/custom foreign override')
        except HemError as error:
            raise partial_failure(error)
        save_selection()
        if qualification_error is not None:
            return deferred(qualification_error)
        if record['overlay'] is None and row is not None:
            deferred_result = qualify()
            if deferred_result is not None:
                return deferred_result
            record['overlay'] = dict(ownership='external', id=model_id)
        if record['overlay'] is None:
            nonce = uuid.uuid4().hex; body = create_body(record['target'], nonce)
            deferred_result = qualify()
            if deferred_result is not None:
                return deferred_result
            stage('create_overlay', body)
            if model_get(model_id) is not None:
                fail('WebUI model appeared before create; pending evidence retained')
            qualify()
            try:
                request('POST', '/api/v1/models/create', lambda: client.create_model(body), body)
            except HemError:
                pass  # Readback, not the status code, establishes the outcome.
            try:
                row = model_get(model_id)
                if row is None:
                    raise UncertainOutcome('WebUI overlay creation unconfirmed; no automatic replay')
                record['overlay'] = dict(ownership='created', nonce=nonce,
                    row=created_row(row, record['target'], nonce))
            except HemError:
                raise UncertainOutcome('WebUI overlay create readback incomplete/conflicting; retain intent') from None
            record['pending'] = None; persist(record, confirmed=True)
            confirmed_effects.append('overlay creation')
        elif document.get('stt') is None:
            persist(record)
        # Overlay creation/finalization is a mutation transition: requalify the
        # deployment before the chat participant can write its own receipt.
        try:
            chat = chat_operation(check=False, disable=False, version=None)
        except HemError as error:
            raise partial_failure(error)
        if chat.exit_code:
            return chat
        if chat.get('accepted') is True:
            confirmed_effects.append('chat update')
        wanted = desired(record['target'])
    else:
        chat = None
        wanted = CLEAR if record['routing'] is not None else None
    if wanted is not None and not same(projection(audio), wanted):
        if not disable:
            deferred_result = qualify()
            if deferred_result is not None:
                return deferred_result
        stage('clear_audio' if disable else 'configure_audio', wanted)
        immediate = audio_get()
        if not same(immediate, audio):
            fail('WebUI audio changed before submission; pending intent retained')
        if not disable:
            qualify()
        body = dict(tts=deepcopy(audio['tts']), stt=dict(deepcopy(audio['stt']), **wanted))
        try:
            request('POST', '/api/v1/audio/config/update', lambda: client.update_audio(body),
                    dict(operation=record['pending']['operation'], routing=wanted))
        except HemError:
            pass  # Upserts may commit before a route error.
        try:
            actual = audio_get()
        except HemError:
            raise UncertainOutcome('WebUI audio update readback incomplete; retain intent') from None
        expected = dict(deepcopy(audio), stt=body['stt'])
        if not same(actual, expected):
            raise UncertainOutcome('WebUI audio update unconfirmed/conflicting; no replay; inspect') from None
        audio = actual
        record['routing'] = None if disable else wanted
        record['pending'] = None; persist(record, confirmed=True)
    elif wanted is not None and record['routing'] is None and not disable:
        # Matching foreign routing without our submitted intent is not ownership.
        fail('Unowned matching STT routing is not eligible for adoption')
    if disable:
        overlay = record['overlay']
        if overlay is not None and overlay['ownership'] == 'created':
            stage('delete_overlay', overlay['row'])
            current = model_get(model_id)
            if current is None or not same(comparable(current), overlay['row']):
                fail('WebUI model changed before deletion; retain evidence')
            try:
                request('POST', '/api/v1/models/model/delete', lambda: client.delete_model(model_id), dict(id=model_id))
            except HemError:
                pass  # Only readback can prove the current outcome.
            if model_get(model_id) is not None:
                raise UncertainOutcome('WebUI model deletion unconfirmed; no automatic replay')
            record['overlay'] = None; record['pending'] = None; persist(record, confirmed=True)
        # External prerequisite is never deleted/adopted; no remaining authority.
        record = None; persist(record, confirmed=True)
        # Audio/overlay effects may cross a deployment transition; observe again.
        try:
            chat = chat_operation(check=False, disable=True, version=None)
        except HemError as error:
            raise partial_failure(error)
    chat['stt'] = dict(configuration='absent' if disable else 'configured',
        selection=None if disable and chat.exit_code == 0 and chat['completed'] else deepcopy(selection), reason=None,
        target=None if disable else deepcopy(record['target']), backend_reachability='operator-required/unverified',
        disconnect='inert-openai/blank-origin-no-provider-fallback' if disable else None)
    chat['changed'] = chat['changed'] or stt_changed
    return chat


def configure_openwebui(root, target, token, deadline, *, check=False, disable=False,
                        client_factory=WebUIClient, display=None,
                        stt_server=None, stt_instance=None) -> OpenWebUIResult:
    """Own complete bounded WebUI setup for CLI and direct Python callers.

    Explicit target/token override their environment defaults. Display callbacks
    receive only fixed routes and owned intent, never whole remote configuration.
    """
    if check and disable:
        raise InvalidInput("check and disable are mutually exclusive")
    if (stt_server is None) != (stt_instance is None) or disable and stt_server is not None:
        raise InvalidInput('Supply both --stt-server and --stt-instance; omit selectors with --disable')
    if stt_server is not None:
        _name(stt_server); _name(stt_instance)
    target = resolve_target(target)
    token = token if token is not None else os.environ.get('OPENWEBUI_TOKEN')
    if not token:
        raise InvalidInput('Supply an administrator bearer credential in OPENWEBUI_TOKEN; no token flag or automatic login')
    if not isinstance(token, str) or re.fullmatch(r"[A-Za-z0-9._~+/=-]+", token) is None:
        raise InvalidInput('OPENWEBUI_TOKEN must contain a nonempty bearer credential')
    value = Path(root if root is not None else os.environ.get('LOOM_SERVE_DIR', str(Path.home() / '.loom')))
    if not value.is_absolute():
        raise InvalidInput('LOOM_SERVE_DIR must be absolute')
    files.private_path(value)
    root = value.resolve()
    deadline.remaining()
    try:
        response = _configure_owned(root, target, token, deadline, check=check, disable=disable,
            client_factory=client_factory, display=display if display is not None else display_request(target),
            stt_server=stt_server, stt_instance=stt_instance)
    except HemError as error:
        # Direct callers receive the same credential-safe diagnostics as the CLI.
        error.args = (str(error).replace(token, '<redacted>'),)
        raise
    response['guidance'] = GUIDANCE
    hem_display.emit(GUIDANCE)
    return response


def _configure_owned(root, target, token, deadline, *, check, disable,
                     client_factory, display, stt_server, stt_instance):
    path = ownership_path(root, target)
    previous = files.read_owned(path)
    document = decode_document(previous, root, target)
    prior_selection = state.stt_selection(document)
    selected_names = (dict(server=stt_server, instance=stt_instance) if stt_server is not None
                      else None if disable else prior_selection)
    if not check and selected_names is not None:
        stt_server, stt_instance = selected_names['server'], selected_names['instance']
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


    registry = None
    operation_target = None
    qualification_error = None
    if not check:
        # Publication and initial server-locked selection precede integration.
        registry = Registry(root, deadline=deadline)
        ensure_endpoints(registry, deadline)
        if stt_server is not None:
            if (not same(selected_names, prior_selection)
                    and not any(row['id'] == stt_server for row in registry.list_servers())):
                raise InvalidInput('Selected STT server must already be saved')
            try:
                operation_target = operation_targets.resolve_operation_target(
                    registry, stt_server, stt_instance, 'transcription', deadline)
            except InvalidInput:
                raise
            except HemError as error:
                # Native proof is independent of names-only intent. The owner
                # admits selection only after applicable preflight/recovery.
                qualification_error = error
    try:
        with nullcontext() if check else integration_lock(root, deadline):
            if not check:
                previous = files.read_owned(path)
                document = decode_document(previous, root, target)
                if not same(state.stt_selection(document), prior_selection):
                    fail('WebUI local STT selection changed while waiting for the integration lock')
            expected = project_endpoints(root, passive_snapshot(root, deadline)['servers'])
            projection = read_endpoints(root)
            if not check and not _same(projection, expected):
                raise InvalidInput("WebUI endpoint projection changed; rerun configure")
            if (not check and not disable and selected_names is not None
                    and not same(selected_names, prior_selection)
                    and stt_server not in {row['server'] for row in expected['servers']}):
                raise InvalidInput('Selected STT server must already be saved')

            def chat_operation(*, check, disable, version):
                # Audio/picker effects may have finalized their receipt since entry.
                current = files.read_owned(path)
                owned = decode_document(current, root, target)
                return _configure_chat(root, target, deadline, check=check, disable=disable,
                    client=client, request=request, expected=expected, projection=projection,
                    previous=current, document=owned, version=version)

            if stt_server is not None or document.get('stt') is not None or state.stt_selection(document) is not None:
                return _configure_audio(root, target, deadline, check=check, disable=disable,
                    client=client, request=request, stt_server=stt_server, stt_instance=stt_instance,
                    previous=previous, document=document, endpoints=expected, chat_operation=chat_operation,
                    registry=registry, operation_target=operation_target, qualification_error=qualification_error)
            return _configure_chat(root, target, deadline, check=check, disable=disable,
                client=client, request=request, expected=expected, projection=projection,
                previous=previous, document=document)
    except DeadlineExpired as error:
        pending = (document.get('stt') or {}).get('pending') or document.get('pending')
        if stt_server is not None and pending is not None and error.accepted is not True:
            raise UncertainOutcome('WebUI pending outcome remains unresolved within the original deadline; retain intent') from None
        raise
