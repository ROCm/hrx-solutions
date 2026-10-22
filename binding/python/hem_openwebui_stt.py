"""One selected WebUI STT target; current owned receipts, never provider history.

Foreign audio/configuration values exist only in memory for lossless carry-through.
The four fixed pending operations are observationally resolved, never replayed.
"""
from __future__ import annotations

from contextlib import nullcontext
from copy import deepcopy
import uuid
from urllib.parse import quote

from .hem_common import HemError, InvalidInput, UncertainOutcome
from . import hem_openwebui_state as state, hem_pi_files as files
from .hem_pi_contracts import provider_id
from .hem_pi_state import integration_lock, passive_snapshot
from .hem_clients_endpoints import ensure_endpoints, project_endpoints
from .hem_state import Registry, _name

ROUTING = frozenset(('ENGINE', 'OPENAI_API_BASE_URL', 'OPENAI_API_KEY', 'OPENAI_API_REQUEST_FORMAT', 'MODEL'))
CLEAR = dict(ENGINE='openai', OPENAI_API_BASE_URL='', OPENAI_API_KEY='', OPENAI_API_REQUEST_FORMAT='multipart', MODEL='')
TTS_STRINGS = frozenset('OPENAI_API_BASE_URL OPENAI_API_KEY API_KEY ENGINE MODEL VOICE SPLIT_ON AZURE_SPEECH_REGION AZURE_SPEECH_BASE_URL AZURE_SPEECH_OUTPUT_FORMAT MISTRAL_API_KEY MISTRAL_API_BASE_URL'.split())
STT_STRINGS = frozenset('OPENAI_API_BASE_URL OPENAI_API_KEY ENGINE MODEL WHISPER_MODEL DEEPGRAM_API_KEY AZURE_API_KEY AZURE_REGION AZURE_LOCALES AZURE_BASE_URL AZURE_MAX_SPEAKERS MISTRAL_API_KEY MISTRAL_API_BASE_URL'.split())
OPTIONAL_TTS = {'OPENAI_PARAMS': (dict, type(None)), 'REALTIME_TTS_PROMPT_TEMPLATE': (str, type(None))}
OPTIONAL_STT = {'OPENAI_API_REQUEST_FORMAT': (str,), 'SUPPORTED_CONTENT_TYPES': (list,), 'ALLOWED_EXTENSIONS': (list,)}
GENERATED = frozenset(('user_id', 'version_id', 'created_at', 'updated_at'))
META_DEFAULTS = frozenset('profile_image_url background_image_url description i18n capabilities knowledge voice voice_avatar'.split())
BODY_KEYS = frozenset(('id', 'base_model_id', 'name', 'meta', 'params', 'access_grants', 'is_active'))


def fail(message='WebUI STT ownership/configuration conflict; retain evidence and inspect'):
    raise InvalidInput(message)


def same(a, b):
    return state._strict(a, b)


def validate_audio(value):
    state._encoded(value)
    if type(value) is not dict or not {'tts', 'stt'} <= value.keys() or value.keys() - {'tts', 'stt', 'realtime'}:
        fail('WebUI audio response has incompatible sections')
    for section, required, optional, extra in (('tts', TTS_STRINGS, OPTIONAL_TTS, {}),
            ('stt', STT_STRINGS, OPTIONAL_STT, {'MISTRAL_USE_CHAT_COMPLETIONS': bool})):
        row = value[section]
        if (type(row) is not dict or not required <= row.keys() or
                row.keys() - required - optional.keys() - extra.keys() or
                any(type(row[key]) is not str for key in required) or
                any(key not in row or type(row[key]) is not kind for key, kind in extra.items()) or
                any(key in row and type(row[key]) not in kinds for key, kinds in optional.items()) or
                any(key in row and any(type(item) is not str for item in row[key])
                    for key in ('SUPPORTED_CONTENT_TYPES', 'ALLOWED_EXTENSIONS'))):
            fail('WebUI audio response has incompatible fields')
    if 'realtime' in value and type(value['realtime']) is not dict:
        fail('WebUI realtime response has incompatible shape')
    return value


def projection(audio):
    # Required request format is defaulted upstream only when omitted.
    return {key: audio['stt'].get(key, 'multipart') for key in ROUTING}


def desired(target):
    return dict(CLEAR, OPENAI_API_BASE_URL=target['base_url'], OPENAI_API_KEY='loom-control', MODEL=target['instance'])


def pristine(audio):
    row = audio['stt']; routing = projection(audio)
    default = dict(CLEAR, ENGINE='', OPENAI_API_BASE_URL='https://api.openai.com/v1')
    return ((same(routing, CLEAR) or same(routing, default) or same(routing, dict(default, OPENAI_API_BASE_URL='')))
            and row['WHISPER_MODEL'] == 'base'
            and all(row[key] == '' for key in ('DEEPGRAM_API_KEY', 'AZURE_API_KEY', 'MISTRAL_API_KEY')))


def upload_prerequisites(audio):
    row = audio['stt']; mimes = row.get('SUPPORTED_CONTENT_TYPES', [])
    extensions = row.get('ALLOWED_EXTENSIONS', [])
    if ((mimes and not all(any(candidate in mimes for candidate in choices)
            for choices in (('audio/*', 'audio/mpeg'), ('audio/*', 'audio/wav')))) or
            extensions and not {'mp3', 'wav'} <= set(extensions)):
        fail('WebUI upload policy must permit MP3 and WAV; adjust it explicitly before configure')


def create_body(target, nonce):
    return dict(id=target['prefixed_id'], base_model_id=None, name=target['instance'],
        meta={'loom_control_stt': dict(owner='loom-control', token=nonce)}, params={},
        access_grants=[], is_active=False)


def comparable(row):
    if type(row) is not dict or row.get('write_access', True) is not True:
        fail('WebUI model row requires administrator write access')
    return {key: deepcopy(value) for key, value in row.items() if key not in ('user', 'write_access')}


def created_row(row, target, nonce):
    row = comparable(row); body = create_body(target, nonce)
    # Pinned ModelMeta.model_dump adds these declared null defaults on create/GET.
    body['meta'] = dict.fromkeys(META_DEFAULTS, None) | body['meta']
    if (not BODY_KEYS <= row.keys() or row.keys() - BODY_KEYS - GENERATED or
            any(not same(row[key], value) for key, value in body.items()) or
            not GENERATED <= row.keys() or any(type(row[key]) not in (str, int) for key in GENERATED)):
        fail('WebUI created model readback differs; retain pending intent')
    return row


def validate_record(record):
    if record is None:
        return
    try:
        if type(record) is not dict or set(record) != {'target', 'routing', 'overlay', 'pending'}:
            fail()
        target = record['target']
        if type(target) is not dict or set(target) != {'server', 'instance', 'base_url', 'prefixed_id'}:
            fail()
        _name(target['server']); _name(target['instance'])
        if target['prefixed_id'] != provider_id(target['server']) + '.' + target['instance']:
            fail()
        from .hem_openwebui_http import canonical_target
        if target['base_url'] != canonical_target(target['base_url'][:-3]) + '/v1':
            fail()
        if record['routing'] is not None and not same(record['routing'], desired(target)):
            fail()
        overlay = record['overlay']
        if overlay is not None:
            if overlay.get('ownership') == 'external':
                if set(overlay) != {'ownership', 'id'} or overlay['id'] != target['prefixed_id']:
                    fail()
            elif overlay.get('ownership') == 'created':
                if set(overlay) != {'ownership', 'nonce', 'row'} or type(overlay['nonce']) is not str or not overlay['nonce']:
                    fail()
                created_row(overlay['row'], target, overlay['nonce'])
            else:
                fail()
        pending = record['pending']
        if pending is not None:
            if (type(pending) is not dict or set(pending) != {'operation', 'intent', 'outcome'} or
                    pending['outcome'] != 'unconfirmed' or pending['operation'] not in (
                        'create_overlay', 'configure_audio', 'clear_audio', 'delete_overlay')):
                fail()
            op, intent = pending['operation'], pending['intent']
            if op == 'create_overlay':
                nonce = intent['meta']['loom_control_stt']['token']
                if type(nonce) is not str or not nonce or not same(intent, create_body(target, nonce)):
                    fail()
            elif op in ('configure_audio', 'clear_audio'):
                if not same(intent, desired(target) if op == 'configure_audio' else CLEAR):
                    fail()
            elif overlay is None or overlay['ownership'] != 'created' or not same(intent, overlay['row']):
                fail()
        state._encoded(record)
    except (KeyError, TypeError, AttributeError, ValueError):
        fail()


def native_target(root, server_id, instance, deadline):
    from .hem_process import ProcessManager, require_build
    from .hem_http import primary_parameters, registration
    registry = Registry(root, deadline=deadline)
    server = registry.get_server(_name(server_id)); _name(instance)
    if server['phase'] != 'running' or server['pending'] or not server['launch_token']:
        fail('Selected STT server is not running with resolved ownership')
    rows = [row for row in registry.list_instances(server_id=server_id)
            if row['launch_token'] == server['launch_token'] and row['name'] == instance]
    if len(rows) != 1 or rows[0]['pending'] or rows[0]['incarnation'] is None:
        fail('Selected STT instance has no resolved tracked incarnation')
    manager = ProcessManager(registry)
    require_build(server, manager.package_root, deadline)
    with manager.pin(server) as process:
        document = manager.client(server, process, deadline).status()
    row = registration(document, instance, rows[0]['incarnation'])
    if row is None:
        fail('Selected STT incarnation is absent')
    loom = row['loom']
    if (loom.get('kind') != 'recognition' or loom.get('endpoint') != '/v1/audio/transcriptions' or
            row.get('input_modalities') != ['audio'] or row.get('output_modalities') != ['text'] or
            loom.get('registration') != 'ready' or loom.get('parameter_operation') != 'idle' or
            primary_parameters(loom).get('active') is not True or loom.get('error') is not None or
            loom.get('status', {}).get('admission') != 'open'):
        fail('Selected STT instance must be ready active recognition with open admission')
    from .hem_openwebui_http import canonical_target
    return dict(server=server_id, instance=instance, base_url=canonical_target(server['endpoint']) + '/v1',
                prefixed_id=provider_id(server_id) + '.' + instance)


def configure_stt(root, target, token, deadline, *, check, disable, client_factory, display,
                  stt_server, stt_instance):
    from .hem_openwebui import configure_openwebui, _result, _same, _replace_before_submission
    from .hem_openwebui_http import QUALIFIED_VERSIONS
    if (stt_server is None) != (stt_instance is None) or disable and stt_server is not None:
        fail('Supply both --stt-server and --stt-instance; omit selectors with --disable')
    if stt_server is not None:
        _name(stt_server); _name(stt_instance)
    path = state.ownership_path(root, target)
    client = client_factory(target, token, deadline)
    if not check:
        ensure_endpoints(Registry(root, deadline=deadline), deadline)
    with nullcontext() if check else integration_lock(root, deadline):
        previous = files.read_owned(path)
        document = state.decode_document(previous, root, target)
        record = deepcopy(document.get('stt'))
        saved_record = deepcopy(record)
        stt_changed = False
        if display: display('GET', '/api/version')
        version = client.version()
        if version['version'] not in QUALIFIED_VERSIONS and not check:
            fail('WebUI version is not qualified for mutation; inspect with --check')
        if document['deployment_id'] and document['deployment_id'] != version['deployment_id']:
            fail('WebUI deployment identity conflict')
        if stt_server is not None:
            if check:
                selected = dict(server=stt_server, instance=stt_instance)
                if record is None or any(record['target'][key] != value for key, value in selected.items()):
                    return _result(target, 'check', configuration='absent', completed=False, exit_code=1,
                                   findings=[dict(status='stt_intent_absent')])
            else:
                selected = native_target(root, stt_server, stt_instance, deadline)
                if record is not None and not same(selected, record['target']):
                    fail('Disable the existing STT target before selecting another')
                if record is None:
                    record = dict(target=selected, routing=None, overlay=None, pending=None)
        if record is None:
            return configure_openwebui(root, target, token, deadline, check=check, disable=disable,
                client_factory=client_factory, display=display, _chat_only=True, _lock_held=not check)
        def audio_get():
            if display: display('GET', '/api/v1/audio/config')
            return validate_audio(client.audio_config())

        def model_get(model_id):
            if display: display('GET', '/api/v1/models/model?id=' + quote(model_id, safe=''))
            value = client.model(model_id)
            if value is not None and (type(value) is not dict or value.get('id') != model_id or value.get('write_access') is not True):
                fail('WebUI model lookup lacks matching identity/write access')
            return value

        audio = audio_get()
        model_id = record['target']['prefixed_id']
        row = model_get(model_id)

        def persist(*, confirmed=False):
            nonlocal previous, document, saved_record, stt_changed
            document = state.decode_document(files.read_owned(path), root, target)
            if not same(document.get('stt'), saved_record):
                fail('WebUI local STT ownership changed before update')
            # The existing chat coordinator may just have finalized its own receipt.
            current = files.read_owned(path)
            document.update(schema_version=2, stt=deepcopy(record), deployment_id=version['deployment_id'])
            state._validate_document(document, root, target)
            try:
                _replace_before_submission(path, files.encoded(document), expected=current, deadline=deadline)
            except (HemError, OSError):
                if confirmed:
                    raise UncertainOutcome('WebUI STT remote outcome observed but local finalization incomplete; retain intent', accepted=True) from None
                raise
            previous = files.read_owned(path)
            saved_record = deepcopy(record)
            stt_changed = True

        def stage(operation, intent):
            record['pending'] = dict(operation=operation, intent=deepcopy(intent), outcome='unconfirmed')
            persist()

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

        had_pending = record['pending'] is not None
        if not observed_pending():
            if check:
                return _result(target, 'check', configuration='pending', pending=True, completed=False,
                    exit_code=1, findings=[dict(status='stt_unconfirmed_no_replay')])
            raise UncertainOutcome('WebUI STT operation unconfirmed; inspect/operator reconcile before retrying')
        if check and had_pending:
            return _result(target, 'check', configuration='pending', pending=True, completed=False,
                exit_code=1, findings=[dict(status='stt_observed_completed_not_finalized')])
        if had_pending:
            persist(confirmed=True)

        def overlay_matches():
            overlay = record['overlay']
            if overlay is None:
                return True
            if overlay['ownership'] == 'created':
                return row is not None and same(comparable(row), overlay['row'])
            return row is not None and row.get('base_model_id') is None and row.get('is_active') is False

        routing_matches = record['routing'] is not None and same(projection(audio), record['routing'])
        if not overlay_matches() or record['routing'] is not None and not routing_matches:
            fail('WebUI owned STT routing/model row drift; preserve evidence')
        if check:
            chat = configure_openwebui(root, target, token, deadline, check=True,
                client_factory=client_factory, display=display, _chat_only=True)
            configured = routing_matches and record['overlay'] is not None and record['pending'] is None
            chat['stt'] = dict(configuration='configured' if configured else 'partial',
                target=deepcopy(record['target']), overlay_ownership=(record['overlay'] or {}).get('ownership'))
            if not configured:
                chat.update(configuration='partial', state='partial', completed=False)
                chat.exit_code = max(chat.exit_code, 1)
            return chat

        # All chat/audio/model conflicts are checked before creating remote state.
        endpoints = project_endpoints(root, passive_snapshot(root, deadline)['servers'])
        remote_chat = client.config()
        plan = state.plan_change(remote_chat, document, state.desired_connections(endpoints),
                                mode='disable' if disable else 'configure', deployment_id=version['deployment_id'])
        if plan.configuration == 'drift' or plan.recovery == 'conflict':
            fail('WebUI chat connection conflict; inspect before configuring STT')
        if not disable:
            if record['routing'] is None and not pristine(audio):
                fail('Foreign STT provider/configuration conflicts; no automatic replacement')
            upload_prerequisites(audio)
            if record['overlay'] is None and row is not None:
                if row.get('write_access', True) is not True or row.get('base_model_id') is not None or row.get('is_active') is not False:
                    fail('WebUI selected model has an active/custom foreign override')
                record['overlay'] = dict(ownership='external', id=model_id)
            if record['overlay'] is None:
                nonce = uuid.uuid4().hex; body = create_body(record['target'], nonce)
                stage('create_overlay', body)
                if model_get(model_id) is not None:
                    fail('WebUI model appeared before create; pending evidence retained')
                try:
                    if display: display('POST', '/api/v1/models/create', body)
                    client.create_model(body)
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
                record['pending'] = None; persist(confirmed=True)
            elif document.get('stt') is None:
                persist()
            chat = configure_openwebui(root, target, token, deadline, client_factory=client_factory,
                display=display, _chat_only=True, _lock_held=True)
            if chat.exit_code:
                return chat
            wanted = desired(record['target'])
        else:
            chat = None
            wanted = CLEAR if record['routing'] is not None else None
        if wanted is not None and not same(projection(audio), wanted):
            stage('clear_audio' if disable else 'configure_audio', wanted)
            immediate = audio_get()
            if not same(immediate, audio):
                fail('WebUI audio changed before submission; pending intent retained')
            body = dict(tts=deepcopy(audio['tts']), stt=dict(deepcopy(audio['stt']), **wanted))
            try:
                if display: display('POST', '/api/v1/audio/config/update', dict(operation=record['pending']['operation'], routing=wanted))
                client.update_audio(body)
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
            record['pending'] = None; persist(confirmed=True)
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
                    if display: display('POST', '/api/v1/models/model/delete', dict(id=model_id))
                    client.delete_model(model_id)
                except HemError:
                    pass  # Only readback can prove the current outcome.
                if model_get(model_id) is not None:
                    raise UncertainOutcome('WebUI model deletion unconfirmed; no automatic replay')
                record['overlay'] = None; record['pending'] = None; persist(confirmed=True)
            # External prerequisite is never deleted/adopted; no remaining authority.
            record = None; persist(confirmed=True)
            chat = configure_openwebui(root, target, token, deadline, disable=True,
                client_factory=client_factory, display=display, _chat_only=True, _lock_held=True)
        chat['stt'] = dict(configuration='absent' if disable else 'configured',
            target=None if disable else deepcopy(record['target']), backend_reachability='operator-required/unverified',
            disconnect='inert-openai/blank-origin-no-provider-fallback' if disable else None)
        chat['changed'] = chat['changed'] or stt_changed
        return chat
