"""Pure schema-2 STT receipt shapes; no effects or target availability policy.

State owns the sole ownership-document decoder. These finite projections are
shared by that decoder and STT policy; validation confers no mutation authority.
Future operation schemas must extend this owner explicitly, not accept unknown
fields or add a second decoder.
"""
from __future__ import annotations

from copy import deepcopy
import re

from ...hem_common import InvalidInput
from ..pi.hem_pi_contracts import provider_id
from .hem_openwebui_wire import canonical_target, strict_equal as same, json_bytes


ROUTING = frozenset(('ENGINE', 'OPENAI_API_BASE_URL', 'OPENAI_API_KEY', 'OPENAI_API_REQUEST_FORMAT', 'MODEL'))
CLEAR = dict(ENGINE='openai', OPENAI_API_BASE_URL='', OPENAI_API_KEY='', OPENAI_API_REQUEST_FORMAT='multipart', MODEL='')
GENERATED = frozenset(('user_id', 'version_id', 'created_at', 'updated_at'))
META_DEFAULTS = frozenset('profile_image_url background_image_url description i18n capabilities knowledge voice voice_avatar'.split())
BODY_KEYS = frozenset(('id', 'base_model_id', 'name', 'meta', 'params', 'access_grants', 'is_active'))


def fail(message='WebUI STT ownership/configuration conflict; retain evidence and inspect'):
    raise InvalidInput(message)


def _name(value):
    # Same lexical ID contract as core, without importing Registry into schemas.
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise InvalidInput("resource IDs must contain letters, digits, '.', '_' or '-'")
    return value


def projection(audio):
    # Required request format is defaulted upstream only when omitted.
    return {key: audio['stt'].get(key, 'multipart') for key in ROUTING}


def desired(target):
    return dict(CLEAR, OPENAI_API_BASE_URL=target['base_url'], OPENAI_API_KEY='loom-control', MODEL=target['instance'])


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
        json_bytes(record, mode='ownership')
    except (KeyError, TypeError, AttributeError, ValueError):
        fail()
