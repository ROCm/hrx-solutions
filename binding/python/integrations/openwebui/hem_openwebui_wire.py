"""Pure WebUI wire admission; no transport, persistence or operation policy.

HTTP and ownership modes deliberately retain their original encoding, errors and
index limits. Foreign credentials may pass through these values, never messages.
"""
from __future__ import annotations

import ipaddress
import json
import math
import re
from urllib.parse import urlsplit

from ...hem_common import InvalidInput

MAX_BYTES = 2 * 1024 * 1024
_FIELDS = {"ENABLE_OPENAI_API", "OPENAI_API_BASE_URLS", "OPENAI_API_KEYS", "OPENAI_API_CONFIGS"}
TTS_STRINGS = frozenset('OPENAI_API_BASE_URL OPENAI_API_KEY API_KEY ENGINE MODEL VOICE SPLIT_ON AZURE_SPEECH_REGION AZURE_SPEECH_BASE_URL AZURE_SPEECH_OUTPUT_FORMAT MISTRAL_API_KEY MISTRAL_API_BASE_URL'.split())
STT_STRINGS = frozenset('OPENAI_API_BASE_URL OPENAI_API_KEY ENGINE MODEL WHISPER_MODEL DEEPGRAM_API_KEY AZURE_API_KEY AZURE_REGION AZURE_LOCALES AZURE_BASE_URL AZURE_MAX_SPEAKERS MISTRAL_API_KEY MISTRAL_API_BASE_URL'.split())
OPTIONAL_TTS = {'OPENAI_PARAMS': (dict, type(None)), 'REALTIME_TTS_PROMPT_TEMPLATE': (str, type(None))}
OPTIONAL_STT = {'OPENAI_API_REQUEST_FORMAT': (str,), 'SUPPORTED_CONTENT_TYPES': (list,), 'ALLOWED_EXTENSIONS': (list,)}


def ownership_error():
    raise InvalidInput("WebUI ownership/schema conflict; retain evidence for inspection")


def canonical_target(value: str) -> str:
    """Return a direct numeric-loopback HTTP origin; never resolve hostnames."""
    try:
        if not isinstance(value, str) or any(c.isspace() or ord(c) < 32 or ord(c) == 127 for c in value):
            raise ValueError
        parsed = urlsplit(value)
        address = ipaddress.ip_address(parsed.hostname or "")
        port = parsed.port
        if (parsed.scheme != "http" or not address.is_loopback or "%" in parsed.netloc
                or parsed.username is not None or parsed.password is not None
                or port is None or port == 0 or parsed.path not in ("", "/")
                or parsed.query or parsed.fragment or "?" in value or "#" in value):
            raise ValueError
        host = f"[{address.compressed}]" if address.version == 6 else address.compressed
        return f"http://{host}:{port}"
    except (ValueError, TypeError):
        raise InvalidInput("WebUI URL must be a numeric-loopback http://address:port origin") from None


def _ownership(mode):
    if mode not in ('http', 'ownership'):
        raise ValueError('unsupported WebUI admission mode')
    return mode == 'ownership'


def _check_json(item):
    if item is None or type(item) in (str, bool, int):
        return
    if type(item) is float and math.isfinite(item):
        return
    if type(item) is list:
        for child in item:
            _check_json(child)
        return
    if type(item) is dict and all(type(key) is str for key in item):
        for child in item.values():
            _check_json(child)
        return
    raise ValueError


def json_bytes(value, *, mode='http'):
    """Admit finite exact JSON values, preserving each boundary's encoded size."""
    owned = _ownership(mode)
    try:
        _check_json(value)
        body = json.dumps(value, ensure_ascii=owned, allow_nan=False, separators=(",", ":")).encode("utf-8")
        if len(body) > MAX_BYTES:
            raise ValueError
        return body
    except (ValueError, TypeError, UnicodeError, RecursionError, OverflowError):
        if owned:
            ownership_error()
        raise InvalidInput("WebUI JSON must be strict and at most 2 MiB") from None


def decode_json(data, *, mode='http'):
    """Decode once with duplicate/nonfinite rejection; caller admits its schema."""
    owned = _ownership(mode)
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError
            result[key] = value
        return result
    def nonfinite(_):
        raise ValueError
    try:
        value = json.loads(data if owned else data.decode('utf-8'),
                           object_pairs_hook=pairs, parse_constant=nonfinite)
        if not owned:
            json_bytes(value)
        return value
    except (ValueError, UnicodeError, RecursionError, InvalidInput):
        if owned:
            ownership_error()
        raise InvalidInput("WebUI response is not bounded strict JSON") from None


def strict_equal(left, right):
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(strict_equal(left[k], right[k]) for k in left)
    if isinstance(left, list):
        return len(left) == len(right) and all(strict_equal(a, b) for a, b in zip(left, right))
    return left == right


def validate_config(value, *, mode='http'):
    """One four-field envelope, with the two existing admission policies."""
    owned = _ownership(mode)
    json_bytes(value, mode=mode)
    def fail(message):
        if owned:
            ownership_error()
        raise InvalidInput(message)
    # Ownership historically uses isinstance; HTTP requires exact containers.
    def is_kind(item, kind):
        return isinstance(item, kind) if owned else type(item) is kind
    if not is_kind(value, dict) or set(value) != _FIELDS:
        fail("WebUI configuration has unsupported top-level fields")
    urls, keys, configs = (value[name] for name in ("OPENAI_API_BASE_URLS", "OPENAI_API_KEYS", "OPENAI_API_CONFIGS"))
    if (type(value["ENABLE_OPENAI_API"]) is not bool or not is_kind(urls, list)
            or not is_kind(keys, list) or len(urls) != len(keys)
            or any(not is_kind(item, str) for item in urls + keys) or not is_kind(configs, dict)):
        fail("WebUI configuration has invalid enable, URL, key or connection fields")
    for index, config in configs.items():
        if owned:
            valid_index = (isinstance(index, str) and index.isascii() and index.isdecimal()
                           and len(index) <= 10 and str(int(index)) == index and int(index) < len(urls))
        else:
            valid_index = (re.fullmatch(r"0|[1-9][0-9]*", index)
                           and len(index) <= len(str(len(urls))) and int(index) < len(urls))
        if not valid_index or not is_kind(config, dict):
            fail("WebUI configuration has an invalid connection index or shape")
    return value


def validate_audio(value):
    # Audio retains ownership-mode encoding/admission even at HTTP boundaries.
    json_bytes(value, mode='ownership')
    if type(value) is not dict or not {'tts', 'stt'} <= value.keys() or value.keys() - {'tts', 'stt', 'realtime'}:
        raise InvalidInput('WebUI audio response has incompatible sections')
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
            raise InvalidInput('WebUI audio response has incompatible fields')
    if 'realtime' in value and type(value['realtime']) is not dict:
        raise InvalidInput('WebUI realtime response has incompatible shape')
    return value
