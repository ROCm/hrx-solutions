"""Finite speech settings and native definitions for the curated Pocket contract."""
import json
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType

from ...hem_common import InvalidInput
from .. import pocket

NATIVE_CONTRACT = 'speech-v1'
PROTOCOL = MappingProxyType({'operation': 'speech', 'native_kind': 'speech',
    'endpoint': '/v1/audio/speech', 'request_encoding': 'json',
    'input_modalities': ('text',), 'output_modalities': ('audio',)})


def validate_contract(serving):
    if (not isinstance(serving, Mapping) or set(serving) != {'pending_requests', 'defaults'}
            or serving['pending_requests'] != [1, 65536]
            or any(type(value) is not int for value in serving['pending_requests'])):
        raise InvalidInput('TTS serving requires pending_requests bounds 1..65536 and defaults')
    validate_layer(serving, serving['defaults'])
    finalize(serving, serving['defaults'], {}, {})


def defaults(serving):
    return {**serving['defaults'], 'options': dict(serving['defaults']['options'])}


def validate_layer(serving, values):
    if not isinstance(values, Mapping):
        raise InvalidInput('TTS settings must be an object')
    for key, value in values.items():
        if key == 'pending_requests':
            pocket._integer(value, key, *serving['pending_requests'])
        elif key == 'options':
            if not isinstance(value, Mapping) or set(value) != set(pocket.RESOURCE_LIMITS):
                raise InvalidInput('TTS options require exactly input_tokens, request_capacity, batch_capacity, threads; measured geometry is read-only')
            for name, limit in pocket.RESOURCE_LIMITS.items():
                pocket._integer(value[name], name, 1, limit)
        else:
            raise InvalidInput(f'Unknown or read-only TTS setting: {key}')


def finalize(serving, settings, sources, ranks):
    if set(settings) != {'pending_requests', 'options'}:
        raise InvalidInput('TTS requires pending_requests and options')
    if settings['options']['batch_capacity'] > settings['options']['request_capacity']:
        raise InvalidInput('TTS batch_capacity must not exceed request_capacity')


def definition(settings, paths, directory, programs):
    if not Path(directory).is_absolute():
        raise InvalidInput('TTS requires an absolute prepared asset directory')
    options = pocket.helper_geometry(paths['prepared_metadata'])
    return {'kind': 'speech', **programs, 'sources': list(programs['sources']),
            'assets': str(directory), 'pending_requests': settings['pending_requests'],
            'options': {**options, **settings['options']}}


def vocabulary():
    return {'pending_requests': {'type': 'integer'},
            'options': {'type': 'object', 'allowed_keys': list(pocket.RESOURCE_LIMITS)}}


def parameters(serving):
    return {'pending_requests': {'type': 'integer', 'default': serving['defaults']['pending_requests'],
             'constraints': {'minimum': 1, 'maximum': 65536, 'allowed_keys': None},
             'active_when': None, 'state': 'configurable'},
            'options': {'type': 'object', 'default': dict(serving['defaults']['options']),
             'constraints': {'allowed_keys': list(pocket.RESOURCE_LIMITS)},
             'active_when': None, 'state': 'configurable'}}


def interpret(event):
    from ..hem_config import json_document
    return json.dumps(json_document(event), sort_keys=True, allow_nan=False)
