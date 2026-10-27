"""ASR field semantics and recognition definitions, driven by serving data."""
from collections.abc import Mapping
import json
from pathlib import Path
from types import MappingProxyType

from ...hem_common import InvalidInput

NATIVE_CONTRACT = 'recognition-v1'
PROTOCOL = MappingProxyType({'operation': 'transcription', 'native_kind': 'recognition',
            'endpoint': '/v1/audio/transcriptions', 'request_encoding': 'multipart',
            'input_modalities': ('audio',), 'output_modalities': ('text',)})


def validate_contract(serving):
    if not isinstance(serving, Mapping) or set(serving) != {'pending_requests', 'defaults'}:
        raise InvalidInput('ASR serving requires pending_requests bounds and defaults')
    interval = serving['pending_requests']
    if (not isinstance(interval, (list, tuple)) or len(interval) != 2
            or any(type(value) is not int for value in interval)
            or not 1 <= interval[0] <= interval[1] <= 2**63 - 1):
        raise InvalidInput('ASR pending_requests requires positive ordered integer bounds')
    validate_layer(serving, serving['defaults'])
    _final_rules(serving['defaults'])


def defaults(serving):
    values = dict(serving['defaults'])
    values['options'] = dict(values['options'])
    return values


def validate_layer(serving, values):
    if not isinstance(values, Mapping):
        raise InvalidInput('ASR settings must be an object')
    for key, value in values.items():
        if key == 'pending_requests':
            minimum, maximum = serving['pending_requests']
            if type(value) is not int or not minimum <= value <= maximum:
                raise InvalidInput(f'pending_requests: expected integer in {minimum}..{maximum}')
        elif key == 'options':
            if not isinstance(value, Mapping) or value:
                raise InvalidInput('options: expected fixed empty object')
        else:
            raise InvalidInput(f'Unknown ASR setting {key}')


def _final_rules(settings):
    if set(settings) != {'pending_requests', 'options'}:
        raise InvalidInput('ASR effective settings require pending_requests and options')


def finalize(serving, settings, sources, ranks):
    _final_rules(settings)


def definition(settings, paths, directory, programs):
    directory = Path(directory)
    if not directory.is_absolute():
        raise InvalidInput('ASR requires an absolute prepared asset directory')
    return {'kind': 'recognition', **programs, 'sources': list(programs['sources']),
            'assets': str(directory), 'pending_requests': settings['pending_requests'],
            'options': dict(settings['options'])}


def vocabulary():
    return {'pending_requests':{'type':'integer'}, 'options':{'type':'object','allowed_keys':[]}}


def parameters(serving):
    minimum, maximum = serving['pending_requests']
    pending = ({'type': 'integer', 'value': minimum, 'state': 'fixed'}
               if minimum == maximum else
               {'type': 'integer', 'default': serving['defaults']['pending_requests'],
                'constraints': {'minimum': minimum, 'maximum': maximum, 'allowed_keys': None},
                'active_when': None, 'state': 'configurable'})
    return {'pending_requests': pending, 'options': {'type': 'object', 'value': {}, 'state': 'fixed'}}


def interpret(event):
    from ..hem_config import json_document
    return json.dumps(json_document(event), ensure_ascii=True, sort_keys=True,
                      separators=(',', ':'), allow_nan=False)
