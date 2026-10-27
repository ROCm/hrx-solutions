"""CHAT field semantics and native text definitions, driven by serving data."""
from collections.abc import Mapping
import json
from pathlib import Path
from types import MappingProxyType

from ...hem_common import InvalidInput

CAPACITIES = ('rows', 'context_capacity', 'pool_capacity', 'prefill_capacity')
NATIVE_CONTRACT = 'text-v1'
PROTOCOL = MappingProxyType({'operation': 'chat', 'native_kind': 'text',
            'endpoint': '/v1/chat/completions', 'request_encoding': 'json',
            'input_modalities': ('text',), 'output_modalities': ('text',)})


def validate_contract(serving):
    """Admit this modality's finite data vocabulary, never a parameter schema."""
    if not isinstance(serving, Mapping) or set(serving) != {'bounds', 'mtp_depth_max', 'defaults'}:
        raise InvalidInput('CHAT serving requires bounds, mtp_depth_max and defaults')
    bounds = serving['bounds']
    if not isinstance(bounds, Mapping) or set(bounds) != set(CAPACITIES):
        raise InvalidInput('CHAT bounds require the four capacity fields')
    for key in CAPACITIES:
        interval = bounds[key]
        if (not isinstance(interval, (list, tuple)) or len(interval) != 2
                or any(type(value) is not int for value in interval)
                or not 1 <= interval[0] <= interval[1] <= 2**63 - 1):
            raise InvalidInput(f'CHAT bounds.{key}: expected positive ordered integer interval')
    maximum = serving['mtp_depth_max']
    if type(maximum) is not int or not 0 <= maximum <= 2**63 - 1:
        raise InvalidInput('CHAT mtp_depth_max must be a nonnegative native integer')
    validate_layer(serving, serving['defaults'])
    values = defaults(serving)
    finalize(serving, values, {key: 'default' for key in values}, {key: 0 for key in values})


def defaults(serving):
    return dict(serving['defaults'])


def validate_layer(serving, values):
    """Admit every supplied field before precedence or cross-setting rules."""
    if not isinstance(values, Mapping):
        raise InvalidInput('CHAT settings must be an object')
    maximum = serving['mtp_depth_max']
    for key, value in values.items():
        if key == 'mtp':
            if type(value) is not bool or maximum == 0 and value:
                raise InvalidInput('mtp: expected boolean' if maximum else 'mtp: expected fixed false')
        elif key in CAPACITIES:
            minimum, limit = serving['bounds'][key]
            if type(value) is not int or not minimum <= value <= limit:
                raise InvalidInput(f'{key}: expected integer in {minimum}..{limit}')
        elif key == 'mtp_depth':
            if maximum == 0:
                raise InvalidInput('mtp_depth: selected executable does not support MTP')
            if type(value) is not int or not 1 <= value <= maximum:
                raise InvalidInput(f'mtp_depth: expected integer in 1..{maximum}')
        else:
            raise InvalidInput(f'Unknown CHAT setting {key}')


def _final_rules(settings):
    required = {*CAPACITIES, 'mtp'}
    if settings.get('mtp'):
        required.add('mtp_depth')
    if set(settings) != required:
        raise InvalidInput('CHAT effective settings are missing required fields or contain inactive mtp_depth')
    if settings['prefill_capacity'] > settings['context_capacity']:
        raise InvalidInput('prefill_capacity must not exceed context_capacity')
    if settings['prefill_capacity'] < (settings['mtp_depth'] if settings['mtp'] else 0) + 1:
        raise InvalidInput('prefill_capacity must fit one anchor plus every MTP proposal')


def finalize(serving, settings, sources, ranks):
    """Apply CHAT precedence semantics and validate the final native setting set."""
    if settings.get('mtp') is False and 'mtp_depth' in settings:
        if ranks['mtp_depth'] > 0 and ranks['mtp_depth'] >= ranks['mtp']:
            raise InvalidInput('configuration.mtp_depth: explicit value conflicts with inactive mtp')
        settings.pop('mtp_depth')
        sources['mtp_depth'] = {'state': 'inactive', 'by': sources['mtp']}
    _final_rules(settings)


def definition(settings, paths, directory, programs):
    """Build text from admitted settings/programs and inspected asset paths."""
    for purpose in ('weights', 'tokenizer'):
        if purpose not in paths or not Path(paths[purpose]).is_absolute():
            raise InvalidInput(f'CHAT requires an absolute prepared {purpose} path')
    return {'kind': 'text', **programs, 'sources': list(programs['sources']),
            'weights': str(paths['weights']), 'tokenizer': str(paths['tokenizer']),
            **{key: settings[key] for key in CAPACITIES},
            'draft_depth': settings['mtp_depth'] if settings['mtp'] else 0,
            'options': {}}


def vocabulary():
    return {**{key:{'type':'integer'} for key in CAPACITIES},
            'mtp':{'type':'boolean'}, 'mtp_depth':{'type':'integer'}}


def parameters(serving):
    """Output-only support projection of the same admitted serving facts."""
    values = {}
    for key in CAPACITIES:
        minimum, maximum = serving['bounds'][key]
        values[key] = ({'type': 'integer', 'value': minimum, 'state': 'fixed'}
                       if minimum == maximum else
                       {'type': 'integer', 'default': serving['defaults'][key],
                        'constraints': {'minimum': minimum, 'maximum': maximum, 'allowed_keys': None},
                        'active_when': None, 'state': 'configurable'})
    maximum = serving['mtp_depth_max']
    values['mtp'] = ({'type': 'boolean', 'default': serving['defaults']['mtp'],
                      'constraints': {'minimum': None, 'maximum': None, 'allowed_keys': None},
                      'active_when': None, 'state': 'configurable'} if maximum else
                     {'type': 'boolean', 'value': False, 'state': 'fixed'})
    values['mtp_depth'] = ({'type': 'integer', 'default': serving['defaults'].get('mtp_depth'),
                            'constraints': {'minimum': 1, 'maximum': maximum, 'allowed_keys': None},
                            'active_when': {'parameter': 'mtp', 'equals': True}, 'state': 'configurable'}
                           if maximum else {'type': 'integer', 'reason': 'Selected executable does not support MTP',
                                            'state': 'unsupported'})
    return values


def interpret(event):
    from ..hem_config import json_document
    return json.dumps(json_document(event), ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), allow_nan=False)
