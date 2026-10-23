"""CHAT parameter, protocol, native-definition and raw-event policy; no I/O."""
from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType

from ...hem_common import InvalidInput
from ..recipe_types import (
    Configurable, Constraints, Fixed, LogView, PipelineDescriptor,
    ProtocolDescriptor, TextPrograms, Unsupported, json_document,
)

# Vocabulary and meanings belong to CHAT. Concrete upper bounds and MTP support
# come from the selected implementation, not the recipe/model name.
DEFAULTS = MappingProxyType({'rows': 4, 'context_capacity': 8192, 'pool_capacity': 32768,
                            'prefill_capacity': 128, 'mtp_depth': 3})
PROTOCOL = ProtocolDescriptor('chat', 'text', '/v1/chat/completions', 'json',
                              ('text',), ('text',))


def describe(binding):
    parameters = {key: Configurable('integer', default,
                                   Constraints(minimum=1, maximum=binding.limits[key]))
                  for key, default in DEFAULTS.items() if key != 'mtp_depth'}
    parameters['mtp'] = (Configurable('boolean', True) if binding.mtp_supported
                         else Fixed('boolean', False))
    parameters['mtp_depth'] = (
        Configurable('integer', DEFAULTS['mtp_depth'],
                     Constraints(minimum=1, maximum=binding.limits['mtp_depth']),
                     active_when={'parameter': 'mtp', 'equals': True})
        if binding.mtp_supported else Unsupported('integer', 'Selected pipeline does not support MTP'))
    return PipelineDescriptor(binding.id, 'chat', parameters, PROTOCOL,
                              binding.artifacts, None, binding.programs,
                              binding.bundle)


def _cross_rules(settings):
    if settings['prefill_capacity'] > settings['context_capacity']:
        raise InvalidInput('prefill_capacity must not exceed context_capacity')
    mtp = settings['mtp']
    if not mtp and 'mtp_depth' in settings:
        raise InvalidInput('mtp_depth is unavailable when MTP is disabled')
    if mtp and 'mtp_depth' not in settings:
        raise InvalidInput('mtp_depth is required when MTP is enabled')
    if settings['prefill_capacity'] < ((settings['mtp_depth'] if mtp else 0) + 1):
        raise InvalidInput('prefill_capacity must fit one anchor plus every MTP proposal')


def validate(settings, descriptor):
    """Validate final effective settings; shared resolution owns layer merging."""
    if descriptor.modality != 'chat' or not isinstance(descriptor.programs, TextPrograms):
        raise InvalidInput('CHAT settings require a CHAT pipeline descriptor')
    if not isinstance(settings, Mapping):
        raise InvalidInput('CHAT settings must be an object')
    unknown = set(settings) - set(descriptor.parameters)
    if unknown:
        raise InvalidInput('Unknown CHAT settings: ' + ', '.join(sorted(map(str, unknown))))
    for key, parameter in descriptor.parameters.items():
        if key not in settings:
            if isinstance(parameter, Unsupported) or (isinstance(parameter, Configurable)
                    and parameter.active_when is not None
                    and settings.get(parameter.active_when['parameter']) != parameter.active_when['equals']):
                continue
            raise InvalidInput(f'Missing CHAT setting {key}')
        if isinstance(parameter, Unsupported):
            raise InvalidInput(f'{key}: {parameter.reason}')
        value = settings[key]
        expected = bool if parameter.type == 'boolean' else int
        if type(value) is not expected:
            raise InvalidInput(f'{key} must be {parameter.type}')
        if isinstance(parameter, Fixed):
            if value != parameter.value:
                raise InvalidInput(f'{key}: selected pipeline requires {parameter.value!r}')
        else:
            constraints = parameter.constraints
            if (constraints.minimum is not None and value < constraints.minimum
                    or constraints.maximum is not None and value > constraints.maximum):
                raise InvalidInput(f'{key} must be in {constraints.minimum}..{constraints.maximum}')
    _cross_rules(settings)


def _asset(prepared, role):
    files = [item for item in prepared.files if item.role == role]
    if len(files) != 1:
        raise InvalidInput(f'CHAT native definition requires exactly one prepared {role} file')
    path = Path(files[0].path)
    if not path.is_absolute():
        raise InvalidInput(f'Prepared {role} path must be absolute')
    # Preserve the declared file's parser-selecting suffix and HF link path.
    return str(path)


def service_definition(settings, prepared, programs):
    """Construct a native definition from already resolved/validated settings."""
    if not isinstance(programs, TextPrograms):
        raise InvalidInput('CHAT native definition requires text programs')
    _cross_rules(settings)
    return {'kind': 'text', 'prepare': programs.prepare, 'control': programs.control,
            'sources': list(programs.providers), 'weights': _asset(prepared, 'weights'),
            'tokenizer': _asset(prepared, 'tokenizer'), 'rows': settings['rows'],
            'context_capacity': settings['context_capacity'],
            'pool_capacity': settings['pool_capacity'],
            'prefill_capacity': settings['prefill_capacity'],
            'draft_depth': settings['mtp_depth'] if settings['mtp'] else 0,
            'options': {}}


def interpret(event, descriptor):
    """No metric producer is qualified yet; retain display-safe raw JSON."""
    if descriptor.modality != 'chat':
        raise InvalidInput('CHAT event interpretation requires a CHAT descriptor')
    return LogView(json.dumps(json_document(event), ensure_ascii=False,
                              sort_keys=True, separators=(',', ':'), allow_nan=False))
