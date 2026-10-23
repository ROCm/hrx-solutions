"""ASR setting vocabulary, recognition definitions and honest raw log views."""
from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from ...hem_common import HemError, InvalidInput
from ..recipe_types import (
    Configurable, Fixed, LogView, PipelineDescriptor, PreparedModel,
    ProtocolDescriptor, RecognitionPrograms, Unsupported, freeze_json, json_document,
)


PROTOCOL = ProtocolDescriptor('transcription', 'recognition', '/v1/audio/transcriptions',
                              'multipart', ('audio',), ('text',))
_TYPES = {'pending_requests': 'integer', 'options': 'object'}


def _descriptor(descriptor):
    if (not isinstance(descriptor, PipelineDescriptor) or descriptor.modality != 'asr'
            or descriptor.protocol != PROTOCOL
            or set(descriptor.parameters) != set(_TYPES)
            or not isinstance(descriptor.programs, RecognitionPrograms)):
        raise HemError('Selected pipeline does not provide the ASR recognition contract')
    for name, parameter in descriptor.parameters.items():
        if (not isinstance(parameter, (Configurable, Fixed, Unsupported))
                or parameter.type != _TYPES[name]):
            raise HemError(f'Invalid ASR parameter contract: {name}')
    return descriptor


def describe(binding):
    """Check the selected binding against the modality-owned vocabulary/protocol."""
    return _descriptor(binding.describe())


def _value(name, value):
    if name == 'pending_requests':
        if type(value) is not int:
            raise InvalidInput('configuration.pending_requests: expected integer')
        return value
    if not isinstance(value, Mapping):
        raise InvalidInput('configuration.options: expected object')
    try:
        return freeze_json(value)
    except ValueError as error:
        raise InvalidInput('configuration.options: expected finite JSON object') from error


def _same(left, right):
    # JSON spelling also keeps bool/int and integer/float assertions distinct.
    return json.dumps(json_document(left), sort_keys=True, allow_nan=False) == json.dumps(
        json_document(right), sort_keys=True, allow_nan=False)


def validate(settings, descriptor):
    """Validate a supplied layer or final settings; ASR has no cross-key rule."""
    descriptor = _descriptor(descriptor)
    if not isinstance(settings, Mapping):
        raise InvalidInput('configuration: expected object')
    for name, value in settings.items():
        if name not in _TYPES:
            raise InvalidInput(f'configuration.{name}: unsupported for modality asr')
        parameter = descriptor.parameters[name]
        if isinstance(parameter, Unsupported):
            raise InvalidInput(f'configuration.{name}: unsupported: {parameter.reason}')
        value = _value(name, value)
        if isinstance(parameter, Fixed):
            if not _same(value, _value(name, parameter.value)):
                raise InvalidInput(f'configuration.{name}: expected fixed value '
                                   + json.dumps(json_document(parameter.value), sort_keys=True))
        else:
            limits = parameter.constraints
            if name == 'pending_requests':
                if (limits.minimum is not None and value < limits.minimum
                        or limits.maximum is not None and value > limits.maximum):
                    raise InvalidInput('configuration.pending_requests: outside supported range')
            elif limits.allowed_keys is not None and set(value) - set(limits.allowed_keys):
                raise InvalidInput('configuration.options: unsupported field')


def service_definition(settings, prepared, programs):
    """Construct recognition directly from checked settings/assets/programs."""
    if not isinstance(settings, Mapping) or set(settings) != set(_TYPES):
        raise InvalidInput('ASR definition requires effective pending_requests and options')
    pending = _value('pending_requests', settings['pending_requests'])
    options = _value('options', settings['options'])
    if pending <= 0:
        raise InvalidInput('configuration.pending_requests: expected positive integer')
    if (not isinstance(prepared, PreparedModel)
            or not isinstance(prepared.directory, Path) or not prepared.directory.is_absolute()
            or not isinstance(programs, RecognitionPrograms)):
        raise InvalidInput('ASR definition requires checked prepared assets and recognition programs')
    return dict(kind='recognition', prepare=programs.prepare, control=programs.control,
                sources=list(programs.sources), assets=str(prepared.directory),
                pending_requests=pending, options=json_document(options))


def interpret(event, descriptor):
    """Preserve validated raw events; native ASR metric fields are unverified."""
    _descriptor(descriptor)
    return LogView(json.dumps(json_document(event), ensure_ascii=True, sort_keys=True,
                              separators=(',', ':'), allow_nan=False))
