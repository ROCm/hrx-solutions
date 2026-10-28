"""Read the selected Pocket Alba component; helper launch fields are not authority."""
import json
import math
from pathlib import Path

from ..hem_common import InvalidInput

GEOMETRY = {'context_capacity', 'rotary_capacity', 'chunk_tokens', 'voices'}
RESOURCE_LIMITS = {'input_tokens': 65536, 'request_capacity': 65536,
                   'batch_capacity': 65536, 'threads': 256}


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise InvalidInput(f'Pocket metadata has duplicate key: {key}')
        result[key] = value
    return result


def _depth(value, depth=0):
    if depth > 16:
        raise InvalidInput('Pocket metadata exceeds depth 16')
    if type(value) is float and not math.isfinite(value):
        raise InvalidInput('Pocket metadata requires finite numbers')
    if isinstance(value, dict):
        for item in value.values():
            _depth(item, depth + 1)
    elif isinstance(value, list):
        for item in value:
            _depth(item, depth + 1)


def _read(path):
    try:
        with Path(path).open('rb') as stream:
            content = stream.read(1024 * 1024 + 1)
        if len(content) > 1024 * 1024:
            raise InvalidInput('Pocket metadata exceeds 1 MiB')
        def constant(token):
            raise InvalidInput(f'Pocket metadata has nonfinite value: {token}')
        value = json.loads(content, object_pairs_hook=_pairs, parse_constant=constant)
        _depth(value)
        return value
    except (OSError, ValueError, RecursionError) as error:
        raise InvalidInput(f'Cannot read Pocket metadata {path}: {error}') from error


def _integer(value, name, low, high):
    if type(value) is not int or not low <= value <= high:
        raise InvalidInput(f'Pocket {name}: expected integer {low}..{high}')


def geometry(value):
    if type(value) is not dict or set(value) != GEOMETRY:
        raise InvalidInput('Pocket geometry requires context_capacity, rotary_capacity, chunk_tokens and voices')
    _integer(value['context_capacity'], 'context_capacity', 1, 65536)
    _integer(value['rotary_capacity'], 'rotary_capacity', 16, 1048576)
    _integer(value['chunk_tokens'], 'chunk_tokens', 50, 50)
    voices = value['voices']
    if (type(voices) is not list or len(voices) != 1 or type(voices[0]) is not dict
            or set(voices[0]) != {'id', 'frames'} or voices[0]['id'] != 'alba'):
        raise InvalidInput('Pocket supports exactly one Alba voice')
    frames = voices[0]['frames']
    _integer(frames, 'alba frames', 1, 65535)
    generation = (25 * 50 + 5) // 6 + 25
    if (value['context_capacity'] < 1 + frames + 50 + generation
            or value['rotary_capacity'] < max(value['context_capacity'], 16 * max(frames, generation))):
        raise InvalidInput('Pocket context/rotary capacities do not cover prepared voice and generation')
    return {**value, 'voices': [dict(voices[0])]}


def helper_geometry(path):
    value = _read(path)
    try:
        options = value['definition']['options']
    except (KeyError, TypeError) as error:
        raise InvalidInput('Pocket model.json requires definition.options geometry') from error
    if type(options) is not dict or set(options) - GEOMETRY - RESOURCE_LIMITS.keys():
        raise InvalidInput('Pocket helper options contain unsupported fields')
    for name, limit in RESOURCE_LIMITS.items():
        if name in options:
            _integer(options[name], name, 1, limit)
    return geometry({name: options[name] for name in GEOMETRY if name in options})


def import_metadata(directory):
    """Own only admitted geometry, accepting the prior portable carrier as input."""
    root = Path(directory)
    helper, portable = root / 'model.json', root / 'prepared-metadata.json'
    measured = helper_geometry(helper) if helper.exists() else None
    if portable.exists():
        value = _read(portable)
        if (type(value) is not dict or set(value) != {'format', 'asset_contract', 'profile', 'geometry'}
                or type(value['format']) is not int or value['format'] != 1
                or value['asset_contract'] != 'pocket-alba-assets-v1' or value['profile'] != 'alba'):
            raise InvalidInput('Unsupported Pocket prepared metadata contract')
        alternative = geometry(value['geometry'])
        if measured is not None and measured != alternative:
            raise InvalidInput('Conflicting Pocket metadata carriers')
        measured = alternative
    if measured is None:
        raise InvalidInput('Pocket import requires model.json or prepared-metadata.json')
    return (json.dumps({'definition': {'options': measured}}, sort_keys=True) + '\n').encode()
