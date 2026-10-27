"""Pure admission of the packaged contract for the fixed trusted operation."""
from __future__ import annotations

import json
from pathlib import Path
import re

from ...hem_common import HemError
from . import _relative

RELATIVE = 'share/loom-serve/preparers/moonshine'
UPSTREAM = 'experimental/loom_serve/models/moonshine'


def _unique(pairs):
    value = {}
    for key, child in pairs:
        if key in value:
            raise ValueError('duplicate key ' + key)
        value[key] = child
    return value


def operation_path():
    package = Path(__file__).resolve().parents[2]
    native = package / '_native'
    return (native / RELATIVE / 'operation.json' if native.is_dir() else
            Path(__file__).with_name('moonshine-operation.json'))


def load_operation(operation='moonshine-assets-v1'):
    """Only known implementation capabilities; data never supplies executable code."""
    if operation != 'moonshine-assets-v1':
        raise HemError(f'Unsupported trusted preparation operation: {operation}')
    try:
        path = operation_path()
        if path.is_symlink():
            raise ValueError('operation contract cannot be an alias')
        content = path.read_bytes()
        if len(content) > 1024 * 1024:
            raise ValueError('operation contract exceeds size limit')
        value = json.loads(content, object_pairs_hook=_unique)
        if (type(value) is not dict or set(value) !=
                {'format', 'id', 'tool_revision', 'tool_files', 'inputs', 'outputs'}
                or type(value['format']) is not int or value['format'] != 1
                or value['id'] != operation
                or type(value['tool_revision']) is not str
                or not re.fullmatch('[0-9a-f]{40}', value['tool_revision'])):
            raise ValueError('invalid operation identity/shape')
        if type(value['tool_files']) is not dict or set(value['tool_files']) != {'prepare_assets.py', 'model.json'}:
            raise ValueError('invalid fixed tool inventory')
        for checksum in value['tool_files'].values():
            if type(checksum) is not str or not re.fullmatch('[0-9a-f]{64}', checksum):
                raise ValueError('invalid tool hash')
        if type(value['inputs']) is not dict or not value['inputs']:
            raise ValueError('invalid operation inputs')
        for name, source in value['inputs'].items():
            _relative(name)
            if (type(source) is not dict or set(source) != {'kind', 'repo_id', 'revision', 'filename'}
                    or source['kind'] != 'huggingface' or source['filename'] != name
                    or type(source['revision']) is not str
                    or not re.fullmatch('[0-9a-f]{40}', source['revision'])):
                raise ValueError('invalid pinned operation input')
            _relative(source['repo_id'])
        if type(value['outputs']) is not dict or not value['outputs']:
            raise ValueError('invalid operation outputs')
        purposes = set()
        for name, output in value['outputs'].items():
            _relative(name)
            if (type(output) is not dict or set(output) != {'purpose', 'format'}
                    or output['format'] not in ('safetensors', 'json', 'gguf')
                    or type(output['purpose']) is not str
                    or not re.fullmatch('[A-Za-z0-9][A-Za-z0-9_.-]*', output['purpose'])
                    or output['purpose'] in purposes):
                raise ValueError('invalid operation output')
            purposes.add(output['purpose'])
        return value
    except (OSError, ValueError, TypeError, RecursionError) as error:
        raise HemError(f'Invalid trusted preparation operation contract: {error}') from error
