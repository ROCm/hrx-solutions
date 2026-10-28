"""Dependency-free admission of the shipped, fixed Pocket helper contract."""
from pathlib import Path
import re

from ...hem_common import HemError
from .. import recipes
from . import _relative


def load_operation():
    package = Path(__file__).resolve().parents[2]
    installed = package / '_native/share/loom-serve/preparers/pocket_tts/operation.json'
    path = installed if (package / '_native').is_dir() else Path(__file__).with_name('pocket-operation.json')
    value, _ = recipes._decode(recipes._read_bytes(path))
    if (type(value) is not dict or set(value) != {'format', 'id', 'tool_revision', 'tool_files', 'inputs', 'outputs'}
            or type(value['format']) is not int or value['format'] != 1
            or value['id'] != 'pocket-alba-assets-v1'
            or type(value['tool_revision']) is not str
            or not re.fullmatch('[0-9a-f]{40}', value['tool_revision'])
            or type(value['tool_files']) is not dict
            or set(value['tool_files']) != {'prepare_assets.py', 'model.json'}
            or any(type(x) is not str or not re.fullmatch('[0-9a-f]{64}', x) for x in value['tool_files'].values())):
        raise HemError('Invalid Pocket preparation contract identity')
    if type(value['inputs']) is not dict or set(value['inputs']) != {'model.safetensors', 'tokenizer.json', 'alba.wav'}:
        raise HemError('Invalid Pocket preparation input inventory')
    for name, source in value['inputs'].items():
        _relative(name)
        if (type(source) is not dict or set(source) != {'kind', 'repo_id', 'revision', 'filename'}
                or source['kind'] != 'huggingface' or type(source['revision']) is not str
                or not re.fullmatch('[0-9a-f]{40}', source['revision'])):
            raise HemError('Invalid Pocket pinned preparation input')
        _relative(source['repo_id'])
        _relative(source['filename'])
    if value['outputs'] != {'rotary.safetensors': {'purpose': 'rotary', 'format': 'safetensors'},
                            'voices.safetensors': {'purpose': 'voices', 'format': 'safetensors'},
                            'model.json': {'purpose': 'prepared_metadata', 'format': 'json'}}:
        raise HemError('Invalid Pocket generated output inventory')
    return value
