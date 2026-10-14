"""Generate sparse GGUF fixtures; no checkpoint payloads or downloads."""

import json
from pathlib import Path
import struct

from binding.python.gguf import tensor_size

PROFILE_PATH = Path(__file__).resolve().parents[1] / 'loom-pipeline/llm/qwen/qwen_27b/compatibility.json'


def profile(architecture='qwen_27b'):
    return json.loads((PROFILE_PATH.parent.parent / architecture / 'compatibility.json').read_text())


def expected_tensors(contract=None):
    contract = contract or profile()
    result = dict(contract['tensors']['global'])
    for layer in range(contract['base_layer_count']):
        family = 'attention' if layer % contract['full_attention_interval'] == 3 else 'gdn'
        entries = dict(contract['tensors'][family])
        entries.update(contract['tensors'].get('layer_overrides', {}).get(str(layer), {}))
        result.update({f'blk.{layer}.{name}': value for name, value in entries.items()})
    if contract['mtp_layer'] is not None:
        result.update({f"blk.{contract['mtp_layer']}.{name}": value for name, value in contract['tensors']['mtp'].items()})
    return result


def string(value):
    encoded = value.encode('utf-8')
    return struct.pack('<Q', len(encoded)) + encoded


def encoded_value(value_type, value):
    formats = {0:'B',1:'b',2:'H',3:'h',4:'I',5:'i',6:'f',7:'B',10:'Q',11:'q',12:'d'}
    if value_type == 8:
        return string(value)
    if value_type == 9:
        subtype, values = value
        return struct.pack('<IQ', subtype, len(values)) + b''.join(encoded_value(subtype, item) for item in values)
    return struct.pack('<'+formats[value_type], value)


def write_gguf(path, *, metadata=(), tensors=(), version=3, alignment=32):
    """Entries are sequences to allow duplicate/malformed-name tests."""
    data = bytearray(b'GGUF' + struct.pack('<IQQ', version, len(tensors), len(metadata)))
    for key, value_type, value in metadata:
        data.extend(string(key) + struct.pack('<I', value_type) + encoded_value(value_type, value))
    end = 0
    for name, shape, ggml_type, offset in tensors:
        data.extend(string(name) + struct.pack('<I', len(shape)))
        data.extend(struct.pack('<'+'Q'*len(shape), *shape))
        data.extend(struct.pack('<IQ', ggml_type, offset))
        try:
            end = max(end, offset + tensor_size(tuple(shape), ggml_type))
        except ValueError:
            pass  # Deliberately malformed descriptors need no backing storage.
    header_bytes = len(data)
    data_offset = (header_bytes + alignment - 1) // alignment * alignment
    with Path(path).open('wb') as stream:
        stream.write(data)
        stream.truncate(data_offset + end)
    return header_bytes, data_offset


def write_matching_gguf(path, *, contract=None, changes=None, missing=(), extra=()):
    contract = contract or profile()
    metadata=[]
    for key,value in contract['metadata'].items():
        if isinstance(value,list):
            kind,value=9,(4,value)
        else:
            kind=8 if isinstance(value,str) else 6 if isinstance(value,float) else 4
        metadata.append((key,kind,value))
    for key,entry in contract['metadata_arrays'].items():
        repeated='' if entry['type']==8 else 0
        metadata.append((key,9,(entry['type'],[repeated]*entry['count'])))
    from binding.python.gguf import GGML_TYPES
    types={value[0]:key for key,value in GGML_TYPES.items()}
    tensors=[]
    offset=0
    expected=expected_tensors(contract)
    for name,update in (changes or {}).items():
        expected[name]={**expected[name],**update}
    for name,spec in expected.items():
        if name in missing:
            continue
        kind=types[spec['type']]
        tensors.append((name,spec['shape'],kind,offset))
        offset+=tensor_size(tuple(spec['shape']),kind)
        offset=(offset+31)//32*32
    tensors.extend(extra)
    return write_gguf(path,metadata=metadata,tensors=tensors)
