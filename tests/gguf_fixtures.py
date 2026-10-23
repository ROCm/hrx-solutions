"""Generate sparse GGUF fixtures; no checkpoint payloads or downloads."""

from pathlib import Path
import struct

from binding.python.models.gguf import tensor_size

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
