"""Bounded, header-only GGUF inspection with no tensor materialization.

The accepted format versions, byte order and GGML block sizes follow the
GGUF parser in HRX revision 6f6176173bb543d0c330a35b78a1837582a76870. Large metadata arrays
are traversed without retaining their contents. No optional dependencies or
native runtime are needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import os
from pathlib import Path
import struct
from typing import Any, BinaryIO, Optional


class GGUFError(ValueError):
    """An unsupported or malformed GGUF file."""


# id -> (name, elements per block, encoded bytes per block). Only types with
# storage traits implemented in pinned HRX are accepted, even for extra tensors.
GGML_TYPES = {
    0: ("F32", 1, 4), 1: ("F16", 1, 2),
    2: ("Q4_0", 32, 18), 3: ("Q4_1", 32, 20),
    6: ("Q5_0", 32, 22), 7: ("Q5_1", 32, 24),
    8: ("Q8_0", 32, 34), 9: ("Q8_1", 32, 40),
    10: ("Q2_K", 256, 84), 11: ("Q3_K", 256, 110),
    12: ("Q4_K", 256, 144), 13: ("Q5_K", 256, 176),
    14: ("Q6_K", 256, 210), 15: ("Q8_K", 256, 292),
    24: ("I8", 1, 1), 25: ("I16", 1, 2),
    26: ("I32", 1, 4), 27: ("I64", 1, 8),
    28: ("F64", 1, 8), 30: ("BF16", 1, 2),
}
_SCALARS = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i",
            6: "f", 7: "B", 10: "Q", 11: "q", 12: "d"}
_MAX_HEADER_BYTES = 64 * 1024 * 1024
_MAX_STRING_BYTES = 1024 * 1024
_MAX_ENTRIES = 100_000
_MAX_ARRAY_ITEMS = 1_000_000
_RETAIN_ARRAY_ITEMS = 4096
_UINT64_MAX = (1 << 64) - 1


@dataclass(frozen=True)
class ArrayInfo:
    value_type: int
    count: int
    values: Optional[tuple[Any, ...]] = None


@dataclass(frozen=True)
class Metadata:
    value_type: int
    value: Any


@dataclass(frozen=True)
class Tensor:
    name: str
    shape: tuple[int, ...]  # GGUF order: dimension zero is the contiguous row.
    ggml_type: int
    offset: int             # Relative to tensor data start.
    size_bytes: int

    @property
    def type_name(self) -> str:
        return GGML_TYPES[self.ggml_type][0]


@dataclass(frozen=True)
class GGUF:
    version: int
    metadata: dict[str, Metadata]
    tensors: dict[str, Tensor]
    header_bytes: int
    data_offset: int
    file_size: int
    tensor_files: dict[str, Path] = field(default_factory=dict)


def tensor_size(shape: tuple[int, ...], ggml_type: int) -> int:
    """Compute native encoded storage without inspecting tensor contents."""
    if ggml_type not in GGML_TYPES:
        raise GGUFError(f"Unsupported GGML tensor type {ggml_type}")
    if not 1 <= len(shape) <= 4 or any(d <= 0 for d in shape):
        raise GGUFError(f"Invalid tensor dimensions {shape}")
    name, block, size = GGML_TYPES[ggml_type]
    if shape[0] % block:
        raise GGUFError(f"Tensor row width {shape[0]} is not divisible by {name} block size {block}")
    elements = math.prod(shape)
    encoded = elements // block * size
    if elements > _UINT64_MAX or encoded > _UINT64_MAX:
        raise GGUFError("Tensor storage size overflows uint64")
    return encoded


class _Reader:
    def __init__(self, stream: BinaryIO, file_size: int):
        self.stream = stream
        self.limit = min(file_size, _MAX_HEADER_BYTES)

    def advance(self, count: int, *, read: bool = True) -> bytes:
        offset = self.stream.tell()
        if count < 0 or offset + count > self.limit:
            raise GGUFError(f"Truncated or oversized GGUF header at byte {offset} (requested {count} bytes)")
        if not read:
            self.stream.seek(count, os.SEEK_CUR)
            return b""
        result = self.stream.read(count)
        if len(result) != count:
            raise GGUFError(f"Truncated GGUF header at byte {offset}")
        return result

    def scalar(self, fmt: str) -> Any:
        return struct.unpack("<" + fmt, self.advance(struct.calcsize(fmt)))[0]

    def string(self, *, retain: bool = True) -> str:
        count = self.scalar("Q")
        if count > _MAX_STRING_BYTES:
            raise GGUFError(f"GGUF string exceeds {_MAX_STRING_BYTES} bytes")
        raw = self.advance(count, read=retain)
        try:
            return raw.decode("utf-8") if retain else ""
        except UnicodeDecodeError as exc:
            raise GGUFError("GGUF string is not valid UTF-8") from exc

    def value(self, value_type: int, *, retain: bool = True) -> Any:
        if value_type == 8:
            return self.string(retain=retain)
        if value_type in _SCALARS:
            result = self.scalar(_SCALARS[value_type])
            if value_type == 7:
                if result not in (0, 1):
                    raise GGUFError(f"Invalid GGUF boolean {result}")
                return bool(result)
            return result
        if value_type != 9:
            raise GGUFError(f"Unsupported GGUF metadata type {value_type}")
        element_type, count = self.scalar("I"), self.scalar("Q")
        if element_type not in (*_SCALARS, 8):
            raise GGUFError(f"Unsupported GGUF array element type {element_type}")
        if count > _MAX_ARRAY_ITEMS:
            raise GGUFError(f"GGUF array exceeds {_MAX_ARRAY_ITEMS} items")
        if count <= _RETAIN_ARRAY_ITEMS:
            values = tuple(self.value(element_type) for _ in range(count))
            return ArrayInfo(element_type, count, values)
        if element_type == 8:
            for _ in range(count):
                self.string(retain=False)
        else:
            self.advance(count * struct.calcsize(_SCALARS[element_type]), read=False)
        return ArrayInfo(element_type, count)


def inspect_gguf(path: Path, *, allow_fragment: bool = False) -> GGUF:
    """Inspect one little-endian GGUF v2/v3 file, reading only its header.

    Limits cap header traversal and retained metadata. Unbuffered reads avoid
    prefetching tensor bytes past the last descriptor. Padding and payloads
    are checked using file size, not read or mapped.
    """
    with Path(path).open("rb", buffering=0) as stream:
        file_size = os.fstat(stream.fileno()).st_size
        reader = _Reader(stream, file_size)
        if reader.advance(4) != b"GGUF":
            raise GGUFError("Missing GGUF magic header")
        version = reader.scalar("I")
        if version not in (2, 3):
            raise GGUFError(f"Unsupported GGUF version or byte order {version}; expected little-endian v2 or v3")
        tensor_count, metadata_count = reader.scalar("Q"), reader.scalar("Q")
        if tensor_count > _MAX_ENTRIES or metadata_count > _MAX_ENTRIES:
            raise GGUFError(f"GGUF entry count exceeds {_MAX_ENTRIES}")
        metadata = {}
        for _ in range(metadata_count):
            key = reader.string()
            if not key or key in metadata:
                raise GGUFError(f"Empty or duplicate GGUF metadata key {key!r}")
            value_type = reader.scalar("I")
            metadata[key] = Metadata(value_type, reader.value(value_type))
        split_count = metadata.get("split.count")
        if any(key.startswith("split.") for key in metadata):
            if split_count is None or (split_count.value != 1 and not allow_fragment):
                raise GGUFError("Split GGUF requires the complete fragment set; inspect all fragments together")
        alignment_entry = metadata.get("general.alignment", Metadata(4, 32))
        alignment = alignment_entry.value
        if alignment_entry.value_type != 4 or not isinstance(alignment, int) or alignment <= 0 or alignment % 8:
            raise GGUFError("general.alignment must be a positive uint32 multiple of 8")
        tensors = {}
        for _ in range(tensor_count):
            name = reader.string()
            if not name or name in tensors:
                raise GGUFError(f"Empty or duplicate GGUF tensor name {name!r}")
            dimension_count = reader.scalar("I")
            if not 1 <= dimension_count <= 4:
                raise GGUFError(f"{name}: invalid dimension count {dimension_count}")
            shape = tuple(reader.scalar("Q") for _ in range(dimension_count))
            ggml_type, offset = reader.scalar("I"), reader.scalar("Q")
            try:
                size = tensor_size(shape, ggml_type)
            except GGUFError as exc:
                raise GGUFError(f"{name}: {exc}") from exc
            if offset % alignment:
                raise GGUFError(f"{name}: tensor offset {offset} is not aligned to {alignment} bytes")
            tensors[name] = Tensor(name, shape, ggml_type, offset, size)
        header_bytes = stream.tell()
    data_offset = (header_bytes + alignment - 1) // alignment * alignment
    if data_offset > file_size:
        raise GGUFError("Truncated GGUF tensor data alignment padding")
    previous_end = 0
    for tensor in sorted(tensors.values(), key=lambda item: item.offset):
        end = tensor.offset + tensor.size_bytes
        if end > _UINT64_MAX or data_offset + end > file_size:
            raise GGUFError(f"{tensor.name}: tensor range exceeds file size")
        if tensor.offset < previous_end:
            raise GGUFError(f"{tensor.name}: tensor storage overlaps another tensor")
        previous_end = end
    return GGUF(version, metadata, tensors, header_bytes, data_offset, file_size)


def inspect_gguf_set(paths) -> GGUF:
    """Inspect all fragments and merge descriptors, never reading tensor data."""
    paths = tuple(Path(path) for path in paths)
    if not paths:
        raise GGUFError("A GGUF set must contain at least one file")
    parts = [inspect_gguf(path, allow_fragment=True) for path in paths]
    first = parts[0]
    tensors = {}
    split_numbers = set()
    tensor_files = {}
    for index, part in enumerate(parts):
        if part.version != first.version:
            raise GGUFError("GGUF fragments have inconsistent format versions")
        for key, value in part.metadata.items():
            if key.startswith("split."):
                continue
            if key in first.metadata and first.metadata[key] != value:
                raise GGUFError(f"GGUF fragments disagree on metadata {key}")
        for key, value in first.metadata.items():
            if key.startswith("split."):
                continue
            if part.metadata.get(key) != value:
                raise GGUFError(f"GGUF fragment {paths[index].name} is missing or disagrees on metadata {key}")
        count = part.metadata.get("split.count")
        number = part.metadata.get("split.no")
        if count is not None or number is not None:
            if count is None or number is None or count.value != len(parts):
                raise GGUFError("Incomplete GGUF fragment set: split.count does not match supplied files")
            if not isinstance(number.value, int) or number.value in split_numbers or not 0 <= number.value < len(parts):
                raise GGUFError("Invalid or duplicate GGUF split.no")
            split_numbers.add(number.value)
        for name, tensor in part.tensors.items():
            if name in tensors:
                raise GGUFError(f"Duplicate tensor across GGUF fragments: {name}")
            tensors[name] = tensor
            tensor_files[name] = paths[index]
    if split_numbers and len(split_numbers) != len(parts):
        raise GGUFError("Some GGUF fragments lack split metadata")
    for part in parts:
        count = part.metadata.get("split.tensors.count")
        if count is not None and count.value != len(tensors):
            raise GGUFError("Incomplete GGUF fragment tensor set: split.tensors.count differs")
    return GGUF(first.version, first.metadata, tensors,
                sum(part.header_bytes for part in parts), first.data_offset,
                sum(part.file_size for part in parts), tensor_files)
