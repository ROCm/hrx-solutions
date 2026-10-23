"""Matching OMP/OpenCode document admission; no ownership or transaction effects.

This decoder retains json.loads nonfinite constants and input-type behavior.
Finite JSON admission belongs to other contracts, not this ownership reader.
"""
import json
import uuid

MAX_BYTES = 1024 * 1024


def _unique(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate field")
        value[key] = item
    return value


def decode_document(data):
    if len(data) > MAX_BYTES:
        raise ValueError()
    value = json.loads(data, object_pairs_hook=_unique)
    def depth(item, level=0):
        if level > 64:
            raise ValueError()
        if isinstance(item, dict):
            for child in item.values(): depth(child, level + 1)
        elif isinstance(item, list):
            for child in item: depth(child, level + 1)
    depth(value)
    if not isinstance(value, dict): raise ValueError()
    return value


def canonical_uuid(value):
    try:
        identifier = uuid.UUID(value)
        return identifier.version == 4 and str(identifier) == value
    except (ValueError, TypeError, AttributeError):
        return False
