"""Compare GGUF descriptors with the packaged Qwen architecture contract."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Optional, TYPE_CHECKING

from .gguf import ArrayInfo, GGUF, Metadata, inspect_gguf_set

if TYPE_CHECKING:
    from .model_store import ModelFiles


_INTEGER_TYPES = {0, 1, 2, 3, 4, 5, 10, 11}


@dataclass(frozen=True)
class ModelInspection:
    """Descriptor and tokenizer facts reusable across candidate profiles."""

    weights: GGUF
    tokenizer_data: Any
    tokenizer_error: Optional[str] = None


def inspect_model(
    model: ModelFiles, *, tokenizer_data: Optional[dict] = None, native_index: bool = False,
) -> ModelInspection:
    """Inspect descriptors/tokenizer, optionally admitting a native serving index.

    Public descriptor validation accepts explicit GGUF fragments independently of
    the native index carrier. Serving selection additionally requires that index.
    """
    weights = inspect_gguf_set(model.gguf_paths)
    if native_index and (model.weights not in model.gguf_paths or len(model.gguf_paths) > 1):
        if not str(model.weights).endswith(".safetensors.index.json"):
            raise ValueError("Fragmented weights require a .safetensors.index.json native index")
        index = json.loads(Path(model.weights).read_text(encoding="utf-8"))
        weight_map = index.get("weight_map") if isinstance(index, dict) else None
        expected = {name: Path(path).absolute() for name, path in weights.tensor_files.items()}
        actual = ({name: (model.weights.parent / path).absolute() for name, path in weight_map.items()}
                  if isinstance(weight_map, dict) and all(isinstance(path, str) for path in weight_map.values()) else {})
        if actual != expected:
            raise ValueError("Native weights index does not match inspected GGUF tensor mappings")
    tokenizer_error = None
    if tokenizer_data is None:
        try:
            tokenizer_data = json.loads(Path(model.tokenizer).read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            tokenizer_error = f"Tokenizer is not valid UTF-8 JSON: {exc}"
    return ModelInspection(weights, tokenizer_data, tokenizer_error)


def compare_profile(inspection: ModelInspection, profile: dict) -> tuple[str, ...]:
    """Compare one inspected asset set with a candidate's exact contract."""
    return compare_gguf(inspection.weights, profile) + (
        (inspection.tokenizer_error,) if inspection.tokenizer_error is not None
        else _compare_tokenizer_data(inspection.tokenizer_data, profile))


def load_profile(catalog: Path) -> dict:
    """Load policy beside the source catalog, independent of native tools."""
    path = Path(catalog) / "compatibility.json"
    profile = json.loads(path.read_text(encoding="utf-8"))
    if profile.get("profile_version") != 1:
        raise ValueError(f"Unsupported Qwen compatibility profile version in {path}")
    return profile


def expected_tensors(profile: dict) -> dict[str, dict]:
    """Expand the reviewed repeated layer schemas into exact tensor names."""
    schemas = profile["tensors"]
    expected = dict(schemas["global"])
    interval = profile["full_attention_interval"]
    overrides = schemas.get("layer_overrides", {})
    valid_layers = {str(layer) for layer in range(profile["base_layer_count"])}
    if set(overrides) - valid_layers:
        raise ValueError("Qwen profile contains overrides for nonexistent base layers")
    for layer in range(profile["base_layer_count"]):
        family = "attention" if (layer + 1) % interval == 0 else "gdn"
        selected = dict(schemas[family])
        changes = overrides.get(str(layer), {})
        if set(changes) - set(selected):
            raise ValueError(f"Qwen profile overrides unknown tensors in layer {layer}")
        selected.update(changes)
        expected.update({f"blk.{layer}.{name}": spec for name, spec in selected.items()})
    mtp_layer = profile["mtp_layer"]
    if mtp_layer is not None:
        expected.update({f"blk.{mtp_layer}.{name}": spec for name, spec in schemas["mtp"].items()})
    elif schemas.get("mtp"):
        raise ValueError("Qwen profile has MTP tensors without an MTP layer")
    return expected


def _matches(entry: Metadata, expected: Any) -> bool:
    actual = entry.value
    if isinstance(expected, list):
        if not isinstance(actual, ArrayInfo) or actual.values is None:
            return False
        if isinstance(expected[0], bool):
            return (actual.value_type in (*_INTEGER_TYPES, 7)
                    and all(value in (0, 1) for value in actual.values)
                    and list(actual.values) == expected)
        return actual.value_type in _INTEGER_TYPES and list(actual.values) == expected
    if isinstance(expected, str):
        return entry.value_type == 8 and actual == expected
    if isinstance(expected, int):
        return entry.value_type in _INTEGER_TYPES and actual == expected
    if isinstance(expected, float):
        return (entry.value_type in (6, 12) and math.isfinite(actual)
                and math.isclose(actual, expected, rel_tol=1e-6, abs_tol=0.0))
    return False


def _describe(entry: Metadata) -> str:
    value = entry.value
    if isinstance(value, ArrayInfo):
        return repr(list(value.values)) if value.values is not None else f"array(type={value.value_type}, count={value.count})"
    return f"{value!r} (metadata type {entry.value_type})"


def compare_gguf(model: GGUF, profile: dict) -> tuple[str, ...]:
    """Return structural mismatches; metadata labels and file_type are advisory.

    All native base/MTP bindings must match exactly. Structurally valid
    auxiliary tensors outside blk.* are unused by this text catalog. Unknown
    qwen35 configuration is rejected so unsupported execution settings cannot
    silently change model semantics.
    """
    problems = []
    required = profile["metadata"]
    optional = profile["optional_metadata"]
    for key, expected in required.items():
        actual = model.metadata.get(key)
        if actual is None:
            problems.append(f"Missing required metadata: {key} (expected {expected!r})")
        elif not _matches(actual, expected):
            problems.append(f"{key}: expected {expected!r}, found {_describe(actual)}")
    for key, expected in optional.items():
        actual = model.metadata.get(key)
        if actual is None:
            continue
        # llama.cpp's default scaling type is linear with factor 1: this is
        # numerically identical to the catalog's unscaled RoPE implementation.
        if key == "qwen35.rope.scaling.type" and actual.value_type == 8 and actual.value in ("none", "linear"):
            continue
        # Zero asks llama.cpp to use 1/sqrt(key_length), also 0.0625 here.
        if key == "qwen35.attention.scale" and actual.value_type in (6, 12) and actual.value == 0:
            continue
        if not _matches(actual, expected):
            problems.append(f"{key}: expected {expected!r}, found {_describe(actual)}")
    for key in model.metadata:
        if key.startswith("qwen35.") and key not in required and key not in optional:
            problems.append(f"Unsupported Qwen configuration: {key}")
    for key, expected in profile["metadata_arrays"].items():
        actual = model.metadata.get(key)
        array = actual.value if actual is not None else None
        if not isinstance(array, ArrayInfo) or array.count != expected["count"] or array.value_type != expected["type"]:
            found = _describe(actual) if actual is not None else "missing"
            problems.append(f"{key}: expected array(type={expected['type']}, count={expected['count']}), found {found}")
    expected = expected_tensors(profile)
    for name, spec in expected.items():
        category = "MTP" if name.startswith(f"blk.{profile['mtp_layer']}.") else "Base model"
        actual = model.tensors.get(name)
        if actual is None:
            problems.append(f"{category}: missing required tensor {name}")
            continue
        if actual.shape != tuple(spec["shape"]):
            problems.append(f"{name}: expected shape {spec['shape']} in GGUF dimension order, found {list(actual.shape)}")
        if actual.type_name != spec["type"]:
            problems.append(f"{name}: expected {spec['type']}, found {actual.type_name}")
    for name in model.tensors:
        if name.startswith("blk.") and name not in expected:
            problems.append(f"Unexpected model-block tensor: {name}")
        elif name == "output.weight" and name not in expected:
            problems.append("Unexpected untied output.weight for this tied-output catalog")
    return tuple(problems)


def compare_tokenizer(path: Path, profile: dict, *, data: Optional[dict] = None) -> tuple[str, ...]:
    """Require the qualified tokenizer configuration, ignoring JSON formatting.

    Native tokenization uses tokenizer.json rather than the GGUF tokenizer
    arrays. This checks that actual runtime input; it does not tokenize text or
    run a model. Weight identity is deliberately not compared.
    """
    try:
        if data is None:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        return (f"Tokenizer is not valid UTF-8 JSON: {exc}",)
    return _compare_tokenizer_data(data, profile)


def _compare_tokenizer_data(data: Any, profile: dict) -> tuple[str, ...]:
    if not isinstance(data, dict) or not isinstance(data.get("model"), dict):
        return ("Tokenizer JSON does not contain a model object",)
    normalized = json.dumps(data, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(normalized).hexdigest()
    if digest != profile["tokenizer_canonical_sha256"]:
        return ("Tokenizer configuration differs from the qualified Qwen tokenizer (vocabulary, token IDs, merges or tokenization settings); use the matching tokenizer.json",)
    return ()
