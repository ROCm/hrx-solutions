"""CPU pipeline qualification, serving configuration and native documents.

No registry or HTTP operations belong here. Lifecycle callers retain the returned
asset dependencies and a configured source bundle until native retirement.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Mapping, Optional

from .gguf import inspect_gguf_set
from .hem_common import HemError, InvalidInput
from .model_store import ModelFiles
from .pipeline_validation import available_pipelines, catalog_modules, resolve_catalog
from .qwen_validation import compare_gguf, compare_tokenizer, load_profile

# Bounds in pinned configuration.c and Qwen prepare.loom, not memory-fit promises.
# Meanings: pinned server/configuration.h and service/text/{model,packing}.h.
# Help and validation consume this same metadata; parser omissions remain None.
SETTINGS = {
    "rows": (4, 16, "Maximum simultaneous execution rows for this model; queued requests may exceed this count"),
    "context_capacity": (8192, 262144, "Logical context token ceiling per row, including retained/input and generated tokens; not eagerly allocated storage"),
    "pool_capacity": (32768, 4194304, "Shared KV cache token positions across rows; must be divisible by source-declared page size, with page-rounded reservations"),
    "prefill_capacity": (128, 262144, "Maximum packed input tokens per model traversal across rows, also the isolated prefill chunk limit"),
    "mtp_depth": (3, 3, "Proposal tokens after each anchor for multi-token prediction (MTP); applicable only when supported and enabled"),
}
LIMITS = {key: metadata[1] for key, metadata in SETTINGS.items()}
DEFAULTS = {key: metadata[0] for key, metadata in SETTINGS.items() if key != "mtp_depth"}
EXPORT_RESPONSIBILITY = (
    "Export does not reserve dependencies or validate a running server. You must "
    "preserve the referenced weights, tokenizer and pipeline modules, and use "
    "a compatible server for the entire raw-curl registration lifetime. Package "
    "upgrades or external file changes may invalidate paths; revalidate/regenerate "
    "before use."
)


def _settings(values: Mapping, source: str) -> dict:
    if not isinstance(values, Mapping):
        raise InvalidInput(f"{source} must be a flat object of serving settings")
    unknown = set(values) - (set(LIMITS) | {"mtp"})
    if unknown:
        raise InvalidInput(f"Unknown {source} settings: {', '.join(sorted(map(str, unknown)))}")
    result = dict(values)
    for key, value in result.items():
        if key == "mtp":
            if type(value) is not bool:
                raise InvalidInput(f"{source} mtp must be a boolean")
        elif type(value) is not int or not 1 <= value <= LIMITS[key]:
            raise InvalidInput(f"{source} {key} must be an integer from 1 through {LIMITS[key]}")
    return result


def load_config(path: Optional[Path]) -> dict:
    """Strict JSON: duplicate/unknown keys, nonfinite values and null fail."""
    if path is None:
        return {}
    def pairs(entries):
        result = {}
        for key, value in entries:
            if key in result:
                raise InvalidInput(f"Duplicate configuration key: {key}")
            result[key] = value
        return result
    def constant(value):
        raise InvalidInput(f"Nonfinite JSON value is unsupported: {value}")
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"),
                           object_pairs_hook=pairs, parse_constant=constant)
    except (OSError, UnicodeError, ValueError) as exc:
        raise InvalidInput(f"Cannot read configuration {path}: {exc}") from exc
    return _settings(value, "configuration")


def pipeline_info(identifier: str) -> dict:
    if identifier not in available_pipelines():
        raise InvalidInput(f"Unknown pipeline {identifier!r}; choose from {', '.join(available_pipelines())}")
    try:
        catalog = resolve_catalog(identifier)
        profile = load_profile(catalog)
    except (OSError, ValueError) as exc:
        raise HemError(f"Cannot inspect packaged pipeline {identifier}: {exc}") from exc
    supported = profile.get("capabilities", {}).get("mtp", profile["mtp_layer"] is not None)
    return {"pipeline": identifier, "profile": profile["identifier"],
            "catalog": str(catalog), "compatibility": profile,
            "capabilities": {"mtp": supported},
            "defaults": {**DEFAULTS, "mtp": supported, "mtp_depth": SETTINGS["mtp_depth"][0] if supported else None},
            "limits": {**LIMITS, "mtp_depth": LIMITS["mtp_depth"] if supported else None}}


def select_pipeline(model: ModelFiles, pipeline: Optional[str] = None) -> str:
    """Inspect headers/tokenizer once and require one compatible packaged profile."""
    choices = (pipeline,) if pipeline is not None else available_pipelines()
    if pipeline is not None and pipeline not in available_pipelines():
        raise InvalidInput(f"Unknown pipeline {pipeline!r}; choose from {', '.join(available_pipelines())}")
    try:
        inspection = inspect_gguf_set(model.gguf_paths)
        if model.weights not in model.gguf_paths or len(model.gguf_paths) > 1:
            if not str(model.weights).endswith(".safetensors.index.json"):
                raise ValueError("Fragmented weights require a .safetensors.index.json native index")
            index = json.loads(Path(model.weights).read_text(encoding="utf-8"))
            weight_map = index.get("weight_map") if isinstance(index, dict) else None
            expected = {name: Path(path).absolute() for name, path in inspection.tensor_files.items()}
            actual = ({name: (model.weights.parent / path).absolute() for name, path in weight_map.items()}
                      if isinstance(weight_map, dict) and all(isinstance(path, str) for path in weight_map.values()) else {})
            if actual != expected:
                raise ValueError("Native weights index does not match inspected GGUF tensor mappings")
        tokenizer = json.loads(Path(model.tokenizer).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise InvalidInput(f"Cannot inspect model assets: {exc}") from exc
    matches, diagnostics = [], []
    for identifier in choices:
        try:
            profile = load_profile(resolve_catalog(identifier))
            problems = compare_gguf(inspection, profile) + compare_tokenizer(
                model.tokenizer, profile, data=tokenizer)
        except (OSError, ValueError) as exc:
            raise HemError(f"Cannot inspect packaged pipeline {identifier}: {exc}") from exc
        if problems:
            diagnostics.append(f"{identifier}: " + "; ".join(problems[:4]))
        else:
            matches.append(identifier)
    if len(matches) > 1:
        raise InvalidInput("Ambiguous compatible pipelines: " + ", ".join(matches) + "; select --pipeline explicitly")
    if not matches:
        raise InvalidInput("Model is incompatible with " + (pipeline or "all packaged pipelines") + ":\n" + "\n".join(diagnostics))
    return matches[0]


@dataclass(frozen=True)
class ResolvedConfiguration:
    model: ModelFiles
    pipeline: str
    catalog: Path
    settings: dict
    sources: dict

    @property
    def dependencies(self) -> tuple[Path, ...]:
        # weights may be an index distinct from the GGUF members.
        return tuple(dict.fromkeys((*self.model.gguf_paths, self.model.weights, self.model.tokenizer)))


def resolve(model: ModelFiles, overrides: Optional[Mapping] = None,
            config_path: Optional[Path] = None, *, pipeline: Optional[str] = None) -> ResolvedConfiguration:
    """Repeat structural qualification, then resolve flags > file > defaults.

    `overrides` contains explicit serving settings only (omit absent CLI values).
    `sources` records the winner for every submitted setting, including defaults.
    No writes, native calls, model hashing or registry reservations occur here.
    """
    file_settings = load_config(config_path)
    flags = _settings(overrides if overrides is not None else {}, "flags")
    selected = select_pipeline(model, pipeline)
    info = pipeline_info(selected)
    settings = dict(info["defaults"])
    sources = dict.fromkeys(settings, "default")
    for label, layer in (("file", file_settings), ("flag", flags)):
        settings.update(layer)
        sources.update(dict.fromkeys(layer, label))
    if not settings["mtp"]:
        if sources["mtp_depth"] != "default":
            raise InvalidInput("Explicit mtp_depth conflicts with disabled MTP; remove mtp_depth or enable MTP with --mtp")
        settings["mtp_depth"] = None
    elif not info["capabilities"]["mtp"]:
        raise InvalidInput(f"Pipeline {selected} does not support MTP; remove --mtp and mtp_depth")
    if settings["prefill_capacity"] > settings["context_capacity"]:
        raise InvalidInput("prefill_capacity must not exceed context_capacity")
    if settings["prefill_capacity"] < (settings["mtp_depth"] or 0) + 1:
        raise InvalidInput("prefill_capacity must fit one anchor plus every MTP proposal")
    return ResolvedConfiguration(model, selected, Path(info["catalog"]), settings, sources)


def deployment_document(resolved: ResolvedConfiguration, name: str, *,
                        catalog: Path, device: int = 0) -> dict:
    """Create the native drained document using an immutable retained catalog.

    The caller supplies a catalog independent of the deployment name, retaining
    its complete source closure and all `resolved.dependencies` through retirement.
    Offline exports instead leave those lifetime obligations with the operator.
    Paths remain server-local, including an index's referenced GGUF fragments.
    """
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
        raise InvalidInput("Instance name must contain letters, digits, '.', '_' or '-'")
    if type(device) is not int or device < 0:
        raise InvalidInput("Device group index must be a nonnegative integer")
    catalog = Path(catalog).absolute()
    try:
        modules = catalog_modules(catalog)
    except (OSError, UnicodeError, ValueError) as exc:
        raise InvalidInput(f"Cannot resolve pipeline modules: {exc}") from exc
    definition = {"kind": "text", "prepare": modules.prepare, "control": modules.control,
                  "sources": list(modules.providers),
                  "weights": str(resolved.model.weights.absolute()),
                  "tokenizer": str(resolved.model.tokenizer.absolute()),
                  **{key: resolved.settings[key] for key in DEFAULTS},
                  "draft_depth": resolved.settings["mtp_depth"] or 0, "options": {}}
    return {"name": name, "device": device, "admission": "drained",
            "base_directory": str(catalog), "definition": definition}


def export_deployment(model: ModelFiles, output: Path, name: str, *,
                      overrides: Optional[Mapping] = None, config_path: Optional[Path] = None,
                      pipeline: Optional[str] = None) -> dict:
    """Validate and atomically export, without registry access or native calls."""
    resolved = resolve(model, overrides, config_path, pipeline=pipeline)
    try:
        document = deployment_document(resolved, name, catalog=resolved.catalog)
        output = Path(output).absolute()
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=output.parent,
                                         prefix=f".{output.name}.", delete=False) as stream:
            staging = Path(stream.name)
            try:
                json.dump(document, stream, indent=2, allow_nan=False)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            except BaseException:
                staging.unlink(missing_ok=True)
                raise
        try:
            os.replace(staging, output)
            parent = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(parent)
            finally:
                os.close(parent)
        finally:
            staging.unlink(missing_ok=True)
    except (OSError, ValueError) as exc:
        raise HemError(f"Cannot export deployment: {exc}") from exc
    print(EXPORT_RESPONSIBILITY, file=sys.stderr)
    return {"state": "exported", "completed": True, "instance": name,
            "pipeline": resolved.pipeline, "output": str(output), "settings": dict(resolved.settings)}
