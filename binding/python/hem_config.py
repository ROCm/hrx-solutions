"""CPU pipeline qualification, serving configuration and native documents.

No registry or HTTP operations belong here. Lifecycle callers retain the returned
asset dependencies and a configured source bundle until native retirement.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
from types import MappingProxyType
from typing import Mapping, Optional

from . import curated_catalog
from .hem_common import Deadline, HemError, InvalidInput
from .model_store import CatalogAssets, ModelFiles
from .pipeline_validation import available_pipelines, catalog_modules, catalog_roots, resolve_catalog, text_definition
from .qwen_validation import compare_profile, inspect_model, load_profile

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
    "preserve all referenced model assets and pipeline programs/policies, and use "
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
    if identifier in curated_catalog.CATALOGS:
        catalog = curated_catalog.load_catalog(identifier)
        settings = {key: catalog.definition[key] for key in ('pending_requests', 'options')
                    if key in catalog.definition}
        return dict(pipeline=identifier, kind=catalog.definition['kind'], catalog=str(catalog.directory),
                    asset_contract=catalog.contract, assets=list(catalog.assets),
                    description=catalog.description, check_kind='acquisition-presence',
                    defaults=settings, limits={})
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
    if pipeline in curated_catalog.CATALOGS:
        raise InvalidInput('Curated pipelines require their declared named asset set')
    choices = (pipeline,) if pipeline is not None else tuple(
        identifier for identifier in available_pipelines() if identifier not in curated_catalog.CATALOGS)
    if pipeline is not None and pipeline not in available_pipelines():
        raise InvalidInput(f"Unknown pipeline {pipeline!r}; choose from {', '.join(available_pipelines())}")
    try:
        inspection = inspect_model(model, native_index=True)
        if inspection.tokenizer_error is not None:
            raise ValueError(inspection.tokenizer_error)
    except (OSError, UnicodeError, ValueError) as exc:
        raise InvalidInput(f"Cannot inspect model assets: {exc}") from exc
    matches, diagnostics = [], []
    for identifier in choices:
        try:
            profile = load_profile(resolve_catalog(identifier))
            problems = compare_profile(inspection, profile)
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


def _freeze_json(value):
    """Copy finite JSON into recursively immutable snapshots."""
    if isinstance(value, Mapping):
        if any(type(key) is not str for key in value):
            raise InvalidInput('Checked serving objects require string keys')
        return MappingProxyType({key: _freeze_json(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item) for item in value)
    if value is None or type(value) in (str, int, bool):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    raise InvalidInput('Checked serving values must be finite JSON')


def _json_snapshot(value):
    if isinstance(value, Mapping):
        return {key: _json_snapshot(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_snapshot(item) for item in value]
    return value


@dataclass(frozen=True)
class CatalogIdentity:
    pipeline: str
    contract: str
    relative_path: Path
    tree_sha256: str


def _catalog_snapshots(pipeline, deadline):
    # Reuse native retention's complete-root names+content identity. Models are
    # external inputs and never part of this digest.
    from .hem_process import tree_identity
    snapshots = {}
    for candidate in catalog_roots(pipeline):
        deadline.remaining()
        if not candidate.is_dir():
            continue
        try:
            root = candidate.resolve(strict=True)
            root.relative_to(candidate.parent.resolve(strict=True))
        except (OSError, ValueError) as error:
            raise InvalidInput('Packaged catalog root escapes its package') from error
        if root not in snapshots:
            snapshots[root] = tree_identity(root, deadline)
    return snapshots


def _checked_catalog_identity(pipeline, contract, catalog, snapshots, deadline):
    from .hem_process import tree_identity
    if not isinstance(contract, str) or not contract:
        raise InvalidInput('Checked catalog contract identity must be a nonempty string')
    try:
        selected = Path(catalog).resolve(strict=True)
        matches = [(root, selected.relative_to(root)) for root in snapshots
                   if selected.is_relative_to(root)]
        if len(matches) != 1 or not selected.is_dir() or matches[0][1] == Path('.'):
            raise ValueError('selected catalog has no authoritative root')
        root, relative = matches[0]
        if tree_identity(root, deadline) != snapshots[root]:
            raise InvalidInput('Pipeline catalog changed during serving resolution')
        return CatalogIdentity(pipeline, contract, relative, snapshots[root])
    except (OSError, ValueError) as error:
        raise InvalidInput('Selected pipeline is outside or missing from the packaged catalog') from error


@dataclass(frozen=True)
class ResolvedConfiguration:
    model: ModelFiles | CatalogAssets
    pipeline: str
    catalog: Path
    settings: Mapping
    sources: Mapping
    definition: Mapping
    catalog_identity: CatalogIdentity
    dependencies: tuple[Path, ...]

    def __post_init__(self):
        for name in ('settings', 'sources', 'definition'):
            object.__setattr__(self, name, _freeze_json(getattr(self, name)))
        object.__setattr__(self, 'dependencies', tuple(dict.fromkeys(self.dependencies)))

    def settings_document(self) -> dict:
        return _json_snapshot(self.settings)

    def sources_document(self) -> dict:
        return _json_snapshot(self.sources)

    def definition_document(self) -> dict:
        return _json_snapshot(self.definition)


def resolve(model: ModelFiles, overrides: Optional[Mapping] = None,
            config_path: Optional[Path] = None, *, pipeline: Optional[str] = None,
            deadline: Optional[Deadline] = None) -> ResolvedConfiguration:
    """Repeat structural qualification, then resolve flags > file > defaults.

    `overrides` contains explicit serving settings only (omit absent CLI values).
    `sources` records the winner for every submitted setting, including defaults.
    No writes, native calls, model hashing or registry reservations occur here.
    """
    budget = deadline or Deadline.after()
    budget.remaining()
    if isinstance(model, CatalogAssets):
        if config_path is not None or overrides:
            raise InvalidInput('Qwen serving settings and --config are unsupported for curated pipelines')
        snapshots = _catalog_snapshots(model.pipeline, budget)
        catalog = curated_catalog.check_assets(model, pipeline)
        from .model_store import readable_file
        if any(not readable_file(asset.path) for asset in model.files):
            raise InvalidInput('Curated model assets are missing or unreadable')
        settings = {key: catalog.definition[key] for key in ('pending_requests', 'options')
                    if key in catalog.definition}
        definition = curated_catalog.definition_for(catalog, model)
        identity = _checked_catalog_identity(model.pipeline, catalog.contract, catalog.directory,
                                             snapshots, budget)
        resolved = ResolvedConfiguration(model, model.pipeline, catalog.directory, settings,
                                         dict.fromkeys(settings, 'catalog'), definition, identity,
                                         tuple(asset.path for asset in model.files))
        budget.remaining()
        return resolved
    file_settings = load_config(config_path)
    flags = _settings(overrides if overrides is not None else {}, "flags")
    snapshots = _catalog_snapshots(pipeline, budget)
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
    catalog = Path(info['catalog'])
    try:
        modules = catalog_modules(catalog)
    except (OSError, UnicodeError, ValueError) as exc:
        raise InvalidInput(f"Cannot resolve pipeline modules: {exc}") from exc
    definition = text_definition(model, modules, rows=settings['rows'],
                                 context_capacity=settings['context_capacity'],
                                 pool_capacity=settings['pool_capacity'],
                                 prefill_capacity=settings['prefill_capacity'],
                                 draft_depth=settings['mtp_depth'] or 0)
    identity = _checked_catalog_identity(selected, info['profile'], catalog, snapshots, budget)
    dependencies = (*model.gguf_paths, model.weights, model.tokenizer)
    resolved = ResolvedConfiguration(model, selected, catalog, settings, sources, definition, identity,
                                     dependencies)
    budget.remaining()
    return resolved


def _matching_catalog(resolved, catalog, deadline):
    """Verify the same checked root/relative program closure, without selecting."""
    from .hem_process import tree_identity
    identity = resolved.catalog_identity
    relative = identity.relative_path
    if (identity.pipeline != resolved.pipeline or relative.is_absolute()
            or not relative.parts or '..' in relative.parts):
        raise InvalidInput('Checked pipeline catalog identity is inconsistent')
    catalog = Path(catalog).absolute()
    root = catalog
    for _part in relative.parts:
        root = root.parent
    try:
        root = root.resolve(strict=True)
        selected = catalog.resolve(strict=True)
        if not selected.is_dir() or selected.relative_to(root) != relative:
            raise ValueError('selected catalog location differs')
        if tree_identity(root, deadline) != identity.tree_sha256:
            raise InvalidInput('Deployment catalog differs from the checked serving interpretation')
    except (OSError, ValueError) as error:
        raise InvalidInput('Deployment catalog is outside or missing from the checked root') from error
    return catalog


def deployment_document(resolved: ResolvedConfiguration, name: str, *,
                        catalog: Path, device: int = 0, deadline: Optional[Deadline] = None) -> dict:
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
    budget = deadline or Deadline.after()
    budget.remaining()
    catalog = _matching_catalog(resolved, catalog, budget)
    definition = resolved.definition_document()
    budget.remaining()
    return {"name": name, "device": device, "admission": "drained",
            "base_directory": str(catalog), "definition": definition}


def export_deployment(model: ModelFiles, output: Path, name: str, *,
                      overrides: Optional[Mapping] = None, config_path: Optional[Path] = None,
                      pipeline: Optional[str] = None, deadline: Optional[Deadline] = None) -> dict:
    """Validate and atomically export, without registry access or native calls."""
    budget = deadline or Deadline.after()
    resolved = resolve(model, overrides, config_path, pipeline=pipeline, deadline=budget)
    try:
        document = deployment_document(resolved, name, catalog=resolved.catalog, deadline=budget)
        output = Path(output).absolute()
        budget.remaining()
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
            budget.remaining()
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
            "pipeline": resolved.pipeline, "output": str(output), "settings": resolved.settings_document()}
