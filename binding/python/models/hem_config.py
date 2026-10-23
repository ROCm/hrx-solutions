"""Pure recipe configuration and checked native documents for every modality."""
from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Mapping, Optional

from ..hem_common import Deadline, HemError, InvalidInput
from . import pipelines, recipes
from .recipe_types import PreparedModel, Recipe, freeze_json, json_document

EXPORT_RESPONSIBILITY = (
    "Export does not reserve dependencies or validate a running server. You must "
    "preserve all referenced model assets and pipeline programs/policies, and use "
    "a compatible server for the entire raw-curl registration lifetime. Package "
    "upgrades or external file changes may invalidate paths; revalidate/regenerate "
    "before use."
)


def load_config(path: Optional[Path]) -> dict:
    """Read strict JSON; descriptor field admission happens for every layer."""
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
        if not isinstance(value, dict):
            raise InvalidInput("configuration must be a flat object of serving settings")
        freeze_json(value)
        return value
    except (OSError, UnicodeError, ValueError, RecursionError) as exc:
        raise InvalidInput(f"Cannot read configuration {path}: {exc}") from exc


def _settings(values, source, descriptor):
    if not isinstance(values, Mapping):
        raise InvalidInput(f"{source} must be an object of serving settings")
    try:
        freeze_json(values)
    except (ValueError, RecursionError) as error:
        raise InvalidInput(f"{source}: expected finite JSON settings") from error
    for key, value in values.items():
        where = f"{source}.{key}"
        parameter = descriptor.parameters.get(key)
        if parameter is None:
            raise InvalidInput(f"{where}: unknown parameter for modality {descriptor.modality}")
        if parameter.state == 'unsupported':
            raise InvalidInput(f"{where}: unsupported for {descriptor.id}: {parameter.reason}")
        valid_type = (type(value) is int if parameter.type == 'integer' else
                      type(value) is bool if parameter.type == 'boolean' else
                      isinstance(value, Mapping) if parameter.type == 'object' else False)
        if not valid_type:
            raise InvalidInput(f"{where}: expected {parameter.type}")
        if parameter.state == 'fixed':
            if value != parameter.value:
                raise InvalidInput(f"{where}: expected fixed value {json_document(parameter.value)!r}")
        elif parameter.state == 'configurable':
            limits = parameter.constraints
            if (limits.minimum is not None and value < limits.minimum or
                    limits.maximum is not None and value > limits.maximum):
                raise InvalidInput(f"{where}: outside permitted range {limits.minimum}..{limits.maximum}")
            if limits.allowed_keys is not None and set(value) - set(limits.allowed_keys):
                raise InvalidInput(f"{where}: unsupported object keys")
        else:
            raise HemError(f"Pipeline {descriptor.id}: invalid parameter state {parameter.state!r}")
    return values


def effective_settings(recipe, descriptor, file_settings=None, overrides=None):
    """Validate fields at each layer, then conditional applicability at the end."""
    values, sources, ranks = {}, {}, {}
    for key, parameter in descriptor.parameters.items():
        if parameter.state in ('configurable', 'fixed'):
            values[key] = parameter.default if parameter.state == 'configurable' else parameter.value
            sources[key], ranks[key] = 'default', 0
    _settings(values, 'default', descriptor)
    for rank, (label, layer) in enumerate((('recipe', recipe.configuration),
                                         ('file', {} if file_settings is None else file_settings),
                                         ('override', {} if overrides is None else overrides)), 1):
        _settings(layer, label, descriptor)
        for key, value in layer.items():
            values[key], sources[key], ranks[key] = value, label, rank
    for key, parameter in descriptor.parameters.items():
        condition = getattr(parameter, 'active_when', None)
        if condition is not None and values.get(condition['parameter']) != condition['equals']:
            controlling = condition['parameter']
            if key in values and ranks[key] > 0 and ranks[key] >= ranks[controlling]:
                raise InvalidInput(f"configuration.{key}: explicit value conflicts with inactive {controlling}")
            values.pop(key, None)
            sources[key] = {'state': 'inactive', 'by': sources[controlling]}
    return freeze_json(values), freeze_json(sources)


def _document(value):
    if is_dataclass(value):
        return {field.name: _document(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {key: _document(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_document(item) for item in value]
    return value


def _catalog(descriptor):
    relative = Path(descriptor.bundle.catalog_relative_path)
    if relative.is_absolute() or not relative.parts or any(part in ('.', '..') for part in relative.parts):
        raise HemError(f"Pipeline {descriptor.id}: invalid catalog path")
    for root in pipelines.catalog_roots():
        candidate = root / relative
        if candidate.is_dir():
            try:
                candidate.resolve(strict=True).relative_to(root.resolve(strict=True))
            except (OSError, ValueError) as error:
                raise HemError(f"Pipeline {descriptor.id}: catalog escapes its root") from error
            return candidate.resolve(strict=True)
    raise HemError(f"No packaged catalog for pipeline {descriptor.id!r}")


def pipeline_info(identifier: str) -> dict:
    descriptor = pipelines.get(identifier).describe()
    return dict(_document(descriptor), pipeline=descriptor.id, catalog=str(_catalog(descriptor)))


def available_pipelines() -> tuple[str, ...]:
    """Discovery validates all recipes; selected serving never calls this."""
    return tuple(sorted({recipe.pipeline for recipe in recipes.list()}))


@dataclass(frozen=True)
class CatalogIdentity:
    pipeline: str
    contract: str
    relative_path: Path
    tree_sha256: str


def _catalog_snapshots(pipeline, deadline):
    # Reuse native retention's complete-root names+content identity. Models are
    # external inputs and never part of this digest.
    from ..control.hem_process import tree_identity
    snapshots = {}
    for candidate in pipelines.catalog_roots():
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
    from ..control.hem_process import tree_identity
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
    model: PreparedModel
    pipeline: str
    catalog: Path
    settings: Mapping
    sources: Mapping
    definition: Mapping
    catalog_identity: CatalogIdentity
    dependencies: tuple[Path, ...]
    recipe_identity: str
    modality: str

    def __post_init__(self):
        for name in ('settings', 'sources', 'definition'):
            object.__setattr__(self, name, freeze_json(getattr(self, name)))
        object.__setattr__(self, 'dependencies', tuple(dict.fromkeys(self.dependencies)))

    def settings_document(self):
        return json_document(self.settings)

    def sources_document(self):
        return json_document(self.sources)

    def definition_document(self):
        return json_document(self.definition)


def _prepared(recipe, model, budget):
    if (recipe.id != model.recipe_id or recipe.identity != model.recipe_identity or
            recipe.pipeline != model.pipeline):
        raise InvalidInput('Prepared model and recipe declaration/pipeline identities differ')
    expected = {item.id: item for item in recipe.artifacts}
    actual = {item.id: item for item in model.files}
    if len(actual) != len(model.files) or not expected.keys() <= actual.keys():
        raise InvalidInput('Prepared model artifact inventory differs from its recipe')
    shards = tuple(item for item in model.files if item.role == 'weights-shard')
    has_index = any(item.role == 'weights' and item.format == 'json' and
                    item.logical_path.endswith('.safetensors.index.json') for item in model.files)
    for item in model.files:
        budget.remaining()
        declared = expected.get(item.id)
        imported = item.ownership in ('caller-reference', 'caller-owned')
        index = (item.role == 'weights' and item.format == 'json' and shards
                 and item.logical_path.endswith('.safetensors.index.json'))
        if declared is None:
            consistent = imported and has_index and item.role == 'weights-shard' and item.format == 'gguf'
        else:
            consistent = (item.role == declared.role and
                          ((item.format, item.logical_path) == (declared.format, declared.path) or index))
            if not imported:
                consistent = consistent and item.ownership == (
                    'hf-reference' if declared.source.kind == 'huggingface' and not index else 'generated-owned')
        logical = Path(item.logical_path)
        if (not consistent or logical.is_absolute() or not logical.parts or
                '..' in logical.parts or '\\' in item.logical_path or
                Path(item.path).absolute() != (Path(model.directory) / logical).absolute() or
                not Path(item.path).is_file() or not os.access(item.path, os.R_OK)):
            raise InvalidInput(f'Prepared artifact {item.id!r} is inconsistent, missing or unreadable')


def resolve(recipe: Recipe, model: PreparedModel, overrides: Optional[Mapping] = None,
            config_path: Optional[Path] = None, *, deadline: Optional[Deadline] = None) -> ResolvedConfiguration:
    """Resolve the selected recipe through one shared merge and modality policy."""
    from . import modalities
    budget = deadline or Deadline.after()
    budget.remaining()
    snapshots = _catalog_snapshots(recipe.pipeline, budget)
    binding = pipelines.get(recipe.pipeline)
    descriptor = binding.describe()
    if descriptor.modality != recipe.modality:
        raise InvalidInput('Recipe and selected pipeline modalities differ')
    binding.validate_artifacts(recipe.artifacts)
    _prepared(recipe, model, budget)
    modality = modalities.get(recipe.modality)
    settings, sources = effective_settings(recipe, descriptor, load_config(config_path), overrides)
    modality.validate(settings, descriptor)
    budget.remaining()
    catalog = _catalog(descriptor)
    definition = modality.service_definition(settings, model, descriptor.programs)
    identity = _checked_catalog_identity(recipe.pipeline, descriptor.id, catalog, snapshots, budget)
    resolved = ResolvedConfiguration(model, recipe.pipeline, catalog, settings, sources, definition,
                                     identity, tuple(Path(item.path) for item in model.files),
                                     recipe.identity, recipe.modality)
    budget.remaining()
    return resolved


def _matching_catalog(resolved, catalog, deadline):
    """Verify the same checked root/relative program closure, without selecting."""
    from ..control.hem_process import tree_identity
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


def export_deployment(recipe: Recipe, model: PreparedModel, output: Path, name: str, *,
                      overrides: Optional[Mapping] = None, config_path: Optional[Path] = None,
                      deadline: Optional[Deadline] = None) -> dict:
    """Validate and atomically export, without registry access or native calls."""
    budget = deadline or Deadline.after()
    resolved = resolve(recipe, model, overrides, config_path, deadline=budget)
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
