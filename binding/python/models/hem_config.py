"""Pure recipe configuration and checked native documents for every modality."""
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

from ..hem_common import Deadline, HemError, InvalidInput
from . import modalities, pipelines, recipes

EXPORT_RESPONSIBILITY = (
    "Export does not reserve dependencies or validate a running server. You must "
    "preserve all referenced model assets and executable programs/policies, and use "
    "a compatible server for the entire raw-curl registration lifetime. Package "
    "upgrades or external file changes may invalidate paths; revalidate/regenerate "
    "before use."
)


def freeze_json(value):
    """Own immutable JSON at the checked boundary; never share caller containers."""
    if isinstance(value, Mapping):
        if any(type(key) is not str for key in value):
            raise ValueError("JSON object keys must be strings")
        return MappingProxyType({key: freeze_json(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(freeze_json(item) for item in value)
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    raise ValueError("expected finite JSON value")


def json_document(value):
    """Copy the checked JSON into existing public/native output containers."""
    if isinstance(value, Mapping):
        return {key: json_document(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [json_document(item) for item in value]
    return value


def load_config(path: Optional[Path]) -> dict:
    """Read strict JSON; selected modality admission happens for every layer."""
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


def effective_settings(recipe, serving, modality, file_settings=None, overrides=None):
    """Admit each layer with concrete policy, then validate final winners once."""
    values, sources, ranks = {}, {}, {}
    layers = (('default', modality.defaults(serving)), ('recipe', recipe['configuration']),
              ('file', {} if file_settings is None else file_settings),
              ('override', {} if overrides is None else overrides))
    for rank, (label, layer) in enumerate(layers):
        if not isinstance(layer, Mapping):
            raise InvalidInput(f'{label}: expected settings object')
        try:
            freeze_json(layer)
            modality.validate_layer(serving, layer)
        except (ValueError, RecursionError) as error:
            raise InvalidInput(f'{label}: expected finite JSON settings') from error
        for key, value in layer.items():
            values[key], sources[key], ranks[key] = value, label, rank
    modality.finalize(serving, values, sources, ranks)
    return freeze_json(values), freeze_json(sources)


def _executable_info(reference, executable, catalog, identity, budget):
    modality = modalities.get(executable['modality'])
    programs = pipelines.programs(executable, catalog, budget)
    if executable['modality'] == 'chat':
        programs = dict(programs)
        programs['providers'] = programs.pop('sources')
    return json_document(freeze_json(dict(
        executable=reference, executable_id=executable['id'], executable_identity=identity,
        modality=executable['modality'], native_contract=executable['native_contract'],
        parameters=modality.parameters(executable['serving']), protocol=modality.PROTOCOL,
        catalog=str(catalog), programs=programs, assets=executable['assets'],
        preparation=executable['preparation'], telemetry={'raw_events':'unverified','metrics':[]})))


def executable_info(reference, *, deadline=None):
    budget = deadline or Deadline.after()
    budget.remaining()
    executable, catalog, identity = pipelines.load(reference)
    return _executable_info(reference, executable, catalog, identity, budget)


def executable_list(modality=None, *, deadline=None):
    budget = deadline or Deadline.after()
    budget.remaining()
    return [_executable_info(reference, executable, catalog, identity, budget)
            for reference, executable, catalog, identity in pipelines.list(modality=modality)]


def _modality_info(name, executables):
    modality = modalities.get(name)
    return json_document(freeze_json(dict(modality=name,
        native_contract=modality.NATIVE_CONTRACT, protocol=modality.PROTOCOL,
        vocabulary=modality.vocabulary(), executables=executables,
        telemetry={'raw_events':'unverified','metrics':[]})))


def modality_info(name, *, deadline=None):
    modalities.get(name)  # Reject unsupported modalities before inventory reads.
    return _modality_info(name, executable_list(modality=name, deadline=deadline))


def modality_list(*, deadline=None):
    rows = executable_list(deadline=deadline)
    return [_modality_info(name, [row for row in rows if row['modality'] == name])
            for name in modalities.NAMES]


def _catalog_snapshots(deadline):
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


def _checked_catalog(catalog, snapshots, deadline):
    from ..control.hem_process import tree_identity
    try:
        selected = Path(catalog).resolve(strict=True)
        matches = [(root, selected.relative_to(root)) for root in snapshots
                   if selected.is_relative_to(root)]
        if len(matches) != 1 or not selected.is_dir() or matches[0][1] == Path('.'):
            raise ValueError('selected catalog has no authoritative root')
        root, relative = matches[0]
        if tree_identity(root, deadline) != snapshots[root]:
            raise InvalidInput('Executable catalog changed during serving resolution')
        return relative, snapshots[root]
    except (OSError, ValueError) as error:
        raise InvalidInput('Selected executable is outside or missing from the packaged catalog') from error


@dataclass(frozen=True)
class ResolvedConfiguration:
    model_id: str
    pipeline: str
    catalog: Path
    settings: Mapping
    sources: Mapping
    definition: Mapping
    catalog_relative_path: Path
    catalog_tree_sha256: str
    dependencies: tuple[Path, ...]
    recipe_identity: str
    modality: str
    evidence: tuple
    executable_id: str
    executable_identity: str

    def __post_init__(self):
        for name in ('settings', 'sources', 'definition', 'evidence'):
            object.__setattr__(self, name, freeze_json(getattr(self, name)))
        object.__setattr__(self, 'dependencies', tuple(dict.fromkeys(self.dependencies)))

    def settings_document(self):
        return json_document(self.settings)

    def sources_document(self):
        return json_document(self.sources)

    def definition_document(self):
        return json_document(self.definition)


def _unchanged_files(paths, dependencies, evidence, budget):
    """A stat race guard on inspected inputs, not another Registry qualification."""
    if (not isinstance(paths, Mapping) or any(type(key) is not str for key in paths)
            or any(not Path(path).is_absolute() for path in (*paths.values(), *dependencies))
            or not set(map(Path, paths.values())) <= set(dependencies)):
        raise InvalidInput('Inspected model paths/dependencies are inconsistent')
    try:
        expected = {item['path']: item for item in evidence}
        if len(expected) != len(evidence) or set(expected) != set(map(str, dependencies)):
            raise InvalidInput('Inspected evidence does not cover all dependencies')
        for path in dependencies:
            budget.remaining()
            if not path.is_file() or not os.access(path, os.R_OK):
                raise InvalidInput(f'Model asset is missing or unreadable: {path}')
            st = path.stat()
            current = dict(path=str(path), device=st.st_dev, inode=st.st_ino,
                           size=st.st_size, mtime_ns=st.st_mtime_ns, ctime_ns=st.st_ctime_ns)
            if current != expected[str(path)]:
                raise InvalidInput('Model files changed after inspection; validate again')
    except (OSError, KeyError, TypeError) as error:
        raise InvalidInput('Cannot verify inspected model evidence') from error


def resolve(record, paths, dependencies, evidence, overrides=None, config_path=None,
            *, deadline=None) -> ResolvedConfiguration:
    """Consume the already-inspected record; no qualification, writes or helpers."""
    budget = deadline or Deadline.after()
    budget.remaining()
    provenance = record['provenance']
    snapshots = _catalog_snapshots(budget)
    recipe, identity = recipes.load(provenance['recipe_id'])
    reference = recipe['executable']
    executable, catalog, executable_identity = pipelines.load(reference)
    association = provenance.get('executable')
    if (recipe['id'] != provenance['recipe_id'] or identity != provenance['recipe_identity']
            or recipe != provenance['declaration'] or reference != provenance['pipeline']
            or reference != record['pipeline'] or not isinstance(association, Mapping)
            or association != {'reference':reference, 'id':executable['id'],
                               'identity':executable_identity, 'declaration':executable}):
        raise InvalidInput('Model recipe or executable association differs; reacquire under a new ID')
    budget.remaining()
    operation = pipelines.validate_files(executable, recipe['files'], recipe.get('preparation_inputs'))
    preparation = ({'id':operation['id'], 'declaration':operation} if operation is not None else None)
    if provenance.get('preparation') != preparation:
        raise InvalidInput('Model preparation association differs; reacquire under a new ID')
    if set(paths) != set(recipe['files']):
        raise InvalidInput('Inspected model purposes differ from the selected recipe')
    directory = Path(record['directory'])
    if not directory.is_absolute():
        raise InvalidInput('Inspected model directory must be absolute')
    # Retained Registry assets determine actual imported paths, including the
    # existing split-GGUF index. Recipe destinations are not substituted for them.
    for purpose, path in paths.items():
        assets = [asset for asset in record['assets'] if asset['role'] == purpose]
        if len(assets) != 1 or Path(path) != directory / assets[0]['logical_path']:
            raise InvalidInput(f'Inspected {purpose} differs from the recorded model mapping')
    dependencies = tuple(dict.fromkeys(map(Path, dependencies)))
    _unchanged_files(paths, dependencies, evidence, budget)
    modality = modalities.get(executable['modality'])
    settings, sources = effective_settings(recipe, executable['serving'], modality,
                                           load_config(config_path), overrides)
    programs = pipelines.programs(executable, catalog, budget)
    definition = modality.definition(settings, paths, directory, programs)
    _unchanged_files(paths, dependencies, evidence, budget)
    budget.remaining()
    if pipelines.validate_files(executable, recipe['files'], recipe.get('preparation_inputs')) != operation:
        raise InvalidInput('Preparation contract changed during serving resolution')
    relative, digest = _checked_catalog(catalog, snapshots, budget)
    resolved = ResolvedConfiguration(record['id'], reference, Path(catalog), settings, sources,
                                     definition, relative, digest, dependencies, identity,
                                     executable['modality'], evidence, executable['id'], executable_identity)
    budget.remaining()
    return resolved


def _matching_catalog(resolved, catalog, deadline):
    """Verify the same checked root/relative program closure, without selecting."""
    from ..control.hem_process import tree_identity
    relative = resolved.catalog_relative_path
    if (relative.is_absolute()
            or not relative.parts or '..' in relative.parts):
        raise InvalidInput('Checked executable catalog identity is inconsistent')
    catalog = Path(catalog).absolute()
    root = catalog
    for _part in relative.parts:
        root = root.parent
    try:
        root = root.resolve(strict=True)
        selected = catalog.resolve(strict=True)
        if not selected.is_dir() or selected.relative_to(root) != relative:
            raise ValueError('selected catalog location differs')
        if tree_identity(root, deadline) != resolved.catalog_tree_sha256:
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


def export_deployment(record, paths, dependencies, evidence, output: Path, name: str, *,
                      overrides: Optional[Mapping] = None, config_path: Optional[Path] = None,
                      deadline: Optional[Deadline] = None) -> dict:
    """Validate and atomically export, without registry access or native calls."""
    budget = deadline or Deadline.after()
    resolved = resolve(record, paths, dependencies, evidence, overrides, config_path, deadline=budget)
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
            "pipeline": resolved.pipeline, "executable": resolved.pipeline,
            "executable_id": resolved.executable_id, "modality": resolved.modality,
            "output": str(output), "settings": resolved.settings_document()}
