"""Concrete Moonshine program, artifact and fixed-preparer binding."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import re

from ..hem_common import HemError, InvalidInput
from . import pipelines
from .modalities import asr
from .moonshine_preparer_identity import (
    FILES, GENERATED_FILES, INPUT_FILES, INPUT_REPOSITORY, INPUT_REVISION, RELATIVE,
)
from .recipe_types import (
    AcquisitionPlan, Artifact, ArtifactRequirement, BundleFile, BundleManifest,
    Fixed, GeneratedOutput, GeneratedSource, HFInputConstraint, HFSource,
    PipelineDescriptor, PreparationRequirement, RecognitionPrograms,
)


PIPELINE = 'moonshine-streaming-tiny'
CATALOG = 'recognition/moonshine/streaming_tiny'
_ARTIFACTS = (
    ('weights', 'safetensors', 'model.safetensors'),
    ('tokenizer', 'json', 'tokenizer.json'),
    ('configuration', 'json', 'config.json'),
    *GENERATED_FILES,
)
_NATIVE_ASSETS = (
    ('checkpoint', 'model.safetensors'), ('tokenizer', 'tokenizer.json'),
    ('configuration', 'config.json'), ('rotary', 'rotary.safetensors'),
)
_PREPARATION = PreparationRequirement(
    tuple(HFInputConstraint(role, format, HFSource(INPUT_REPOSITORY, INPUT_REVISION, path))
          for role, format, path in _ARTIFACTS if path in INPUT_FILES),
    tuple(GeneratedOutput(*item) for item in GENERATED_FILES),
)


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate catalog field')
        result[key] = value
    return result


def _document(path):
    return json.loads(path.read_text(encoding='utf-8'), object_pairs_hook=_object,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite catalog value')))


def _relative(value):
    if (type(value) is not str or not value or '\\' in value
            or PurePosixPath(value).is_absolute()
            or any(part in ('', '.', '..') for part in value.split('/'))):
        raise ValueError('Program locator must be a normalized relative file')
    return value


def _catalog_directory():
    try:
        for root in pipelines.catalog_roots():
            candidate = root / CATALOG
            if candidate.is_dir():
                resolved = candidate.resolve(strict=True)
                resolved.relative_to(root.resolve(strict=True))
                return resolved
    except (OSError, ValueError, RuntimeError) as error:
        raise HemError(f'Cannot inspect Moonshine pipeline catalog {PIPELINE}: {error}') from error
    raise HemError(f'No packaged catalog for pipeline {PIPELINE}')


def _programs(root):
    """Inspect concrete program metadata/integrity, never model tensor contents."""
    try:
        data = _document(root / 'catalog.json')
        if (type(data) is not dict
                or set(data) != {'format', 'id', 'contract', 'description', 'assets', 'definition'}
                or type(data['format']) is not int or data['format'] != 1
                or data['id'] != PIPELINE or data['contract'] != 'moonshine-streaming-tiny-assets-v1'
                or type(data['description']) is not str):
            raise ValueError('Invalid Moonshine catalog identity')
        expected = [dict(role=role, path=path, required=True) for role, path in _NATIVE_ASSETS]
        if (type(data['assets']) is not list or len(data['assets']) != len(expected)
                or any(type(item) is not dict or set(item) != {'role', 'path', 'required'}
                       or type(item['required']) is not bool for item in data['assets'])
                or data['assets'] != expected):
            raise ValueError('Moonshine native asset manifest differs from the fixed binding')
        definition = data['definition']
        if (type(definition) is not dict or not {'prepare', 'control', 'sources'} <= set(definition)
                or set(definition) - {'prepare', 'control', 'sources', 'kind', 'assets'}
                or ('kind' in definition and definition['kind'] != 'recognition')
                or ('assets' in definition and definition['assets'] != '.')
                or type(definition['sources']) is not list or not definition['sources']):
            raise ValueError('Catalog must contain only recognition program locators')
        programs = RecognitionPrograms(_relative(definition['prepare']),
                                       _relative(definition['control']),
                                       tuple(_relative(item) for item in definition['sources']))
        locators = (programs.prepare, programs.control, *programs.sources)
        if len(set(locators)) != len(locators):
            raise ValueError('Duplicate Moonshine program locator')
        for name in locators:
            path = root / name
            path.resolve(strict=True).relative_to(root)
            if not path.is_file() or path.suffix not in ('.loom', '.loombc'):
                raise ValueError('Missing or unsupported Moonshine program')
        compiled = any(Path(name).suffix == '.loombc' for name in locators)
        policy = 'weights.loombc' if compiled else 'weights.loom'
        policy_path = root / policy
        policy_path.resolve(strict=True).relative_to(root)
        if not policy_path.is_file():
            raise ValueError('Missing Moonshine weights policy')
        manifest = root / 'artifacts.json'
        if compiled and not manifest.is_file():
            raise ValueError('Compiled Moonshine programs require integrity metadata')
        if manifest.exists():
            metadata = _document(manifest)
            required = {*locators, policy}
            if (type(metadata) is not dict or set(metadata) != {'format', 'policy', 'files', 'roots'}
                    or type(metadata['format']) is not int or metadata['format'] != 1
                    or metadata['policy'] != policy or type(metadata['files']) is not dict
                    or type(metadata['roots']) is not dict
                    or set(metadata['files']) != required or set(metadata['roots']) != required):
                raise ValueError('Invalid Moonshine artifact integrity manifest')
            for name, checksum in metadata['files'].items():
                path = root / _relative(name)
                path.resolve(strict=True).relative_to(root)
                content = path.read_bytes()
                if (type(checksum) is not str or re.fullmatch('[0-9a-f]{64}', checksum) is None
                        or not content.startswith(b'LOOM') or hashlib.sha256(content).hexdigest() != checksum):
                    raise ValueError('Moonshine program integrity mismatch')
            actual = {path.relative_to(root).as_posix() for path in root.rglob('*')
                      if path.suffix in ('.loom', '.loombc')}
            if actual != required:
                raise ValueError('Undeclared Moonshine executable module')
        # Bundling describes the installed payload even when discovery is
        # inspecting source programs before the finite bytecode build.
        names = ('prepare.loombc', 'control.loombc', 'providers.loombc',
                 'weights.loombc', 'artifacts.json', 'catalog.json', 'sources.json',
                 'LICENSE.moonshine', 'LICENSE-APACHE-2.0',
                 'recipes/moonshine-streaming-tiny.json')
        return programs, names
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as error:
        raise HemError(f'Cannot inspect Moonshine pipeline programs: {error}') from error


class _MoonshineBinding:
    def describe(self):
        root = _catalog_directory()
        programs, names = _programs(root)
        bundle = tuple(BundleFile(f'loom-pipeline/{CATALOG}/{name}', 'catalog') for name in names)
        helpers = tuple(BundleFile(f'_native/{RELATIVE}/{name}', 'preparer')
                        for name in (*FILES, 'LICENSE', 'provenance.json'))
        return PipelineDescriptor(
            PIPELINE, 'asr', {'pending_requests': Fixed('integer', 4), 'options': Fixed('object', {})},
            asr.PROTOCOL,
            tuple(ArtifactRequirement(role, (format,), 1, 1, (path,))
                  for role, format, path in _ARTIFACTS),
            _PREPARATION, programs, BundleManifest(CATALOG, bundle + helpers))

    def validate_artifacts(self, artifacts):
        if (type(artifacts) is not tuple or len(artifacts) != len(_ARTIFACTS)
                or any(not isinstance(item, Artifact) for item in artifacts)
                or len({item.id for item in artifacts}) != len(artifacts)
                or len({item.path for item in artifacts}) != len(artifacts)):
            raise InvalidInput('artifacts: unsupported-preparation: Moonshine requires its exact four artifacts')
        actual = {item.role: item for item in artifacts}
        if len(actual) != len(artifacts) or set(actual) != {item[0] for item in _ARTIFACTS}:
            raise InvalidInput('artifacts: unsupported-preparation: missing or extra Moonshine role')
        for role, format, path in _ARTIFACTS:
            item = actual[role]
            source = (HFSource(INPUT_REPOSITORY, INPUT_REVISION, path)
                      if path in INPUT_FILES else GeneratedSource())
            if (item.format, item.path, item.source) != (format, path, source):
                raise InvalidInput(f'artifacts.{item.id}: unsupported-preparation: fixed Moonshine input/output differs')

    def _plan(self, plan):
        if not isinstance(plan, AcquisitionPlan) or plan.pipeline != PIPELINE or plan.preparation != _PREPARATION:
            raise InvalidInput('unsupported-preparation: plan differs from fixed Moonshine pipeline')
        self.validate_artifacts(plan.inputs + plan.generated)
        if (any(not isinstance(item.source, HFSource) for item in plan.inputs)
                or any(not isinstance(item.source, GeneratedSource) for item in plan.generated)):
            raise InvalidInput('unsupported-preparation: invalid Moonshine input/output selection')

    def preflight(self, plan, deadline):
        self._plan(plan)
        deadline.remaining()
        from . import moonshine_preparation
        moonshine_preparation.preflight(deadline=deadline)

    def prepare(self, plan, inputs, attempt, deadline):
        self._plan(plan)
        if set(inputs) != {item.id for item in plan.inputs}:
            raise InvalidInput('unsupported-preparation: resolved Moonshine inputs do not match the plan')
        deadline.remaining()
        from . import moonshine_preparation
        return moonshine_preparation.prepare(
            {item.path: inputs[item.id] for item in plan.inputs}, attempt, deadline)


def get(pipeline_id):
    if pipeline_id != PIPELINE:
        raise InvalidInput(f'Unknown Moonshine pipeline {pipeline_id!r}')
    return _MoonshineBinding()
