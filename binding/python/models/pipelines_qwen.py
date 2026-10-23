"""Concrete Qwen capabilities, artifact admission and program integrity.

No tensor/profile inspection, acquisition, state or native execution occurs here.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from types import MappingProxyType

from ..hem_common import HemError, InvalidInput
from . import pipelines
from .modalities import chat
from .recipe_types import ArtifactRequirement, BundleFile, BundleManifest, HFSource, TextPrograms

# Existing catalog_sources.bzl compiled identities, independent of model tensors.
_VARIANTS = {
    'qwen-08b': ('qwen_08b', 'qwen35-08b-ud-q5-k-xl-v1', False),
    'qwen': ('qwen_27b', 'qwen38-27b-ud-q5-k-xl-v1', True),
}
_LIMITS = MappingProxyType({'rows': 16, 'context_capacity': 262144,
                           'pool_capacity': 4194304, 'prefill_capacity': 262144,
                           'mtp_depth': 3})
_ARTIFACTS = (ArtifactRequirement('weights', ('gguf',), 1, 1),
              ArtifactRequirement('tokenizer', ('json',), 1, 1))


def _pairs(entries):
    result = {}
    for key, value in entries:
        if key in result:
            raise ValueError(f'Duplicate program descriptor field: {key}')
        result[key] = value
    return result


def _inside(path, root):
    path.resolve(strict=True).relative_to(root.resolve(strict=True))
    if not path.is_file():
        raise ValueError(f'Pipeline program must be a file: {path}')


def _programs(catalog, variant):
    """Admit the existing source or compiled closure, without shape profiles."""
    descriptor = catalog / 'pipeline.json'
    if descriptor.exists():
        metadata = json.loads(descriptor.read_text(encoding='utf-8'), object_pairs_hook=_pairs,
                              parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite program descriptor')))
        if (not isinstance(metadata, dict) or set(metadata) != {'format', 'variant', 'module', 'sha256'}
                or type(metadata['format']) is not int or metadata['format'] != 1
                or metadata['variant'] != variant or metadata['module'] != 'pipeline.loombc'):
            raise ValueError('Invalid singleton pipeline descriptor or variant')
        module = catalog / 'pipeline.loombc'
        _inside(module, catalog)
        content = module.read_bytes()
        if not content.startswith(b'LOOM') or hashlib.sha256(content).hexdigest() != metadata['sha256']:
            raise ValueError('Invalid Loom bytecode module or hash')
        executable = {p.relative_to(catalog).as_posix() for p in catalog.rglob('*')
                      if p.suffix in ('.loom', '.loombc')}
        if executable != {'pipeline.loombc'}:
            raise ValueError('Singleton catalog contains unexpected executable modules')
        return TextPrograms('pipeline.loombc', 'pipeline.loombc', ('pipeline.loombc',)), (module, descriptor)
    entries = tuple(line.strip() for line in (catalog / 'sources.txt').read_text(encoding='utf-8').splitlines()
                    if line.strip() and not line.strip().startswith('#'))
    root = catalog.parents[1]  # Existing authoritative llm source boundary.
    roles = {role: [] for role in ('prepare', 'control', 'weights')}
    identities, files, providers = set(), [], []
    for entry in entries:
        relative = Path(entry)
        if relative.is_absolute() or '\\' in entry or relative.suffix != '.loom':
            raise ValueError(f'Unsupported pipeline module: {entry}')
        module = catalog / relative
        _inside(module, root)
        identity = module.absolute().as_posix()
        # Normalize parent components without resolving intentional source links.
        identity = os.path.normpath(identity)
        if identity in identities:
            raise ValueError(f'Duplicate pipeline module: {entry}')
        identities.add(identity)
        files.append(module)
        if relative.stem in roles:
            roles[relative.stem].append(entry)
        else:
            providers.append(entry)
    for role, matches in roles.items():
        if len(matches) != 1:
            raise ValueError(f'Pipeline catalog must contain exactly one {role} module')
    return TextPrograms(roles['prepare'][0], roles['control'][0], tuple(providers)), (*files, catalog / 'sources.txt')


@dataclass(frozen=True)
class QwenBinding:
    id: str
    catalog: Path
    mtp_supported: bool
    programs: TextPrograms
    bundle: BundleManifest
    limits = _LIMITS
    artifacts = _ARTIFACTS

    def describe(self):
        return chat.describe(self)

    def validate_artifacts(self, artifacts):
        if len(artifacts) != 2:
            raise InvalidInput('Qwen requires one weights artifact and one tokenizer artifact')
        for requirement in self.artifacts:
            matches = [item for item in artifacts if item.role == requirement.role]
            if (len(matches) != 1 or matches[0].format not in requirement.formats
                    or not isinstance(matches[0].source, HFSource)):
                raise InvalidInput(f'Qwen {requirement.role} requires one pinned HF {requirement.formats[0]} artifact')
        weights = next(item for item in artifacts if item.role == 'weights')
        tokenizer = next(item for item in artifacts if item.role == 'tokenizer')
        if not weights.path.endswith('.gguf') or not tokenizer.path.endswith('.json'):
            raise InvalidInput('Qwen artifact paths must preserve .gguf and .json parser suffixes')

    def preflight(self, plan, deadline):
        # No generated offline preparation: genuinely no tool or budget access.
        return None

    def prepare(self, plan, inputs, attempt, deadline):
        if plan.pipeline != self.id or plan.generated or plan.preparation is not None:
            raise InvalidInput('Qwen has no generated offline preparation')
        self.validate_artifacts(plan.inputs)
        return ()


def get(pipeline_id):
    if type(pipeline_id) is not str or pipeline_id not in _VARIANTS:
        raise InvalidInput(f'Unknown Qwen pipeline {pipeline_id!r}')
    folder, variant, supported = _VARIANTS[pipeline_id]
    relative = Path('llm/qwen') / folder
    for root in pipelines.catalog_roots():
        catalog = root / relative
        if catalog.is_dir():
            break
    else:
        raise HemError(f'No pipeline catalog is installed for pipeline {pipeline_id!r}')
    try:
        catalog = catalog.resolve(strict=True)
        catalog.relative_to(root.resolve(strict=True))
        programs, program_files = _programs(catalog, variant)
        recipes = sorted((catalog / 'recipes').glob('*.json'))
        files = [*program_files, root / 'llm/LICENSE-APACHE-2.0', *recipes]
        for path in files:
            _inside(path, root)
        # Shipping manifest, independent of the current source-program form.
        # Family stagers own producing the singleton and proving compiled closure.
        prefix = 'loom-pipeline/' + relative.as_posix()
        bundle = BundleManifest(relative.as_posix(), (
            BundleFile(prefix + '/pipeline.loombc', 'program'),
            BundleFile(prefix + '/pipeline.json', 'integrity'),
            *(BundleFile(prefix + '/recipes/' + path.name, 'recipe') for path in recipes),
            BundleFile('loom-pipeline/llm/LICENSE-APACHE-2.0', 'notice')))
        return QwenBinding(pipeline_id, catalog, supported, programs, bundle)
    except (OSError, UnicodeError, ValueError, TypeError, KeyError) as error:
        raise HemError(f'Cannot inspect Qwen pipeline {pipeline_id}: {error}') from error
