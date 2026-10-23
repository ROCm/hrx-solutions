"""Data-only curated asset layouts and native definitions, not model validators."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
import hashlib
from pathlib import Path, PurePosixPath
import re

from ..hem_common import InvalidInput
from .model_store import CatalogAssets, NamedAsset

# Working entries are admitted with their separately reviewed packaged closure.
# Relative to the source/installed loom-pipeline directory, never an external repo.
CATALOGS: dict[str, str] = {
    "moonshine-streaming-tiny": "recognition/moonshine/streaming_tiny",
}
_NAME = re.compile(r'[A-Za-z0-9][A-Za-z0-9_.-]*\Z')


def _name(value):
    if not isinstance(value, str) or not _NAME.fullmatch(value) or value in ('.', '..'):
        raise InvalidInput('Invalid curated catalog identity or role')
    return value


def relative_file(value):
    if (not isinstance(value, str) or not value or '\\' in value or
            PurePosixPath(value).is_absolute() or
            any(part in ('', '.', '..') for part in value.split('/'))):
        raise InvalidInput('Catalog paths must be normalized relative files')
    return value


def catalog_roots():
    """The fixed installed/source locations, not arbitrary catalog ancestors."""
    package = Path(__file__).resolve().parents[1]
    return package / 'loom-pipeline', package.parents[1] / 'loom-pipeline'


def catalog_path(identifier):
    relative = relative_file(CATALOGS[identifier])
    for root in catalog_roots():
        candidate = root / relative
        if candidate.is_dir():
            candidate.resolve(strict=True).relative_to(root.resolve(strict=True))
            return candidate.resolve()
    raise InvalidInput(f'No packaged curated catalog for {identifier!r}')


def _object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError('Duplicate catalog field')
        value[key] = item
    return value


@dataclass(frozen=True)
class Catalog:
    identifier: str
    contract: str
    directory: Path
    description: str
    assets: tuple[dict, ...]
    definition: dict
    recipe: dict | None = None


def load_catalog(identifier, *, directory=None):
    """Check packaged data/paths only; native loading judges actual model bytes."""
    if identifier not in CATALOGS:
        raise InvalidInput(f'Unknown curated pipeline {identifier!r}')
    try:
        root = Path(directory) if directory is not None else catalog_path(identifier)
        root = root.resolve(strict=True)
        data = json.loads((root / 'catalog.json').read_text(encoding='utf-8'),
                          object_pairs_hook=_object,
                          parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite catalog value')))
        required = {'format', 'id', 'contract', 'description', 'assets', 'definition'}
        if (not isinstance(data, dict) or not required.issubset(data) or
                set(data) - required - {'download_recipe'} or
                type(data['format']) is not int or data['format'] != 1 or
                data['id'] != identifier or not isinstance(data['description'], str)):
            raise ValueError('Invalid curated catalog format or identity')
        _name(data['id'])
        _name(data['contract'])
        assets = data['assets']
        if not isinstance(assets, list) or not assets:
            raise ValueError('Catalog requires declared assets')
        paths = []
        for asset in assets:
            if not isinstance(asset, dict) or set(asset) != {'role', 'path', 'required'} or type(asset['required']) is not bool:
                raise ValueError('Invalid named asset declaration')
            _name(asset['role'])
            path = relative_file(asset['path'])
            if any(path == other or Path(path) in Path(other).parents or Path(other) in Path(path).parents for other in paths):
                raise ValueError('Duplicate or overlapping catalog asset paths')
            paths.append(path)
        definition = data['definition']
        if (not isinstance(definition, dict) or
                set(definition) != {'kind', 'prepare', 'control', 'sources', 'assets', 'pending_requests', 'options'} or
                definition.get('assets') != '.' or
                not isinstance(definition.get('sources'), list) or
                not isinstance(definition.get('options'), dict)):
            raise ValueError('Invalid curated native definition')
        _name(definition.get('kind'))
        # These are native program fields, not a generic template/path language.
        for locator in (definition.get('prepare'), definition.get('control'), *definition['sources']):
            path = root / relative_file(locator)
            path.resolve(strict=True).relative_to(root)
            if not path.is_file() or path.suffix not in ('.loom', '.loombc'):
                raise ValueError('Missing or unsupported curated program')
        if 'pending_requests' in definition and (type(definition['pending_requests']) is not int or definition['pending_requests'] <= 0):
            raise ValueError('Invalid native pending_requests')
        manifest = root / 'artifacts.json'
        if not manifest.exists() and any(Path(locator).suffix == '.loombc' for locator in (definition['prepare'], definition['control'], *definition['sources'])):
            raise ValueError('Compiled curated catalog requires artifact integrity metadata')
        if manifest.exists():
            metadata = json.loads(manifest.read_text(), object_pairs_hook=_object)
            if (not isinstance(metadata, dict) or set(metadata) != {'format','policy','files','roots'} or
                    type(metadata['format']) is not int or metadata['format'] != 1 or
                    not isinstance(metadata['files'], dict) or not isinstance(metadata['roots'], dict)):
                raise ValueError('Invalid packaged artifact integrity metadata')
            policy = relative_file(metadata['policy'])
            expected = {definition['prepare'],definition['control'],*definition['sources'],policy}
            if set(metadata['files']) != expected or set(metadata['roots']) != expected:
                raise ValueError('Packaged program/policy inventory differs from native definition')
            for name, checksum in metadata['files'].items():
                path = root / relative_file(name)
                path.resolve(strict=True).relative_to(root)
                content = path.read_bytes()
                if not content.startswith(b'LOOM') or hashlib.sha256(content).hexdigest() != checksum:
                    raise ValueError('Packaged program/policy integrity mismatch')
            actual = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.suffix in ('.loom','.loombc')}
            if actual != expected:
                raise ValueError('Packaged catalog contains undeclared executable modules')
        recipe = data.get('download_recipe')
        if recipe is not None and not isinstance(recipe, dict):
            raise ValueError('Invalid curated download recipe')
        return Catalog(identifier, data['contract'], root, data['description'],
                       tuple(assets), definition, recipe)
    except (OSError, ValueError, TypeError, KeyError) as error:
        raise InvalidInput(f'Cannot inspect curated catalog {identifier}: {error}') from error


def is_curated_record(record):
    provenance = record.get('provenance')
    return isinstance(provenance, dict) and ('asset_contract' in provenance or 'format' in provenance)


def record_files(record, directory=None):
    """Decode only the reserved new-record format; never reinterpret as Qwen."""
    provenance = record['provenance']
    if (not isinstance(provenance, dict) or type(provenance.get('format')) is not int or
            provenance['format'] != 1 or provenance.get('kind') not in ('import', 'download')):
        raise InvalidInput('Unsupported named asset record format')
    pipeline, contract = _name(provenance.get('pipeline')), _name(provenance.get('asset_contract'))
    root = Path(directory or record['directory'])
    files, paths = [], set()
    for asset in record['assets']:
        role, logical = _name(asset.get('role')), relative_file(asset.get('logical_path'))
        if logical in paths or asset.get('ownership') not in ('owned', 'reference') or not Path(asset.get('source_path', '')).is_absolute():
            raise InvalidInput('Invalid named asset mapping')
        paths.add(logical)
        files.append(NamedAsset(role, logical, root / logical))
    return CatalogAssets(pipeline, contract, root, tuple(files))


def check_assets(files, pipeline=None):
    """Require declared mapping completeness; do not inspect model contents."""
    if pipeline is not None and pipeline != files.pipeline:
        raise InvalidInput('Selected pipeline differs from the immutable asset record')
    catalog = load_catalog(files.pipeline)
    if catalog.contract != files.contract:
        raise InvalidInput('Asset contract changed; acquire under a new local ID')
    declarations = {asset['path']: asset for asset in catalog.assets}
    actual = {asset.logical_path: asset for asset in files.files}
    if (any(path not in declarations or item.role != declarations[path]['role'] for path, item in actual.items()) or
            any(asset['required'] and asset['path'] not in actual for asset in catalog.assets)):
        raise InvalidInput('Named asset mappings do not complete the declared catalog contract')
    return catalog


def definition_for(catalog, files):
    definition = deepcopy(catalog.definition)
    definition['assets'] = str(files.directory.absolute())
    return definition
