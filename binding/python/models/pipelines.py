"""Strict direct executable declarations and source/bytecode closure admission."""
from __future__ import annotations

import builtins
import hashlib
from pathlib import Path
import re

from ..hem_common import HemError, InvalidInput
from . import modalities, recipes

_SHA = re.compile(r'[0-9a-f]{64}\Z')


def catalog_roots() -> tuple[Path, ...]:
    package = Path(__file__).resolve().parents[1]
    return package / 'loom-pipeline', package.parents[1] / 'loom-pipeline'


def _root():
    try:
        for root in catalog_roots():
            if root.is_dir():
                return root.resolve(strict=True)
    except (OSError, ValueError, RuntimeError) as error:
        raise HemError('Cannot inspect packaged executable catalog root') from error
    raise HemError('No packaged executable catalog is available')


def _inside(path, root):
    try:
        path.resolve(strict=True).relative_to(root.resolve(strict=True))
        if not path.is_file():
            raise ValueError('expected a regular file')
    except (OSError, ValueError, RuntimeError) as error:
        raise HemError(f'Packaged declaration/program escapes its root or is not a file: {path}') from error


def _manifest(reference):
    recipes._relative(reference, 'executable', locator=True)
    if not reference.endswith('/executable.json'):
        raise InvalidInput('executable: expected a direct catalog-relative executable.json reference')
    root = _root()
    path = root / reference
    _inside(path, root)
    return path, root


def _hashes(value, where):
    if type(value) is not dict or not value:
        recipes._fail(where, 'expected a nonempty path-to-SHA256 object')
    for name, checksum in value.items():
        recipes._relative(name, where, locator=True)
        if type(checksum) is not str or not _SHA.fullmatch(checksum):
            recipes._fail(where + '.' + name, 'expected lowercase SHA256')


def _locators(value, where, *, empty=False):
    if type(value) is not builtins.list or not empty and not value:
        recipes._fail(where, 'expected an ordered locator array')
    for locator in value:
        recipes._relative(locator, where, locator=True)
    if len(set(value)) != len(value):
        recipes._fail(where, 'duplicate program locator')


def parse(data: bytes) -> tuple[dict, str]:
    value, identity = recipes._decode(data)
    recipes._object(value, ('format', 'id', 'modality', 'native_contract', 'serving',
                           'assets', 'programs', 'preparation', 'provenance'), 'executable')
    if type(value['format']) is not int or value['format'] != 1:
        recipes._fail('executable.format', 'expected integer 1')
    recipes._name(value['id'], 'executable.id')
    modality = value['modality']
    if type(modality) is not str or modality not in ('chat', 'asr'):
        recipes._fail('executable.modality', 'expected chat or asr')
    contract = 'text-v1' if modality == 'chat' else 'recognition-v1'
    if value['native_contract'] != contract:
        recipes._fail('executable.native_contract', 'unsupported modality/native contract pair')
    modalities.get(modality).validate_contract(value['serving'])
    assets = value['assets']
    if type(assets) is not dict or not assets:
        recipes._fail('executable.assets', 'expected a nonempty purpose object')
    fixed = []
    for purpose, asset in assets.items():
        recipes._name(purpose, 'executable.assets purpose')
        recipes._object(asset, ('format', 'path'), 'executable.assets.' + purpose)
        if type(asset['format']) is not str or asset['format'] not in ('gguf', 'safetensors', 'json'):
            recipes._fail('executable.assets.' + purpose, 'unsupported file format')
        if asset['path'] is not None:
            recipes._relative(asset['path'], 'executable.assets.' + purpose + '.path')
            if not asset['path'].endswith('.' + asset['format']):
                recipes._fail('executable.assets.' + purpose, 'fixed path suffix differs from format')
            path = Path(asset['path'])
            if any(path == other or path in other.parents or other in path.parents for other in fixed):
                recipes._fail('executable.assets', 'overlapping fixed destinations')
            fixed.append(path)
        elif modality == 'asr':
            recipes._fail('executable.assets.' + purpose, 'recognition requires a fixed destination')
    if modality == 'chat' and (set(assets) != {'weights', 'tokenizer'} or
            assets['weights']['format'] != 'gguf' or assets['tokenizer']['format'] != 'json'):
        recipes._fail('executable.assets', 'text-v1 requires weights/gguf and tokenizer/json')
    programs = value['programs']
    recipes._object(programs, ('root', 'encoding', 'prepare', 'control', 'sources', 'support', 'files'), 'executable.programs')
    recipes._relative(programs['root'], 'executable.programs.root', locator=True)
    if type(programs['encoding']) is not str or programs['encoding'] not in ('source', 'bytecode'):
        recipes._fail('executable.programs.encoding', 'expected source or bytecode')
    for role in ('prepare', 'control'):
        recipes._relative(programs[role], 'executable.programs.' + role, locator=True)
    _locators(programs['sources'], 'executable.programs.sources')
    _locators(programs['support'], 'executable.programs.support', empty=True)
    _hashes(programs['files'], 'executable.programs.files')
    native = {programs['prepare'], programs['control'], *programs['sources']}
    if set(programs['support']) & native:
        recipes._fail('executable.programs.support', 'support duplicates native roles')
    if not (native | set(programs['support'])).issubset(programs['files']):
        recipes._fail('executable.programs.files', 'missing declared role or support file')
    suffix = '.loom' if programs['encoding'] == 'source' else '.loombc'
    if any(not path.endswith(suffix) for path in programs['files']):
        recipes._fail('executable.programs.files', 'encoding and program suffix differ')
    if programs['encoding'] == 'bytecode' and set(programs['files']) != native | set(programs['support']):
        recipes._fail('executable.programs.files', 'undeclared compiled role or support file')
    from .preparation import validate_binding
    validate_binding(value['preparation'], assets)
    provenance = value['provenance']
    if programs['encoding'] == 'source':
        if provenance is not None:
            recipes._fail('executable.provenance', 'authored source provenance must be null')
    else:
        recipes._object(provenance, ('source_declaration_sha256', 'source_files'), 'executable.provenance')
        if type(provenance['source_declaration_sha256']) is not str or not _SHA.fullmatch(provenance['source_declaration_sha256']):
            recipes._fail('executable.provenance.source_declaration_sha256', 'expected lowercase SHA256')
        _hashes(provenance['source_files'], 'executable.provenance.source_files')
        if any(not name.endswith('.loom') for name in provenance['source_files']):
            recipes._fail('executable.provenance.source_files', 'expected source .loom closure')
    return value, identity


def load(reference):
    """Selected metadata only; operation callers explicitly admit current programs."""
    manifest, root = _manifest(reference)
    value, identity = parse(recipes._read_bytes(manifest))
    try:
        program_root = (root / value['programs']['root']).resolve(strict=True)
        program_root.relative_to(root)
        manifest.resolve(strict=True).relative_to(program_root)
        if not program_root.is_dir():
            raise ValueError('program root is not a directory')
    except (OSError, ValueError, RuntimeError) as error:
        raise HemError('Executable declaration/program root is outside its authoritative catalog') from error
    return value, program_root, identity


def validate_files(executable, files):
    """Associate admitted recipe files with data-owned assets and trusted operation."""
    assets = executable['assets']
    if type(files) is not dict or set(files) != set(assets):
        raise InvalidInput('Recipe files differ from executable asset purposes')
    for purpose, asset in assets.items():
        item = files[purpose]
        if not item['path'].endswith('.' + asset['format']) or asset['path'] not in (None, item['path']):
            raise InvalidInput(f'files.{purpose}: destination differs from executable format/path')
    from .preparation import validate_preparation
    return validate_preparation(executable['preparation'], files, assets)


def programs(executable, catalog_path, deadline):
    """Admit all declared source/compiled bytes under the caller's original budget."""
    declared = executable['programs']
    root = Path(catalog_path)
    for name, checksum in declared['files'].items():
        deadline.remaining()
        path = root / name
        _inside(path, root)
        try:
            content = path.read_bytes()
        except OSError as error:
            raise HemError(f'Cannot read executable program {name}') from error
        if hashlib.sha256(content).hexdigest() != checksum:
            raise InvalidInput(f'Executable program hash mismatch: {name}')
        if declared['encoding'] == 'bytecode' and not content.startswith(b'LOOM'):
            raise InvalidInput(f'Executable program is not Loom bytecode: {name}')
    if declared['encoding'] == 'bytecode':
        actual = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.suffix in ('.loom', '.loombc')}
        if actual != set(declared['files']):
            raise InvalidInput('Executable compiled program closure differs from declared inventory')
    deadline.remaining()
    return {role: declared[role] for role in ('prepare', 'control')} | {'sources': builtins.list(declared['sources'])}


def list(modality=None):
    """Strict complete executable inventory, independent of recipe membership."""
    if modality is not None:
        modalities.get(modality)
    root = _root()
    found, ids = [], set()
    for manifest in sorted(root.glob('**/executable.json')):
        reference = manifest.relative_to(root).as_posix()
        value, catalog, identity = load(reference)
        if value['id'] in ids:
            raise InvalidInput(f'Duplicate executable ID {value["id"]!r}')
        ids.add(value['id'])
        if modality is None or value['modality'] == modality:
            found.append((reference, value, catalog, identity))
    return tuple(found)
