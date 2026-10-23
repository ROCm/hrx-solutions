"""Shared artifact acquisition, immutable references and owned publication.

Registry owns reservations and no-replace publication. Pipeline bindings own
artifact admission and preparation; acquisition never selects a model family.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import tempfile

from . import model_download, pipelines, recipes
from .gguf import inspect_gguf_set
from ..hem_common import Deadline, HemError, InvalidInput
from .model_store import ModelImportError, generated_outputs, readable_file, validate_model_id
from .recipe_types import (AcquisitionPlan, AcquisitionResult, ArtifactLocation,
                           GeneratedSource, HFSource, PreparedArtifact, PreparedModel,
                           json_document)

_GENERATED_RECEIPT = '.loom-acquisition.json'
_PREFLIGHT_TIMEOUT = 30
_PREPARATION_TIMEOUT = 300


def _admit_paths(paths):
    """Keep declared files disjoint from publication bookkeeping and partials."""
    members = [Path(item) for item in paths]
    reserved = [Path(_GENERATED_RECEIPT),
                *(path.with_name('.' + path.name + '.partial') for path in members)]
    if any(path == other or path in other.parents or other in path.parents
           for path in members for other in reserved):
        raise InvalidInput('Artifact destination conflicts with acquisition bookkeeping')


def _plan(recipe, binding):
    binding.validate_artifacts(recipe.artifacts)
    _admit_paths(item.path for item in recipe.artifacts)
    descriptor = binding.describe()
    inputs = tuple(item for item in recipe.artifacts if isinstance(item.source, HFSource))
    generated = tuple(item for item in recipe.artifacts if isinstance(item.source, GeneratedSource))
    preparation = descriptor.preparation
    expected = {(item.role, item.format, item.path) for item in generated}
    declared = set() if preparation is None else {
        (item.role, item.format, item.path) for item in preparation.outputs}
    if (expected != declared or len(expected) != len(generated)
            or preparation is not None and len(declared) != len(preparation.outputs)):
        raise InvalidInput('Generated artifacts differ from selected preparation outputs')
    return AcquisitionPlan(recipe.id, recipe.pipeline, inputs, generated, preparation)


def _declaration(recipe):
    return dict(format=recipe.format, id=recipe.id, name=recipe.name,
                modality=recipe.modality, pipeline=recipe.pipeline,
                artifacts=[dict(id=a.id, role=a.role, format=a.format, path=a.path,
                                source=(dict(kind='huggingface', repo_id=a.source.repo_id,
                                             revision=a.source.revision, filename=a.source.filename)
                                        if isinstance(a.source, HFSource) else dict(kind='generated')))
                           for a in recipe.artifacts],
                configuration=json_document(recipe.configuration))


def _provenance(recipe, kind):
    return dict(format=1, kind=kind, recipe_id=recipe.id, recipe_identity=recipe.identity,
                pipeline=recipe.pipeline, declaration=_declaration(recipe))


def _recipe(record, pipeline=None):
    provenance = record['provenance']
    recipe = recipes.load(provenance['recipe_id'])
    if (recipe.identity != provenance['recipe_identity'] or
            _declaration(recipe) != provenance['declaration'] or
            recipe.pipeline != provenance['pipeline'] or pipeline not in (None, recipe.pipeline)):
        raise InvalidInput('Model recipe declaration or selected pipeline differs; acquire under a new ID')
    return recipe


def _files(record, directory=None):
    root = Path(directory or record['directory'])
    provenance = record['provenance']
    if root.is_symlink():
        raise InvalidInput('Published model directory cannot be an alias')
    for asset in record['assets']:
        path = root / asset['logical_path']
        relative = path.relative_to(root)
        if any((root / Path(*relative.parts[:i])).is_symlink() for i in range(1, len(relative.parts))):
            raise InvalidInput('Model artifact parent cannot be an alias')
        if asset['ownership'] == 'reference':
            if not path.is_symlink() or os.readlink(path) != asset['source_path']:
                raise InvalidInput(f'Model reference mapping changed or is missing: {path}')
        elif path.is_symlink():
            raise InvalidInput(f'Owned model artifact cannot be an alias: {path}')
    evidence = _generated_receipt(record, directory)
    _check_index(record, root)
    return PreparedModel(record['id'], provenance['recipe_id'], provenance['recipe_identity'],
                         provenance['pipeline'], root,
                         tuple(PreparedArtifact(a['id'], a['role'], a['format'], a['logical_path'],
                                                root / a['logical_path'], a['prepared_ownership'])
                               for a in record['assets']), evidence)


def _evidence(files):
    result = []
    for item in files.files:
        if not readable_file(item.path):
            raise InvalidInput(f'Model asset is missing or unreadable: {item.path}')
        stat = item.path.stat()
        result.append(dict(path=str(item.path), device=stat.st_dev, inode=stat.st_ino,
                           size=stat.st_size, mtime_ns=stat.st_mtime_ns, ctime_ns=stat.st_ctime_ns))
    return result


def _atomic_file(destination, write):
    temporary = destination.with_name('.' + destination.name + '.partial')
    if os.path.lexists(temporary):
        if temporary.is_symlink() or not temporary.is_file():
            raise HemError(f'Unexpected partial asset preserved: {temporary}')
        temporary.unlink()  # Only our exact unpublished partial filename under the model lock.
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        with os.fdopen(descriptor, 'wb') as stream:
            write(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        descriptor = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def _generated_receipt(record, directory=None, *, create=False):
    """Prove the first owned generated identity independently of revalidation."""
    generated = [a for a in record['assets'] if a['prepared_ownership'] == 'generated-owned']
    if not generated:
        return {}
    root = Path(directory or record['directory'])
    receipt = root / _GENERATED_RECEIPT
    try:
        identities = {}
        for asset in generated:
            path = root / asset['logical_path']
            relative = path.relative_to(root)
            if (path.is_symlink() or any((root / Path(*relative.parts[:i])).is_symlink()
                                        for i in range(1, len(relative.parts)))
                    or not readable_file(path)):
                raise ValueError('owned generated output is missing or aliased')
            stat = path.stat()
            identities[asset['logical_path']] = dict(device=stat.st_dev, inode=stat.st_ino,
                size=stat.st_size, mtime_ns=stat.st_mtime_ns, ctime_ns=stat.st_ctime_ns)
        expected = dict(format=1, recipe_identity=record['provenance']['recipe_identity'],
                        generated=identities)
        if create and not os.path.lexists(receipt):
            pending = record.get('pending') or {}
            if (pending.get('stage') != 'staging' or str(root) != pending.get('path') or
                    root.is_symlink() or not root.is_dir() or os.path.lexists(record['directory'])):
                raise ValueError('first generated receipt requires unpublished owned staging')
            content = (json.dumps(expected, sort_keys=True) + '\n').encode()
            _atomic_file(receipt, lambda stream: stream.write(content))
        if receipt.is_symlink() or json.loads(receipt.read_text()) != expected:
            raise ValueError('first generated receipt is invalid or generated output changed')
        return expected
    except (OSError, ValueError) as error:
        raise InvalidInput('Generated acquisition identity is unproven; preserve the model and '
                           f'acquire under a new ID: {error}') from error


def _populate(record, *, prepared=None, omit=()):
    pending = record['pending']
    if not pending or pending['stage'] != 'staging':
        return
    stage = Path(pending['path'])
    # publish_model handles a crash after rename; never recreate that staging path.
    if not stage.exists() and pending.get('inode'):
        return
    prepared = prepared or {}
    # A fresh result is one associated set. Ordinary retry preservation below
    # must not retain an older member or adopt content appearing during prepare.
    conflicts = [stage / name for name in prepared if os.path.lexists(stage / name)]
    if conflicts:
        raise HemError('Conflicting staged generated outputs retained; inspect incomplete acquisition before retrying: '
                       + ', '.join(map(str, conflicts)))
    for asset in record['assets']:
        if asset['logical_path'] in omit:
            continue
        source = prepared.get(asset['logical_path'], Path(asset['source_path']))
        destination = stage / asset['logical_path']
        if os.path.lexists(destination):
            if asset['ownership'] == 'reference':
                if not destination.is_symlink() or os.readlink(destination) != str(source):
                    raise HemError(f'Conflicting staged alias preserved: {destination}')
            elif destination.is_symlink() or not readable_file(destination):
                raise HemError(f'Conflicting staged file preserved: {destination}')
            continue
        if not readable_file(source):
            raise HemError(f'Source is missing, empty or unreadable: {source}; incomplete acquisition retained')
        try:
            destination.parent.resolve().relative_to(stage.resolve(strict=True))
        except ValueError as exc:
            raise HemError('Staged asset parent escapes its reserved directory') from exc
        destination.parent.mkdir(parents=True, exist_ok=True)
        if asset['ownership'] == 'reference':
            destination.symlink_to(source)
        else:
            def copy(stream):
                with source.open('rb') as incoming:
                    shutil.copyfileobj(incoming, stream, length=1024 * 1024)
            _atomic_file(destination, copy)


def _qualify(registry, model_id, pipeline=None):
    record = registry.get_model(model_id)
    try:
        recipe = _recipe(record, pipeline)
        files = _files(record)
        before = _evidence(files)
        # Resolving the selected binding retains its catalog/program integrity
        # checks. No architecture or tensor-shape qualification occurs here.
        pipelines.get(recipe.pipeline).describe()
        after = _evidence(files)
        if before != after:
            raise InvalidInput('Model assets changed during presence checks; retry with stable inputs')
    except (OSError, ValueError, HemError) as exc:
        registry.record_validation(model_id, valid=False, evidence={'error': str(exc)})
        if isinstance(exc, HemError):
            raise
        raise InvalidInput(f'Cannot inspect model {model_id}: {exc}') from exc
    return registry.record_validation(model_id, valid=True, pipeline=recipe.pipeline, evidence=after)


def _preparation_attempt(root, model_id):
    parent = Path(root) / 'preparation-attempts' / model_id
    try:
        parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        if parent.is_symlink() or parent.resolve() != parent.absolute():
            raise OSError('preparation parent must be owned without aliases')
        return Path(tempfile.mkdtemp(prefix='attempt-', dir=parent))
    except OSError as error:
        raise HemError(f'Could not reserve private preparation output: {error}') from error


def _acquire(registry, model_id, assets, provenance, *, obtain=None, plan=None, binding=None, stage_prepare=None):
    with registry.lock('model', model_id):
        record = registry.prepare_model(model_id, assets=assets, provenance=provenance)
        pending = record['pending']
        stage = Path(pending['path']) if pending and pending['stage'] == 'staging' else None
        needs_preparation = (plan is not None and plan.preparation is not None and
                             stage is not None and stage.exists() and any(
                                 not os.path.lexists(stage / item.path) for item in plan.generated))
        if needs_preparation:
            binding.preflight(plan, Deadline.after(_PREFLIGHT_TIMEOUT))
        if obtain is not None:
            obtain()
        prepared = {}
        if needs_preparation:
            attempt = _preparation_attempt(registry.root, model_id)
            inputs = {item.id: model_download._cache_path(item.source) for item in plan.inputs}
            result = binding.prepare(plan, inputs, attempt, Deadline.after(_PREPARATION_TIMEOUT))
            try:
                prepared = generated_outputs(attempt, result, plan.generated)
            except ModelImportError as error:
                raise InvalidInput(f'Incomplete preparation result; attempt retained at {attempt}: {error}') from error
        if stage is None or not stage.exists():
            _generated_receipt(record)
        if stage is not None:
            try:
                if stage.exists() and stage_prepare is not None:
                    _populate(record, omit=(provenance['fragment_index']['path'],))
                    try:
                        stage_prepare(stage)
                    except ValueError as error:
                        raise InvalidInput(f'Invalid fragment format; incomplete assets retained: {error}') from error
                _populate(record, prepared=prepared)
                if stage.exists():
                    _generated_receipt(record, stage, create=bool(prepared) or stage_prepare is not None)
            except OSError as exc:
                raise HemError(f'Could not acquire model {model_id}; incomplete assets retained: {exc}') from exc
            registry.publish_model(model_id)
        return _qualify(registry, model_id)


def _asset(item, source, ownership, prepared_ownership):
    return dict(id=item.id, role=item.role, format=item.format, logical_path=item.path,
                source_path=str(source), ownership=ownership, prepared_ownership=prepared_ownership)


def _locations(recipe):
    return tuple(ArtifactLocation(a.id, a.path, a.source,
                                  model_download._cache_path(a.source) if isinstance(a.source, HFSource) else None,
                                  readable_file(model_download._cache_path(a.source)) if isinstance(a.source, HFSource) else False,
                                  'hf-reference' if isinstance(a.source, HFSource) else 'generated-owned')
                 for a in recipe.artifacts)


def available():
    return [dict(id=recipe.id, name=recipe.name, modality=recipe.modality, pipeline=recipe.pipeline,
                 files=[dict(id=item.id, logical_path=item.logical_path, cached=item.cached,
                             cache_path=str(item.cache_path) if item.cache_path else None,
                             ownership=item.ownership) for item in _locations(recipe)])
            for recipe in recipes.list()]


def download(registry, recipe_id, *, model_id, pipeline=None, offline=False, dry_run=False):
    try:
        validate_model_id(model_id)
    except ModelImportError as error:
        raise InvalidInput(str(error)) from error
    recipe = recipes.load(recipe_id)
    if pipeline not in (None, recipe.pipeline):
        raise InvalidInput('Selected pipeline differs from recipe pipeline')
    binding = pipelines.get(recipe.pipeline)
    plan = _plan(recipe, binding)  # Pure admission precedes cache, state and helper work.
    if dry_run:
        return AcquisitionResult(model_id, recipe.id, recipe.pipeline, 'preview', _locations(recipe))
    offline = offline or os.environ.get('HF_HUB_OFFLINE', '').upper() in ('1', 'ON', 'YES', 'TRUE')
    assets = [_asset(item, model_download._cache_path(item.source), 'reference', 'hf-reference')
              if isinstance(item.source, HFSource) else
              _asset(item, registry.root / 'models' / model_id / item.path, 'owned', 'generated-owned')
              for item in recipe.artifacts]
    def obtain():
        for item in plan.inputs:
            source = item.source
            if not readable_file(model_download._cache_path(source)):
                if offline:
                    raise HemError(f'Offline cache is missing {source.repo_id}@{source.revision}/{source.filename}')
                try:
                    model_download._run_hf(model_download.command(source))
                except ModelImportError as error:
                    raise HemError(str(error)) from error
                if not readable_file(model_download._cache_path(source)):
                    raise HemError('hf did not populate the pinned snapshot; incomplete record retained')
    _acquire(registry, model_id, assets, _provenance(recipe, 'download'),
             obtain=obtain, plan=plan, binding=binding)
    return AcquisitionResult(model_id, recipe.id, recipe.pipeline, 'acquired', _locations(recipe))


def _import_recipe(pipeline, formats):
    if pipeline is None:
        declarations = recipes.list()
    else:
        # The binding already owns its catalog location. Read that catalog's
        # declaration metadata before resolving any other implementation; full
        # support discovery deliberately keeps its stronger all-entry checks.
        descriptor = pipelines.get(pipeline).describe()
        declarations = []
        for root in pipelines.catalog_roots():
            if not root.is_dir():
                continue
            directory = root / descriptor.bundle.catalog_relative_path / 'recipes'
            try:
                directory.resolve().relative_to(root.resolve(strict=True))
            except (ValueError, OSError, RuntimeError) as error:
                raise InvalidInput('Selected recipe catalog escapes its packaged root') from error
            for path in sorted(directory.glob('*.json')):
                try:
                    path.resolve(strict=True).relative_to(root.resolve(strict=True))
                except (ValueError, OSError, RuntimeError) as error:
                    raise InvalidInput('Selected recipe declaration escapes its packaged root') from error
                try:
                    with path.open('rb') as stream:
                        metadata = recipes.parse(stream.read(recipes.MAX_BYTES + 1))
                except OSError as error:
                    raise InvalidInput(f'Cannot read selected recipe declaration {path.name!r}') from error
                if metadata.pipeline == pipeline:
                    if path.name != metadata.id + '.json':
                        raise InvalidInput('Selected recipe filename and declared ID differ')
                    declarations.append(recipes.load(metadata.id))
            break
    candidates = [recipe for recipe in declarations if
                  all(any(a.format == fmt for a in recipe.artifacts) for fmt in formats)]
    if len(candidates) != 1:
        raise InvalidInput('Import requires an unambiguous packaged recipe; select an existing --pipeline')
    return candidates[0]


def import_model(registry, model_id, *, ggufs, tokenizer, alias=False, pipeline=None):
    """Preserve the existing explicit GGUF import command without shape inference."""
    if isinstance(ggufs, (str, Path)):
        ggufs = [ggufs]
    paths = [Path(path).expanduser().absolute() for path in ggufs]
    token = Path(tokenizer).expanduser().absolute()
    if not paths or len(set(paths + [token])) != len(paths) + 1 or any(p.suffix.lower() != '.gguf' for p in paths):
        raise InvalidInput('GGUF members and tokenizer require distinct mappings and .gguf weight files')
    recipe = _import_recipe(pipeline, ('gguf', 'json'))
    weights = [a for a in recipe.artifacts if a.role == 'weights' and a.format == 'gguf']
    tokens = [a for a in recipe.artifacts if a.role == 'tokenizer']
    if len(weights) != 1 or len(tokens) != 1 or len(recipe.artifacts) != 2:
        raise InvalidInput('Selected recipe does not admit this GGUF/tokenizer import')
    if len(paths) > 1:
        # Keep explicit split-file format/index support, never compare tensors
        # with a model architecture or guess the selected pipeline from them.
        return _import_fragments(registry, model_id, recipe, paths, token, alias)
    selected = {weights[0].id: paths[0], tokens[0].id: token}
    assets = [_asset(a, selected[a.id], 'reference' if alias else 'owned',
                     'caller-reference' if alias else 'caller-owned') for a in recipe.artifacts]
    _admit_paths(a['logical_path'] for a in assets)
    return _acquire(registry, model_id, assets, _provenance(recipe, 'import'))


def _index_content(paths):
    inspection = inspect_gguf_set(paths)
    return (json.dumps({'weight_map': {name: path.name for name, path in
                                      inspection.tensor_files.items()}}, sort_keys=True) + '\n').encode()


def _check_index(record, root):
    declaration = record['provenance'].get('fragment_index')
    if declaration is None:
        return
    index = root / declaration['path']
    if (index.is_symlink() or index.read_bytes() !=
            _index_content(tuple(root / name for name in declaration['members']))):
        raise InvalidInput('Existing fragment index differs from inspected members; preserved for inspection')


def _import_fragments(registry, model_id, recipe, paths, tokenizer, alias):
    from .recipe_types import Artifact
    weights = next(a for a in recipe.artifacts if a.role == 'weights')
    token = next(a for a in recipe.artifacts if a.role == 'tokenizer')
    # This existing local import command supplies explicit members. Only its
    # format-derived index is generated; HF recipe destinations are never guessed.
    names = [f'model-{i+1:05d}-of-{len(paths):05d}.gguf' for i in range(len(paths))]
    index_name = 'model.safetensors.index.json'
    index = Artifact(weights.id, weights.role, 'json', index_name, GeneratedSource())
    assets = [_asset(index, registry.root / 'models' / model_id / index_name, 'owned', 'generated-owned')]
    for i, (name, path) in enumerate(zip(names, paths)):
        item = Artifact(f'fragment-{i+1}', 'weights-shard', 'gguf', name, GeneratedSource())
        assets.append(_asset(item, path, 'reference' if alias else 'owned',
                             'caller-reference' if alias else 'caller-owned'))
    assets.append(_asset(token, tokenizer, 'reference' if alias else 'owned',
                         'caller-reference' if alias else 'caller-owned'))
    _admit_paths(a['logical_path'] for a in assets)
    provenance = _provenance(recipe, 'import')
    provenance['fragment_index'] = dict(path=index_name, members=names)
    def prepare(stage):
        destination = stage / index_name
        content = _index_content(tuple(stage / name for name in names))
        if os.path.lexists(destination):
            if destination.is_symlink() or destination.read_bytes() != content:
                raise InvalidInput('Existing fragment index differs from inspected members; preserved for inspection')
        else:
            _atomic_file(destination, lambda stream: stream.write(content))
    return _acquire(registry, model_id, assets, provenance, stage_prepare=prepare)


def import_assets(registry, model_id, *, directory, pipeline, alias=False):
    """Import a complete caller-prepared set, without acquiring or preparing it."""
    recipe = _import_recipe(pipeline, ())
    root = Path(directory).expanduser().absolute()
    _admit_paths(a.path for a in recipe.artifacts)
    assets = [_asset(a, root / a.path, 'reference' if alias else 'owned',
                     'caller-reference' if alias else 'caller-owned') for a in recipe.artifacts]
    return _acquire(registry, model_id, assets, _provenance(recipe, 'import'))


def validate(registry, model_id, *, pipeline=None):
    with registry.lock('model', model_id):
        record = registry.get_model(model_id)
        if record['pending'] and record['pending']['stage'] == 'staging':
            raise HemError('Acquisition is incomplete; repeat matching import/download before validation')
        return _qualify(registry, model_id, pipeline)


def model_files(registry, model_id, *, pipeline=None):
    """Caller keeps its model lock through dependency reservation/export."""
    record = registry.get_model(model_id)
    if record['pending'] and record['pending']['stage'] == 'staging':
        raise HemError('Model acquisition is incomplete')
    return _files(_qualify(registry, model_id, pipeline))


def show(registry, model_id):
    record = registry.get_model(model_id)
    try:
        _recipe(record)
        files = _files(record)
        evidence = _evidence(files)
        accessible, error = True, None
    except (OSError, ValueError, HemError) as exc:
        files, evidence, accessible, error = None, None, False, str(exc)
    validation = record['validation']
    state = ('incomplete' if record['pending'] and record['pending']['stage'] == 'staging' else
             'missing' if not accessible else
             'ready' if validation and validation['valid'] and validation['evidence'] == evidence else
             'invalid' if validation and not validation['valid'] else 'unvalidated')
    return dict(record, model=model_id, state=state, accessible=accessible, error=error,
                intended_pipeline=record['provenance']['pipeline'],
                check_kind='acquisition-presence', native_readiness='unprobed',
                artifacts=[dict(id=a.id, role=a.role, format=a.format, logical_path=a.logical_path,
                                path=str(a.path), ownership=a.ownership) for a in files.files] if files else [])


def inventory(registry):
    return [show(registry, record['id']) for record in registry.list_models()]
