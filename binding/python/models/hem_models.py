"""Direct recipe files -> cache/owned publication -> inspected Registry records.

Selected executable declarations admit files; Registry owns state and no-replace publication.
There is no parallel acquisition plan, prepared-file or result object model.
"""
from __future__ import annotations

from copy import deepcopy
from functools import partial
import json
import os
from pathlib import Path
import shutil
import tempfile

from . import model_download, pipelines, recipes, preparation
from .gguf import inspect_gguf_set
from .preparation import validate_preparation
from ..hem_common import Deadline, HemError, InvalidInput
from .model_store import ModelImportError, generated_outputs, readable_file, validate_model_id

_GENERATED_RECEIPT = '.loom-acquisition.json'
_PREFLIGHT_TIMEOUT = 30
_PREPARATION_TIMEOUT = 300


def _admit_paths(paths):
    members = [Path(item) for item in paths]
    reserved = [Path(_GENERATED_RECEIPT),
                *(path.with_name('.' + path.name + '.partial') for path in members)]
    if any(path == other or path in other.parents or other in path.parents
           for path in members for other in reserved):
        raise InvalidInput('Artifact destination conflicts with acquisition bookkeeping')


def _provenance(recipe, identity, kind, selected, executable_identity):
    return dict(format=1, kind=kind, recipe_id=recipe['id'], recipe_identity=identity,
                pipeline=recipe['executable'], declaration=deepcopy(recipe),
                executable=dict(reference=recipe['executable'], id=selected['id'],
                                identity=executable_identity, declaration=deepcopy(selected)),
                preparation=_operation_association(selected, recipe))


def _operation_association(selected, recipe):
    operation = validate_preparation(selected['preparation'], recipe['files'], selected['assets'], recipe.get('preparation_inputs'))
    return dict(id=operation['id'], declaration=operation) if operation is not None else None


def _recipe(record, executable=None, modality=None):
    provenance = record['provenance']
    recipe, identity = recipes.load(provenance['recipe_id'])
    selected, _, executable_identity = pipelines.load(recipe['executable'])
    expected = dict(reference=recipe['executable'], id=selected['id'],
                    identity=executable_identity, declaration=selected)
    if (identity != provenance['recipe_identity'] or recipe != provenance['declaration'] or
            recipe['executable'] != provenance['pipeline'] or
            record['pipeline'] not in (None, recipe['executable']) or
            provenance.get('executable') != expected or
            provenance.get('preparation') != _operation_association(selected, recipe) or
            executable not in (None, recipe['executable'])):
        raise InvalidInput('Model recipe or executable association differs; acquire under a new ID')
    if modality not in (None, selected['modality']):
        raise InvalidInput('--modality differs from the selected executable modality')
    pipelines.validate_files(selected, recipe['files'], recipe.get('preparation_inputs'))
    return recipe, selected


def _evidence(dependencies):
    result = []
    for path in dependencies:
        if not readable_file(path):
            raise InvalidInput(f'Model asset is missing or unreadable: {path}')
        stat = path.stat()
        result.append(dict(path=str(path), device=stat.st_dev, inode=stat.st_ino,
                           size=stat.st_size, mtime_ns=stat.st_mtime_ns, ctime_ns=stat.st_ctime_ns))
    return result


def inspect_record(record, *, executable=None, modality=None):
    """Inspect an existing record without Registry writes or saved-valid trust."""
    try:
        recipe, selected = _recipe(record, executable, modality)
        root = Path(record['directory'])
        dependencies = tuple(root / a['logical_path'] for a in record['assets'])
        before = _evidence(dependencies)
        if root.is_symlink():
            raise InvalidInput('Published model directory cannot be an alias')
        paths = {}
        expected = recipe['files']
        fragment_index = record['provenance'].get('fragment_index')
        for asset, path in zip(record['assets'], dependencies):
            purpose = asset['role']
            declaration = expected.get(asset['id'])
            is_shard = purpose == 'weights-shard' and fragment_index is not None
            is_index = (fragment_index is not None and purpose == 'weights' and
                        asset['logical_path'] == fragment_index['path'] and asset['format'] == 'json')
            if not is_shard and (declaration is None or purpose != asset['id'] or
                    not is_index and (asset['logical_path'] != declaration['path'] or
                                      asset['format'] != selected['assets'][purpose]['format'])):
                raise InvalidInput('Model mappings differ from the admitted recipe')
            if is_shard and (asset['format'] != 'gguf' or
                             asset['logical_path'] not in fragment_index['members']):
                raise InvalidInput('Fragment mapping differs from its explicit index')
            if not is_shard:
                if purpose in paths:
                    raise InvalidInput('Duplicate model purpose mapping')
                paths[purpose] = path
            relative = path.relative_to(root)
            if any((root / Path(*relative.parts[:i])).is_symlink() for i in range(1, len(relative.parts))):
                raise InvalidInput('Model artifact parent cannot be an alias')
            if asset['ownership'] == 'reference':
                if not path.is_symlink() or os.readlink(path) != asset['source_path']:
                    raise InvalidInput(f'Model reference mapping changed or is missing: {path}')
            elif path.is_symlink():
                raise InvalidInput(f'Owned model artifact cannot be an alias: {path}')
        if set(paths) != set(expected):
            raise InvalidInput('Model file inventory differs from the admitted recipe')
        _generated_receipt(record)
        _check_index(record, root)
        after = _evidence(dependencies)
        if before != after:
            raise InvalidInput('Model assets changed during presence checks; retry with stable inputs')
        return paths, dependencies, after
    except (OSError, ValueError) as error:
        raise InvalidInput(f'Cannot inspect model {record["id"]}: {error}') from error


def _admit_programs(reference, deadline):
    """Fresh admission at an operation boundary, under the caller's budget."""
    deadline.remaining()
    selected, catalog, identity = pipelines.load(reference)
    pipelines.programs(selected, catalog, deadline)
    deadline.remaining()
    return selected, identity


def _select(recipe_id, *, executable=None, modality=None, deadline=None):
    """Read and admit this operation's exact documents before Registry/effects."""
    if recipe_id is None:
        raise InvalidInput('Local import requires an explicit --recipe')
    recipe, identity = recipes.load(recipe_id)
    if executable not in (None, recipe['executable']):
        raise InvalidInput('--executable differs from the selected recipe executable')
    selected, executable_identity = _admit_programs(recipe['executable'], deadline or Deadline.after())
    if modality not in (None, selected['modality']):
        raise InvalidInput('--modality differs from the selected executable modality')
    pipelines.validate_files(selected, recipe['files'], recipe.get('preparation_inputs'))
    _admit_paths(item['path'] for item in recipe['files'].values())
    return recipe, identity, selected, executable_identity


def _qualify_record(registry, record, executable=None, modality=None):
    # Acquire/import already admitted programs before their effects. Reuse that
    # operation's admission; independent qualify below performs a fresh one.
    model_id = record['id']
    if record['pending'] and record['pending']['stage'] == 'staging':
        raise HemError('Acquisition is incomplete; repeat matching import/download before validation')
    try:
        paths, dependencies, evidence = inspect_record(record, executable=executable, modality=modality)
    except HemError as error:
        registry.record_validation(model_id, valid=False, evidence={'error': str(error)})
        raise
    record = registry.record_validation(model_id, valid=True,
                                        pipeline=record['provenance']['pipeline'], evidence=evidence)
    return record, paths, dependencies, evidence


def qualify(registry, model_id, *, executable=None, modality=None):
    """Caller holds model lock; freshly admit programs then persist inspection."""
    record = registry.get_model(model_id)
    if record['pending'] and record['pending']['stage'] == 'staging':
        raise HemError('Acquisition is incomplete; repeat matching import/download before validation')
    reference = record['provenance']['pipeline']
    if executable not in (None, reference):
        raise InvalidInput('Selected executable association differs from the model')
    recorded_modality = record['provenance'].get('executable', {}).get('declaration', {}).get('modality')
    if modality is not None and recorded_modality in ('chat', 'asr', 'tts') and modality != recorded_modality:
        raise InvalidInput('--modality differs from the recorded executable modality')
    try:
        _admit_programs(reference, registry.deadline or Deadline.after())
    except HemError as error:
        registry.record_validation(model_id, valid=False, evidence={'error': str(error)})
        raise
    # Live modality drift is an association failure, not a wrong caller assertion.
    # Full association checks stay inside _qualify_record's failure persistence.
    return _qualify_record(registry, record, executable, modality)


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


def _preparation_attempt(root, model_id):
    parent = Path(root) / 'preparation-attempts' / model_id
    try:
        parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        if parent.is_symlink() or parent.resolve() != parent.absolute():
            raise OSError('preparation parent must be owned without aliases')
        return Path(tempfile.mkdtemp(prefix='attempt-', dir=parent))
    except OSError as error:
        raise HemError(f'Could not reserve private preparation output: {error}') from error


def _acquire(registry, model_id, assets, provenance, *, obtain=None, generated=(),
             prepare=None, preflight=None, inputs=None, stage_prepare=None, stage_outputs=()):
    with registry.lock('model', model_id):
        record = registry.prepare_model(model_id, assets=assets, provenance=provenance)
        pending = record['pending']
        stage = Path(pending['path']) if pending and pending['stage'] == 'staging' else None
        needs_preparation = (bool(generated) and stage is not None and stage.exists() and
                             any(not os.path.lexists(stage / name) for name in generated))
        if needs_preparation:
            preflight(Deadline.after(_PREFLIGHT_TIMEOUT))
        if obtain is not None:
            obtain(needs_preparation)
        prepared = {}
        if needs_preparation:
            attempt = _preparation_attempt(registry.root, model_id)
            result = prepare(inputs, attempt, Deadline.after(_PREPARATION_TIMEOUT))
            try:
                prepared = generated_outputs(attempt, result, generated)
            except ModelImportError as error:
                raise InvalidInput(f'Incomplete preparation result; attempt retained at {attempt}: {error}') from error
        if stage is None or not stage.exists():
            _generated_receipt(record)
        if stage is not None:
            try:
                if stage.exists() and stage_prepare is not None:
                    _populate(record, omit=stage_outputs)
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
        record, _, _, evidence = _qualify_record(registry, registry.get_model(model_id))
        return _view(record, evidence)


def _asset(purpose, format, logical_path, source, ownership, prepared_ownership):
    return dict(id=purpose, role=purpose, format=format, logical_path=logical_path,
                source_path=str(source), ownership=ownership, prepared_ownership=prepared_ownership)


def _preview(recipe):
    artifacts = []
    for purpose, item in recipe['files'].items():
        source = item['source']
        path = model_download._cache_path(source) if source['kind'] == 'huggingface' else None
        artifacts.append(dict(id=purpose, logical_path=item['path'], source=source,
                              cache_path=str(path) if path else None,
                              cached=readable_file(path) if path else False,
                              ownership='hf-reference' if path else 'generated-owned'))
    return artifacts


def available(*, deadline=None):
    budget = deadline or Deadline.after()
    admitted = {}
    result = []
    for recipe, _ in recipes.list():
        reference = recipe['executable']
        if reference not in admitted:
            admitted[reference], _ = _admit_programs(reference, budget)
        selected = admitted[reference]
        pipelines.validate_files(selected, recipe['files'], recipe.get('preparation_inputs'))
        result.append(dict(id=recipe['id'], recipe_id=recipe['id'], name=recipe['name'],
                           modality=selected['modality'], executable_id=selected['id'],
                           executable=reference, pipeline=reference,
                           files=[{k: v for k, v in item.items() if k != 'source'}
                                  for item in _preview(recipe)]))
    return result


def download(registry, recipe_id, *, model_id, executable=None, modality=None, offline=False, dry_run=False, deadline=None, _selection=None):
    try:
        validate_model_id(model_id)
    except ModelImportError as error:
        raise InvalidInput(str(error)) from error
    budget = deadline or (registry.deadline if registry is not None else None) or Deadline.after()
    recipe, identity, selected, executable_identity = (_selection if _selection is not None else
        _select(recipe_id, executable=executable, modality=modality, deadline=budget))
    if dry_run:
        return dict(model_id=model_id, recipe_id=recipe_id, pipeline=recipe['executable'],
                    executable=recipe['executable'], executable_id=selected['id'], modality=selected['modality'],
                    state='preview', artifacts=_preview(recipe))
    offline = offline or os.environ.get('HF_HUB_OFFLINE', '').upper() in ('1', 'ON', 'YES', 'TRUE')
    assets, inputs, generated = [], {}, []
    for purpose, item in recipe['files'].items():
        source, path = item['source'], item['path']
        if source['kind'] == 'huggingface':
            cache = model_download._cache_path(source)
            assets.append(_asset(purpose, selected['assets'][purpose]['format'], path, cache, 'reference', 'hf-reference'))

        else:
            generated.append(path)
            assets.append(_asset(purpose, selected['assets'][purpose]['format'], path,
                                 registry.root / 'models' / model_id / path, 'owned', 'generated-owned'))
    sources = [item['source'] for item in recipe['files'].values()]
    def obtain(needs_preparation):
        required = sources + (list(recipe.get('preparation_inputs', {}).values()) if needs_preparation else [])
        for source in required:
            if source['kind'] != 'huggingface':
                continue
            if not readable_file(model_download._cache_path(source)):
                if offline:
                    raise HemError(f'Offline cache is missing {source["repo_id"]}@{source["revision"]}/{source["filename"]}')
                try:
                    model_download._run_hf(model_download.command(source))
                except ModelImportError as error:
                    raise HemError(str(error)) from error
                if not readable_file(model_download._cache_path(source)):
                    raise HemError('hf did not populate the pinned snapshot; incomplete record retained')
    if generated:
        binding = selected['preparation']
        preflight = partial(preparation.preflight, binding['operation'])
        prepare = partial(preparation.prepare, binding['operation'])
        inputs = {name: model_download._cache_path(recipe['files'][purpose]['source'])
                  for name, purpose in binding['inputs'].items()}
        inputs.update({name: model_download._cache_path(source)
                       for name, source in binding.get('preparation_inputs', {}).items()})
    else:
        preflight = prepare = None
    return _acquire(registry, model_id, assets,
                    _provenance(recipe, identity, 'download', selected, executable_identity),
                    obtain=obtain, generated=tuple(generated), inputs=inputs,
                    preflight=preflight, prepare=prepare)


def import_model(registry, model_id, *, ggufs, tokenizer, recipe_id=None, alias=False,
                 executable=None, modality=None, _selection=None):
    if isinstance(ggufs, (str, Path)):
        ggufs = [ggufs]
    paths = [Path(path).expanduser().absolute() for path in ggufs]
    token = Path(tokenizer).expanduser().absolute()
    if not paths or len(set(paths + [token])) != len(paths) + 1 or any(p.suffix.lower() != '.gguf' for p in paths):
        raise InvalidInput('GGUF members and tokenizer require distinct mappings and .gguf weight files')
    recipe, identity, selected, executable_identity = (_selection if _selection is not None else
        _select(recipe_id, executable=executable, modality=modality, deadline=registry.deadline))
    if (set(recipe['files']) != {'weights', 'tokenizer'} or
            {purpose: item['format'] for purpose, item in selected['assets'].items()} !=
            {'weights': 'gguf', 'tokenizer': 'json'}):
        raise InvalidInput('Selected recipe does not admit this GGUF/tokenizer import')
    if len(paths) > 1:
        return _import_fragments(registry, model_id, recipe, identity, selected, executable_identity,
                                 paths, token, alias)
    sources = {'weights': paths[0], 'tokenizer': token}
    assets = [_asset(purpose, selected['assets'][purpose]['format'], item['path'], sources[purpose],
                     'reference' if alias else 'owned', 'caller-reference' if alias else 'caller-owned')
              for purpose, item in recipe['files'].items()]
    _admit_paths(a['logical_path'] for a in assets)
    return _acquire(registry, model_id, assets,
                    _provenance(recipe, identity, 'import', selected, executable_identity))


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


def _import_fragments(registry, model_id, recipe, identity, selected, executable_identity, paths, tokenizer, alias):
    names = [f'model-{i+1:05d}-of-{len(paths):05d}.gguf' for i in range(len(paths))]
    index_name = 'model.safetensors.index.json'
    assets = [_asset('weights', 'json', index_name, registry.root / 'models' / model_id / index_name,
                     'owned', 'generated-owned')]
    for i, (name, path) in enumerate(zip(names, paths)):
        item = _asset(f'fragment-{i+1}', 'gguf', name, path, 'reference' if alias else 'owned',
                      'caller-reference' if alias else 'caller-owned')
        item['role'] = 'weights-shard'
        assets.append(item)
    assets.append(_asset('tokenizer', 'json', recipe['files']['tokenizer']['path'], tokenizer,
                         'reference' if alias else 'owned', 'caller-reference' if alias else 'caller-owned'))
    _admit_paths(a['logical_path'] for a in assets)
    provenance = _provenance(recipe, identity, 'import', selected, executable_identity)
    provenance['fragment_index'] = dict(path=index_name, members=names)
    def prepare(stage):
        destination = stage / index_name
        content = _index_content(tuple(stage / name for name in names))
        if os.path.lexists(destination):
            if destination.is_symlink() or destination.read_bytes() != content:
                raise InvalidInput('Existing fragment index differs from inspected members; preserved for inspection')
        else:
            _atomic_file(destination, lambda stream: stream.write(content))
    return _acquire(registry, model_id, assets, provenance, stage_prepare=prepare, stage_outputs=(index_name,))


def import_assets(registry, model_id, *, directory, recipe_id=None, alias=False,
                  executable=None, modality=None, _selection=None):
    recipe, identity, selected, executable_identity = (_selection if _selection is not None else
        _select(recipe_id, executable=executable, modality=modality, deadline=registry.deadline))
    root = Path(directory).expanduser().absolute()
    _admit_paths(a['path'] for a in recipe['files'].values())
    operation = selected['preparation']
    metadata = None
    if operation is not None and operation['operation'] == 'pocket-alba-assets-v1':
        from .pocket import import_metadata
        metadata = import_metadata(root)
    assets = [_asset(purpose, selected['assets'][purpose]['format'], item['path'], root / item['path'],
                     'owned' if metadata is not None and purpose == 'prepared_metadata' else 'reference' if alias else 'owned',
                     'generated-owned' if metadata is not None and purpose == 'prepared_metadata' else 'caller-reference' if alias else 'caller-owned')
              for purpose, item in recipe['files'].items()]
    def snapshot(stage):
        destination = stage / 'model.json'
        if os.path.lexists(destination):
            if destination.is_symlink() or destination.read_bytes() != metadata:
                raise InvalidInput('Existing Pocket metadata differs; incomplete import preserved')
        else:
            _atomic_file(destination, lambda stream: stream.write(metadata))
    return _acquire(registry, model_id, assets,
                    _provenance(recipe, identity, 'import', selected, executable_identity),
                    stage_prepare=snapshot if metadata is not None else None,
                    stage_outputs=('model.json',) if metadata is not None else ())


def validate(registry, model_id, *, executable=None, modality=None):
    with registry.lock('model', model_id):
        record, _, _, evidence = qualify(registry, model_id, executable=executable, modality=modality)
        return _view(record, evidence)


def _view(record, evidence, error=None):
    validation = record['validation']
    accessible = error is None
    state = ('incomplete' if record['pending'] and record['pending']['stage'] == 'staging' else
             'missing' if not accessible else
             'ready' if validation and validation['valid'] and validation['evidence'] == evidence else
             'invalid' if validation and not validation['valid'] else 'unvalidated')
    return dict(record, model=record['id'], state=state, accessible=accessible, error=error,
                intended_pipeline=record['provenance']['pipeline'],
                recipe_id=record['provenance']['recipe_id'],
                executable=record['provenance']['pipeline'],
                executable_id=record['provenance'].get('executable', {}).get('id'),
                modality=record['provenance'].get('executable', {}).get('declaration', {}).get('modality'),
                check_kind='acquisition-presence', native_readiness='unprobed',
                artifacts=[dict(id=a['id'], role=a['role'], format=a['format'], logical_path=a['logical_path'],
                                path=str(Path(record['directory']) / a['logical_path']),
                                ownership=a['prepared_ownership']) for a in record['assets']] if accessible else [])


def _show_record(record, admission_error=None):
    if admission_error is not None:
        return _view(record, None, error=admission_error)
    try:
        _, _, evidence = inspect_record(record)
        return _view(record, evidence)
    except HemError as error:
        return _view(record, None, error=str(error))


def show(registry, model_id):
    record = registry.get_model(model_id)
    try:
        _admit_programs(record['provenance']['pipeline'], registry.deadline or Deadline.after())
    except HemError as error:
        return _show_record(record, str(error))
    return _show_record(record)


def inventory(registry):
    budget = registry.deadline or Deadline.after()
    admitted = {}
    result = []
    for record in registry.list_models():
        pipeline = record['provenance']['pipeline']
        if pipeline not in admitted:
            try:
                _admit_programs(pipeline, budget)
                admitted[pipeline] = None
            except HemError as error:
                admitted[pipeline] = str(error)
        result.append(_show_record(record, admitted[pipeline]))
    return result
