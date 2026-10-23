"""Scoped current model acquisition; payload ownership never follows aliases.

Callers pass a Registry. No CLI, daemon, history or legacy manifests are involved.
Acquisition keeps its model lock through I/O, but never a database transaction.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import tempfile

from . import curated_catalog, hem_config, hem_display, model_download
from .gguf import inspect_gguf_set
from .hem_common import Deadline, HemError, InvalidInput
from .model_store import (AcquisitionInput, RuntimeMapping, PreparationAttempt,
    CatalogAssets, ModelFiles, ModelImportError, mapped_assets, prepared_outputs,
    readable_file, validate_model_id)

_INDEX = 'model.safetensors.index.json'
_GENERATED_RECEIPT = '.loom-acquisition.json'


def _assets(ggufs, tokenizer, alias):
    paths = [Path(path).expanduser().absolute() for path in ggufs]
    token = Path(tokenizer).expanduser().absolute()
    inputs = tuple(AcquisitionInput(f'input-{index}', path)
                   for index, path in enumerate(paths + [token]))
    return _qwen_assets(inputs, alias)


def _qwen_assets(inputs, alias):
    paths, token = [item.path for item in inputs[:-1]], inputs[-1].path
    if not paths or len(set(paths + [token])) != len(paths) + 1:
        raise InvalidInput('GGUF members and tokenizer must have distinct source mappings')
    if any(path.suffix.lower() != '.gguf' for path in paths):
        raise InvalidInput('Every GGUF member must have a .gguf extension')
    names = (['model.gguf'] if len(paths) == 1 else
             [f'model-{i+1:05d}-of-{len(paths):05d}.gguf' for i in range(len(paths))])
    mappings = tuple(RuntimeMapping('gguf' if index < len(paths) else 'tokenizer', name,
                                   'reference' if alias else 'owned', input_id=item.input_id)
                     for index, (name, item) in enumerate(zip(names + ['tokenizer.json'], inputs)))
    return _mapped_assets(inputs, mappings, named=False)


def _mapped_assets(inputs, mappings, **kwargs):
    try:
        return mapped_assets(inputs, mappings, **kwargs)
    except ModelImportError as error:
        raise InvalidInput(str(error)) from error


def _preparation_attempt(root, model_id):
    """Reserve caller-owned scratch while the acquisition holds its model lock."""
    parent = Path(root) / 'preparation-attempts' / model_id
    try:
        parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        return PreparationAttempt(Path(tempfile.mkdtemp(prefix='attempt-', dir=parent)))
    except OSError as error:
        raise HemError(f'Could not reserve private preparation output; incomplete acquisition retained: {error}') from error


def _files(record, directory=None):
    if curated_catalog.is_curated_record(record):
        return curated_catalog.record_files(record, directory)
    directory = Path(directory or record['directory'])
    members = tuple(directory / item['logical_path'] for item in record['assets'][:-1])
    return ModelFiles(members[0] if len(members) == 1 else directory / _INDEX,
                      directory / record['assets'][-1]['logical_path'], members)


def _evidence(files):
    result = []
    paths = (tuple(asset.path for asset in files.files) if isinstance(files, CatalogAssets)
             else (*files.gguf_paths, files.weights, files.tokenizer))
    for path in dict.fromkeys(paths):
        if not readable_file(path):
            raise InvalidInput(f'Model asset is missing or unreadable: {path}')
        stat = path.stat()
        result.append(dict(path=str(path), device=stat.st_dev, inode=stat.st_ino,
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


def _index(files):
    if len(files.gguf_paths) == 1:
        return
    inspection = inspect_gguf_set(files.gguf_paths)
    document = {'weight_map': {name: path.name for name, path in inspection.tensor_files.items()}}
    content = (json.dumps(document, sort_keys=True) + '\n').encode()
    if os.path.lexists(files.weights):
        if files.weights.is_symlink() or files.weights.read_bytes() != content:
            raise InvalidInput('Existing fragment index differs from inspected members; preserved for inspection')
    else:
        _atomic_file(files.weights, lambda stream: stream.write(content))


def _generated_receipt(record, directory=None, *, create=False):
    """Keep the first published generated identity separate from revalidation."""
    provenance = record.get('provenance', {})
    if (not isinstance(provenance, dict) or provenance.get('kind') != 'download' or
            provenance.get('pipeline') != 'moonshine-streaming-tiny'):
        return
    root = Path(directory or record['directory'])
    rotary = root / 'rotary.safetensors'
    receipt = root / _GENERATED_RECEIPT
    try:
        if rotary.is_symlink() or not readable_file(rotary):
            raise ValueError('owned generated rotary is missing or aliased')
        stat = rotary.stat()
        expected = dict(format=1, generated={'rotary.safetensors': dict(
            device=stat.st_dev, inode=stat.st_ino, size=stat.st_size,
            mtime_ns=stat.st_mtime_ns, ctime_ns=stat.st_ctime_ns)})
        if create and not os.path.lexists(receipt):
            pending = record.get('pending') or {}
            if (pending.get('stage') != 'staging' or str(root) != pending.get('path') or
                    root.is_symlink() or not root.is_dir() or os.path.lexists(record['directory']) or
                    any(asset['logical_path'] == _GENERATED_RECEIPT for asset in record['assets'])):
                raise ValueError('first generated receipt requires an unpublished owned staging directory')
            content = (json.dumps(expected, sort_keys=True) + '\n').encode()
            _atomic_file(receipt, lambda stream: stream.write(content))
        if receipt.is_symlink() or json.loads(receipt.read_text()) != expected:
            raise ValueError('first generated receipt is invalid or rotary changed')
    except (OSError, ValueError) as error:
        raise InvalidInput(f'Generated rotary acquisition identity is unproven; preserve the model and acquire under a new ID: {error}') from error


def _populate(record, *, prepared=None):
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
        files = _files(record)
        _generated_receipt(record)
        if not isinstance(files, CatalogAssets):
            _index(files)
        before = _evidence(files)
        if isinstance(files, CatalogAssets):
            curated_catalog.check_assets(files, pipeline)
            selected = files.pipeline
        else:
            selected = hem_config.select_pipeline(files, pipeline)
        after = _evidence(files)
        if before != after:
            raise InvalidInput('Model assets changed during qualification; retry after making inputs stable')
    except (OSError, ValueError, HemError) as exc:
        registry.record_validation(model_id, valid=False, evidence={'error': str(exc)})
        if isinstance(exc, HemError):
            raise
        raise InvalidInput(f'Cannot qualify model {model_id}: {exc}') from exc
    return registry.record_validation(model_id, valid=True, pipeline=selected, evidence=after)


def _acquire(registry, model_id, assets, provenance, pipeline, obtain=None, *,
             inputs=(), generated=(), preparer=None):
    """Own reservation, optional fixed preparation, staging and publication once."""
    with registry.lock('model', model_id):
        record = registry.prepare_model(model_id, assets=assets, provenance=provenance)
        pending = record['pending']
        stage = Path(pending['path']) if pending and pending['stage'] == 'staging' else None
        needs_preparation = preparer is not None and stage is not None and stage.exists() and any(
            not os.path.lexists(stage / item.logical_path) for item in generated)
        if needs_preparation:
            preparer.preflight()  # Existing 30s cap, before unbounded acquisition.
        if obtain is not None:
            obtain()  # Matching published references may need a missing pinned cache member again.
        prepared = {}
        if needs_preparation:
            attempt = _preparation_attempt(registry.root, model_id)
            deadline = Deadline.after(preparer.TIMEOUT)  # Fresh only after inputs are ready.
            result = preparer.prepare(inputs, attempt, deadline)
            try:
                outputs = prepared_outputs(attempt, result, tuple(
                    (item.generated_role, item.logical_path) for item in generated))
            except ModelImportError as error:
                raise InvalidInput(f'Incomplete preparation result; attempt retained at {attempt.directory}: {error}') from error
            prepared = {item.logical_path: outputs[item.generated_role] for item in generated}
        if stage is None or not stage.exists():
            # Published generated identities, including rename-before-DB recovery,
            # must be proven as-is. Never recreate their first-publication proof.
            _generated_receipt(record)
        if stage is not None:
            try:
                _populate(record, prepared=prepared)
                if stage.exists():
                    _generated_receipt(record, stage, create=True)
            except OSError as exc:
                raise HemError(f'Could not acquire model {model_id}; incomplete assets retained: {exc}') from exc
            # Registry owns no-replace rename and recoverable DB publication.
            registry.publish_model(model_id)
        return _qualify(registry, model_id, pipeline)


def import_model(registry, model_id, *, ggufs, tokenizer, alias=False, pipeline=None):
    """Import copies (default) or immutable external references, then qualify."""
    if pipeline in curated_catalog.CATALOGS:
        raise InvalidInput('Curated pipelines require --assets-dir with their complete prepared file set')
    if isinstance(ggufs, (str, Path)):
        ggufs = [ggufs]
    assets = _assets(ggufs, tokenizer, alias)
    return _acquire(registry, model_id, assets, {'kind': 'import'}, pipeline)


def import_assets(registry, model_id, *, directory, pipeline, alias=False):
    """Copy or alias an explicitly selected, complete caller-prepared file set."""
    catalog = curated_catalog.load_catalog(pipeline)
    directory = Path(directory).expanduser().absolute()
    declarations = [asset for asset in catalog.assets
                    if asset['required'] or os.path.lexists(directory / asset['path'])]
    inputs = tuple(AcquisitionInput(f'input-{index}', directory / asset['path'])
                   for index, asset in enumerate(declarations))
    mappings = tuple(RuntimeMapping(asset['role'], asset['path'], 'reference' if alias else 'owned',
                                   input_id=item.input_id)
                     for asset, item in zip(declarations, inputs))
    assets = _mapped_assets(inputs, mappings)
    provenance = dict(kind='import', format=1, asset_contract=catalog.contract,
                      pipeline=pipeline, source={'kind': 'caller-prepared'})
    return _acquire(registry, model_id, assets, provenance, pipeline)


def available():
    """Pinned recipes and local cache state only; never legacy registrations."""
    return [{**recipe, 'files': [model_download.source_status(source)
                               for source in model_download._sources(recipe)]}
            for recipe in model_download.recipes()]


def download(registry, recipe_id, *, model_id, pipeline=None, offline=False, dry_run=False):
    """Download into pinned HF cache, referencing snapshot locators in scoped storage.

    Dry-run never constructs a Registry: callers may pass None and preview the
    proposed namespace separately. Normal HF credentials/cache/environment survive.
    """
    try:
        validate_model_id(model_id)
        recipe = model_download.recipe_by_id(recipe_id)
        sources = model_download._sources(recipe)
    except ModelImportError as exc:
        raise InvalidInput(str(exc).replace('loom-serve-setup models', 'loom-control model')) from exc
    offline = offline or os.environ.get('HF_HUB_OFFLINE', '').upper() in ('1', 'ON', 'YES', 'TRUE')
    files = [model_download.source_status(source) for source in sources]
    named = 'sources' in recipe
    inputs = model_download.acquisition_inputs(sources)
    try:
        mappings = model_download.runtime_mappings(recipe, sources) if named else ()
    except ModelImportError as error:
        raise InvalidInput(str(error)) from error
    assets = (_mapped_assets(inputs, mappings) if named else _qwen_assets(inputs, True))
    selected = pipeline or recipe['pipeline']
    commands = [['hf', 'download', source['repo_id'], source['filename'], '--revision',
                 source['revision'], '--cache-dir', str(model_download.cache_root())] for source in sources]
    if dry_run:
        return dict(model=model_id, recipe=recipe, files=files, assets=assets,
                    commands=[hem_display.command(command) for command in commands],
                    offline=offline, dry_run=True, completed=True, state='preview',
                    **({'generated': recipe.get('generated', []), 'check_kind': 'acquisition-presence'} if named else {}))
    def obtain():
        for source, command in zip(sources, commands):
            if not model_download.source_status(source)['cached']:
                if offline:
                    raise HemError(f'Offline cache is missing {source["repo_id"]}@{source["revision"]}/{source["filename"]}')
                # Existing streamer prints this exact quoted command once and redacts HF credentials.
                try:
                    model_download._run_hf(command)
                except ModelImportError as exc:
                    raise HemError(str(exc)) from exc
                if not model_download.source_status(source)['cached']:
                    raise HemError('hf did not populate the pinned snapshot; incomplete record retained')
    if named:
        if recipe['id'] != 'moonshine-streaming-tiny' or selected != recipe['pipeline']:
            raise InvalidInput('Unsupported named preparation recipe or pipeline override')
        from . import moonshine_preparation
        generated_mappings = tuple(RuntimeMapping(item['role'], item['path'], 'owned',
                                                   generated_role=item['role'])
                                   for item in recipe['generated'])
        if tuple((item.generated_role, item.logical_path) for item in generated_mappings) != moonshine_preparation.GENERATED:
            raise InvalidInput('Named preparation outputs differ from the fixed adapter contract')
        assets = _mapped_assets(inputs, (*mappings, *generated_mappings),
                                generated_directory=registry.root / 'models' / model_id)
        provenance = dict(kind='download', format=1, pipeline=selected,
                          asset_contract=recipe['asset_contract'], source=recipe,
                          preparation=dict(revision=moonshine_preparation.REVISION,
                                           files=moonshine_preparation.TOOL_FILES))
        return _acquire(registry, model_id, assets, provenance, selected, obtain,
                        inputs=inputs, generated=generated_mappings, preparer=moonshine_preparation)
    return _acquire(registry, model_id, assets, {'kind': 'download', 'recipe': recipe}, selected, obtain)


def validate(registry, model_id, *, pipeline=None):
    """Recheck accessible published inputs; explicit acquisition resumes staging."""
    with registry.lock('model', model_id):
        record = registry.get_model(model_id)
        if record['pending'] and record['pending']['stage'] == 'staging':
            raise HemError('Acquisition is incomplete; repeat the matching import/download before validation')
        return _qualify(registry, model_id, pipeline)


def model_files(registry, model_id, *, pipeline=None):
    """Consumer helper repeats qualification before returning immutable mappings.

    Lifecycle consumers keep the model lock while preparing their dependency
    reservation; this helper therefore does not acquire a second lock.
    """
    record = registry.get_model(model_id)
    if record['pending'] and record['pending']['stage'] == 'staging':
        raise HemError('Model acquisition is incomplete')
    record = _qualify(registry, model_id, pipeline)
    return _files(record)


def show(registry, model_id):
    """Observational inventory preserves missing aliases and saved qualification."""
    record = registry.get_model(model_id)
    files = None
    named = curated_catalog.is_curated_record(record)
    try:
        files = _files(record)
        evidence = _evidence(files)
        if named:
            curated_catalog.check_assets(files)
            _generated_receipt(record)
        accessible, error = True, None
    except (OSError, HemError) as exc:
        evidence, accessible, error = None, False, str(exc)
    validation = record['validation']
    state = ('incomplete' if record['pending'] and record['pending']['stage'] == 'staging' else
             'missing' if not accessible else
             'ready' if validation and validation['valid'] and validation['evidence'] == evidence else
             'invalid' if validation and not validation['valid'] else 'unvalidated')
    if named:
        return dict(record, model=model_id, state=state, accessible=accessible, error=error,
                    intended_pipeline=record['provenance'].get('pipeline'),
                    check_kind='acquisition-presence', native_readiness='unprobed',
                    named_assets=[dict(role=item.role, logical_path=item.logical_path, path=str(item.path))
                                  for item in files.files] if files else [])
    return dict(record, model=model_id, state=state, accessible=accessible, error=error,
                weights=str(files.weights), tokenizer=str(files.tokenizer),
                fragments=[str(path) for path in files.gguf_paths])


def inventory(registry):
    return [show(registry, record['id']) for record in registry.list_models()]
