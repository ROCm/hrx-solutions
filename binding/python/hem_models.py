"""Scoped current model acquisition; payload ownership never follows aliases.

Callers pass a Registry. No CLI, daemon, history or legacy manifests are involved.
Acquisition keeps its model lock through I/O, but never a database transaction.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil

from . import hem_config, hem_display, model_download
from .gguf import inspect_gguf_set
from .hem_common import HemError, InvalidInput
from .model_store import ModelFiles, ModelImportError, readable_file, validate_model_id

_INDEX = 'model.safetensors.index.json'


def _assets(ggufs, tokenizer, alias):
    paths = [Path(path).expanduser().absolute() for path in ggufs]
    token = Path(tokenizer).expanduser().absolute()
    if not paths or len(set(paths + [token])) != len(paths) + 1:
        raise InvalidInput('GGUF members and tokenizer must have distinct source mappings')
    if any(path.suffix.lower() != '.gguf' for path in paths):
        raise InvalidInput('Every GGUF member must have a .gguf extension')
    names = (['model.gguf'] if len(paths) == 1 else
             [f'model-{i+1:05d}-of-{len(paths):05d}.gguf' for i in range(len(paths))])
    return [dict(logical_path=name, source_path=str(path),
                 ownership='reference' if alias else 'owned')
            for name, path in zip(names + ['tokenizer.json'], paths + [token])]


def _files(record, directory=None):
    directory = Path(directory or record['directory'])
    members = tuple(directory / item['logical_path'] for item in record['assets'][:-1])
    return ModelFiles(members[0] if len(members) == 1 else directory / _INDEX,
                      directory / record['assets'][-1]['logical_path'], members)


def _evidence(files):
    result = []
    for path in dict.fromkeys((*files.gguf_paths, files.weights, files.tokenizer)):
        if not readable_file(path):
            raise InvalidInput(f'Model asset is missing, empty or unreadable: {path}')
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


def _populate(record):
    pending = record['pending']
    if not pending or pending['stage'] != 'staging':
        return
    stage = Path(pending['path'])
    # publish_model handles a crash after rename; never recreate that staging path.
    if not stage.exists() and pending.get('inode'):
        return
    for asset in record['assets']:
        source, destination = Path(asset['source_path']), stage / asset['logical_path']
        if os.path.lexists(destination):
            if asset['ownership'] == 'reference':
                if not destination.is_symlink() or os.readlink(destination) != str(source):
                    raise HemError(f'Conflicting staged alias preserved: {destination}')
            elif destination.is_symlink() or not readable_file(destination):
                raise HemError(f'Conflicting staged file preserved: {destination}')
            continue
        if not readable_file(source):
            raise HemError(f'Source is missing, empty or unreadable: {source}; incomplete acquisition retained')
        if asset['ownership'] == 'reference':
            destination.symlink_to(source)
        else:
            def copy(stream):
                with source.open('rb') as incoming:
                    shutil.copyfileobj(incoming, stream, length=1024 * 1024)
            _atomic_file(destination, copy)


def _qualify(registry, model_id, pipeline=None):
    record = registry.get_model(model_id)
    files = _files(record)
    try:
        _index(files)
        before = _evidence(files)
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


def _acquire(registry, model_id, assets, provenance, pipeline, obtain=None):
    with registry.lock('model', model_id):
        record = registry.prepare_model(model_id, assets=assets, provenance=provenance)
        if obtain is not None:
            obtain()  # Matching published references may need a missing pinned cache member again.
        pending = record['pending']
        if pending and pending['stage'] == 'staging':
            try:
                _populate(record)
            except OSError as exc:
                raise HemError(f'Could not acquire model {model_id}; incomplete assets retained: {exc}') from exc
            # Publication itself is deliberately recoverable independently of validation.
            registry.publish_model(model_id)
        return _qualify(registry, model_id, pipeline)


def import_model(registry, model_id, *, ggufs, tokenizer, alias=False, pipeline=None):
    """Import copies (default) or immutable external references, then qualify."""
    if isinstance(ggufs, (str, Path)):
        ggufs = [ggufs]
    assets = _assets(ggufs, tokenizer, alias)
    hem_display.emit('Local model acquisition and packaged pipeline validation (no native equivalent)')
    return _acquire(registry, model_id, assets, {'kind': 'import'}, pipeline)


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
    paths = [Path(item['cache_path']) for item in files]
    assets = _assets(paths[:-1], paths[-1], True)
    selected = pipeline or recipe['pipeline']
    commands = [['hf', 'download', source['repo_id'], source['filename'], '--revision',
                 source['revision'], '--cache-dir', str(model_download.cache_root())] for source in sources]
    if dry_run:
        return dict(model=model_id, recipe=recipe, files=files, assets=assets,
                    commands=[hem_display.command(command) for command in commands],
                    offline=offline, dry_run=True, completed=True, state='preview')
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
    files = _files(record)
    try:
        evidence = _evidence(files)
        accessible, error = True, None
    except (OSError, HemError) as exc:
        evidence, accessible, error = None, False, str(exc)
    validation = record['validation']
    state = ('incomplete' if record['pending'] and record['pending']['stage'] == 'staging' else
             'missing' if not accessible else
             'ready' if validation and validation['valid'] and validation['evidence'] == evidence else
             'invalid' if validation and not validation['valid'] else 'unvalidated')
    return dict(record, model=model_id, state=state, accessible=accessible, error=error,
                weights=str(files.weights), tokenizer=str(files.tokenizer),
                fragments=[str(path) for path in files.gguf_paths])


def inventory(registry):
    return [show(registry, record['id']) for record in registry.list_models()]
