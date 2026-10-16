"""Dedicated OMP drop-in ownership and recoverable file replacement.

Only current installation bytes live here. No model roster, client registry or
history is stored. Readers refuse the pending marker until the full generation
has committed. Shared projection publication happens outside this target lock.
"""
from __future__ import annotations

import base64
import hashlib
from importlib.resources import files
import json
import os
from pathlib import Path
import uuid

from .hem_common import InvalidInput, result
from .hem_pi_files import (atomic_replace, encoded, private_path, read_owned,
                           remove_owned, staged_path, sync_dir, validate_stage)
from .hem_state import ResourceLock
from .hem_omp_yaml import scan, selected_file, edit_plan

ASSETS = ("index.ts", "discovery.ts", "installation.ts", "activation.ts", "stream.ts", "profile.json", "package.json")
ARTIFACTS = (*ASSETS, "config.json")
FILES = (*ARTIFACTS, "manifest.json")
PLUGIN = "extensions/loom-control-omp"
MAX_BYTES = 1024 * 1024
REPAIR = "inspect/export the retained files before repair; rerun loom-control omp configure --check"


class OMPResult(dict):
    def __init__(self, *args, exit_code=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.exit_code = exit_code


def bundled_assets():
    resource = files(__package__).joinpath("omp_extension")
    return {name: resource.joinpath(name).read_bytes() for name in ASSETS}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def _unique(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate field")
        value[key] = item
    return value


def document(data):
    try:
        if len(data) > MAX_BYTES:
            raise ValueError()
        value = json.loads(data, object_pairs_hook=_unique)
        def depth(item, level=0):
            if level > 64:
                raise ValueError()
            if isinstance(item, dict):
                for child in item.values(): depth(child, level + 1)
            elif isinstance(item, list):
                for child in item: depth(child, level + 1)
        depth(value)
        if not isinstance(value, dict): raise ValueError()
        return value
    except (ValueError, TypeError, UnicodeError, RecursionError) as error:
        raise InvalidInput("malformed OMP ownership data; " + REPAIR) from error


def _uuid(value):
    try:
        identifier = uuid.UUID(value)
        return identifier.version == 4 and str(identifier) == value
    except (ValueError, TypeError, AttributeError):
        return False


def _canonical(path):
    path = Path(path)
    if not path.is_absolute() or str(path.resolve()) != str(path):
        raise InvalidInput("OMP target/state root must be canonical and absolute")
    private_path(path)
    return path


def validate_set(values, target, root):
    """Prove one complete owned generation, including root and exact whitelist."""
    if not values:
        return None
    if set(values) != set(FILES) or any(not isinstance(data, bytes) or len(data) > MAX_BYTES for data in values.values()):
        raise InvalidInput("unsupported OMP owned file set; " + REPAIR)
    manifest, config = document(values["manifest.json"]), document(values["config.json"])
    if (set(manifest) != {"schema_version", "integration_version", "enabled", "generation", "integration_id", "target", "artifacts", "native_activation"} or
        type(manifest["schema_version"]) is not int or manifest["schema_version"] != 1 or
        type(manifest["integration_version"]) is not int or manifest["integration_version"] != 1 or
        manifest["enabled"] is not True or not _uuid(manifest["generation"]) or not _uuid(manifest["integration_id"]) or
        manifest["target"] != str(target) or not isinstance(manifest["artifacts"], dict) or
        set(manifest["artifacts"]) != set(ARTIFACTS) or
        any(manifest["artifacts"][name] != digest(values[name]) for name in ARTIFACTS)):
        raise InvalidInput("foreign, edited or unsupported OMP manifest; " + REPAIR)
    activation = manifest['native_activation']
    if (not isinstance(activation, dict) or set(activation) != {'path', 'providers', 'heading'} or
        activation['path'] not in [str(target / name) for name in ('models.yml', 'models.yaml')] or
        not isinstance(activation['providers'], dict)):
        raise InvalidInput('invalid OMP activation ownership')
    for name, record in activation['providers'].items():
        import re
        if (not re.fullmatch(r'loom-hem-[A-Za-z0-9_-]+', name) or not isinstance(record, dict) or
            set(record) != {'ownership', 'sha256'} or record['ownership'] not in ('owned', 'borrowed') or
            not isinstance(record['sha256'], str) or not re.fullmatch(r'[0-9a-f]{64}', record['sha256'])):
            raise InvalidInput('invalid OMP provider ownership')
    heading = activation['heading']
    if heading is not None and (not isinstance(heading, dict) or set(heading) != {'original', 'replacement', 'created_file'} or
        not isinstance(heading['original'], str) or not isinstance(heading['replacement'], str) or
        type(heading['created_file']) is not bool or len(heading['original']) > 4096 or
        not __import__('re').fullmatch(r'(?:\n)?providers: *(?:#[^\r\n]*)?\r?\n', heading['replacement']) or
        heading['original'] and not __import__('re').fullmatch(r'providers:(?: *(?:\{\})? *(?:#[^\r\n]*)?)?(?:\r?\n)?', heading['original'])):
        raise InvalidInput('invalid OMP heading ownership')
    expected = dict(schema_version=1, generation=manifest["generation"], integration_id=manifest["integration_id"],
                    target=str(target), hem_root=str(root), endpoint_file=str(root / "clients/endpoints.json"), native_models_file=manifest["native_activation"]["path"])
    if config != expected or type(config.get("schema_version")) is not int:
        raise InvalidInput("OMP installation belongs to another root/target or generation; " + REPAIR)
    return manifest


def _read(path):
    data = read_owned(path)
    if data is not None and len(data) > MAX_BYTES:
        raise InvalidInput("oversized OMP owned file; " + REPAIR)
    return data


def _inspect(target):
    directory = target / PLUGIN
    private_path(directory)
    for path in (target, target/"extensions", directory):
        if path.exists() and path.stat().st_mode & 0o077:
            raise InvalidInput("OMP target/owned directories must be private: " + str(path))
    allowed = set(FILES) | {"pending.json"}
    # Fixed recovery temporaries are inspected even before a pending commit.
    allowed |= {staged_path(Path(name)).name for name in (*FILES, "pending.json")}
    if directory.exists():
        unknown = [path.name for path in directory.iterdir() if path.name not in allowed]
        if unknown:
            raise InvalidInput("unknown OMP plugin content retained: " + ", ".join(sorted(unknown)))
    values = {name: data for name in FILES if (data := _read(directory / name)) is not None}
    pending = _read(directory / "pending.json")
    stages = {name: _read(staged_path(directory / name)) for name in (*FILES, "pending.json")}
    return directory, values, pending, {name: data for name, data in stages.items() if data is not None}


def _hash(data):
    return None if data is None else digest(data)


def _plan(previous, desired, target, root, native):
    old, new = validate_set(previous, target, root), validate_set(desired, target, root)
    if not old and not new: raise InvalidInput('empty OMP recovery plan')
    if old and new and old['integration_id'] != new['integration_id']:
        raise InvalidInput('OMP recovery identity changed')
    return encoded(dict(schema_version=1, target=str(target), hem_root=str(root),
        previous={name: base64.b64encode(data).decode('ascii') for name, data in previous.items()},
        desired={name: base64.b64encode(data).decode('ascii') for name, data in desired.items()}, native=native))


def _decode_plan(data, target, root):
    value = document(data)
    if (set(value) != {'schema_version', 'target', 'hem_root', 'previous', 'desired', 'native'} or
        type(value['schema_version']) is not int or value['schema_version'] != 1 or
        value['target'] != str(target) or value['hem_root'] != str(root)):
        raise InvalidInput('foreign OMP recovery plan; ' + REPAIR)
    try:
        sets = [{name: base64.b64decode(item, validate=True) for name, item in value[key].items()}
                for key in ('previous', 'desired')]
        native = value['native']
        if set(native) != {'path', 'before', 'after', 'patches'} or native['path'] not in [str(target/n) for n in ('models.yml','models.yaml')]: raise ValueError()
        for patch in native['patches']:
            if not isinstance(patch, dict) or not {'start','old','new'} <= set(patch) <= {'start','old','new','final'}: raise ValueError()
            if type(patch['start']) is not int or patch['start'] < 0 or not all(isinstance(patch[k],str) for k in ('old','new')): raise ValueError()
        if _plan(*sets, target, root, native) != data: raise ValueError()
    except (ValueError, TypeError, AttributeError, KeyError) as error:
        raise InvalidInput('malformed OMP recovery plan; ' + REPAIR) from error
    return *sets, native


def _native_desired(current, native):
    if _hash(current) == native['after']: return current
    if _hash(current) != native['before']:
        raise InvalidInput('native OMP YAML changed during recovery; retained; ' + REPAIR)
    text = '' if current is None else current.decode('utf8')
    ordinary = [p for p in native['patches'] if not p.get('final')]
    final = [p for p in native['patches'] if p.get('final')]
    for patch in sorted(ordinary, key=lambda p:p['start'], reverse=True) + final:
        at = patch['start']
        if text[at:at+len(patch['old'])] != patch['old']: raise InvalidInput('OMP YAML recovery anchor changed')
        text = text[:at] + patch['new'] + text[at+len(patch['old']):]
    data = None if native['after'] is None else text.encode()
    if _hash(data) != native['after']: raise InvalidInput('OMP YAML recovery digest mismatch')
    scan(text)
    return data


def _resume(directory, pending, target, root, deadline):
    previous, desired, native = _decode_plan(pending, target, root)
    if selected_file(target) != Path(native['path']): raise InvalidInput('selected native OMP YAML changed')
    current = {name: _read(directory / name) for name in FILES}
    native_path = Path(native['path'])
    old_yaml = _read(native_path)
    new_yaml = _native_desired(old_yaml, native)
    for name in FILES:
        if current[name] not in (previous.get(name), desired.get(name)):
            raise InvalidInput('edited OMP recovery file retained: ' + name + '; ' + REPAIR)
        if name in desired: validate_stage(directory/name, desired[name], allow_prefix=True)
        elif _read(staged_path(directory/name)) is not None: raise InvalidInput('unexpected OMP removal temporary')
    if new_yaml is not None: validate_stage(native_path, new_yaml, allow_prefix=True)
    elif _read(staged_path(native_path)) is not None: raise InvalidInput('unexpected OMP YAML removal temporary')
    if _read(staged_path(directory/'pending.json')) is not None:
        validate_stage(directory/'pending.json', pending)
        remove_owned(staged_path(directory/'pending.json'), pending, deadline)
    if old_yaml != new_yaml or _read(staged_path(native_path)) is not None:
        if new_yaml is None: remove_owned(native_path, old_yaml, deadline)
        else: atomic_replace(native_path, new_yaml, expected=old_yaml, deadline=deadline, allow_prefix=True)
    # Pending blocks the loader throughout. Manifest and entrypoint publish last.
    for name in [n for n in FILES if n not in ('manifest.json','index.ts')] + ['manifest.json','index.ts']:
        path = directory/name
        if name in desired:
            if current[name] != desired[name] or _read(staged_path(path)) is not None:
                atomic_replace(path, desired[name], expected=current[name], deadline=deadline, allow_prefix=True)
        else: remove_owned(path, current[name], deadline)
    remove_owned(directory/'pending.json', pending, deadline)


def _desired(target, root, assets, previous, identity, activation):
    old = validate_set(previous, target, root)
    if old and old['native_activation'] == activation and all(previous[n] == data for n,data in assets.items()): return previous
    generation = str(uuid.uuid4())
    config = encoded(dict(schema_version=1, generation=generation, integration_id=identity,
        target=str(target), hem_root=str(root), endpoint_file=str(root/'clients/endpoints.json'), native_models_file=activation['path']))
    values = dict(assets, **{'config.json':config})
    values['manifest.json'] = encoded(dict(schema_version=1, integration_version=1, enabled=True,
        generation=generation, integration_id=identity, target=str(target),
        artifacts={name:digest(data) for name,data in values.items()}, native_activation=activation))
    validate_set(values,target,root)
    return values


def _prepare(root, target, values, providers, disable, assets):
    manifest = validate_set(values,target,root)
    identity = manifest['integration_id'] if manifest else str(uuid.uuid4())
    path = selected_file(target)
    old_activation = manifest['native_activation'] if manifest else dict(path=str(path),providers={},heading=None)
    if old_activation['path'] != str(path): raise InvalidInput('selected native OMP YAML changed; ' + REPAIR)
    before = _read(path)
    if disable and not manifest:
        return {}, dict(path=str(path),before=_hash(before),after=_hash(before),patches=[])
    try:
        after, patches, heading = edit_plan(before, identity, providers, old_activation['providers'],
                                           disable=disable, heading_state=old_activation['heading'])
    except UnicodeError as error: raise InvalidInput('native OMP YAML must be UTF8') from error
    if before is None and after == b'' and not patches: after = None
    records = {}
    if not disable:
        for name,row in scan(after.decode() if after else '').items():
            if name in providers or name in old_activation['providers']:
                if not row['valid'] or row['owner'] not in (None,identity): raise InvalidInput('conflicting OMP native activation')
                records[name] = dict(ownership='owned' if row['owner'] else 'borrowed',sha256=digest(row['text'].encode()))
    activation = dict(path=str(path),providers=records,heading=heading)
    desired = {} if disable else _desired(target,root,assets,values,identity,activation)
    native = dict(path=str(path),before=_hash(before),after=_hash(after),patches=patches)
    return desired,native


def configure_omp(root,target,deadline,*,providers=(),check=False,disable=False):
    root,target = _canonical(root),_canonical(target)
    assets = bundled_assets()
    if check:
        try:
            _,values,pending,stages = _inspect(target)
            manifest = validate_set(values,target,root) if not pending and not stages else None
            if pending or stages: state='pending'
            elif not manifest: state='absent'
            else:
                desired,_ = _prepare(root,target,values,providers,False,assets)
                state='configured' if desired==values else 'outdated'
            return OMPResult(result(state,completed=True,configured=state=='configured',target=str(target),
                pending=bool(pending or stages),findings=[] if state=='configured' else [state],
                next_action=('Inspect/export retained recovery files; an incomplete unpublished plan temporary cannot be safely adopted. ' + str(staged_path(target/PLUGIN/'pending.json')) + '; after verifying its ownership and that no pending.json was published, manually move the temporary aside and rerun configure.' if pending is None and 'pending.json' in stages else REPAIR) if state=='pending' else None),exit_code=0 if state=='configured' else 1)
        except InvalidInput as error:
            return OMPResult(result('conflict',completed=True,configured=False,target=str(target),findings=[str(error)],next_action=REPAIR),exit_code=2)
    _,values,pending,stages = _inspect(target)
    if pending is None and not stages: _prepare(root,target,values,providers,disable,assets)
    if disable and not values and pending is None and not stages:
        return OMPResult(result('disabled',completed=True,changed=False,configured=False,target=str(target),affected_paths=[]))
    private_path(target,create=True)
    with ResourceLock(target,'target','unused',deadline,path=target/'.loom-control-omp.lock'):
        directory,values,pending,stages = _inspect(target)
        if pending is None and 'pending.json' in stages:
            pending=stages['pending.json']
            try: _decode_plan(pending,target,root)
            except InvalidInput as error:
                raise InvalidInput('Unpublished OMP plan temporary is incomplete or foreign and retained: ' + str(staged_path(directory/'pending.json')) + '; inspect/export it, verify ownership and absence of published pending.json, then manually move it aside before retry. No native YAML or extension update was started from this plan.') from error
            atomic_replace(directory/'pending.json',pending,expected=None,deadline=deadline)
        elif pending is None and stages: raise InvalidInput('orphan OMP staging files retained; '+REPAIR)
        if pending is not None:
            _resume(directory,pending,target,root,deadline)
            _,values,_,_=_inspect(target)
        desired,native=_prepare(root,target,values,providers,disable,assets)
        changed=values!=desired or native['before']!=native['after']
        if changed:
            plan=_plan(values,desired,target,root,native)
            if len(plan)>MAX_BYTES: raise InvalidInput('OMP recovery plan exceeds 1 MiB')
            atomic_replace(directory/'pending.json',plan,expected=None,deadline=deadline)
            _resume(directory,plan,target,root,deadline)
        if disable and directory.exists(): directory.rmdir();sync_dir(directory.parent)
        return OMPResult(result('disabled' if disable else 'configured',completed=True,changed=changed,
            configured=not disable,target=str(target),affected_paths=[str(directory/n) for n in FILES] if changed else []))
