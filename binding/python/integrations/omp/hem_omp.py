"""Dedicated OMP drop-in ownership and recoverable file replacement.

Only current installation bytes live here. No model roster, client registry or
history is stored. Readers refuse the pending marker until the full generation
has committed. Shared projection publication happens outside this target lock.
"""
from __future__ import annotations

import base64
import hashlib
from importlib.resources import files
import os
from pathlib import Path
import uuid
import re
import shlex
import shutil
import subprocess

from ...hem_common import InvalidInput, result, HemError, DeadlineExpired, UncertainOutcome
from ... import hem_display
from ...hem_client_documents import MAX_BYTES, decode_document, canonical_uuid as _uuid
from ..pi.hem_pi_state import passive_snapshot
from ...hem_clients_endpoints import project_endpoints, ensure_endpoints, read_endpoints
from ...hem_state import Registry
from ..pi.hem_pi_files import (atomic_replace, encoded, private_path, read_owned,
                           remove_owned, staged_path, sync_dir, validate_stage)
from ...hem_state import ResourceLock
from .hem_omp_yaml import scan, selected_file, edit_plan

ASSETS = ("index.ts", "discovery.ts", "native_protocol.ts", "installation.ts", "activation.ts", "stream.ts", "profile.json", "package.json")
ARTIFACTS = (*ASSETS, "config.json")
FILES = (*ARTIFACTS, "manifest.json")
PLUGIN = "extensions/loom-control-omp"
REPAIR = "inspect/export the retained files before repair; rerun loom-control omp configure --check"


VERSION = '18.11.2'
BUN_VERSION = '1.3.14'
BUN_SHA256 = '9fd36f87e4b90b07632b987a2e4ec81ca15a62c81bf983190cea6d715be2ad74'
PACKAGE = '@oh-my-pi/pi-coding-agent'


def normalize_profile(value):
    # Released @oh-my-pi/pi-utils/dirs normalization, kept passive.
    value=value.strip()
    if not value or value=='default':return ''
    if (not re.fullmatch(r'[a-z0-9][a-z0-9._-]{0,63}',value) or value.endswith('.') or
        re.fullmatch(r'(?:CON|PRN|AUX|NUL|COM[0-9]|LPT[0-9])(?:\..*)?',value,re.I)):
        raise InvalidInput('invalid OMP profile name; use the qualified lowercase profile grammar')
    return value


def resolve_target(override=None):
    profile=normalize_profile(os.environ.get('OMP_PROFILE',os.environ.get('PI_PROFILE','')))
    if override is not None and profile:
        raise InvalidInput('explicit --omp-dir conflicts with named OMP_PROFILE/PI_PROFILE; clear profile selectors for explicit target')
    if override is not None: path=Path(override).expanduser()
    elif profile: path=Path.home()/'.omp/profiles'/profile/'agent'
    else: path=Path(os.environ.get('PI_CODING_AGENT_DIR',str(Path.home()/'.omp/agent'))).expanduser()
    if not path.is_absolute(): raise InvalidInput('OMP agent directory must be absolute')
    private_path(path)
    return path.resolve()


def inspect_runtime(deadline,*,passive=False):
    selected=shutil.which('omp')
    if selected is None: raise InvalidInput('OMP absent from PATH; select qualified @oh-my-pi/pi-coding-agent18.11.2 with Bun1.3.14; no automatic install')
    executable=Path(selected).resolve()
    package=None
    for parent in executable.parents:
        path=parent/'package.json'
        if path.is_file():
            # Metadata may live in a shared package installation; no write or execution.
            if path.stat().st_size>1024*1024: raise InvalidInput('oversized OMP package metadata')
            metadata=document(path.read_bytes())
            if metadata.get('name')==PACKAGE: package=parent;break
    if package is None or metadata.get('version')!=VERSION: raise InvalidInput('unsupported OMP package; require18.11.2')
    bins=metadata.get('bin',{})
    if not isinstance(bins,dict) or not isinstance(bins.get('omp'),str) or (package/bins['omp']).resolve()!=executable:
        raise InvalidInput('ambiguous OMP launcher; select genuine package npm bin')
    try:
        with executable.open('rb') as stream: shebang=stream.readline(512).decode().strip()
    except (OSError,UnicodeError) as error: raise InvalidInput('cannot inspect OMP launcher') from error
    if shebang!='#!/usr/bin/env bun': raise InvalidInput('unqualified OMP launcher; require Bun npm route')
    bun=shutil.which('bun')
    if bun is None: raise InvalidInput('qualified Bun1.3.14 absent from PATH')
    if passive:
        hasher=hashlib.sha256()
        if Path(bun).stat().st_size>512*1024*1024: raise InvalidInput('oversized Bun identity')
        with Path(bun).open('rb') as stream:
            while True:
                deadline.remaining();chunk=stream.read(1024*1024)
                if not chunk:break
                hasher.update(chunk)
        if hasher.hexdigest()!=BUN_SHA256:raise InvalidInput('Bun identity not passively qualified; version not executed')

    environment={k:v for k,v in os.environ.items() if k not in ('BUN_OPTIONS','NODE_OPTIONS','NODE_PATH')}
    try:
        observed=None if passive else subprocess.run([str(Path(bun).resolve()),'--version'],env=environment,stdin=subprocess.DEVNULL,capture_output=True,text=True,timeout=deadline.remaining(),check=False)
    except subprocess.TimeoutExpired as error: raise HemError('Bun identity check timed out before writes',accepted=False) from error
    except OSError as error: raise InvalidInput('cannot inspect selected Bun runtime') from error
    if not passive and (observed.returncode or observed.stdout.strip()!=BUN_VERSION): raise InvalidInput('require qualified Bun1.3.14; runtime unchanged')
    dependencies=metadata.get('dependencies',{})
    if dependencies.get('@oh-my-pi/pi-ai')!=VERSION: raise InvalidInput('unsupported OMP pi-ai dependency')
    ai=next((parent/'node_modules/@oh-my-pi/pi-ai' for parent in (package,*package.parents)
             if (parent/'node_modules/@oh-my-pi/pi-ai/package.json').is_file()),None)
    if ai is None or (ai/'package.json').stat().st_size>1024*1024:raise InvalidInput('qualified pi-ai package missing')
    ai_metadata=document((ai/'package.json').read_bytes())
    if ai_metadata.get('name')!='@oh-my-pi/pi-ai' or ai_metadata.get('version')!=VERSION:raise InvalidInput('unsupported selected pi-ai package')
    return dict(package=PACKAGE,version=VERSION,executable=str(executable),bun=str(Path(bun).resolve()),bun_version=BUN_VERSION,qualification='passive qualified Bun/package identity' if passive else 'qualified Bun npm route')


class OMPResult(dict):
    def __init__(self, *args, exit_code=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.exit_code = exit_code


def bundled_assets():
    package = files(__package__.rsplit(".", 2)[0])
    resource = package.joinpath("omp_extension")
    return {name: (package.joinpath(name) if name == "native_protocol.ts" else resource.joinpath(name)).read_bytes()
            for name in ASSETS}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def document(data):
    try:
        return decode_document(data)
    except (ValueError, TypeError, UnicodeError, RecursionError) as error:
        raise InvalidInput("malformed OMP ownership data; " + REPAIR) from error


def validate_set(values, target, root):
    """Prove one complete owned generation, including root and exact whitelist."""
    if not values:
        return None
    # The sole prior format predates the shared companion; no partial set is owned.
    artifacts = ARTIFACTS if "native_protocol.ts" in values else tuple(name for name in ARTIFACTS if name != "native_protocol.ts")
    if set(values) != {*artifacts, "manifest.json"} or any(not isinstance(data, bytes) or len(data) > MAX_BYTES for data in values.values()):
        raise InvalidInput("unsupported OMP owned file set; " + REPAIR)
    manifest, config = document(values["manifest.json"]), document(values["config.json"])
    if (set(manifest) != {"schema_version", "integration_version", "enabled", "generation", "integration_id", "target", "artifacts", "native_activation"} or
        type(manifest["schema_version"]) is not int or manifest["schema_version"] != 1 or
        type(manifest["integration_version"]) is not int or manifest["integration_version"] != 1 or
        manifest["enabled"] is not True or not _uuid(manifest["generation"]) or not _uuid(manifest["integration_id"]) or
        manifest["target"] != str(target) or not isinstance(manifest["artifacts"], dict) or
        set(manifest["artifacts"]) != set(artifacts) or
        any(manifest["artifacts"][name] != digest(values[name]) for name in artifacts)):
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
    if old and old['native_activation'] == activation and all(previous.get(n) == data for n,data in assets.items()): return previous
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


def _check_owned(root,target,assets,providers):
    """Inspect current file/activation state without publishing or recovering."""
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


def _preflight(root,target,assets,providers,disable):
    """Refuse known file/YAML conflicts before root/endpoint publication."""
    inspected = _inspect(target)
    _,values,pending,stages = inspected
    if pending is None and not stages:
        _prepare(root,target,values,providers,disable,assets)
    return inspected


def _apply_owned(root,target,deadline,assets,inspected,*,providers,disable=False):
    """Reinspect and mutate one target under its own lock, preserving recovery."""
    _,values,pending,stages = inspected
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


def configure_omp(root,target,deadline,*,check=False,disable=False):
    """Own the complete qualified OMP operation for CLI and direct callers."""
    if check and disable:
        raise InvalidInput('OMP check and disable are mutually exclusive')
    deadline.remaining()
    value=Path(root if root is not None else os.environ.get('LOOM_SERVE_DIR',str(Path.home()/'.loom')))
    if not value.is_absolute(): raise InvalidInput('LOOM_SERVE_DIR must be absolute')
    private_path(value);root=value.resolve();target=resolve_target(target)
    runtime=None;compatibility=dict(state='not_checked',reason='client unnecessary for owned removal')
    if not disable:
        try:
            runtime=inspect_runtime(deadline,passive=check)
            compatibility=dict(state='compatible',reason=runtime['qualification'])
        except (HemError,OSError) as error:
            if not check: raise
            compatibility=dict(state='unqualified',reason=str(error))
    assets=bundled_assets()
    projection=None;providers=[]
    if not disable:
        projection=project_endpoints(root,passive_snapshot(root,deadline)['servers'])
        if len(projection['servers'])>256: raise InvalidInput('OMP supports at most256 candidate endpoints')
        providers=[row['provider'] for row in projection['servers']]
    hem_display.emit('No native Loom equivalent: owned OMP extension and native keyless-provider activation')
    try:
        if check:
            response=_check_owned(root,target,assets,providers)
        else:
            inspected=_preflight(root,target,assets,providers,disable)
            if not disable:
                ensure_endpoints(Registry(root,deadline=deadline),deadline)
            response=_apply_owned(root,target,deadline,assets,inspected,providers=providers,disable=disable)
    except (OSError,DeadlineExpired) as error:
        if check: raise HemError('OMP inspection failed without writes',accepted=False) from error
        raise UncertainOutcome('OMP update interrupted; '+REPAIR) from error
    response.update(runtime=runtime,runtime_compatibility=compatibility,
        availability=dict(state='not_observed',reason='startup or explicit refresh/restart discovers ready models'))
    if check:
        try:
            valid=read_endpoints(root)==projection and projection is not None
            response['endpoint_projection']='current' if valid else 'absent_or_outdated'
            if not valid: response['findings'].append('root endpoint projection absent/outdated; rerun configure');response.exit_code=response.exit_code or 1
        except (HemError,OSError) as error:
            response['findings'].append(str(error));response.exit_code=response.exit_code or 2
        if compatibility['state']!='compatible':response.exit_code=response.exit_code or 1
    if not disable:
        prefix='PI_CODING_AGENT_DIR='+shlex.quote(str(target))+' OMP_PROFILE= PI_PROFILE= '
        response.update(launch=prefix+'omp',refresh=prefix+'omp models refresh --json',
            guidance='Adding a saved server requires configure, then refresh and restart OMP. Model changes on configured servers require refresh and restart only. No polling. Export/backups are your responsibility. Advanced YAML outside the narrow block subset requires manual editing.')
        hem_display.emit('Launch: '+response['launch']);hem_display.emit('Refresh: '+response['refresh']+'; then restart OMP')
    return response
