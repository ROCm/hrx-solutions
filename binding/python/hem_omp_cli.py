"""Optional OMP setup, never client/server lifecycle ownership."""
from __future__ import annotations

import os
import hashlib
from pathlib import Path
import re
import shlex
import shutil
import subprocess
from . import hem_arguments as arguments, hem_display
from .hem_common import InvalidInput, HemError, DeadlineExpired, UncertainOutcome
from .hem_omp import bundled_assets, configure_omp, document, REPAIR
from .hem_pi_files import private_path, read_owned
from .hem_pi_state import passive_snapshot
from .hem_clients_endpoints import project_endpoints, ensure_endpoints, read_endpoints
from .hem_state import Registry

VERSION = '18.11.2'
BUN_VERSION = '1.3.14'
BUN_SHA256 = '9fd36f87e4b90b07632b987a2e4ec81ca15a62c81bf983190cea6d715be2ad74'
PACKAGE = '@oh-my-pi/pi-coding-agent'


def configure(parser):
    commands=parser.add_subparsers(dest='command',required=True)
    child=commands.add_parser('configure',help='configure or inspect Oh My Pi integration')
    arguments.output(child);arguments.lifecycle(child)
    mode=child.add_mutually_exclusive_group()
    mode.add_argument('--check',action='store_true',help='inspect without writes or client execution')
    mode.add_argument('--disable',action='store_true',help='remove unchanged owned entries without client')
    child.add_argument('--omp-dir',type=Path,help='explicit default-profile agent directory')


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


def run(args,deadline):
    value=Path(os.environ.get('LOOM_SERVE_DIR',str(Path.home()/'.loom')))
    if not value.is_absolute(): raise InvalidInput('LOOM_SERVE_DIR must be absolute')
    private_path(value);root=value.resolve();target=resolve_target(args.omp_dir)
    runtime=None;compatibility=dict(state='not_checked',reason='client unnecessary for owned removal')
    if not args.disable:
        try:
            runtime=inspect_runtime(deadline,passive=args.check)
            compatibility=dict(state='compatible',reason=runtime['qualification'])
        except (HemError,OSError) as error:
            if not args.check: raise
            compatibility=dict(state='unqualified',reason=str(error))
    bundled_assets()
    projection=None;providers=[]
    if not args.disable:
        projection=project_endpoints(root,passive_snapshot(root,deadline)['servers'])
        if len(projection['servers'])>256: raise InvalidInput('OMP supports at most256 candidate endpoints')
        providers=[row['provider'] for row in projection['servers']]
    hem_display.emit('No native Loom equivalent: owned OMP extension and native keyless-provider activation')
    try:
        if not args.check and not args.disable:
            preview=configure_omp(root,target,deadline,providers=providers,check=True)
            if preview['state']=='conflict': raise InvalidInput('; '.join(preview['findings']))
            ensure_endpoints(Registry(root,deadline=deadline),deadline)
        response=configure_omp(root,target,deadline,providers=providers,check=args.check,disable=args.disable)
    except (OSError,DeadlineExpired) as error:
        if args.check: raise HemError('OMP inspection failed without writes',accepted=False) from error
        raise UncertainOutcome('OMP update interrupted; '+REPAIR) from error
    response.update(runtime=runtime,runtime_compatibility=compatibility,
        availability=dict(state='not_observed',reason='startup or explicit refresh/restart discovers ready models'))
    if args.check:
        try:
            valid=read_endpoints(root)==projection and projection is not None
            response['endpoint_projection']='current' if valid else 'absent_or_outdated'
            if not valid: response['findings'].append('root endpoint projection absent/outdated; rerun configure');response.exit_code=response.exit_code or 1
        except (HemError,OSError) as error:
            response['findings'].append(str(error));response.exit_code=response.exit_code or 2
        if compatibility['state']!='compatible':response.exit_code=response.exit_code or 1
    if not args.disable:
        prefix='PI_CODING_AGENT_DIR='+shlex.quote(str(target))+' OMP_PROFILE= PI_PROFILE= '
        response.update(launch=prefix+'omp',refresh=prefix+'omp models refresh --json',
            guidance='Adding a saved server requires configure, then refresh and restart OMP. Model changes on configured servers require refresh and restart only. No polling. Export/backups are your responsibility. Advanced YAML outside the narrow block subset requires manual editing.')
        hem_display.emit('Launch: '+response['launch']);hem_display.emit('Refresh: '+response['refresh']+'; then restart OMP')
    return response
