"""Explicit selected Moonshine source/bytecode packaging, independent of Qwen."""
from __future__ import annotations

import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

TARGET = '//loom-pipeline:moonshine_bytecode'
RELATIVE = 'recognition/moonshine/streaming_tiny'
ARTIFACTS = ('prepare.loombc', 'control.loombc', 'weights.loombc', 'providers.loombc')
DATA = ('catalog.json', 'sources.json', 'LICENSE.moonshine', 'LICENSE-APACHE-2.0')
REVISION = '61417137f334cfa828b38d6e9b0f2d7190503f16'


def digest(content):
    return hashlib.sha256(content).hexdigest()


def inventory(repository):
    path = repository / 'loom-pipeline/moonshine_sources.bzl'
    tree = ast.parse(path.read_text())
    if (len(tree.body) != 1 or not isinstance(tree.body[0], ast.Assign) or
            len(tree.body[0].targets) != 1 or not isinstance(tree.body[0].targets[0], ast.Name) or
            tree.body[0].targets[0].id != 'MOONSHINE_ARTIFACTS'):
        raise ValueError('Expected one literal Moonshine artifact inventory')
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            keys = [ast.literal_eval(key) for key in node.keys]
            if len(keys) != len(set(keys)):
                raise ValueError('Duplicate Moonshine inventory key')
    groups = ast.literal_eval(tree.body[0].value)
    if (not isinstance(groups, dict) or set(groups) != set(ARTIFACTS) or
            any(not isinstance(group, list) for group in groups.values()) or
            groups['prepare.loombc'] != ['prepare.loom'] or
            groups['control.loombc'] != ['control.loom'] or
            groups['weights.loombc'] != ['weights.loom'] or len(groups['providers.loombc']) != 13):
        raise ValueError('Moonshine requires the reviewed four-artifact mapping')
    sources = [source for group in groups.values() for source in group]
    if (len(sources) != 16 or len(set(sources)) != 16 or
            any(not isinstance(source, str) or Path(source).name != source or not source.endswith('.loom') for source in sources)):
        raise ValueError('Expected sixteen distinct local Moonshine sources')
    return groups


def validate_sources(repository):
    groups = inventory(repository)
    root = repository / 'loom-pipeline' / RELATIVE
    sources = {source for group in groups.values() for source in group}
    if {p.name for p in root.glob('*.loom')} != sources:
        raise ValueError('Undeclared or missing Moonshine source')
    provenance = json.loads((root / 'sources.json').read_text())
    if (provenance.get('format') != 1 or provenance.get('revision') != REVISION or
            provenance.get('repository') != 'https://github.com/ROCm/hrx-system' or
            provenance.get('upstream_directory') != 'experimental/loom_serve/models/moonshine' or
            provenance.get('sources') != {name:digest((root / name).read_bytes()) for name in sources} or
            provenance.get('notices') != {name:digest((root / name).read_bytes()) for name in DATA[2:]} or
            provenance.get('projection') != {'prepare.loom':{'policy_from':'weights.loom','policy_to':'weights.loombc'}}):
        raise ValueError('Moonshine source/projection/notice provenance mismatch')
    originals = provenance.get('upstream_sources', {})
    if (set(originals) != sources or set(provenance.get('local_patches', {})) != {'prepare.loom','control.loom'} or
            any(not isinstance(checksum,str) or not re.fullmatch('[0-9a-f]{64}',checksum) for checksum in originals.values()) or
            any(originals[name] != provenance['sources'][name] for name in sources - {'prepare.loom','control.loom'})):
        raise ValueError('Moonshine original/local compatibility patch provenance mismatch')
    if any((root / source).is_symlink() for source in (*sources, *DATA)):
        raise ValueError('Moonshine catalog source/data must be physically owned files')
    return groups


def installed_definition(document):
    value = deepcopy(document)
    value['definition'].update(prepare='prepare.loombc', control='control.loombc', sources=['providers.loombc'])
    return value


def roots(text):
    patterns = {'functions':r'^func\.def public @([^\s(]+)',
                'commands':r'^command\.program\.def public @([^\s(]+)',
                'configs':r'^config\.(?:def|decl) @([^\s=:(]+)'}
    return {kind:sorted(set(re.findall(pattern,text,re.MULTILINE))) for kind,pattern in patterns.items()}


def validate_artifact(path, sources, root, decoder, policy=False):
    inputs = '\n'.join((root / source).read_text() for source in sources)
    with tempfile.TemporaryDirectory() as temporary:
        output = Path(temporary) / 'decoded.loom'
        subprocess.run([str(decoder),str(path),'--from=bc','--to=text','--output='+str(output)],check=True)
        text = output.read_text()
    expected = roots(inputs)
    actual = roots(text)
    if (actual != expected or len(re.findall(r'^func\.def ',inputs,re.MULTILINE)) !=
            len(re.findall(r'^func\.def ',text,re.MULTILINE))):
        raise ValueError('Moonshine bytecode public function/command/configuration closure differs from authored inputs')
    if policy:
        declaration = 'global.rodata.def @policy = bytes("'+b'weights.loombc'.hex()+'")'
        if text.count(declaration) != 1 or b'weights.loom'.hex()+'"' in text:
            raise ValueError('Moonshine bootstrap must reference installed weights.loombc policy')
    return expected


def query_outputs(query):
    """Interpret ordered build-bound outputs using Moonshine physical artifacts."""
    outputs = {}
    for path in query(TARGET):
        name = path.name
        if name not in ARTIFACTS or name in outputs or not path.as_posix().endswith('/'+RELATIVE+'/'+name):
            raise ValueError('Unexpected or duplicate Moonshine artifact output')
        path = path.resolve()
        if not path.is_file() or not path.read_bytes().startswith(b'LOOM'):
            raise ValueError('Missing or invalid Moonshine bytecode')
        outputs[name] = path
    if set(outputs) != set(ARTIFACTS):
        raise ValueError('Missing required Moonshine bytecode output')
    return outputs


def stage(destination, outputs, repository, *, decoder):
    groups = validate_sources(repository)
    if set(outputs) != set(ARTIFACTS):
        raise ValueError('Unexpected Moonshine artifact inventory')
    root = repository / 'loom-pipeline' / RELATIVE
    metadata = {'format':1,'policy':'weights.loombc','files':{},'roots':{}}
    for name,sources in groups.items():
        metadata['roots'][name] = validate_artifact(outputs[name],sources,root,decoder,name=='prepare.loombc')
        metadata['files'][name] = digest(outputs[name].read_bytes())
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    for name in ARTIFACTS:
        shutil.copyfile(outputs[name],destination/name)
    for name in DATA[1:]:
        shutil.copyfile(root/name,destination/name)
    (destination/'catalog.json').write_text(json.dumps(installed_definition(json.loads((root/'catalog.json').read_text())),indent=2)+'\n')
    (destination/'artifacts.json').write_text(json.dumps(metadata,indent=2,sort_keys=True)+'\n')
    return sorted(destination.iterdir())


def stage_package(package_root, outputs, repository, *, decoder):
    """Own Moonshine's installed catalog destination and source interpretation."""
    return stage(package_root / 'loom-pipeline' / RELATIVE, outputs, repository, decoder=decoder)


def validate_archive(archive, repository, *, prefix='hrx_solutions/loom-pipeline/'):
    groups = validate_sources(repository)
    root = repository / 'loom-pipeline' / RELATIVE
    prefix += RELATIVE+'/'
    expected = {prefix+name for name in (*ARTIFACTS,*DATA,'artifacts.json')}
    actual = {name for name in archive.namelist() if name.startswith('hrx_solutions/loom-pipeline/recognition/') and not name.endswith('/')}
    if actual != expected:
        raise ValueError('Moonshine installed catalog allowlist mismatch')
    metadata = json.loads(archive.read(prefix+'artifacts.json'))
    expected_metadata = {'format':1,'policy':'weights.loombc',
        'files':{name:digest(archive.read(prefix+name)) for name in ARTIFACTS},
        'roots':{name:roots('\n'.join((root/source).read_text() for source in sources)) for name,sources in groups.items()}}
    if metadata != expected_metadata or any(not archive.read(prefix+name).startswith(b'LOOM') for name in ARTIFACTS):
        raise ValueError('Moonshine installed artifact integrity/closure metadata mismatch')
    if json.loads(archive.read(prefix+'catalog.json')) != installed_definition(json.loads((root/'catalog.json').read_text())):
        raise ValueError('Moonshine installed native configuration differs from curated source')
    if any(archive.read(prefix+name) != (root/name).read_bytes() for name in DATA[1:]):
        raise ValueError('Moonshine installed source/notice provenance differs from authored input')


def validate_source_archive(archive, repository):
    """Require owned authored inputs/notices and exclude engineering payloads."""
    groups = validate_sources(repository)
    root = repository / 'loom-pipeline' / RELATIVE
    files = {source for group in groups.values() for source in group} | set(DATA)
    for name in files:
        suffix = '/loom-pipeline/'+RELATIVE+'/'+name
        matches = [member for member in archive.getmembers() if member.name.endswith(suffix)]
        if (len(matches) != 1 or not matches[0].isfile() or
                archive.extractfile(matches[0]).read() != (root/name).read_bytes()):
            raise ValueError('Source archive missing/changed physically owned Moonshine input: '+name)
    for member in archive.getmembers():
        if Path(member.name).suffix.lower() in ('.gguf','.safetensors','.pt','.pth','.ckpt','.wav','.mp3','.flac','.pcm'):
            raise ValueError('Source archive contains model/audio engineering payload')
