"""Loom compiler admission and source-to-bytecode catalog normalization."""
from __future__ import annotations

import argparse
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import posixpath
import re
import subprocess
import tempfile

CATALOG_TARGET = '//loom-pipeline:bytecode_catalogs'
DECODER_TARGET = '@hrx_system//loom/src/loom/tools/loom-format'


def relative(value):
    if (type(value) is not str or not value or '\\' in value or '\0' in value
            or PurePosixPath(value).is_absolute() or PureWindowsPath(value).drive
            or any(part in ('', '.', '..') or part.startswith('-') for part in value.split('/'))):
        raise ValueError('Unsafe build catalog path: ' + repr(value))
    return value


def source_path(root, name):
    path = root / relative(name)
    path.resolve(strict=True).relative_to(root.resolve(strict=True))
    if not path.is_file():
        raise ValueError('Missing regular catalog input: ' + name)
    return path


def digest(content):
    return hashlib.sha256(content).hexdigest()


def unique(pairs):
    result = {}
    for name, value in pairs:
        if name in result:
            raise ValueError('Duplicate catalog JSON key: ' + name)
        result[name] = value
    return result


def document(path):
    with path.open('rb') as stream:
        content = stream.read(1024 * 1024 + 1)
    if len(content) > 1024 * 1024:
        raise ValueError('Catalog declaration exceeds size limit')
    return json.loads(content, object_pairs_hook=unique), content


def load_builds(path):
    """One literal build declaration; names never select Python implementations."""
    tree = ast.parse(path.read_text())
    if (len(tree.body) != 1 or not isinstance(tree.body[0], ast.Assign)
            or len(tree.body[0].targets) != 1 or not isinstance(tree.body[0].targets[0], ast.Name)
            or tree.body[0].targets[0].id != 'CATALOG_BUILDS'):
        raise ValueError('Expected one literal CATALOG_BUILDS assignment')
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            keys = [ast.literal_eval(key) for key in node.keys]
            if len(keys) != len(set(keys)):
                raise ValueError('Duplicate catalog build key')
    builds = ast.literal_eval(tree.body[0].value)
    if type(builds) is not dict or not builds:
        raise ValueError('Missing catalog build declarations')
    jobs = {}
    for reference, build in builds.items():
        relative(reference)
        if not reference.endswith('/executable.json') or type(build) is not dict or set(build) != {'root', 'groups', 'roles', 'notices', 'helper'}:
            raise ValueError('Invalid catalog build declaration')
        relative(build['root'])
        if type(build['groups']) is not dict or not build['groups']:
            raise ValueError('Missing compiler groups')
        for output, group in build['groups'].items():
            relative(output)
            if not output.endswith('.loombc') or type(group) is not dict or set(group) != {'sources', 'policy', 'exact_configs'}:
                raise ValueError('Invalid compiler group')
            sources = group['sources']
            if (type(sources) is not list or not sources or not all(type(name) is str for name in sources)
                    or len(sources) != len(set(sources))
                    or type(group['exact_configs']) is not bool):
                raise ValueError('Invalid compiler source closure')
            for source in sources:
                relative(source)
                if not source.endswith('.loom'):
                    raise ValueError('Compiler inputs must be Loom source')
            policy = group['policy']
            if policy is not None:
                if type(policy) is not dict or set(policy) != {'source', 'from', 'to'} | ({'symbol'} if 'symbol' in policy else set()) or policy['source'] not in sources:
                    raise ValueError('Invalid policy projection')
                for value in policy.values(): relative(value)
                if not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_.]*', policy.get('symbol', 'policy')):
                    raise ValueError('Invalid policy symbol')
            name = build['root'] + '/' + output
            if name in jobs and jobs[name] != group:
                raise ValueError('Conflicting compiler producers: ' + name)
            jobs[name] = group
        roles = build['roles']
        if type(roles) is not dict or set(roles) != {'prepare', 'control', 'sources', 'support'}:
            raise ValueError('Invalid compiled native roles')
        for key in ('prepare', 'control'): relative(roles[key])
        for key in ('sources', 'support'):
            if (type(roles[key]) is not list or (key == 'sources' and not roles[key])
                    or not all(type(name) is str for name in roles[key])
                    or len(roles[key]) != len(set(roles[key]))):
                raise ValueError('Invalid compiled native role sequence')
            for name in roles[key]: relative(name)
        native = {roles['prepare'], roles['control'], *roles['sources']}
        if native & set(roles['support']) or native | set(roles['support']) != set(build['groups']):
            raise ValueError('Compiled roles do not close over compiler products')
        if (type(build['notices']) is not list or not all(type(name) is str for name in build['notices'])
                or len(build['notices']) != len(set(build['notices']))):
            raise ValueError('Invalid catalog notice inventory')
        for name in build['notices']:
            relative(name)
            if Path(name).suffix.lower() in ('.loom', '.loombc', '.gguf', '.safetensors', '.pt', '.pth', '.ckpt', '.wav', '.mp3', '.flac', '.pcm'):
                raise ValueError('Notices cannot ship model or executable payloads')
        helper = build['helper']
        if helper is not None:
            if type(helper) is not dict or set(helper) != {'contract', 'upstream', 'destination'}:
                raise ValueError('Invalid trusted payload build input')
            for name in helper.values(): relative(name)
            if not helper['destination'].startswith('_native/'):
                raise ValueError('Trusted payload must stage in native package namespace')
    return builds


def admit_source(repository, reference, build):
    """Compiler-specific source checks; runtime serving admission stays shared."""
    root = repository / 'loom-pipeline'
    value, content = document(source_path(root, reference))
    if (type(value) is not dict or set(value) != {'format', 'id', 'modality', 'native_contract', 'serving', 'assets', 'programs', 'preparation', 'provenance'}
            or type(value['format']) is not int or value['format'] != 1 or value['provenance'] is not None):
        raise ValueError('Expected authored executable declaration')
    programs = value['programs']
    if (type(programs) is not dict or set(programs) != {'root', 'encoding', 'prepare', 'control', 'sources', 'support', 'files'}
            or programs['encoding'] != 'source' or type(programs['files']) is not dict or not programs['files']):
        raise ValueError('Expected explicit source program closure')
    for key in ('prepare', 'control'):
        relative(programs[key])
    for key in ('sources', 'support'):
        if (type(programs[key]) is not list or (key == 'sources' and not programs[key])
                or not all(type(name) is str for name in programs[key])
                or len(programs[key]) != len(set(programs[key]))):
            raise ValueError('Invalid authored native role sequence')
        for name in programs[key]: relative(name)
    roles = {programs['prepare'], programs['control'], *programs['sources']}
    if roles & set(programs['support']) or not (roles | set(programs['support'])) <= programs['files'].keys():
        raise ValueError('Authored native roles are not closed over source files')
    source_root = root / relative(programs['root'])
    expected = set()
    for name, checksum in programs['files'].items():
        relative(name)
        if not name.endswith('.loom') or type(checksum) is not str or not re.fullmatch('[0-9a-f]{64}', checksum):
            raise ValueError('Invalid authored source identity')
        source = source_path(source_root, name)
        if digest(source.read_bytes()) != checksum:
            raise ValueError('Authored source hash mismatch: ' + name)
        expected.add(programs['root'] + '/' + name)
    actual = {name for group in build['groups'].values() for name in group['sources']}
    if actual != expected:
        raise ValueError('Build source closure differs from authored declaration')
    for name in build['notices']: source_path(root, name)
    if (value['preparation'] is None) != (build['helper'] is None):
        raise ValueError('Selected preparation/payload build mismatch')
    return value, digest(content)


def query_decoder(query):
    paths = query(DECODER_TARGET)
    if len(paths) != 1:
        raise ValueError('Expected exactly one pinned Loom decoder executable')
    return paths[0].resolve(strict=True)


def query_outputs(query, repository, builds, selected):
    all_names = {build['root'] + '/' + name for build in builds.values() for name in build['groups']}
    needed = {builds[ref]['root'] + '/' + name for ref in selected for name in builds[ref]['groups']}
    outputs, seen = {}, set()
    for path in query(CATALOG_TARGET):
        matches = [name for name in all_names if (path.as_posix().endswith('/loom-pipeline/bytecode/' + name) or path.as_posix() == 'loom-pipeline/bytecode/' + name)]
        if len(matches) != 1 or matches[0] in seen:
            raise ValueError('Unexpected or duplicate compiler output: ' + str(path))
        name = matches[0]
        seen.add(name)
        if name not in needed: continue
        output = path if path.is_absolute() else repository / path
        if not output.is_file() or not _is_bytecode(output):
            raise ValueError('Missing or invalid selected compiled output: ' + name)
        outputs[name] = output.resolve()
    if outputs.keys() != needed:
        raise ValueError('Missing selected compiler outputs: ' + repr(sorted(needed - outputs.keys())))
    return outputs


def _is_bytecode(path):
    with path.open('rb') as stream:
        return stream.read(4) == b'LOOM'


def program_roots(text: str) -> dict:
    """Compiler observations, independent of model names or runtime modality."""
    patterns = {
        "functions": r"^func\.def public @([^\s(]+)",
        "commands": r"^command\.program\.def public @([^\s(]+)",
        "configs": r"^config\.(?:def|decl) @([^\s=:(]+)",
    }
    return {kind: sorted(set(re.findall(pattern, text, re.MULTILINE)))
            for kind, pattern in patterns.items()}


def validate_compiled_program(artifact: Path, sources, root: Path, decoder: Path,
                              *, policy=None, exact_configs=True, policy_symbol='policy') -> dict:
    """Prove a linked closure and its policy rewrite before normalizing metadata.

    Source identities are explicit compiler inputs. A singleton and a multi-file
    native catalog use this same proof; no runtime identity selects Python code.
    """
    sources = tuple(sources)
    if not sources or len(sources) != len(set(sources)):
        raise ValueError("Missing or duplicate compiler source identity")
    content = []
    for source in sources:
        if (not isinstance(source, str) or posixpath.normpath(source) != source
                or source.startswith(("/", "../")) or "\\" in source
                or not source.endswith(".loom")):
            raise ValueError("Unsafe compiler source identity")
        path = root / source
        path.resolve(strict=True).relative_to(root.resolve(strict=True))
        content.append(path.read_text(encoding="utf-8"))
    if not artifact.is_file() or not _is_bytecode(artifact):
        raise ValueError("Missing or invalid compiled program")
    with tempfile.TemporaryDirectory() as temporary:
        decoded = Path(temporary) / "decoded.loom"
        subprocess.run([str(decoder), str(artifact), "--from=bc", "--to=text",
                        "--output=" + str(decoded)], check=True)
        text = decoded.read_text(encoding="utf-8")
    inputs = "\n".join(content)
    expected, actual = program_roots(inputs), program_roots(text)
    configs_match = (actual["configs"] == expected["configs"] if exact_configs
                     else set(expected["configs"]) <= set(actual["configs"]))
    source_count = len(re.findall(r"^func\.def ", inputs, re.MULTILINE))
    compiled_count = len(re.findall(r"^func\.def ", text, re.MULTILINE))
    if (actual["functions"] != expected["functions"]
            or actual["commands"] != expected["commands"] or not configs_match
            or source_count != compiled_count):
        raise ValueError("Compiled program public/private/config closure differs from source")
    if policy is not None:
        before, after = policy
        for locator in policy:
            if (not isinstance(locator, str) or not locator
                    or posixpath.normpath(locator) != locator
                    or locator.startswith(("/", "../")) or "\\" in locator):
                raise ValueError("Unsafe compiled policy locator")
        old = 'global.rodata.def @' + policy_symbol + ' = bytes("' + before.encode().hex() + '")'
        new = 'global.rodata.def @' + policy_symbol + ' = bytes("' + after.encode().hex() + '")'
        if inputs.count(old) != 1 or text.count(new) != 1 or old in text:
            raise ValueError("Compiled program does not preserve exact policy projection")
    return {"source_roots": expected, "compiled_roots": actual,
            "source_function_count": source_count, "compiled_function_count": compiled_count}



def preparation_bytecode_source(source, old_policy, policy, symbol='policy'):
    relative(old_policy)
    relative(policy)
    if not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_.]*', symbol):
        raise ValueError('Invalid policy symbol')
    old = 'global.rodata.def @' + symbol + ' = bytes("' + old_policy.encode().hex() + '")'
    new = 'global.rodata.def @' + symbol + ' = bytes("' + policy.encode().hex() + '")'
    declarations = re.findall(r'^global\.rodata\.def\s+@' + re.escape(symbol) + r'\s*=.*$', source, re.MULTILINE)
    if declarations != [old]:
        raise ValueError('Expected exactly one authored policy locator')
    return source.replace(old, new, 1)


def write_once(package_root, name, content, outputs):
    """Materialize logical package files once, permitting identical shared closure."""
    target = package_root / relative(name)
    target.resolve().relative_to(package_root.resolve())
    if target.is_symlink():
        raise ValueError('Package output cannot be an alias: ' + name)
    if target.exists():
        if not target.is_file() or target.read_bytes() != content:
            raise ValueError('Conflicting selected package output: ' + name)
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    outputs[name] = target
    return target


def normalize_catalog(package_root, repository, reference, build, source, source_identity, artifacts, decoder, outputs):
    """Lower physical locators only; authored executable semantics are never inferred."""
    root = repository / 'loom-pipeline'
    proofs, hashes = {}, {}
    for output, group in build['groups'].items():
        name = build['root'] + '/' + output
        artifact = artifacts[name]
        before = digest(artifact.read_bytes())
        policy = group['policy']
        proof = validate_compiled_program(artifact, group['sources'], root, decoder,
            policy=(policy['from'], policy['to']) if policy else None,
            exact_configs=group['exact_configs'], policy_symbol=policy.get('symbol', 'policy') if policy else 'policy')
        content = artifact.read_bytes()
        if digest(content) != before:
            raise ValueError('Compiled program changed during decode: ' + name)
        hashes[output] = before
        proofs[output] = dict(proof, sha256=before, policy=policy)
        write_once(package_root, 'loom-pipeline/' + name, content, outputs)
    current, identity = admit_source(repository, reference, build)
    if identity != source_identity or current != source:
        raise ValueError('Authored catalog changed during compilation: ' + reference)
    compiled = deepcopy(source)
    compiled['programs'] = dict(build['roles'], root=build['root'], encoding='bytecode', files=hashes)
    compiled['provenance'] = dict(source_declaration_sha256=source_identity,
                                  source_files=source['programs']['files'])
    encoded = (json.dumps(compiled, sort_keys=True, indent=2) + '\n').encode()
    write_once(package_root, 'loom-pipeline/' + reference, encoded, outputs)
    proof = dict(format=1, source_declaration_sha256=source_identity,
                 source_files=source['programs']['files'], programs=proofs)
    write_once(package_root, 'loom-pipeline/' + str(PurePosixPath(reference).parent / 'build.json'),
               (json.dumps(proof, sort_keys=True, indent=2) + '\n').encode(), outputs)
    for name in build['notices']:
        write_once(package_root, 'loom-pipeline/' + name, source_path(root, name).read_bytes(), outputs)
    return compiled


def stage_trusted_payload(package_root, repository, hrx_source, helper, operation_id, outputs):
    """Copy a selected trusted tool contract; never execute data-selected commands."""
    contract_path = source_path(repository, helper['contract'])
    operation, operation_bytes = document(contract_path)
    if (type(operation) is not dict or set(operation) != {'format', 'id', 'tool_revision', 'tool_files', 'inputs', 'outputs'}
            or type(operation['format']) is not int or operation['format'] != 1
            or operation['id'] != operation_id or type(operation['tool_files']) is not dict
            or not operation['tool_files'] or type(operation['tool_revision']) is not str
            or not re.fullmatch('[0-9a-f]{40}', operation['tool_revision'])):
        raise ValueError('Invalid selected trusted payload contract')
    destination = helper['destination']
    for name, checksum in operation['tool_files'].items():
        relative(name)
        if type(checksum) is not str or not re.fullmatch('[0-9a-f]{64}', checksum):
            raise ValueError('Invalid trusted tool identity')
        content = source_path(hrx_source, helper['upstream'] + '/' + name).read_bytes()
        if digest(content) != checksum:
            raise ValueError('Trusted tool differs from selected operation: ' + name)
        write_once(package_root, destination + '/' + name, content, outputs)
    if contract_path.read_bytes() != operation_bytes:
        raise ValueError('Trusted operation changed during staging')
    write_once(package_root, destination + '/operation.json', operation_bytes, outputs)
    write_once(package_root, destination + '/LICENSE', source_path(hrx_source, 'LICENSE').read_bytes(), outputs)
    provenance = dict(format=1, repository='https://github.com/ROCm/hrx-system',
                      revision=operation['tool_revision'], upstream_directory=helper['upstream'], files=operation['tool_files'])
    write_once(package_root, destination + '/provenance.json',
               (json.dumps(provenance, indent=2) + '\n').encode(), outputs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    validate = commands.add_parser('validate')
    validate.add_argument('--inventory', type=Path, required=True)
    validate.add_argument('--stamp', type=Path, required=True)
    prepare = commands.add_parser('prepare')
    prepare.add_argument('--source', type=Path, required=True)
    prepare.add_argument('--output', type=Path, required=True)
    prepare.add_argument('--from-policy', required=True)
    prepare.add_argument('--policy', required=True)
    prepare.add_argument('--policy-symbol', default='policy')
    args = parser.parse_args()
    if args.command == 'validate':
        builds = load_builds(args.inventory)
        for reference, build in builds.items():
            admit_source(args.inventory.resolve().parent.parent, reference, build)
        args.stamp.write_text('validated\n')
    else:
        args.output.write_text(preparation_bytecode_source(args.source.read_text(), args.from_policy, args.policy, args.policy_symbol))


if __name__ == '__main__':
    main()
