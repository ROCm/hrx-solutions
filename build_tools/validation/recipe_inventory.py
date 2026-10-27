"""Independent reviewed release DATA, never producer inputs or emitted inventory."""
import hashlib
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import stat
import re
import importlib.util
import marshal
import io
import types


def _unique(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise RuntimeError('Duplicate release/declaration key: ' + key)
        value[key] = item
    return value


def _json(data):
    return json.loads(data, object_pairs_hook=_unique,
                      parse_constant=lambda token: (_ for _ in ()).throw(RuntimeError('Nonfinite JSON')))


def _same(actual, expected):
    """JSON facts compare exact scalar kinds, including bool versus integer."""
    return json.dumps(actual, sort_keys=True, allow_nan=False) == json.dumps(expected, sort_keys=True, allow_nan=False)


def _relative(value):
    if (type(value) is not str or not value or '\\' in value or '\0' in value
            or PurePosixPath(value).is_absolute() or PureWindowsPath(value).drive
            or any(part in ('', '.', '..') for part in value.split('/'))):
        raise RuntimeError('Unsafe release path: ' + repr(value))
    return value


def load_release(path=None):
    """Literal DATA has its own authority; producers neither read nor generate it."""
    value = _json(Path(path or Path(__file__).with_name('release_catalogs.json')).read_bytes())
    required = {'format', 'recipes', 'executables', 'operations'}
    if (type(value) is not dict or not required <= set(value)
            or set(value) - required - {'source_files', 'qualification'}
            or type(value['format']) is not int or value['format'] != 1
            or any(type(value[key]) is not dict for key in ('recipes', 'executables', 'operations'))):
        raise RuntimeError('Unsupported independent release expectations')
    recipe_ids, executable_ids = set(), set()
    for path, expected in value['recipes'].items():
        _relative(path)
        if (expected['id'] in recipe_ids or expected['executable'] not in value['executables']
                or PurePosixPath(path).name != expected['id'] + '.json'):
            raise RuntimeError('Inconsistent independent recipe expectations')
        recipe_ids.add(expected['id'])
    for reference, expected in value['executables'].items():
        _relative(reference)
        if (not reference.endswith('/executable.json')
                or set(expected) != {'source', 'source_sha256', 'compiled', 'notices', 'proofs'}
                or expected['source']['id'] in executable_ids):
            raise RuntimeError('Inconsistent independent executable expectations')
        executable_ids.add(expected['source']['id'])
        for path in expected['notices']:
            _relative(path)
    for path, expected in value['operations'].items():
        _relative(path)
        if set(expected) != {'source', 'declaration', 'provenance'}:
            raise RuntimeError('Inconsistent independent operation expectations')
        _relative(expected['source'])
    for path, checksum in value.get('source_files', {}).items():
        _relative(path)
        if type(checksum) is not str or not re.fullmatch('[0-9a-f]{64}', checksum):
            raise RuntimeError('Invalid independent source-only file identity')
    for identifier in value.get('qualification', {}).values():
        if identifier not in recipe_ids:
            raise RuntimeError('Qualification fixture must name a reviewed recipe')
    return value


def _release(value):
    return load_release() if value is None else value


def validate_values(read, paths, parse, *, release=None):
    """Compare common-parser results with independently authored expected documents."""
    expected = _release(release)['recipes']
    paths = tuple(paths)
    if len(paths) != len(set(paths)) or set(paths) != set(expected):
        raise RuntimeError('Authored recipe inventory missing, duplicate or unregistered: ' + repr(paths))
    for path, wanted in expected.items():
        document, _ = parse(read(path))
        if not _same(document, wanted) or list(document['files']) != list(wanted['files']):
            raise RuntimeError('Authored recipe identity/coordinates differ from release contract: ' + path)


def validate_wheel(archive, repository, parse, *, release=None):
    prefix = 'hrx_solutions/loom-pipeline/'
    members = [info for info in archive.infolist()
               if info.filename.startswith(prefix) and '/recipes/' in info.filename and not info.is_dir()]
    paths = [info.filename[len(prefix):] for info in members]
    validate_values(lambda path: archive.read(prefix + path), paths, parse, release=release)
    for info in members:
        path = info.filename[len(prefix):]
        if (stat.S_IFMT(info.external_attr >> 16) not in (0, stat.S_IFREG)
                or archive.read(info) != (repository / 'loom-pipeline' / path).read_bytes()):
            raise RuntimeError('Wheel recipe is not the exact regular authored file: ' + path)


def validate_sdist(archive, repository, parse, *, release=None):
    members = [member for member in archive.getmembers() if '/loom-pipeline/' in member.name
               and '/recipes/' in member.name and not member.isdir()]
    paths = [member.name.split('/loom-pipeline/', 1)[1] for member in members]
    if any(not member.isfile() for member in members):
        raise RuntimeError('Source recipe must be a regular authored file')
    by_path = dict(zip(paths, members))
    read = lambda path: archive.extractfile(by_path[path]).read()
    validate_values(read, paths, parse, release=release)
    for path in paths:
        if read(path) != (repository / 'loom-pipeline' / path).read_bytes():
            raise RuntimeError('Source recipe differs from authored file: ' + path)


def validate_installed(package, parse, source_root=None, *, release=None):
    root = package / 'loom-pipeline'
    paths = [path.relative_to(root).as_posix() for path in root.glob('**/recipes/*') if not path.is_dir()]
    validate_values(lambda path: (root / path).read_bytes(), paths, parse, release=release)
    for relative in paths:
        path = root / relative
        _regular(path, root)
        if source_root is not None and path.read_bytes() != (source_root / 'loom-pipeline' / relative).read_bytes():
            raise RuntimeError('Installed recipe differs from authored file: ' + relative)


def _regular(path, root):
    path.resolve(strict=True).relative_to(root.resolve(strict=True))
    if not path.is_file() or path.is_symlink():
        raise RuntimeError('Installed catalog payload is not a regular file: ' + str(path))
    for parent in path.parents:
        if parent == root:
            break
        if parent.is_symlink():
            raise RuntimeError('Installed catalog payload has an aliased parent: ' + str(path))


def catalog_paths(release=None):
    """Expected installed catalog paths, from reviewed DATA, never staged outputs."""
    release = _release(release)
    paths = set(release['recipes']) | set(release['executables'])
    for reference, expected in release['executables'].items():
        paths.add(str(PurePosixPath(reference).parent / 'build.json'))
        programs = expected['compiled']
        paths.update(programs['root'] + '/' + name for name in programs['files'])
        paths.update(expected['notices'])
    return paths


def validate_executables(read, references, parse, *, release=None, encoding=None):
    release = _release(release)
    references = tuple(references)
    if len(references) != len(set(references)) or set(references) != set(release['executables']):
        raise RuntimeError('Executable inventory missing, duplicate or unregistered')
    documents = {}
    for reference, expected in release['executables'].items():
        raw = read(reference)
        document, _ = parse(raw)
        documents[reference] = document
        actual_encoding = document['programs']['encoding']
        if encoding is not None and actual_encoding != encoding:
            raise RuntimeError('Unexpected executable encoding: ' + reference)
        if actual_encoding == 'source':
            if (not _same(document, expected['source']) or
                    hashlib.sha256(raw).hexdigest() != expected['source_sha256']):
                raise RuntimeError('Source executable differs from reviewed declaration: ' + reference)
        elif actual_encoding == 'bytecode':
            if set(document) != set(expected['source']):
                raise RuntimeError('Compiled executable field inventory differs')
            for key in document:
                if key not in ('programs', 'provenance') and not _same(document[key], expected['source'][key]):
                    raise RuntimeError('Compiled executable semantic drift: ' + reference + ':' + key)
            wanted = expected['compiled']
            programs = document['programs']
            if set(programs) != set(wanted):
                raise RuntimeError('Compiled program contract fields differ')
            if any(programs[key] != wanted[key] for key in wanted if key != 'files'):
                raise RuntimeError('Compiled program roles/root differ from reviewed data')
            if set(programs['files']) != set(wanted['files']):
                raise RuntimeError('Compiled program file inventory differs')
            if document['provenance'] != {
                    'source_declaration_sha256': expected['source_sha256'],
                    'source_files': expected['source']['programs']['files']}:
                raise RuntimeError('Compiled executable source provenance differs')
        else:
            raise RuntimeError('Unknown executable encoding')
        for relative, checksum in document['programs']['files'].items():
            path = _relative(document['programs']['root']) + '/' + _relative(relative)
            content = read(path)
            if hashlib.sha256(content).hexdigest() != checksum:
                raise RuntimeError('Program hash mismatch: ' + path)
            if actual_encoding == 'bytecode' and (not path.endswith('.loombc') or not content.startswith(b'LOOM')):
                raise RuntimeError('Program is not Loom bytecode: ' + path)
    return documents


def validate_build_proofs(read, documents, *, release=None):
    """Compare compiler observations with reviewed source facts, never build DATA."""
    for reference, expected in _release(release)['executables'].items():
        proof = _json(read(str(PurePosixPath(reference).parent / 'build.json')))
        if (type(proof) is not dict or set(proof) != {
                'format', 'source_declaration_sha256', 'source_files', 'programs'}
                or type(proof['format']) is not int or proof['format'] != 1
                or proof['source_declaration_sha256'] != expected['source_sha256']
                or not _same(proof['source_files'], expected['source']['programs']['files'])
                or type(proof['programs']) is not dict
                or set(proof['programs']) != set(expected['proofs'])):
            raise RuntimeError('Compiler receipt identity/inventory differs: ' + reference)
        for name, wanted in expected['proofs'].items():
            observed = proof['programs'][name]
            if (type(observed) is not dict or set(observed) != {
                    'source_roots', 'compiled_roots', 'source_function_count',
                    'compiled_function_count', 'sha256', 'policy'}
                    or observed['sha256'] != documents[reference]['programs']['files'][name]
                    or not _same(observed['policy'], wanted['policy'])
                    or not _same(observed['source_roots'], wanted['source_roots'])
                    or any(type(observed[key]) is not int or observed[key] != wanted['function_count']
                           for key in ('source_function_count', 'compiled_function_count'))):
                raise RuntimeError('Compiler receipt source/hash/policy/count differs: ' + reference)
            actual = observed['compiled_roots']
            if type(actual) is not dict or set(actual) != {'functions', 'commands', 'configs'}:
                raise RuntimeError('Compiler receipt root vocabulary differs')
            for kind, values in actual.items():
                if (type(values) is not list or any(type(x) is not str for x in values)
                        or values != sorted(set(values))):
                    raise RuntimeError('Compiler receipt root inventory is malformed')
                source = wanted['source_roots'][kind]
                if ((kind != 'configs' or wanted['exact_configs']) and values != source
                        or kind == 'configs' and not set(source) <= set(values)):
                    raise RuntimeError('Compiler receipt public/command/config roots differ')


def source_paths(release=None):
    release = _release(release)
    paths = set(release['recipes']) | set(release['executables']) | set(release.get('source_files', {}))
    for expected in release['executables'].values():
        source = expected['source']['programs']
        paths.update(source['root'] + '/' + name for name in source['files'])
        paths.update(expected['notices'])
    return paths


def validate_source_proofs(read, *, release=None):
    """Independently check reviewed source-group membership and root observations."""
    patterns = {'functions': r'^func\.def public @([^\s(]+)',
                'commands': r'^command\.program\.def public @([^\s(]+)',
                'configs': r'^config\.(?:def|decl) @([^\s=:(]+)'}
    for expected in _release(release)['executables'].values():
        programs = expected['source']['programs']
        closure = {programs['root'] + '/' + path for path in programs['files']}
        seen = set()
        if set(expected['proofs']) != set(expected['compiled']['files']):
            raise RuntimeError('Reviewed compiler group inventory differs')
        for wanted in expected['proofs'].values():
            paths = wanted['sources']
            if not paths or len(paths) != len(set(paths)) or not set(paths) <= closure:
                raise RuntimeError('Reviewed source group is outside selected closure')
            seen.update(paths)
            text = '\n'.join(read(_relative(path)).decode('utf-8') for path in paths)
            roots = {key: sorted(set(re.findall(pattern, text, re.MULTILINE)))
                     for key, pattern in patterns.items()}
            count = len(re.findall(r'^func\.def ', text, re.MULTILINE))
            if not _same(roots, wanted['source_roots']) or not _same(count, wanted['function_count']):
                raise RuntimeError('Reviewed source roots/count differ from authored files')
            policy = wanted['policy']
            if policy is not None:
                if policy['source'] not in paths:
                    raise RuntimeError('Reviewed policy source is outside compiler group')
                old = 'global.rodata.def @policy = bytes("' + policy['from'].encode().hex() + '")'
                if read(policy['source']).decode('utf-8').count(old) != 1:
                    raise RuntimeError('Reviewed source policy observation differs')
        if seen != closure:
            raise RuntimeError('Reviewed compiler groups omit authored source closure')


def operation_paths(release=None):
    paths = set()
    for reference, expected in _release(release)['operations'].items():
        directory = reference.rsplit('/', 1)[0]
        paths.add(reference)
        paths.update(directory + '/' + name for name in
                     (*expected['declaration']['tool_files'], 'LICENSE', 'provenance.json'))
    return paths


def validate_operations(read, paths, *, release=None):
    release = _release(release)
    paths = tuple(paths)
    if len(paths) != len(set(paths)) or set(paths) != operation_paths(release):
        raise RuntimeError('Trusted operation payload inventory differs')
    for reference, expected in release['operations'].items():
        directory = reference.rsplit('/', 1)[0]
        if not _same(_json(read(reference)), expected['declaration']):
            raise RuntimeError('Trusted operation contract differs from independent expectations')
        for name, checksum in expected['declaration']['tool_files'].items():
            if hashlib.sha256(read(directory + '/' + name)).hexdigest() != checksum:
                raise RuntimeError('Trusted operation tool differs: ' + name)
        if not _same(_json(read(directory + '/provenance.json')), expected['provenance']):
            raise RuntimeError('Trusted operation provenance differs')
        if read(directory + '/LICENSE') != read('_native/share/loom-serve/LICENSE'):
            raise RuntimeError('Trusted operation license differs from selected native runtime')


def validate_installed_operations(package, *, release=None):
    """Admit authored tools exactly and verify this interpreter's derived caches."""
    release = _release(release)
    root = package / '_native/share/loom-serve/preparers'
    paths = tuple(root.rglob('*'))
    if any(path.is_symlink() for path in paths):
        raise RuntimeError('Installed helper payload/cache cannot contain aliases')
    members = {path.relative_to(package).as_posix() for path in paths if not path.is_dir()}
    for relative in members:
        _regular(package / relative, package)
    caches = {}
    authored = operation_paths(release)
    for relative in authored:
        if relative.endswith('.py'):
            for optimization in ('', '1', '2'):
                cache = importlib.util.cache_from_source(relative, optimization=optimization)
                caches[Path(cache).as_posix()] = (relative, int(optimization or '0'))
    generated = members & caches.keys()
    validate_operations(lambda relative: (package / relative).read_bytes(),
                        members - generated, release=release)
    for relative in generated:
        source_name, optimization = caches[relative]
        source = package / source_name
        content = source.read_bytes()
        cache = (package / relative).read_bytes()
        if len(cache) < 16 or cache[:4] != importlib.util.MAGIC_NUMBER:
            raise RuntimeError('Invalid installed helper cache header: ' + relative)
        flags = int.from_bytes(cache[4:8], 'little')
        if flags == 0:
            expected = ((int(source.stat().st_mtime) & 0xffffffff).to_bytes(4, 'little')
                        + (len(content) & 0xffffffff).to_bytes(4, 'little'))
        elif flags in (1, 3):
            expected = importlib.util.source_hash(content)
        else:
            raise RuntimeError('Invalid installed helper cache flags: ' + relative)
        code = compile(content, str(source), 'exec', dont_inherit=True, optimize=optimization)
        try:
            stream = io.BytesIO(cache[16:])
            observed = marshal.load(stream)
        except (EOFError, ValueError, TypeError) as error:
            raise RuntimeError('Invalid installed helper cache code: ' + relative) from error
        if (cache[8:16] != expected or type(observed) is not types.CodeType
                or observed.co_filename != str(source) or observed != code
                or stream.tell() != len(cache) - 16):
            raise RuntimeError('Installed helper cache differs from admitted source: ' + relative)


def validate_installed_catalogs(package, recipe_parse, executable_parse, source_root=None, *, release=None):
    release = _release(release)
    validate_installed(package, recipe_parse, source_root, release=release)
    root = package / 'loom-pipeline'
    actual = {path.relative_to(root).as_posix() for path in root.rglob('*') if not path.is_dir()}
    if actual != catalog_paths(release):
        raise RuntimeError('Installed catalog payload inventory differs from independent expectations')
    for relative in actual:
        _regular(root / relative, root)
    documents = validate_executables(lambda relative: (root / relative).read_bytes(),
        [path for path in actual if path.endswith('/executable.json')], executable_parse,
        release=release, encoding='bytecode')
    validate_build_proofs(lambda relative: (root / relative).read_bytes(), documents, release=release)
    for document in documents.values():
        programs = document['programs']
        directory = root / programs['root']
        actual_programs = {path.relative_to(directory).as_posix() for path in directory.rglob('*')
                           if path.suffix in ('.loom', '.loombc')}
        if actual_programs != set(programs['files']):
            raise RuntimeError('Installed compiled closure has missing or extra executables')
    validate_installed_operations(package, release=release)
    if source_root is not None:
        for reference, expected in release['operations'].items():
            source = (source_root / expected['source']).read_bytes()
            if not _same(_json(source), expected['declaration']) or source != (package / reference).read_bytes():
                raise RuntimeError('Installed operation differs from authored source contract')
    return documents


def validate_wheel_catalogs(archive, repository, recipe_parse, executable_parse, *, release=None):
    release = _release(release)
    validate_wheel(archive, repository, recipe_parse, release=release)
    prefix = 'hrx_solutions/'
    catalog_prefix = prefix + 'loom-pipeline/'
    members = [info for info in archive.infolist() if info.filename.startswith(catalog_prefix) and not info.is_dir()]
    paths = [info.filename[len(catalog_prefix):] for info in members]
    if len(paths) != len(set(paths)) or set(paths) != catalog_paths(release):
        raise RuntimeError('Wheel catalog payload inventory differs from independent expectations')
    for info in members:
        if stat.S_IFMT(info.external_attr >> 16) not in (0, stat.S_IFREG):
            raise RuntimeError('Wheel catalog member is not regular')
    for expected in release['executables'].values():
        for relative in expected['notices']:
            if archive.read(catalog_prefix + relative) != (repository / 'loom-pipeline' / relative).read_bytes():
                raise RuntimeError('Wheel notice differs from authored source: ' + relative)
    documents = validate_executables(lambda path: archive.read(catalog_prefix + path),
        [path for path in paths if path.endswith('/executable.json')], executable_parse,
        release=release, encoding='bytecode')
    validate_build_proofs(lambda path: archive.read(catalog_prefix + path), documents, release=release)
    operation_members = [info for info in archive.infolist()
        if info.filename.startswith(prefix + '_native/share/loom-serve/preparers/') and not info.is_dir()]
    if any(stat.S_IFMT(info.external_attr >> 16) not in (0, stat.S_IFREG) for info in operation_members):
        raise RuntimeError('Wheel trusted operation member is not regular')
    validate_operations(lambda path: archive.read(prefix + path),
                        [info.filename[len(prefix):] for info in operation_members], release=release)
    for reference, expected in release['operations'].items():
        if archive.read(prefix + reference) != (repository / expected['source']).read_bytes():
            raise RuntimeError('Wheel operation differs from authored source')
    return documents


def validate_sdist_catalogs(archive, repository, recipe_parse, executable_parse, *, release=None):
    """Exact authored catalog closure and selected operation contracts in an sdist."""
    release = _release(release)
    validate_sdist(archive, repository, recipe_parse, release=release)
    members = [member for member in archive.getmembers()
               if '/loom-pipeline/' in member.name and not member.isdir()]
    paths = [member.name.split('/loom-pipeline/', 1)[1] for member in members]
    # Build source metadata is source-only, with parity enforced separately below.
    metadata = {path.name for path in (repository / 'loom-pipeline').iterdir()
                if path.is_file() and (path.suffix == '.bzl' or path.name == 'BUILD.bazel')}
    if (len(paths) != len(set(paths)) or set(paths) != source_paths(release) | metadata
            or any(not member.isfile() for member in members)):
        raise RuntimeError('Source catalog closure missing, duplicate, extra or nonregular')
    by_path = dict(zip(paths, members))
    read = lambda path: archive.extractfile(by_path[path]).read()
    for path in paths:
        _relative(path)
        if read(path) != (repository / 'loom-pipeline' / path).read_bytes():
            raise RuntimeError('Source catalog differs from authored file: ' + path)
    validate_executables(read, release['executables'], executable_parse, release=release, encoding='source')
    validate_source_proofs(read, release=release)
    for path, checksum in release.get('source_files', {}).items():
        if hashlib.sha256(read(path)).hexdigest() != checksum:
            raise RuntimeError('Source-only backing file identity differs: ' + path)
    for expected in release['operations'].values():
        suffix = '/' + expected['source']
        contracts = [member for member in archive.getmembers() if member.name.endswith(suffix)]
        if len(contracts) != 1 or not contracts[0].isfile():
            raise RuntimeError('Source operation contract missing, duplicate or nonregular')
        content = archive.extractfile(contracts[0]).read()
        if (content != (repository / expected['source']).read_bytes()
                or not _same(_json(content), expected['declaration'])):
            raise RuntimeError('Source operation contract differs')
