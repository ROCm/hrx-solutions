"""Independent release expectations and corruption oracles, without model inference."""
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import stat
import tarfile
import tempfile
import unittest
import zipfile
import py_compile
import os

from build_tools.validation import recipe_inventory as inventory


def encoded(value):
    return (json.dumps(value, indent=2) + '\n').encode()


def parse(data):
    # Tests of independent DATA oracles; real product parser composition is
    # separate and must not be inferred from this structural callback.
    return json.loads(data), hashlib.sha256(data).hexdigest()


def fixture():
    release = {'format': 1, 'recipes': {}, 'executables': {}, 'operations': {}}
    source, installed = {}, {}
    for modality, native in (('chat', 'text-v1'), ('asr', 'recognition-v1')):
        directory = 'unseen/fresh-' + modality
        reference = directory + '/executable.json'
        if modality == 'chat':
            assets = {'weights': {'format': 'gguf', 'path': None}, 'tokenizer': {'format': 'json', 'path': None}}
            serving = {'bounds': {name: [1, 99] for name in
                       ('rows', 'context_capacity', 'pool_capacity', 'prefill_capacity')},
                       'mtp_depth_max': 5, 'defaults': {'rows': 3, 'context_capacity': 30,
                       'pool_capacity': 60, 'prefill_capacity': 7, 'mtp': True, 'mtp_depth': 4}}
        else:
            assets = {'audio-parameters': {'format': 'safetensors', 'path': 'parameters.safetensors'}}
            serving = {'pending_requests': [1, 9], 'defaults': {'pending_requests': 2, 'options': {}}}
        text = b'func.def public @structural_fixture() {}\n'
        compiled = b'LOOM structural fixture, not native qualification ' + modality.encode()
        declaration = {'format': 1, 'id': 'novel-' + modality, 'modality': modality,
                       'native_contract': native, 'serving': serving, 'assets': assets,
                       'programs': {'root': directory, 'encoding': 'source', 'prepare': 'main.loom',
                           'control': 'main.loom', 'sources': ['main.loom'], 'support': [],
                           'files': {'main.loom': hashlib.sha256(text).hexdigest()}},
                       'preparation': None, 'provenance': None}
        raw = encoded(declaration)
        source[reference], source[directory + '/main.loom'] = raw, text
        expected = {'source': deepcopy(declaration), 'source_sha256': hashlib.sha256(raw).hexdigest(),
                    'compiled': {'root': directory, 'encoding': 'bytecode', 'prepare': 'main.loombc',
                                 'control': 'main.loombc', 'sources': ['main.loombc'],
                                 'support': [], 'files': ['main.loombc']}, 'notices': [directory + '/LICENSE']}
        release['executables'][reference] = expected
        roots = {'functions': ['structural_fixture'], 'commands': [], 'configs': []}
        expected['proofs'] = {'main.loombc': {
            'sources': [directory + '/main.loom'], 'source_roots': deepcopy(roots),
            'function_count': 1, 'policy': None, 'exact_configs': True}}
        normalized = deepcopy(declaration)
        normalized['programs'] = deepcopy(expected['compiled'])
        normalized['programs']['files'] = {'main.loombc': hashlib.sha256(compiled).hexdigest()}
        normalized['provenance'] = {'source_declaration_sha256': expected['source_sha256'],
                                    'source_files': deepcopy(declaration['programs']['files'])}
        installed[directory + '/build.json'] = encoded(dict(
            format=1, **normalized['provenance'], programs={'main.loombc': dict(
                source_roots=roots, compiled_roots=deepcopy(roots), source_function_count=1,
                compiled_function_count=1, policy=None,
                sha256=hashlib.sha256(compiled).hexdigest())}))
        installed[reference], installed[directory + '/main.loombc'] = encoded(normalized), compiled
        installed[directory + '/LICENSE'] = b'fixture notice\n'
        source[directory + '/LICENSE'] = b'fixture notice\n'
        for variant in range(2 if modality == 'chat' else 1):
            identifier = 'choice-' + modality + '-' + str(variant)
            files = {}
            for purpose, fact in assets.items():
                path = fact['path'] or (purpose + ('.gguf' if fact['format'] == 'gguf' else '.json'))
                files[purpose] = {'path': path, 'source': {'kind': 'huggingface', 'repo_id': 'unseen/model',
                    'revision': 'a' * 40, 'filename': 'variant-' + str(variant) + '/' + path}}
            recipe = {'format': 3, 'id': identifier, 'name': identifier, 'executable': reference,
                      'files': files, 'configuration': {}}
            relative = directory + '/recipes/' + identifier + '.json'
            release['recipes'][relative] = recipe
            installed[relative] = source[relative] = encoded(recipe)
    return release, source, installed


class IndependentReleaseTest(unittest.TestCase):
    def setUp(self):
        self.release, self.source, self.installed = fixture()

    def test_default_data_has_complete_consistent_current_expectations(self):
        release = inventory.load_release()
        self.assertEqual(set(release), {'format', 'recipes', 'executables', 'operations', 'source_files', 'qualification'})
        self.assertTrue(release['recipes'])
        self.assertTrue(release['executables'])
        self.assertTrue(release['operations'])
        for recipe in release['recipes'].values():
            self.assertIn(recipe['executable'], release['executables'])

    def test_fresh_chat_asr_and_second_recipe_need_no_checker_code_change(self):
        original = Path(inventory.__file__).read_bytes()
        inventory.validate_values(self.source.__getitem__, self.release['recipes'], parse, release=self.release)
        inventory.validate_executables(self.source.__getitem__, self.release['executables'], parse,
                                      release=self.release, encoding='source')
        inventory.validate_executables(self.installed.__getitem__, self.release['executables'], parse,
                                      release=self.release, encoding='bytecode')
        self.assertEqual(set(self.installed), inventory.catalog_paths(self.release))
        self.assertEqual(Path(inventory.__file__).read_bytes(), original)

    def test_recipe_inventory_and_pin_mutations_cannot_be_blessed_by_produced_set(self):
        paths = list(self.release['recipes'])
        for values in (paths[:-1], paths + [paths[0]], paths + ['unregistered/recipes/new.json']):
            with self.subTest(values=values), self.assertRaises(RuntimeError):
                inventory.validate_values(self.source.__getitem__, values, parse, release=self.release)
        changed = dict(self.source)
        recipe = deepcopy(self.release['recipes'][paths[0]])
        next(iter(recipe['files'].values()))['source']['revision'] = 'b' * 40
        changed[paths[0]] = encoded(recipe)
        with self.assertRaisesRegex(RuntimeError, 'coordinates'):
            inventory.validate_values(changed.__getitem__, paths, parse, release=self.release)

    def test_compiled_capability_identity_provenance_and_hash_mutations_fail(self):
        reference = next(iter(self.release['executables']))
        for mutate in ('serving', 'id', 'provenance', 'hash', 'root', 'role'):
            changed = dict(self.installed)
            declaration = json.loads(changed[reference])
            if mutate == 'serving':
                declaration['serving']['mtp_depth_max'] = 77
            elif mutate == 'id':
                declaration['id'] = 'swapped-id'
            elif mutate == 'provenance':
                declaration['provenance']['source_declaration_sha256'] = '0' * 64
            elif mutate == 'hash':
                declaration['programs']['files']['main.loombc'] = '0' * 64
            elif mutate == 'root':
                declaration['programs']['root'] = 'other'
            else:
                declaration['programs']['prepare'] = 'other.loombc'
            changed[reference] = encoded(declaration)
            with self.subTest(mutate=mutate), self.assertRaises(RuntimeError):
                inventory.validate_executables(changed.__getitem__, self.release['executables'], parse,
                                              release=self.release, encoding='bytecode')

    def test_receipt_provenance_roots_private_counts_policy_hash_and_inventory(self):
        documents = inventory.validate_executables(self.installed.__getitem__, self.release['executables'],
                                                  parse, release=self.release, encoding='bytecode')
        inventory.validate_build_proofs(self.installed.__getitem__, documents, release=self.release)
        inventory.validate_source_proofs(self.source.__getitem__, release=self.release)
        reference = next(iter(self.release['executables']))
        path = str(Path(reference).parent / 'build.json')
        for kind in ('source', 'functions', 'commands', 'configs', 'private', 'bool-count',
                     'policy', 'hash', 'missing', 'extra'):
            changed = dict(self.installed)
            receipt = json.loads(changed[path])
            program = receipt['programs']['main.loombc']
            if kind == 'source': receipt['source_files'] = {}
            elif kind in ('functions', 'commands', 'configs'): program['compiled_roots'][kind] = ['foreign']
            elif kind == 'private': program['compiled_function_count'] = 2
            elif kind == 'bool-count': program['compiled_function_count'] = True
            elif kind == 'policy': program['policy'] = {'source': 'foreign', 'from': 'old', 'to': 'new'}
            elif kind == 'hash': program['sha256'] = '0' * 64
            elif kind == 'missing': del receipt['programs']['main.loombc']
            else: receipt['programs']['extra.loombc'] = program
            changed[path] = encoded(receipt)
            with self.subTest(kind=kind), self.assertRaises(RuntimeError):
                inventory.validate_build_proofs(changed.__getitem__, documents, release=self.release)

    def test_source_archive_full_closure_and_adversarial_members(self):
        # A checked-in internal alias target is a source distribution input,
        # never a compiled/installed catalog member.
        backing = 'shared/control.loom'
        self.source[backing] = b'reviewed internal source alias backing\n'
        self.release['source_files'] = {backing: hashlib.sha256(self.source[backing]).hexdigest()}
        self.assertIn(backing, inventory.source_paths(self.release))
        self.assertNotIn(backing, inventory.catalog_paths(self.release))
        with tempfile.TemporaryDirectory() as temporary:
            repository = Path(temporary)
            for relative, raw in self.source.items():
                path = repository / 'loom-pipeline' / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(raw)
            for kind in (None, 'missing', 'extra', 'duplicate', 'nonregular', 'drift', 'backing-drift'):
                content = dict(self.source)
                first = next(path for path in content if path.endswith('.loom'))
                if kind == 'missing': del content[first]
                elif kind == 'extra': content['extra.loom'] = b'extra'
                elif kind == 'drift': content[first] += b'changed'
                elif kind == 'backing-drift': content[backing] += b'changed'
                stream = io.BytesIO()
                with tarfile.open(fileobj=stream, mode='w') as archive:
                    for relative, raw in content.items():
                        info = tarfile.TarInfo('source/loom-pipeline/' + relative)
                        info.size = len(raw)
                        if kind == 'nonregular' and relative == first:
                            info.type = tarfile.SYMTYPE; info.linkname = '/outside'
                        archive.addfile(info, io.BytesIO(raw))
                    if kind == 'duplicate':
                        info = tarfile.TarInfo('source/loom-pipeline/' + first)
                        info.size = len(content[first]); archive.addfile(info, io.BytesIO(content[first]))
                stream.seek(0)
                with tarfile.open(fileobj=stream) as archive:
                    if kind is None:
                        inventory.validate_sdist_catalogs(archive, repository, parse, parse, release=self.release)
                    else:
                        with self.subTest(kind=kind), self.assertRaises(RuntimeError):
                            inventory.validate_sdist_catalogs(archive, repository, parse, parse, release=self.release)

    def test_installed_inventory_regular_files_and_source_recipe_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            package = root / 'installed'
            source = root / 'source'
            for relative, raw in self.installed.items():
                path = package / 'loom-pipeline' / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(raw)
            for relative in self.release['recipes']:
                path = source / 'loom-pipeline' / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(self.source[relative])
            inventory.validate_installed_catalogs(package, parse, parse, source, release=self.release)
            extra = package / 'loom-pipeline/extra.loombc'
            extra.write_bytes(b'LOOM extra')
            with self.assertRaisesRegex(RuntimeError, 'inventory'):
                inventory.validate_installed_catalogs(package, parse, parse, source, release=self.release)
            extra.unlink()
            relative = next(iter(self.release['recipes']))
            target = package / 'loom-pipeline' / relative
            original = target.read_bytes()
            target.unlink()
            outside = root / 'outside.json'
            outside.write_bytes(original)
            target.symlink_to(outside)
            with self.assertRaises((RuntimeError, ValueError)):
                inventory.validate_installed_catalogs(package, parse, parse, source, release=self.release)

    def test_wheel_recipe_bytes_duplicate_nonregular_and_swapped_semantics(self):
        with tempfile.TemporaryDirectory() as temporary:
            repository = Path(temporary)
            for relative in self.source:
                path = repository / 'loom-pipeline' / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(self.source[relative])
            for mutation in (None, 'duplicate', 'symlink', 'swapped'):
                content = dict(self.installed)
                first = next(iter(self.release['recipes']))
                if mutation == 'swapped':
                    content[first] = self.source[list(self.release['recipes'])[1]]
                stream = io.BytesIO()
                with zipfile.ZipFile(stream, 'w') as archive:
                    for relative, data in content.items():
                        info = zipfile.ZipInfo('hrx_solutions/loom-pipeline/' + relative)
                        info.external_attr = ((stat.S_IFLNK if mutation == 'symlink' and relative == first else stat.S_IFREG) | 0o644) << 16
                        archive.writestr(info, data)
                    if mutation == 'duplicate':
                        archive.writestr('hrx_solutions/loom-pipeline/' + first, content[first])
                stream.seek(0)
                with zipfile.ZipFile(stream) as archive:
                    if mutation is None:
                        inventory.validate_wheel_catalogs(archive, repository, parse, parse, release=self.release)
                    else:
                        with self.subTest(mutation=mutation), self.assertRaises(RuntimeError):
                            inventory.validate_wheel_catalogs(archive, repository, parse, parse, release=self.release)

    def test_sdist_members_are_regular_exact_authored_recipe_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            repository = Path(temporary)
            stream = io.BytesIO()
            with tarfile.open(fileobj=stream, mode='w') as archive:
                for relative in self.release['recipes']:
                    raw = self.source[relative]
                    path = repository / 'loom-pipeline' / relative
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(raw)
                    info = tarfile.TarInfo('source/loom-pipeline/' + relative)
                    info.size = len(raw)
                    archive.addfile(info, io.BytesIO(raw))
            stream.seek(0)
            with tarfile.open(fileobj=stream) as archive:
                inventory.validate_sdist(archive, repository, parse, release=self.release)

    def test_expectation_file_rejects_duplicate_and_unsafe_members(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'release.json'
            path.write_text('{"format":1,"format":1,"recipes":{},"executables":{},"operations":{}}')
            with self.assertRaisesRegex(RuntimeError, 'Duplicate'):
                inventory.load_release(path)
            value = deepcopy(self.release)
            value['recipes']['../unsafe.json'] = next(iter(value['recipes'].values()))
            path.write_bytes(encoded(value))
            with self.assertRaisesRegex(RuntimeError, 'Unsafe'):
                inventory.load_release(path)

    def test_trusted_tool_contract_hash_provenance_license_and_complete_inventory(self):
        release = inventory.load_release()
        reference, expected = next(iter(release['operations'].items()))
        directory = reference.rsplit('/', 1)[0]
        # Fixed synthetic tool bytes exercise the independent oracle, never an
        # external preparation or model qualification claim.
        contents = {directory + '/' + name: b'reviewed fixture tool ' + name.encode()
                    for name in expected['declaration']['tool_files']}
        hashes = {name: hashlib.sha256(contents[directory + '/' + name]).hexdigest()
                  for name in expected['declaration']['tool_files']}
        expected['declaration']['tool_files'] = hashes
        expected['provenance']['files'] = deepcopy(hashes)
        contents[reference] = encoded(expected['declaration'])
        contents[directory + '/provenance.json'] = encoded(expected['provenance'])
        contents[directory + '/LICENSE'] = b'reviewed license'
        contents['_native/share/loom-serve/LICENSE'] = b'reviewed license'
        paths = sorted(inventory.operation_paths(release))
        inventory.validate_operations(contents.__getitem__, paths, release=release)
        for kind in ('contract', 'bool-format', 'tool', 'provenance', 'license'):
            changed = dict(contents)
            if kind in ('contract', 'bool-format'):
                doc = deepcopy(expected['declaration'])
                if kind == 'contract':
                    next(iter(doc['inputs'].values()))['revision'] = '0' * 40
                else:
                    doc['format'] = True
                changed[reference] = encoded(doc)
            elif kind == 'tool':
                changed[directory + '/' + next(iter(hashes))] = b'changed tool'
            elif kind == 'provenance':
                changed[directory + '/provenance.json'] = encoded({**expected['provenance'], 'revision': '0' * 40})
            else:
                changed[directory + '/LICENSE'] = b'changed license'
            with self.subTest(kind=kind), self.assertRaises(RuntimeError):
                inventory.validate_operations(changed.__getitem__, paths, release=release)
        for actual in (paths[:-1], paths + [paths[0]], paths + [directory + '/extra']):
            with self.assertRaisesRegex(RuntimeError, 'inventory'):
                inventory.validate_operations(contents.__getitem__, actual, release=release)

    def test_installed_helper_caches_are_derived_regular_and_exact_not_authored_payload(self):
        release = inventory.load_release()
        reference, expected = next(iter(release['operations'].items()))
        directory = reference.rsplit('/', 1)[0]
        with tempfile.TemporaryDirectory() as temporary:
            package = Path(temporary)
            hashes = {}
            for name in expected['declaration']['tool_files']:
                content = b'"""Independent helper cache fixture."""\nassert True\nx = 42\n' if name.endswith('.py') else b'fixture data'
                path = package / directory / name
                path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(content)
                hashes[name] = hashlib.sha256(content).hexdigest()
            expected['declaration']['tool_files'] = hashes
            expected['provenance']['files'] = deepcopy(hashes)
            (package / reference).write_bytes(encoded(expected['declaration']))
            (package / directory / 'provenance.json').write_bytes(encoded(expected['provenance']))
            (package / directory / 'LICENSE').write_bytes(b'license')
            (package / '_native/share/loom-serve/LICENSE').write_bytes(b'license')
            inventory.validate_installed_operations(package, release=release)
            source = next((package / directory).glob('*.py'))
            for mode in py_compile.PycInvalidationMode:
                for optimization in (0, 1, 2):
                    cache = Path(py_compile.compile(str(source), doraise=True,
                        optimize=optimization, invalidation_mode=mode))
                    inventory.validate_installed_operations(package, release=release)
            # Archives/authored-tool checks still reject generated caches;
            # only the installed namespace admits checked derived files.
            members = [p.relative_to(package).as_posix() for p in (package / directory).rglob('*') if p.is_file()]
            with self.assertRaisesRegex(RuntimeError, 'inventory'):
                inventory.validate_operations(lambda name: (package / name).read_bytes(), members, release=release)
            original = cache.read_bytes()
            for kind in ('header', 'flags', 'identity', 'code', 'trailing', 'unknown', 'symlink', 'aliased-directory', 'source'):
                extra = cache.with_name('unknown.cpython-314.pyc')
                source_bytes = source.read_bytes()
                source_stat = source.stat()
                cache_backup = package / 'cache-backup'
                try:
                    if kind == 'header': cache.write_bytes(b'wrong' + original[5:])
                    elif kind == 'flags': cache.write_bytes(original[:4] + b'\xff'*4 + original[8:])
                    elif kind == 'identity': cache.write_bytes(original[:8] + b'\0'*8 + original[16:])
                    elif kind == 'code': cache.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
                    elif kind == 'trailing': cache.write_bytes(original + b'trailing garbage')
                    elif kind == 'unknown': extra.write_bytes(original)
                    elif kind == 'source': source.write_bytes(source_bytes + b'changed')
                    elif kind == 'aliased-directory':
                        cache.parent.rename(cache_backup); cache.parent.symlink_to(cache_backup, target_is_directory=True)
                    else:
                        extra.write_bytes(original); cache.unlink(); cache.symlink_to(extra)
                    with self.subTest(kind=kind), self.assertRaises((RuntimeError, ValueError)):
                        inventory.validate_installed_operations(package, release=release)
                finally:
                    if cache.parent.is_symlink():
                        cache.parent.unlink(); cache_backup.rename(cache.parent)
                    if cache.is_symlink(): cache.unlink()
                    cache.write_bytes(original)
                    extra.unlink(missing_ok=True)
                    if kind == 'source':
                        source.write_bytes(source_bytes)
                        os.utime(source, ns=(source_stat.st_atime_ns, source_stat.st_mtime_ns))
                    inventory.validate_installed_operations(package, release=release)


if __name__ == '__main__':
    unittest.main()
