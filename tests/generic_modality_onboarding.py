"""Real generic runtime/checker onboarding in isolated source/installed fixtures.

The fixed harness copies DATA only after initial admission. Compiled bytes come
from a checked-in archive of accepted native build inputs; there is no fake compiler or decoder.
The producer case also executes the real uniform staging API and native decoder.
These fixtures never execute a native model or claim genuine wheel qualification.
"""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / 'tests/fixtures/generic_modalities'


def _python_identity(root):
    return {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for region in ('binding/python', 'build_tools')
            for path in (root / region).rglob('*.py')}


def _worker(package, release_path, mode, action, build_path=None):
    # -I removes inherited import paths; this is an ordinary unchanged package
    # copy under its real installed name, not a replacement reader/module table.
    from hrx_solutions.hem_common import Deadline, HemError
    from hrx_solutions.control.hem_state import Registry
    from hrx_solutions.models import pipelines, recipes, hem_models, hem_config
    from build_tools.validation import recipe_inventory as checker
    assert Path(checker.__file__).resolve().is_relative_to(package.parent / 'build_tools')
    assert Path(pipelines.__file__).resolve().is_relative_to(package)
    release = checker.load_release(release_path)
    catalog = package / 'loom-pipeline'
    read = lambda relative: (catalog / relative).read_bytes()
    budget = Deadline.after(30)
    if action == 'produce':
        repository = package.parent
        shutil.copytree(package / 'loom-pipeline', repository / 'loom-pipeline', dirs_exist_ok=True)
        shutil.copyfile(build_path, repository / 'loom-pipeline/catalog_sources.bzl')
        shutil.copytree(package, repository / 'binding/python', dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns('__pycache__', 'loom-pipeline'))
        from build_tools.model_tools import package_families, loom_catalog
        builds = loom_catalog.load_builds(repository / 'loom-pipeline/catalog_sources.bzl')
        artifacts = []
        with zipfile.ZipFile(os.environ['GENERIC_MODALITY_FIXTURE_ARCHIVE']) as archive:
            for case in json.loads((FIXTURES / 'cases.json').read_bytes()):
                if case['reference'] not in builds: continue
                declared = json.loads((FIXTURES / case['phase'] / 'installed' / case['reference']).read_bytes())
                for name, member in case['compiled_members'].items():
                    content = archive.read(member)
                    assert hashlib.sha256(content).hexdigest() == declared['programs']['files'][name]
                    target = repository / 'reused/loom-pipeline/bytecode' / builds[case['reference']]['root'] / name
                    target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(content); artifacts.append(target)
        decoder = Path(os.environ['GENERIC_MODALITY_FIXTURE_DECODER']).resolve(strict=True)
        def query(target):
            if target == loom_catalog.DECODER_TARGET: return [decoder]
            if target == loom_catalog.CATALOG_TARGET: return artifacts
            raise AssertionError('Unexpected producer query target: ' + target)
        prepared = package_families.prepare(query, repository)
        destination = repository / 'produced/hrx_solutions'
        shutil.copytree(package, destination, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns('__pycache__', 'loom-pipeline'))
        shutil.copytree(repository / 'build_tools', destination.parent / 'build_tools',
                        dirs_exist_ok=True, ignore=shutil.ignore_patterns('__pycache__'))
        paths = package_families.stage(destination, prepared, repository)
        assert len(paths) == len(set(paths))
        try:
            checker.validate_installed_catalogs(destination, recipes.parse, pipelines.parse, release=release)
        except RuntimeError as error:
            actual = {p.relative_to(destination / 'loom-pipeline').as_posix()
                      for p in (destination / 'loom-pipeline').rglob('*') if p.is_file()}
            expected = checker.catalog_paths(release)
            raise RuntimeError(f'{error}; extra={sorted(actual - expected)}; missing={sorted(expected - actual)}') from error
        return {'produced': str(destination), 'outputs': len(paths), 'decoder': str(decoder)}
    if action in ('check', 'exercise'):
        if mode == 'source':
            paths = [p.relative_to(catalog).as_posix() for p in catalog.glob('**/recipes/*.json')]
            refs = [p.relative_to(catalog).as_posix() for p in catalog.glob('**/executable.json')]
            checker.validate_values(read, paths, recipes.parse, release=release)
            checker.validate_executables(read, refs, pipelines.parse, release=release, encoding='source')
            checker.validate_source_proofs(read, release=release)
        else:
            checker.validate_installed_catalogs(package, recipes.parse, pipelines.parse, release=release)
    if action == 'check':
        return {'checked': mode}
    if action == 'read':
        for _, declaration, directory, _ in pipelines.list():
            pipelines.programs(declaration, directory, budget)
        recipes.list()
        return {'admitted': mode}
    if action == 'selected':
        try:
            pipelines.list()
        except HemError:
            pass
        else:
            raise AssertionError('Global discovery accepted malformed unrelated metadata')
        documents = (recipes.load('orbit-base-recipe'),)
    else:
        documents = recipes.list()
        assert {ref for ref, *_ in pipelines.list()} == set(release['executables'])
    registry = Registry(package.parent / 'state', deadline=budget)
    results = []
    for recipe, _ in documents:
        selected, directory, _ = pipelines.load(recipe['executable'])
        pipelines.programs(selected, directory, budget)
        prepared = package.parent / 'inputs' / recipe['id']; prepared.mkdir(parents=True, exist_ok=True)
        for item in recipe['files'].values():
            path = prepared / item['path']; path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'prepared bookkeeping fixture; no model loading\n')
        acquired = hem_models.import_assets(registry, recipe['id'], directory=prepared,
                                           recipe_id=recipe['id'], executable=recipe['executable'],
                                           modality=selected['modality'])
        record = registry.get_model(recipe['id'])
        paths, dependencies, evidence = hem_models.inspect_record(record, executable=recipe['executable'],
                                                                 modality=selected['modality'])
        if action == 'drift':
            manifest = catalog / recipe['executable']; original = manifest.read_bytes()
            changed = json.loads(original)
            changed['serving']['defaults']['rows'] = 3
            manifest.write_text(json.dumps(changed))
            try:
                for consume in (lambda: hem_models.inspect_record(record),
                                lambda: hem_config.resolve(record, paths, dependencies, evidence, deadline=budget)):
                    try: consume()
                    except HemError: pass
                    else: raise AssertionError('Stale serving association was reused')
            finally:
                manifest.write_bytes(original)
            return {'drift_rejected': mode}
        resolved = hem_config.resolve(record, paths, dependencies, evidence, deadline=budget)
        native = hem_config.deployment_document(resolved, recipe['id'], catalog=resolved.catalog, deadline=budget)
        output = package.parent / 'exports' / (recipe['id'] + '.json')
        output.parent.mkdir(exist_ok=True)
        exported = hem_config.export_deployment(record, paths, dependencies, evidence, output,
                                                recipe['id'], deadline=budget)
        assert exported['completed'] and output.is_file() and acquired['state'] == 'ready'
        assert native['definition']['kind'] == ('text' if selected['modality'] == 'chat' else 'recognition')
        results.append({'recipe': recipe['id'], 'executable': recipe['executable'],
                        'executable_id': selected['id'], 'modality': selected['modality']})
    return {'mode': mode, 'results': results}


class GenericModalityOnboardingTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.cases = json.loads((FIXTURES / 'cases.json').read_bytes())
        # Real native bytes are reused as inputs, never fabricated or executed.
        archive = os.environ.get('GENERIC_MODALITY_FIXTURE_ARCHIVE')
        self.assertTrue(archive, 'Run source_tests onboarding to provision the frozen genuine native fixture')
        self.archive = Path(archive).resolve(strict=True)
        source_root = os.environ.get('GENERIC_MODALITY_FIXTURE_SOURCE_ROOT')
        self.assertTrue(source_root, 'Set GENERIC_MODALITY_FIXTURE_SOURCE_ROOT to the immutable loom-pipeline source matching the frozen native fixture')
        self.source_root = Path(source_root).resolve(strict=True)
        self.code_before = _python_identity(ROOT)
        for relative in self.code_before:
            content = (ROOT / relative).read_bytes()
            for name in (b'comet-text-v1', b'rain-recognition-v1', b'astronomy/comet', b'weather/rain'):
                self.assertNotIn(name, content, relative)

    def _layout(self, mode, phase='initial'):
        home = self.root / (mode + '-' + phase)
        package = home / 'hrx_solutions'
        shutil.copytree(ROOT / 'binding/python', package,
                        ignore=shutil.ignore_patterns('__pycache__', '_native', 'loom-pipeline'))
        shutil.copytree(ROOT / 'build_tools', home / 'build_tools', ignore=shutil.ignore_patterns('__pycache__'))
        (package / 'loom-pipeline').mkdir()
        self._add(package, mode, 'initial')
        if phase == 'expanded': self._add(package, mode, 'addition')
        return package

    def _add(self, package, mode, phase):
        catalog = package / 'loom-pipeline'
        layout = 'source' if mode == 'source' else 'installed'
        shutil.copytree(FIXTURES / phase / layout, catalog, dirs_exist_ok=True)
        for case in self.cases:
            if case['phase'] != phase: continue
            declaration = json.loads((catalog / case['reference']).read_bytes())
            root = catalog / declaration['programs']['root']
            if mode == 'source':
                for name in case['source_files']:
                    target = root / name; target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(self.source_root / case['source_root'] / name, target)
            else:
                with zipfile.ZipFile(self.archive) as archive:
                    for name, member in case['compiled_members'].items():
                        content = archive.read(member)
                        self.assertEqual(hashlib.sha256(content).hexdigest(), declaration['programs']['files'][name])
                        target = root / name; target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(content)

    def _run(self, package, mode, action='exercise', *, release=None, reject=None, build=None):
        release = release or (FIXTURES / 'expanded-release.json')
        environment = {k: v for k, v in os.environ.items() if k not in ('PYTHONPATH', 'PYTHONHOME')}
        environment.update(LOOM_SERVE_DIR=str(package.parent / ('loom-' + action)),
                           HF_HUB_CACHE=str(package.parent / 'empty-hf-cache'), HF_HUB_OFFLINE='1')
        process = subprocess.run([sys.executable, '-I', str(Path(__file__).resolve()), '--worker',
                                  str(package), str(release), mode, action, str(build) if build else ''], cwd=package.parent,
                                  env=environment, capture_output=True, text=True, timeout=45)
        if reject is None:
            self.assertEqual(process.returncode, 0, process.stderr + process.stdout)
        else:
            self.assertNotEqual(process.returncode, 0)
            self.assertNotIn('Traceback', process.stderr)
            self.assertRegex(process.stdout, reject)
        return json.loads(process.stdout)

    def test_data_only_addition_real_runtime_checker_and_exact_python_identity(self):
        for mode in ('source', 'bytecode'):
            with self.subTest(mode=mode):
                package = self._layout(mode)
                before = self._run(package, mode, release=FIXTURES / 'initial-release.json')
                self.assertEqual(len(before['results']), 1)
                initial_python = {p.relative_to(package.parent).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in package.parent.rglob('*.py')}
                self._add(package, mode, 'addition')  # Addition stage writes DATA only.
                after = self._run(package, mode)
                self.assertEqual({r['recipe'] for r in after['results']},
                                 {'orbit-base-recipe', 'orbit-extra-recipe', 'comet-first', 'rain-first'})
                self.assertEqual({r['modality'] for r in after['results']}, {'chat', 'asr'})
                self.assertEqual(initial_python, {p.relative_to(package.parent).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                                                 for p in package.parent.rglob('*.py')})
                code_after = _python_identity(ROOT)
                self.assertEqual(self.code_before, code_after)
                digest = lambda values: hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()
                print(json.dumps({'layout': mode, 'python_files': len(code_after),
                                  'python_before': digest(self.code_before), 'python_after': digest(code_after),
                                  'isolated_python_before': digest(initial_python),
                                  'isolated_python_after': digest({p.relative_to(package.parent).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                                                                 for p in package.parent.rglob('*.py')}),
                                  'recipes': [r['recipe'] for r in after['results']],
                                  'scope': 'real runtime/checker structural fixture; producer case separate, no inference'}))

    def test_real_uniform_producer_decoder_and_independent_checker_data_only(self):
        self.assertTrue(os.environ.get('GENERIC_MODALITY_FIXTURE_DECODER'),
                        'Set the real accepted Loom decoder; no decoder/compiler double is allowed')
        package = self._layout('source')
        initial = self._run(package, 'source', action='produce', release=FIXTURES / 'initial-release.json',
                            build=FIXTURES / 'initial-build.bzl')
        self._run(Path(initial['produced']), 'bytecode', release=FIXTURES / 'initial-release.json')
        before = _python_identity(ROOT)
        copied_before = {p.relative_to(package.parent).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in package.parent.rglob('*.py')}
        self._add(package, 'source', 'addition')
        expanded = self._run(package, 'source', action='produce', build=FIXTURES / 'expanded-build.bzl')
        exercised = self._run(Path(expanded['produced']), 'bytecode')
        self.assertEqual(len(exercised['results']), 4)
        self.assertEqual(before, _python_identity(ROOT))
        copied_after = {p.relative_to(package.parent).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in package.parent.rglob('*.py')}
        self.assertEqual(copied_before, copied_after)
        digest = lambda values: hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()
        print(json.dumps({'producer': 'real uniform API and real native decoder with reused accepted bytes',
                          'python_before': digest(before), 'python_after': digest(_python_identity(ROOT)),
                          'isolated_before': digest(copied_before), 'isolated_after': digest(copied_after),
                          'produced_recipes': [item['recipe'] for item in exercised['results']],
                          'inference_claim': False}))

    def test_selected_unrelated_isolation_and_global_strictness(self):
        for mode in ('source', 'bytecode'):
            package = self._layout(mode, 'expanded')
            unrelated = package / 'loom-pipeline/unrelated/executable.json'; unrelated.parent.mkdir()
            unrelated.write_text('{malformed')
            result = self._run(package, mode, action='selected')
            self.assertEqual([item['recipe'] for item in result['results']], ['orbit-base-recipe'])

    def test_unknown_modality_abi_operation_and_source_closure_negatives(self):
        for field, value in (('modality', 'future'), ('native_contract', 'future-v1'),
                             ('preparation', {'operation':'arbitrary-module','inputs':{},'outputs':{}})):
            package = self._layout('source', 'expanded')
            manifest = package / 'loom-pipeline/astronomy/comet/executable.json'
            document = json.loads(manifest.read_bytes()); document[field] = value
            manifest.write_text(json.dumps(document))
            self._run(package, 'source', action='read', reject='rejected')
            shutil.rmtree(package.parent)
        for mutation in ('missing', 'escaped'):
            package = self._layout('source', 'expanded')
            manifest = package / 'loom-pipeline/astronomy/comet/executable.json'
            programs = json.loads(manifest.read_bytes())['programs']
            program = package / 'loom-pipeline' / programs['root'] / programs['control']
            content = program.read_bytes(); program.unlink()
            if mutation == 'escaped':
                outside = self.root / 'outside.loom'; outside.write_bytes(content); program.symlink_to(outside)
            self._run(package, 'source', action='read', reject='rejected')
            shutil.rmtree(package.parent)

    def test_compiled_missing_extra_hash_and_stale_provenance_negatives(self):
        for mutation in ('missing', 'extra', 'hash', 'provenance', 'serving'):
            package = self._layout('bytecode', 'expanded')
            root = package / 'loom-pipeline/astronomy/comet'
            manifest = root / 'executable.json'; document = json.loads(manifest.read_bytes())
            if mutation == 'missing': (root / 'pipeline.loombc').unlink()
            elif mutation == 'extra': (root / 'extra.loombc').write_bytes((root / 'pipeline.loombc').read_bytes())
            elif mutation == 'hash':
                with (root / 'pipeline.loombc').open('ab') as stream: stream.write(b'changed')
            elif mutation == 'provenance': document['provenance']['source_declaration_sha256'] = '0' * 64
            else: document['serving']['defaults']['rows'] = 3
            manifest.write_text(json.dumps(document))
            self._run(package, 'bytecode', action='check', reject='rejected')
            shutil.rmtree(package.parent)

    def test_receipt_observations_and_independent_source_roots_cannot_self_approve(self):
        for mutation in ('source_count', 'compiled_count', 'public_root', 'policy', 'source_identity'):
            package = self._layout('bytecode', 'expanded')
            receipt = package / 'loom-pipeline/astronomy/comet/build.json'
            document = json.loads(receipt.read_bytes()); observed = document['programs']['pipeline.loombc']
            if mutation == 'source_count': observed['source_function_count'] += 1
            elif mutation == 'compiled_count': observed['compiled_function_count'] += 1
            elif mutation == 'public_root': observed['compiled_roots']['functions'] = []
            elif mutation == 'policy': observed['policy'] = None
            else: document['source_declaration_sha256'] = '0' * 64
            receipt.write_text(json.dumps(document))
            self._run(package, 'bytecode', action='check', reject='Compiler receipt')
            shutil.rmtree(package.parent)
        package = self._layout('bytecode', 'expanded')
        release = json.loads((FIXTURES / 'expanded-release.json').read_bytes())
        expected = release['executables']['astronomy/comet/executable.json']['proofs']['pipeline.loombc']
        expected['source_roots']['functions'].append('invented_expected_public_root')
        path = self.root / 'forged-expected-roots.json'; path.write_text(json.dumps(release))
        self._run(package, 'bytecode', action='check', release=path, reject='Compiler receipt')

    def test_independent_expected_inventory_cannot_be_derived_from_actual_outputs(self):
        package = self._layout('bytecode', 'expanded')
        release = json.loads((FIXTURES / 'expanded-release.json').read_bytes())
        del release['recipes']['orbit/base/recipes/orbit-extra-recipe.json']
        path = self.root / 'wrong-expected-set.json'; path.write_text(json.dumps(release))
        self._run(package, 'bytecode', action='check', release=path, reject='inventory')

    def test_saved_serving_association_drift_rejects_both_layouts(self):
        for mode in ('source', 'bytecode'):
            package = self._layout(mode, 'expanded')
            self.assertTrue(self._run(package, mode, action='drift')['drift_rejected'])


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--worker':
        sys.path.insert(0, str(Path(sys.argv[2]).parent))
        from hrx_solutions.hem_common import HemError
        try:
            answer = _worker(Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4], sys.argv[5], Path(sys.argv[6]) if sys.argv[6] else None)
        except (HemError, RuntimeError, ValueError, OSError) as error:
            print(json.dumps({'rejected': type(error).__name__, 'message': str(error)}))
            raise SystemExit(1)
        else:
            print(json.dumps(answer))
    else:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(GenericModalityOnboardingTest)
        if suite.countTestCases() != 8:
            raise SystemExit("Expected all eight onboarding tests")
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        raise SystemExit(not result.wasSuccessful() or bool(result.skipped))
