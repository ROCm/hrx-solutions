"""Source composition proof using real generic callers and test-only preparation.

The ASR programs retain their real source closure. Tiny assets exercise setup,
not inference. Python is copied unchanged into an explicit source fixture; this
is deliberately separate from genuine installed-wheel qualification.
"""
from copy import deepcopy
import hashlib
import importlib.abc
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / 'tests/fixtures/generic_modalities'


def _identity(root):
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob('*.py')}


def _worker(home, action):
    class NoConverter(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if (('moonshine' in fullname and action != 'helper-tamper') or
                    fullname.split('.')[0] in ('torch', 'safetensors', 'huggingface_hub')):
                raise AssertionError('Concrete conversion dependency imported: ' + fullname)
    sys.meta_path.insert(0, NoConverter())
    sys.path.insert(0, str(home))
    from hrx_solutions.hem_common import Deadline, HemError
    from hrx_solutions.control.hem_state import Registry
    from hrx_solutions.models import hem_models, hem_config, recipes, pipelines, preparation, model_download
    package = home / 'hrx_solutions'
    assert Path(hem_models.__file__).resolve().is_relative_to(package)
    if action == 'helper-tamper':
        # Run the real concrete integrity check through generic acquisition.
        # No helper or optional conversion dependency gets to execute.
        catalog = package / 'loom-pipeline'
        shutil.rmtree(catalog)
        shutil.copytree(ROOT / 'loom-pipeline/recognition', catalog / 'recognition')
        tool = package / '_native/share/loom-serve/preparers/moonshine'
        tool.mkdir(parents=True)
        shutil.copyfile(package / 'models/preparation/moonshine-operation.json', tool / 'operation.json')
        (tool / 'prepare_assets.py').write_bytes(b'corrupted reviewed helper')
        (tool / 'model.json').write_bytes(b'{}')
        registry = Registry(home / 'state', deadline=Deadline.after(30))
        try:
            hem_models.download(registry, 'moonshine-streaming-tiny', model_id='downloaded', offline=True)
        except HemError as error:
            assert 'tool integrity differs' in str(error), str(error)
            assert not (registry.root / 'models/downloaded').exists()
            assert registry.get_model('downloaded')['pending']['stage'] == 'staging'
            assert not (home / 'hf').exists(), 'Tool rejection happened after acquisition'
            assert not list((registry.root / 'models').glob('**/.loom-acquisition.json'))
            return {'rejected': action, 'error': str(error), 'unpublished': True}
        raise AssertionError('Tampered real helper was accepted')
    manifest = package / 'loom-pipeline/weather/rain/executable.json'
    recipe_path = manifest.parent / 'recipes/rain-first.json'
    selected = json.loads(manifest.read_bytes())
    recipe = json.loads(recipe_path.read_bytes())
    events = []
    contract = None
    if action != 'no-preparer':
        # This deliberately uses a different input inventory and algorithm. Only
        # the preparation boundary's test seam changes; generic callers do not.
        contract = {'format': 1, 'id': 'test-only-byte-transform',
                    'tool_revision': '2' * 40, 'tool_files': {},
                    'inputs': {'model.safetensors': deepcopy(recipe['files']['weights']['source'])},
                    'outputs': {'rotary.safetensors': {'purpose': 'rotary', 'format': 'safetensors'}}}
        selected['preparation'] = {'operation': contract['id'],
                                   'inputs': {'model.safetensors': 'weights'},
                                   'outputs': {'rotary.safetensors': 'rotary'}}
        recipe['files']['rotary']['source'] = {'kind': 'generated'}

        class ByteTransform:
            @staticmethod
            def preflight(deadline):
                deadline.remaining()
                events.append('preflight')

            @staticmethod
            def prepare(inputs, attempt, deadline):
                deadline.remaining()
                events.append('prepare')
                assert set(inputs) == {'model.safetensors'}
                if action == 'failed-preparation':
                    raise HemError('test transformation failed')
                result = attempt / 'output/rotary.safetensors'
                result.parent.mkdir()
                result.write_bytes(b'test-transform:' + inputs['model.safetensors'].read_bytes()[::-1])
                return {'rotary.safetensors': result}

        preparation._IMPLEMENTATIONS[contract['id']] = (lambda: deepcopy(contract), lambda: ByteTransform)
        if action == 'unsupported': selected['preparation']['operation'] = 'not-trusted'
        if action == 'bad-binding': selected['preparation']['inputs']['model.safetensors'] = 'rotary'
        if action == 'bad-pin': recipe['files']['weights']['source']['revision'] = '3' * 40
        manifest.write_text(json.dumps(selected))
        recipe_path.write_text(json.dumps(recipe))

    before = _identity(home)
    registry = Registry(home / 'state', deadline=Deadline.after(30))
    publication = registry.root / 'models' / 'downloaded'
    if action in ('unsupported', 'bad-binding', 'bad-pin'):
        try:
            hem_models.download(registry, recipe['id'], model_id='downloaded', offline=True)
        except HemError as error:
            assert not publication.exists() and events == []
            assert registry.list_models() == []
            return {'rejected': action, 'error': str(error), 'events': events}
        raise AssertionError('Invalid admission reached acquisition')

    discovered = hem_models.available()
    assert len(discovered) == 1 and discovered[0]['modality'] == 'asr'
    assert discovered[0]['executable_id'] == 'rain-recognition-v1'
    assert len(pipelines.list('asr')) == 1 and len(recipes.list()) == 1
    for purpose, item in recipe['files'].items():
        if item['source']['kind'] == 'huggingface':
            cached = model_download._cache_path(item['source'])
            cached.parent.mkdir(parents=True, exist_ok=True)
            cached.write_bytes(('fixture-' + purpose).encode())
    if action == 'failed-preparation':
        try:
            hem_models.download(registry, recipe['id'], model_id='downloaded', offline=True)
        except HemError as error:
            assert str(error) == 'test transformation failed'
            assert not publication.exists() and events == ['preflight', 'prepare']
            record = registry.get_model('downloaded')
            assert record['pending']['stage'] == 'staging'
            assert not list((registry.root / 'models').glob('**/.loom-acquisition.json'))
            return {'rejected': action, 'events': events, 'unpublished': True}
        raise AssertionError('Failed preparation published a model')

    downloaded = hem_models.download(registry, recipe['id'], model_id='downloaded', offline=True)
    assert downloaded['state'] == 'ready'
    if contract:
        assert events == ['preflight', 'prepare']
        assert (publication / 'rotary.safetensors').read_bytes() == b'test-transform:sthgiew-erutxif'
        assert (publication / '.loom-acquisition.json').is_file()
        assert registry.get_model('downloaded')['provenance']['preparation']['id'] == contract['id']
    else:
        assert events == []
        assert registry.get_model('downloaded')['provenance']['preparation'] is None
    records = [registry.get_model('downloaded')]
    if not contract:
        prepared = home / 'ready-assets'
        prepared.mkdir()
        for item in recipe['files'].values():
            (prepared / item['path']).write_bytes(b'ready-to-use structural fixture')
        imported = hem_models.import_assets(registry, 'imported', directory=prepared,
                                            recipe_id=recipe['id'], modality='asr')
        assert imported['state'] == 'ready'
        records.append(registry.get_model('imported'))
    for record in records:
        paths, dependencies, evidence = hem_models.inspect_record(record)
        resolved = hem_config.resolve(record, paths, dependencies, evidence, deadline=registry.deadline)
        document = hem_config.deployment_document(resolved, record['id'],
                                                  catalog=resolved.catalog, deadline=registry.deadline)
        assert document['definition']['kind'] == 'recognition'
        assert document['definition']['pending_requests'] == 4
        destination = home / (record['id'] + '.json')
        exported = hem_config.export_deployment(record, paths, dependencies, evidence,
                                                destination, record['id'], deadline=registry.deadline)
        assert exported['completed'] and json.loads(destination.read_bytes()) == document
    assert before == _identity(home), 'Onboarding changed Python product/checker code'
    assert not any('moonshine' in name for name in sys.modules)
    return {'action': action, 'events': events, 'models': [r['id'] for r in records],
            'python_files': len(before), 'unchanged_python': True, 'scope': 'source setup; no inference'}


class GenericAsrPreparationTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.home = Path(temporary.name)
        package = self.home / 'hrx_solutions'
        shutil.copytree(ROOT / 'binding/python', package,
                        ignore=shutil.ignore_patterns('__pycache__', '_native', 'loom-pipeline'))
        self.python_before = _identity(ROOT / 'binding/python')
        self.checker_before = _identity(ROOT / 'build_tools')
        catalog = package / 'loom-pipeline'
        shutil.copytree(FIXTURES / 'addition/source/weather', catalog / 'weather')
        case = next(c for c in json.loads((FIXTURES / 'cases.json').read_bytes())
                    if c['reference'] == 'weather/rain/executable.json')
        for name in case['source_files']:
            shutil.copyfile(ROOT / 'loom-pipeline' / case['source_root'] / name,
                            catalog / 'weather/rain' / name)

    def _run(self, action):
        environment = {k: v for k, v in os.environ.items() if k not in ('PYTHONPATH', 'PYTHONHOME')}
        environment.update(LOOM_SERVE_DIR=str(self.home / 'loom'), HF_HUB_OFFLINE='1',
                           HF_HUB_CACHE=str(self.home / 'hf'), PYTHONNOUSERSITE='1')
        result = subprocess.run([sys.executable, '-I', str(Path(__file__).resolve()),
                                 '--worker', str(self.home), action], cwd=self.home,
                                env=environment, text=True, capture_output=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.python_before, _identity(ROOT / 'binding/python'))
        self.assertEqual(self.checker_before, _identity(ROOT / 'build_tools'))
        return json.loads(result.stdout)

    def test_no_preparer_asr_discovery_download_import_configuration_export_without_converter(self):
        result = self._run('no-preparer')
        self.assertEqual(result['models'], ['downloaded', 'imported'])
        self.assertEqual(result['events'], [])

    def test_distinct_preparation_through_unchanged_acquisition_and_configuration(self):
        result = self._run('prepared')
        self.assertEqual(result['events'], ['preflight', 'prepare'])
        self.assertTrue(result['unchanged_python'])

    def test_bad_bindings_unknown_operation_and_bad_pins_reject_before_effects(self):
        for action in ('unsupported', 'bad-binding', 'bad-pin'):
            with self.subTest(action=action):
                result = self._run(action)
                self.assertEqual(result['rejected'], action)
                self.assertEqual(result['events'], [])

    def test_preparation_failure_does_not_publish_assets_or_receipt(self):
        self.assertTrue(self._run('failed-preparation')['unpublished'])

    @unittest.skipIf(sys.version_info < (3, 12), 'Concrete Moonshine preparation requires Python >=3.12')
    def test_real_moonshine_helper_tamper_rejected_by_generic_acquisition_before_fetch(self):
        self.assertTrue(self._run('helper-tamper')['unpublished'])


if __name__ == '__main__':
    if len(sys.argv) == 4 and sys.argv[1] == '--worker':
        print(json.dumps(_worker(Path(sys.argv[2]), sys.argv[3])))
    else:
        unittest.main()
