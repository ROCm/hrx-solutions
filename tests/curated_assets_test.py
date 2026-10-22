"""Synthetic catalog/files exercise acquisition boundaries, never model validity."""
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
import io
import json
import os
import shutil
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from binding.python import curated_catalog as catalogs, hem_cli, hem_config, hem_models, hem_render, model_download
from binding.python.hem_common import Deadline, HemError, InvalidInput
from binding.python.hem_state import Registry
from binding.python.model_store import CatalogAssets
from binding.python.pipeline_validation import available_pipelines, resolve_catalog


class CuratedAssetsTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.catalog = self.root / 'package/loom-pipeline/recognition/fixture'
        self.catalog.mkdir(parents=True)
        self.module_file = self.root / 'package/binding/python/curated_catalog.py'
        self.module_file.parent.mkdir(parents=True)
        self.input = self.root / 'input'
        self.input.mkdir()
        self.state = self.root / 'state'
        self.registry = Registry(self.state)
        self.descriptor = dict(format=1, id='fixture', contract='fixture-assets-v1', description='Synthetic only',
            assets=[dict(role='checkpoint', path='weights.bin', required=True),
                    dict(role='checkpoint', path='extra/rotary.bin', required=True),
                    dict(role='metadata', path='optional.json', required=False)],
            definition=dict(kind='recognition', prepare='prepare.loom', control='control.loom',
                            sources=['provider.loom'], assets='.', pending_requests=4, options={}))
        self.write_catalog()
        for name in ('prepare.loom', 'control.loom', 'provider.loom'):
            (self.catalog / name).write_text('// Synthetic program path fixture only')
        (self.input / 'weights.bin').write_bytes(b'')  # Presence checks intentionally accept arbitrary bytes.
        (self.input / 'extra').mkdir()
        (self.input / 'extra/rotary.bin').write_bytes(b'not safetensors')
        for context in (patch.dict(catalogs.CATALOGS, {'fixture': 'recognition/fixture'}, clear=True),
                        patch.object(catalogs, '__file__', str(self.module_file)),
                        redirect_stderr(io.StringIO())):
            context.__enter__()
            self.addCleanup(context.__exit__, None, None, None)

    def write_catalog(self):
        (self.catalog / 'catalog.json').write_text(json.dumps(self.descriptor))

    def acquire(self, name='sample', **kwargs):
        return hem_models.import_assets(self.registry, name, directory=self.input, pipeline='fixture', **kwargs)

    def test_source_and_installed_catalog_lookup_and_inventory(self):
        self.assertIn('fixture', available_pipelines())
        self.assertEqual(resolve_catalog('fixture'), self.catalog)
        info = hem_config.pipeline_info('fixture')
        self.assertEqual(info['kind'], 'recognition')
        self.assertNotIn('compatibility', info)
        self.assertNotIn('mtp', info['defaults'])
        installed = self.root / 'installed/hrx_solutions'
        installed.mkdir(parents=True)
        (installed / 'loom-pipeline').symlink_to(self.root / 'package/loom-pipeline', target_is_directory=True)
        with patch.object(catalogs, '__file__', str(installed / 'curated_catalog.py')):
            self.assertEqual(resolve_catalog('fixture'), self.catalog)

    def test_copy_repeated_roles_optional_presence_and_retry(self):
        row = self.acquire()
        files = hem_models.model_files(self.registry, 'sample')
        self.assertIsInstance(files, CatalogAssets)
        self.assertEqual([file.role for file in files.files], ['checkpoint', 'checkpoint'])
        self.assertEqual(len(files.files), 2)
        self.assertEqual(row['provenance']['source'], {'kind': 'caller-prepared'})
        self.assertEqual(row['assets'], self.acquire()['assets'])
        (self.input / 'weights.bin').write_bytes(b'changed outside')
        self.assertEqual((files.directory / 'weights.bin').read_bytes(), b'')
        (self.input / 'optional.json').write_text('{}')
        with self.assertRaisesRegex(HemError, 'different mappings'):
            self.acquire()
        self.acquire('optional')
        self.assertEqual(len(hem_models.model_files(self.registry, 'optional').files), 3)

    def test_alias_and_passive_inventory_missing_changed(self):
        self.acquire(alias=True)
        before = self.registry.get_model('sample')
        info = hem_models.show(self.registry, 'sample')
        self.assertEqual(info['state'], 'ready')
        self.assertEqual(info['native_readiness'], 'unprobed')
        self.assertNotIn('weights', info)
        self.assertEqual(before, self.registry.get_model('sample'))
        (self.input / 'weights.bin').write_bytes(b'changed')
        self.assertEqual(hem_models.show(self.registry, 'sample')['state'], 'unvalidated')
        (self.input / 'weights.bin').unlink()
        self.assertEqual(hem_models.show(self.registry, 'sample')['state'], 'missing')
        with self.assertRaises(InvalidInput):
            hem_models.validate(self.registry, 'sample')
        self.assertTrue((Path(before['directory']) / 'weights.bin').is_symlink())

    def test_incomplete_retry_and_publish_before_check(self):
        (self.input / 'extra/rotary.bin').unlink()
        with self.assertRaisesRegex(HemError, 'incomplete acquisition retained'):
            self.acquire()
        self.assertEqual(hem_models.show(self.registry, 'sample')['state'], 'incomplete')
        (self.input / 'extra/rotary.bin').write_bytes(b'prepared')
        def check(files, pipeline=None):
            current = self.registry.get_model('sample')
            self.assertEqual(current['pending']['stage'], 'qualification')
            self.assertTrue(Path(current['directory']).is_dir())
            raise InvalidInput('synthetic post-publication failure')
        with patch.object(catalogs, 'check_assets', side_effect=check):
            with self.assertRaisesRegex(InvalidInput, 'post-publication'):
                self.acquire()
        self.assertEqual(self.registry.get_model('sample')['pending']['stage'], 'published-invalid')
        self.assertEqual(hem_models.show(self.registry, 'sample')['state'], 'invalid')
        self.acquire()
        self.assertEqual(hem_models.show(self.registry, 'sample')['state'], 'ready')

    def test_crash_after_rename_recovery_and_immutable_contract(self):
        update = self.registry.update_model
        def crash(model_id, changes, **kwargs):
            if changes.get('pending') == {'stage': 'qualification'}:
                raise RuntimeError('crash')
            return update(model_id, changes, **kwargs)
        with patch.object(self.registry, 'update_model', side_effect=crash):
            with self.assertRaisesRegex(RuntimeError, 'crash'):
                self.acquire()
        self.acquire()
        self.descriptor['contract'] = 'fixture-assets-v2'
        self.write_catalog()
        with self.assertRaisesRegex(HemError, 'different mappings'):
            self.acquire()
        with self.assertRaisesRegex(InvalidInput, 'contract changed'):
            hem_models.model_files(self.registry, 'sample')

    def test_observed_mutation_is_never_published_valid(self):
        original = catalogs.check_assets
        def mutate(files, pipeline=None):
            result = original(files, pipeline)
            (files.directory / 'weights.bin').write_bytes(b'changed while checking')
            return result
        with patch.object(catalogs, 'check_assets', side_effect=mutate):
            with self.assertRaisesRegex(InvalidInput, 'changed during qualification'):
                self.acquire()
        self.assertFalse(self.registry.get_model('sample')['validation']['valid'])

    def test_unknown_record_format_inspectable_and_cannot_start(self):
        self.acquire()
        # Synthetic future-version row; normal immutable API cannot mutate provenance.
        record = self.registry.get_model('sample')
        record.pop('revision')
        record['provenance']['format'] = 99
        with self.registry._transaction() as connection:
            connection.execute('UPDATE models SET data=? WHERE id=?', (json.dumps(record), 'sample'))
        info = hem_models.show(self.registry, 'sample')
        self.assertFalse(info['accessible'])
        self.assertIn('Unsupported', info['error'])
        with self.assertRaisesRegex(InvalidInput, 'Unsupported'):
            hem_models.model_files(self.registry, 'sample')

    def test_descriptor_paths_and_versions_fail_before_reservation(self):
        original = deepcopy(self.descriptor)
        variants = []
        for path in ('/absolute', '../escape', 'a//b', './a', 'a/../b'):
            variant = deepcopy(original)
            variant['assets'][0]['path'] = path
            variants.append(variant)
        variant = deepcopy(original); variant['format'] = 2; variants.append(variant)
        variant = deepcopy(original); variant['assets'].append(dict(variant['assets'][0])); variants.append(variant)
        variant = deepcopy(original); variant['definition']['prepare'] = '../escape.loom'; variants.append(variant)
        outside = self.root / 'outside.loom'; outside.write_text('x')
        (self.catalog / 'escape.loom').symlink_to(outside)
        variant = deepcopy(original); variant['definition']['prepare'] = 'escape.loom'; variants.append(variant)
        for variant in variants:
            self.descriptor = variant; self.write_catalog()
            with self.subTest(variant=variant), self.assertRaises(InvalidInput):
                self.acquire()
            self.assertEqual(self.registry.list_models(), [])
        (self.catalog / 'catalog.json').write_text('{"format":1,"format":1}')
        with self.assertRaises(InvalidInput):
            catalogs.load_catalog('fixture')

    def test_resolve_native_fixed_slots_export_and_dependency_set(self):
        self.acquire()
        files = hem_models.model_files(self.registry, 'sample')
        resolved = hem_config.resolve(files)
        self.assertEqual(set(resolved.dependencies), {asset.path for asset in files.files})
        self.assertEqual(resolved.sources, {'pending_requests': 'catalog', 'options': 'catalog'})
        document = hem_config.deployment_document(resolved, 'dictation', catalog=self.catalog, device=2)
        self.assertEqual(document['definition'], dict(self.descriptor['definition'], assets=str(files.directory)))
        self.assertEqual((document['name'], document['device'], document['admission']), ('dictation', 2, 'drained'))
        self.assertEqual(self.descriptor['definition']['assets'], '.')
        output = self.root / 'deployment.json'
        hem_config.export_deployment(files, output, 'raw')
        self.assertEqual(json.loads(output.read_text())['definition']['assets'], str(files.directory))
        for overrides, config in (({'rows':4}, None), ({'mtp':False}, None), ({}, self.root / 'empty.json')):
            absent = self.root / 'absent.json'
            with self.assertRaisesRegex(InvalidInput, 'unsupported'):
                hem_config.export_deployment(files, absent, 'raw', overrides=overrides, config_path=config)
            self.assertFalse(absent.exists())
        with self.assertRaisesRegex(InvalidInput, 'differs'):
            hem_config.resolve(files, pipeline='qwen')
        (self.catalog / 'provider.loom').unlink()
        with self.assertRaises(InvalidInput):
            hem_config.export_deployment(files, self.root / 'missing.json', 'raw')
        self.assertFalse((self.root / 'missing.json').exists())

    def test_cli_input_exclusions_and_named_rendering(self):
        with patch.dict(os.environ, {'LOOM_SERVE_DIR': str(self.state)}), redirect_stderr(io.StringIO()), redirect_stdout(io.StringIO()):
            self.assertEqual(hem_cli.main(['model','import','cli','--pipeline','fixture','--assets-dir',str(self.input),'--json']), 0)
        parser = hem_cli.parser()
        for arguments in (['--gguf','a.gguf','--assets-dir',str(self.input)], []):
            with self.assertRaises(InvalidInput):
                parser.parse_args(['model','import','bad',*arguments])
        args = parser.parse_args(['model','show','cli'])
        lines = str(hem_render.model(args, hem_models.show(self.registry,'cli'), 120))
        self.assertIn('file presence', lines)
        self.assertNotIn('structural validation passed', lines)
        self.assertNotIn('Structurally valid', lines)
        args = parser.parse_args(['pipeline','show','fixture'])
        lines = str(hem_render.pipeline(args, hem_config.pipeline_info('fixture'), 120))
        self.assertIn('rotary.bin', lines)
        self.assertNotIn('Tokenizer fingerprint', lines)

    def test_managed_lifecycle_reserves_named_files_and_retained_catalog(self):
        from binding.python import hem_instances, hem_process
        from tests.hem_instances_test import Processes
        from tests.hem_lifecycle_fixture import fixture, inventory, model, reply
        self.acquire()
        manager = hem_instances.InstanceManager(self.registry, processes=Processes(self.root / 'package'))
        rows, calls = [], []
        def respond(handler, requests):
            method, path, body, _ = requests[-1]
            calls.append((method, path))
            if method == 'GET':
                reply(handler, inventory(*rows)); return
            payload = json.loads(body)
            if path.endswith('/register'):
                reservation = self.registry.list_instances()[0]
                self.assertEqual(set(reservation['dependencies']), {
                    str(self.state / 'models/sample/weights.bin'),
                    str(self.state / 'models/sample/extra/rotary.bin'), str(retained)})
                self.assertEqual(payload['definition']['kind'], 'recognition')
                self.assertEqual(payload['base_directory'], str(retained))
                self.assertTrue((retained / payload['definition']['control']).is_file())
                rows.append(model(17, registration='registered'))
                rows[0]['loom']['kind'] = 'recognition'
                rows[0]['loom'].pop('parameters', None)
            elif path.endswith('/activate'):
                rows[0]['loom'].update(registration='ready', parameters={'primary': {'active': True}})
            elif path.endswith('/undrain'):
                rows[0]['loom']['status']['admission'] = 'open'
            elif path.endswith('/unregister'):
                rows.clear()
            else:
                raise AssertionError(path)
            reply(handler, {'accepted':True}, 202)
        with fixture(respond) as (endpoint, requests):
            self.registry.create_server('local', saved_settings={})
            server = self.registry.reserve_launch('local', launch_token='launch', current_settings={}, endpoint=endpoint)
            launch = self.state / 'servers/local/launch'
            shutil.copytree(self.root / 'package/loom-pipeline', launch / 'loom-pipeline')
            retained = launch / 'loom-pipeline/recognition/fixture'
            server = self.registry.update_server('local', {'phase':'running','pending':None,'build':{'directory':str(launch)}},
                expected_token='launch', expected_revision=server['revision'])
            with patch.object(hem_process, 'require_build', return_value=launch / 'loom-pipeline'):
                answer = manager.start(server,'sample',Deadline.after(),name='chat')
                self.assertEqual(answer['state'],'running')
                # A retained closure survives removal of the current package catalog.
                shutil.rmtree(self.catalog)
                self.assertTrue(manager.stop(server,'chat',Deadline.after())['completed'])
                self.assertEqual(self.registry.references('sample'), [])
                self.assertTrue((self.state / 'models/sample/weights.bin').is_file())
                self.assertTrue((retained / 'provider.loom').is_file())
        self.assertEqual([path for method,path in calls if method=='POST'],
                         ['/v1/models/register','/v1/models/activate','/v1/models/undrain','/v1/models/unregister'])

    def test_named_recipe_preview_only_has_no_fetch_or_state(self):
        self.descriptor['download_recipe'] = dict(id='fixture-recipe', description='Synthetic',
            sources=[dict(role='checkpoint',kind='huggingface',repo_id='example/model',revision='a'*40,filename='weights.bin')],
            generated=[dict(role='checkpoint',path='extra/rotary.bin')])
        self.write_catalog()
        with patch.object(model_download, '_run_hf') as network:
            preview = hem_models.download(None,'fixture-recipe',model_id='dry',dry_run=True)
            self.assertEqual(preview['generated'][0]['path'], 'extra/rotary.bin')
            self.assertEqual(preview['assets'][0]['role'], 'checkpoint')
            with self.assertRaisesRegex(InvalidInput, 'Unsupported named preparation'):
                hem_models.download(self.registry,'fixture-recipe',model_id='unsupported')
            network.assert_not_called()
        self.assertEqual(self.registry.list_models(), [])


if __name__ == '__main__':
    unittest.main()
