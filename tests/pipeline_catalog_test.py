"""Source/bytecode role resolution and immutable deployment export contracts."""
from contextlib import redirect_stderr
import io
import hashlib
import json
import shutil
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from binding.python.models import hem_config, pipeline_validation
from binding.python.models.model_store import ModelFiles
from binding.python.hem_common import Deadline, DeadlineExpired, InvalidInput
from binding.python import hem_process


class PipelineCatalogTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.catalog = self.root / 'loom-pipeline/llm/qwen/qwen_08b'
        self.catalog.mkdir(parents=True)
        self.weights, self.tokenizer = self.root / 'weights.gguf', self.root / 'tokenizer.json'
        self.weights.write_bytes(b'fixture')
        self.tokenizer.write_text('{}')
        location = patch.object(pipeline_validation, '__file__', str(self.root / 'binding/python/models/pipeline_validation.py'))
        location.start()
        self.addCleanup(location.stop)

    def populate(self, suffix):
        for old in self.catalog.iterdir():old.unlink()
        (self.catalog / 'compatibility.json').write_text(json.dumps({'profile_version':1,'identifier':'fixture-variant','mtp_layer':None}))
        if suffix == '.loombc':
            content = b'LOOMfixture'
            (self.catalog / 'pipeline.loombc').write_bytes(content)
            (self.catalog / 'compatibility.json').write_text(json.dumps({'profile_version':1,'identifier':'fixture-variant','mtp_layer':None}))
            (self.catalog / 'pipeline.json').write_text(json.dumps({'format':1,'variant':'fixture-variant',
                'module':'pipeline.loombc','sha256':hashlib.sha256(content).hexdigest()}))
            return ['pipeline.loombc']
        entries = [name+suffix for name in ('config', 'prepare', 'control', 'weights', 'provider')]
        for entry in entries:
            (self.catalog / entry).write_bytes(b'fixture source')
        (self.catalog / 'sources.txt').write_text('\n'.join(entries)+'\n')
        return entries

    def resolved(self, overrides=None):
        # Model qualification is a separate fixture seam; catalog/profile/module
        # admission and the checked producer remain the production implementation.
        with patch.object(hem_config, 'select_pipeline', return_value='qwen-08b'):
            return hem_config.resolve(ModelFiles(self.weights,self.tokenizer), overrides, pipeline='qwen-08b')

    def test_checked_identity_covers_full_authoritative_source_and_installed_roots(self):
        self.populate('.loom')
        root = self.root / 'loom-pipeline'
        extra = root / 'recognition/other/policy.loom'
        extra.parent.mkdir(parents=True)
        extra.write_text('outside selected Qwen leaf')
        first = self.resolved()
        self.assertEqual(first.catalog_identity.relative_path, Path('llm/qwen/qwen_08b'))
        self.assertEqual(first.catalog_identity.contract, 'fixture-variant')
        self.assertEqual(first.catalog_identity.tree_sha256, hem_process.tree_identity(root, Deadline.after()))
        extra.write_text('changed outside selected Qwen leaf')
        second = self.resolved()
        self.assertNotEqual(first.catalog_identity, second.catalog_identity)
        installed = self.root / 'installed'
        shutil.copytree(root, installed / 'loom-pipeline')
        with patch.object(pipeline_validation, '__file__', str(installed / 'models/pipeline_validation.py')):
            copied = self.resolved()
        self.assertEqual(copied.catalog, installed / 'loom-pipeline/llm/qwen/qwen_08b')
        self.assertEqual(copied.catalog_identity, second.catalog_identity)

    def test_catalog_identity_brackets_profile_and_definition_reads(self):
        for stage, suffix in (('profile', '.loom'), ('definition', '.loom'), ('definition', '.loombc')):
            with self.subTest(stage=stage, suffix=suffix):
                self.populate(suffix)
                outside = self.root / 'loom-pipeline/recognition/policy.loom'
                outside.parent.mkdir(parents=True, exist_ok=True)
                outside.write_text('original policy')
                operation = 'pipeline_info' if stage == 'profile' else 'catalog_modules'
                original = getattr(hem_config, operation)
                def drift(*args, **kwargs):
                    answer = original(*args, **kwargs)
                    target = outside if stage == 'profile' else self.catalog / (
                        'pipeline.json' if suffix == '.loombc' else 'provider.loom')
                    target.write_text(target.read_text() + ' ')
                    return answer
                with patch.object(hem_config, operation, side_effect=drift):
                    with self.assertRaisesRegex(InvalidInput, 'changed during serving resolution'):
                        self.resolved()

    def test_checked_value_refuses_catalog_without_authoritative_root(self):
        self.populate('.loom')
        outside = self.root / 'untrusted/loom-pipeline/llm/qwen/qwen_08b'
        shutil.copytree(self.catalog, outside)
        original = hem_config.pipeline_info
        def foreign(pipeline):
            answer = original(pipeline)
            return dict(answer, catalog=str(outside))
        with patch.object(hem_config, 'pipeline_info', side_effect=foreign):
            with self.assertRaisesRegex(InvalidInput, 'outside or missing'):
                self.resolved()

    def test_root_identity_and_definition_share_the_original_deadline(self):
        self.populate('.loom')
        budget = Deadline.after()
        original = hem_process.tree_identity
        budgets = []
        def observe(directory, deadline):
            budgets.append(deadline)
            return original(directory, deadline)
        with patch.object(hem_config, 'select_pipeline', return_value='qwen-08b'), \
             patch.object(hem_process, 'tree_identity', side_effect=observe):
            hem_config.resolve(ModelFiles(self.weights, self.tokenizer), pipeline='qwen-08b', deadline=budget)
        self.assertEqual(len(budgets), 2)
        self.assertTrue(all(item is budget for item in budgets))
        clock = [0.0]
        budget = Deadline(1.0, clock=lambda: clock[0])
        original_info = hem_config.pipeline_info
        def expire(pipeline):
            answer = original_info(pipeline)
            clock[0] = 2.0
            return answer
        with patch.object(hem_config, 'select_pipeline', return_value='qwen-08b'), \
             patch.object(hem_config, 'pipeline_info', side_effect=expire):
            with self.assertRaises(DeadlineExpired):
                hem_config.resolve(ModelFiles(self.weights, self.tokenizer), pipeline='qwen-08b', deadline=budget)

    def test_emission_only_rebases_checked_definition_without_reselecting(self):
        self.populate('.loombc')
        checked = self.resolved()
        retained_root = self.root / 'retained/loom-pipeline'
        shutil.copytree(self.root / 'loom-pipeline', retained_root)
        retained = retained_root / checked.catalog_identity.relative_path
        with patch.object(hem_config, 'catalog_modules', side_effect=AssertionError('selected modules again')), \
             patch.object(hem_config, 'pipeline_info', side_effect=AssertionError('selected defaults again')), \
             patch.object(hem_config.curated_catalog, 'load_catalog', side_effect=AssertionError('selected catalog again')):
            emitted = hem_config.deployment_document(checked, 'alternate-name', catalog=retained, device=2)
        self.assertEqual(emitted, dict(name='alternate-name', device=2, admission='drained',
                                       base_directory=str(retained), definition=checked.definition_document()))
        emitted['definition']['sources'].append('caller.loom')
        self.assertEqual(checked.definition_document()['sources'], ['pipeline.loombc'])

    def test_retained_complete_root_drift_and_different_program_form_are_refused(self):
        for suffix, changed in (('.loom', 'provider.loom'), ('.loombc', 'pipeline.json'),
                                ('.loom', '../../../recognition/policy.loom')):
            with self.subTest(suffix=suffix, changed=changed):
                self.populate(suffix)
                root = self.root / 'loom-pipeline'
                policy = root / 'recognition/policy.loom'
                policy.parent.mkdir(parents=True, exist_ok=True)
                policy.write_text('policy')
                checked = self.resolved()
                with tempfile.TemporaryDirectory() as temporary:
                    retained_root = Path(temporary) / 'loom-pipeline'
                    shutil.copytree(root, retained_root)
                    retained = retained_root / checked.catalog_identity.relative_path
                    target = retained / changed
                    target.write_text(target.read_text() + ' ')
                    with self.assertRaisesRegex(InvalidInput, 'checked serving interpretation'):
                        hem_config.deployment_document(checked, 'alias', catalog=retained)
        self.populate('.loom')
        source_checked = self.resolved()
        self.populate('.loombc')
        with self.assertRaisesRegex(InvalidInput, 'checked serving interpretation'):
            hem_config.deployment_document(source_checked, 'alias', catalog=self.catalog)

    def test_emission_requires_exact_relative_location_before_reading_root(self):
        self.populate('.loom')
        checked = self.resolved()
        foreign = self.catalog.with_name('different-location')
        shutil.copytree(self.catalog, foreign)
        with patch.object(hem_process, 'tree_identity', side_effect=AssertionError('read wrong root')):
            with self.assertRaisesRegex(InvalidInput, 'outside or missing'):
                hem_config.deployment_document(checked, 'alias', catalog=foreign)

    def test_emission_root_reads_do_not_refresh_expired_invocation_budget(self):
        self.populate('.loombc')
        checked = self.resolved()
        clock = [0.0]
        budget = Deadline(1.0, clock=lambda: clock[0])
        digest_file = hem_process._digest_file
        seen = []
        def expire(path, deadline):
            seen.append(deadline)
            answer = digest_file(path, deadline)
            clock[0] = 2.0
            return answer
        with patch.object(hem_process, '_digest_file', side_effect=expire):
            with self.assertRaises(DeadlineExpired):
                hem_config.deployment_document(checked, 'alias', catalog=self.catalog, deadline=budget)
        self.assertTrue(seen)
        self.assertTrue(all(item is budget for item in seen))

    def test_shared_role_partition_drives_both_native_document_builders(self):
        for suffix in ('.loom', '.loombc'):
            with self.subTest(suffix=suffix):
                self.populate(suffix)
                modules = pipeline_validation.catalog_modules(self.catalog)
                self.assertEqual(modules.providers, ('pipeline.loombc',) if suffix=='.loombc' else ('config.loom', 'provider.loom'))
                with patch.object(pipeline_validation,'runtime_cache_dir',return_value=self.root/'cache'):
                    path = pipeline_validation.deployment_for_model(self.catalog,ModelFiles(self.weights,self.tokenizer),'alias-A',draft_depth=0)
                legacy = json.loads(path.read_text())
                modern = hem_config.deployment_document(self.resolved(),'alias-B',catalog=self.catalog)
                for document in (legacy, modern):
                    self.assertEqual(document['definition']['prepare'],'pipeline.loombc' if suffix=='.loombc' else 'prepare.loom')
                    self.assertEqual(document['definition']['control'],'pipeline.loombc' if suffix=='.loombc' else 'control.loom')
                    self.assertEqual(document['definition']['sources'],list(modules.providers))
                    if suffix=='.loom':self.assertNotIn('weights.loom',document['definition']['sources'])
                self.assertEqual(legacy['name'],'alias-A')
                self.assertEqual(modern['name'],'alias-B')

    def test_shared_text_assembly_preserves_distinct_defaults_explicit_values_and_cache(self):
        fields = ('rows', 'context_capacity', 'pool_capacity', 'prefill_capacity', 'draft_depth')
        for suffix in ('.loom', '.loombc'):
            with self.subTest(suffix=suffix):
                self.populate(suffix)
                files = ModelFiles(self.weights, self.tokenizer)
                with patch.object(pipeline_validation, 'runtime_cache_dir', return_value=self.root / 'cache'):
                    path = pipeline_validation.deployment_for_model(self.catalog, files, 'legacy')
                    content, modified = path.read_bytes(), path.stat().st_mtime_ns
                    repeated = pipeline_validation.deployment_for_model(self.catalog, files, 'legacy')
                    custom = pipeline_validation.deployment_for_model(
                        self.catalog, files, 'legacy', rows=2, context_capacity=2048,
                        pool_capacity=8192, prefill_capacity=64, draft_depth=0)
                self.assertEqual(repeated, path)
                self.assertEqual(path.read_bytes(), content)
                self.assertEqual(path.stat().st_mtime_ns, modified)
                self.assertNotEqual(custom, path)
                legacy, explicit = json.loads(content), json.loads(custom.read_text())
                default_managed = hem_config.deployment_document(self.resolved(), 'managed', catalog=self.catalog)
                checked = self.resolved({'rows': 3, 'context_capacity': 4096,
                                         'pool_capacity': 16384, 'prefill_capacity': 96})
                managed = hem_config.deployment_document(checked, 'managed-explicit', catalog=self.catalog, device=7)
                self.assertEqual(tuple(legacy['definition'][key] for key in fields), (4, 16384, 65536, 512, 0))
                self.assertEqual(tuple(explicit['definition'][key] for key in fields), (2, 2048, 8192, 64, 0))
                self.assertEqual(tuple(default_managed['definition'][key] for key in fields), (4, 8192, 32768, 128, 0))
                self.assertEqual(tuple(managed['definition'][key] for key in fields), (3, 4096, 16384, 96, 0))
                self.assertNotIn('admission', legacy)
                self.assertNotIn('admission', explicit)
                self.assertEqual((legacy['device'], explicit['device']), (0, 0))
                self.assertEqual((managed['device'], managed['admission']), (7, 'drained'))
                self.assertEqual(managed['definition'], checked.definition_document())
                for document in (legacy, explicit, default_managed, managed):
                    self.assertEqual(document['definition']['options'], {})
                    self.assertEqual(document['definition']['weights'], str(self.weights))
                    self.assertEqual(document['definition']['tokenizer'], str(self.tokenizer))
                    self.assertEqual(document['base_directory'], str(self.catalog))

    def test_offline_exports_two_names_without_native_calls_or_named_catalogs(self):
        self.populate('.loombc')
        before = {p.name:p.read_bytes() for p in self.catalog.iterdir()}
        checked = self.resolved()
        with patch.object(hem_config,'resolve',return_value=checked), \
             patch.object(hem_config, 'catalog_modules', side_effect=AssertionError('export selected modules again')), \
             patch.object(hem_config, 'pipeline_info', side_effect=AssertionError('export selected defaults again')), \
             patch('subprocess.run',side_effect=AssertionError('offline export ran native tool')), \
             patch.object(pipeline_validation,'runtime_cache_dir',side_effect=AssertionError('named catalog cache')), \
             redirect_stderr(io.StringIO()) as stderr:
            for name in ('custom.alias-A','other-alias_B'):
                output = self.root / (name+'.json')
                hem_config.export_deployment(ModelFiles(self.weights,self.tokenizer),output,name)
                document = json.loads(output.read_text())
                self.assertEqual(document['name'],name)
                self.assertEqual(document['definition'], checked.definition_document())
                self.assertEqual(document['base_directory'],str(self.catalog))
                self.assertTrue(all(p.endswith('.loombc') for p in document['definition']['sources']))
        self.assertIn('does not reserve dependencies',stderr.getvalue())
        self.assertIn('Package upgrades',stderr.getvalue())
        self.assertEqual(before,{p.name:p.read_bytes() for p in self.catalog.iterdir()})

    def test_descriptor_drift_extra_executables_and_cross_variant_fail(self):
        self.populate('.loombc')
        descriptor=self.catalog/'pipeline.json';original=descriptor.read_text()
        for field,value in [('variant','other'),('module','../other/pipeline.loombc'),('format',2)]:
            data=json.loads(original);data[field]=value;descriptor.write_text(json.dumps(data))
            with self.subTest(field=field),self.assertRaises(ValueError):
                pipeline_validation.catalog_modules(self.catalog)
        descriptor.write_text(original)
        for name in ('extra.loombc','control.loom'):
            extra=self.catalog/name;extra.write_bytes(b'LOOMold')
            with self.assertRaisesRegex(ValueError,'unexpected executable'):
                pipeline_validation.catalog_modules(self.catalog)
            extra.unlink()

    def test_non_object_descriptor_fails_resolver_and_offline_export_cleanly(self):
        self.populate('.loombc')
        checked = self.resolved()
        for metadata in (None, 3, 'invalid', [{'module':'pipeline.loombc'}]):
            (self.catalog/'pipeline.json').write_text(json.dumps(metadata))
            with self.subTest(metadata=metadata), self.assertRaisesRegex(ValueError,'Invalid singleton'):
                pipeline_validation.catalog_modules(self.catalog)
            with patch.object(hem_config,'resolve',return_value=checked):
                with self.assertRaisesRegex(hem_config.InvalidInput,'checked serving interpretation'):
                    hem_config.export_deployment(ModelFiles(self.weights,self.tokenizer),self.root/'bad.json','alias')
            self.assertFalse((self.root/'bad.json').exists())

    def test_missing_corrupt_and_escaped_bytecode_fail_without_source_fallback(self):
        self.populate('.loombc')
        module=self.catalog/'pipeline.loombc'
        module.write_bytes(b'not bytecode')
        with self.assertRaisesRegex(ValueError,'Invalid Loom bytecode'):
            pipeline_validation.catalog_modules(self.catalog)
        module.unlink()
        with self.assertRaises(FileNotFoundError):
            pipeline_validation.catalog_modules(self.catalog)
        module.symlink_to(self.weights)
        with self.assertRaises(ValueError):
            pipeline_validation.catalog_modules(self.catalog)


if __name__=='__main__':
    unittest.main()
