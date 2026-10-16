"""Source/bytecode role resolution and immutable deployment export contracts."""
from contextlib import redirect_stderr
import io
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from binding.python import hem_config, pipeline_validation
from binding.python.model_store import ModelFiles


class PipelineCatalogTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.catalog = self.root / 'llm/qwen/test'
        self.catalog.mkdir(parents=True)
        self.weights, self.tokenizer = self.root / 'weights.gguf', self.root / 'tokenizer.json'
        self.weights.write_bytes(b'fixture')
        self.tokenizer.write_text('{}')

    def populate(self, suffix):
        for old in self.catalog.iterdir():old.unlink()
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

    def resolved(self):
        return hem_config.ResolvedConfiguration(ModelFiles(self.weights,self.tokenizer), 'qwen-08b',
            self.catalog, {**hem_config.DEFAULTS, 'mtp':False, 'mtp_depth':None}, {})

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

    def test_offline_exports_two_names_without_native_calls_or_named_catalogs(self):
        self.populate('.loombc')
        before = {p.name:p.read_bytes() for p in self.catalog.iterdir()}
        with patch.object(hem_config,'resolve',return_value=self.resolved()), \
             patch('subprocess.run',side_effect=AssertionError('offline export ran native tool')), \
             patch.object(pipeline_validation,'runtime_cache_dir',side_effect=AssertionError('named catalog cache')), \
             redirect_stderr(io.StringIO()) as stderr:
            for name in ('custom.alias-A','other-alias_B'):
                output = self.root / (name+'.json')
                hem_config.export_deployment(ModelFiles(self.weights,self.tokenizer),output,name)
                document = json.loads(output.read_text())
                self.assertEqual(document['name'],name)
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
        for metadata in (None, 3, 'invalid', [{'module':'pipeline.loombc'}]):
            (self.catalog/'pipeline.json').write_text(json.dumps(metadata))
            with self.subTest(metadata=metadata), self.assertRaisesRegex(ValueError,'Invalid singleton'):
                pipeline_validation.catalog_modules(self.catalog)
            with patch.object(hem_config,'resolve',return_value=self.resolved()):
                with self.assertRaisesRegex(hem_config.InvalidInput,'Cannot resolve pipeline modules'):
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
