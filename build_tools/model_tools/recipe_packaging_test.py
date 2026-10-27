"""Actual selected source catalogs staged through one generic producer."""
from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch

from binding.python.models import recipes, pipelines
from build_tools.model_tools import package_families, loom_catalog
from build_tools.model_tools.catalog_test_fixture import CatalogFixture
from build_tools.validation import recipe_inventory


class SelectedPackagingTest(unittest.TestCase):
    def setUp(self):
        temporary=tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup)
        self.fixture=CatalogFixture(Path(temporary.name))
        self.package=self.fixture.root/'package'

    def test_each_selected_catalog_stages_own_programs_recipes_notices_and_helper_only(self):
        for reference,build in self.fixture.builds.items():
            package=self.package/Path(reference).parent
            outputs=self.fixture.stage(package,selected=(reference,))
            expected={'loom-pipeline/'+reference,
                      'loom-pipeline/'+str(Path(reference).parent/'build.json'),
                      *('loom-pipeline/'+name for name in build['notices']),
                      *('loom-pipeline/'+build['root']+'/'+name for name in build['groups'])}
            source=self.fixture.repository/'loom-pipeline'/Path(reference).parent
            expected.update('loom-pipeline/'+p.relative_to(self.fixture.repository/'loom-pipeline').as_posix()
                            for p in (source/'recipes').glob('*.json'))
            if build['helper']:
                expected.update(build['helper']['destination']+'/'+name for name in
                                (*self.fixture.tools,'operation.json','LICENSE','provenance.json'))
            self.assertEqual({p.relative_to(package).as_posix() for p in outputs},expected)
            self.assertEqual(set(outputs),{p for p in package.rglob('*') if p.is_file()})
            self.assertFalse(list(package.rglob('*.loom')))
            if not build['helper']:self.assertFalse((package/'_native').exists())

    def test_source_recipe_oracle_independently_requires_complete_release(self):
        root=self.fixture.repository/'loom-pipeline'
        paths=[p.relative_to(root).as_posix() for p in root.glob('**/recipes/*.json')]
        read=lambda name:(root/name).read_bytes()
        recipe_inventory.validate_values(read,paths,recipes.parse)
        for invalid in (paths[:-1],paths+paths[:1]):
            with self.assertRaisesRegex(RuntimeError,'recipe inventory'):
                recipe_inventory.validate_values(read,invalid,recipes.parse)
        changed=json.loads(read(paths[0]));changed['name']='unreviewed name'
        with self.assertRaises(RuntimeError):recipe_inventory.validate_values(
            lambda name:json.dumps(changed).encode() if name==paths[0] else read(name),paths,recipes.parse)

    def test_unknown_or_duplicate_selected_catalog_and_recipe_fail_before_queries(self):
        references=tuple(self.fixture.builds)
        for selected in (('unknown/executable.json',),references+references[:1]):
            with self.assertRaises(ValueError):package_families.prepare(self.fixture.query,self.fixture.repository,selected=selected)
        self.assertEqual(self.fixture.queried,[])
        path=next((self.fixture.repository/'loom-pipeline').glob('**/recipes/*.json'))
        doc=json.loads(path.read_bytes());doc['executable']=next(ref for ref in references if ref!=doc['executable'])
        path.write_text(json.dumps(doc))
        with self.assertRaisesRegex(ValueError,'cross-catalog'):package_families.prepare(self.fixture.query,self.fixture.repository)
        self.assertEqual(self.fixture.queried,[])

    def test_changed_recipe_or_source_after_prepare_is_rejected(self):
        for mutation in ('recipe','source'):
            with tempfile.TemporaryDirectory() as temporary:
                fixture=CatalogFixture(Path(temporary))
                prepared=package_families.prepare(fixture.query,fixture.repository)
                if mutation=='recipe':path=next((fixture.repository/'loom-pipeline').glob('**/recipes/*.json'))
                else:path=next((fixture.repository/'loom-pipeline').glob('**/prepare.loom'))
                path.write_bytes(path.read_bytes()+b'\n')
                with patch.object(loom_catalog.subprocess,'run',side_effect=fixture.decode),self.assertRaisesRegex(ValueError,'changed|hash mismatch'):
                    package_families.stage(fixture.root/'package',prepared,fixture.repository)

    def test_normalized_current_catalogs_pass_actual_compiled_parser(self):
        outputs=self.fixture.stage(self.package)
        for path in outputs:
            if path.name=='executable.json':
                doc,_=pipelines.parse(path.read_bytes())
                root=self.package/'loom-pipeline'/doc['programs']['root']
                self.assertTrue(all((root/name).is_file() for name in doc['programs']['files']))
            elif path.parent.name=='recipes':recipes.parse(path.read_bytes())


if __name__=='__main__':unittest.main()
