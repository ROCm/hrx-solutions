"""Literal build DATA and actual authored source closure admission."""
from copy import deepcopy
from pathlib import Path
import pprint
import tempfile
import unittest

from build_tools.model_tools import loom_catalog as catalog
from build_tools.model_tools.catalog_test_fixture import CatalogFixture


class CatalogTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        self.fixture = CatalogFixture(Path(temporary.name))
        self.root = self.fixture.repository
        self.inventory = self.root / 'loom-pipeline/catalog_sources.bzl'
        self.builds = self.fixture.builds

    def save(self, value):
        self.inventory.write_text('CATALOG_BUILDS = ' + pprint.pformat(value, sort_dicts=False) + '\n')

    def test_current_authored_closures_and_flattened_internal_aliases(self):
        for reference, build in self.builds.items():
            document, identity = catalog.admit_source(self.root, reference, build)
            self.assertEqual(identity, catalog.digest((self.root / 'loom-pipeline' / reference).read_bytes()))
            for name in document['programs']['files']:
                path = self.root / 'loom-pipeline' / document['programs']['root'] / name
                if path.is_symlink():
                    content = path.read_bytes(); path.unlink(); path.write_bytes(content)
            self.assertEqual(catalog.admit_source(self.root, reference, build), (document, identity))

    def test_one_literal_assignment_no_expressions_or_duplicate_keys(self):
        original = self.inventory.read_text()
        for text in (original + '\nX = 1', original.replace('CATALOG_BUILDS = ', 'X = '),
                     'CATALOG_BUILDS = dict()', "CATALOG_BUILDS = {'x': {}, 'x': {}}"):
            with self.subTest(text=text[:50]), self.assertRaises((ValueError, SyntaxError)):
                self.inventory.write_text(text); catalog.load_builds(self.inventory)

    def test_build_paths_group_membership_roles_and_policy_remain_strict(self):
        for mutation in ('source', 'duplicate', 'output', 'role', 'support', 'helper', 'notice', 'projection'):
            builds = deepcopy(self.builds)
            build = next(iter(builds.values())); group = next(iter(build['groups'].values()))
            if mutation == 'source': group['sources'][0] = '../escape.loom'
            elif mutation == 'duplicate': group['sources'].append(group['sources'][0])
            elif mutation == 'output': build['groups']['raw.loom'] = group
            elif mutation == 'role': build['roles']['control'] = 'missing.loombc'
            elif mutation == 'support': build['roles']['support'].append(build['roles']['control'])
            elif mutation == 'helper': build['helper'] = dict(contract='x.json', upstream='tools', destination='outside')
            elif mutation == 'notice': build['notices'].append('model.gguf')
            else: group['policy']['source'] = 'absent.loom'
            self.save(builds)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError): catalog.load_builds(self.inventory)

    def test_missing_hash_drift_and_escaped_sources_rejected(self):
        reference, build = next(iter(self.builds.items()))
        document, _ = catalog.admit_source(self.root, reference, build)
        name = next(iter(document['programs']['files']))
        path = self.root / 'loom-pipeline' / document['programs']['root'] / name
        original = path.read_bytes()
        path.write_bytes(b'changed source')
        with self.assertRaisesRegex(ValueError, 'source hash'): catalog.admit_source(self.root, reference, build)
        path.unlink()
        with self.assertRaises(FileNotFoundError): catalog.admit_source(self.root, reference, build)
        external = self.fixture.root / 'outside.loom'; external.write_bytes(original); path.symlink_to(external)
        with self.assertRaises(ValueError): catalog.admit_source(self.root, reference, build)

    def test_build_closure_cannot_omit_or_add_source_even_when_it_exists(self):
        reference, build = next(iter(self.builds.items()))
        for added in (False, True):
            changed = deepcopy(build); group = next(iter(changed['groups'].values()))
            if added:
                path = self.root / 'loom-pipeline/extra.loom'; path.write_text('extra')
                group['sources'].append('extra.loom')
            else: group['sources'].pop()
            with self.subTest(added=added), self.assertRaisesRegex(ValueError, 'closure'):
                catalog.admit_source(self.root, reference, changed)

    def test_source_order_retained_and_sandbox_inventory_symlink_usable(self):
        self.assertEqual(catalog.load_builds(self.inventory), self.builds)
        alias = self.fixture.root / 'sandbox/catalog_sources.bzl'; alias.parent.mkdir(); alias.symlink_to(self.inventory)
        self.assertEqual(catalog.load_builds(alias), self.builds)
        self.assertEqual(alias.resolve().parent.parent, self.root)

    def test_product_query_rejects_duplicate_extra_missing_or_wrong_encoding(self):
        references = tuple(self.builds)
        original = tuple(self.fixture.products.values())
        for outputs in (original[:-1], original + original[:1], original + (self.fixture.root / 'extra.loombc',)):
            with self.subTest(outputs=outputs), self.assertRaises(ValueError):
                catalog.query_outputs(lambda _: outputs, self.root, self.builds, references)
        original[0].write_bytes(b'wrong header')
        with self.assertRaises(ValueError): catalog.query_outputs(lambda _: original, self.root, self.builds, references)

    def test_write_once_complete_outputs_and_conflicting_or_escaped_destination(self):
        root = self.fixture.root / 'package'; outputs = {}
        for _ in range(2): catalog.write_once(root, 'shared/NOTICE', b'notice', outputs)
        self.assertEqual(list(outputs), ['shared/NOTICE'])
        with self.assertRaisesRegex(ValueError, 'Conflicting'): catalog.write_once(root, 'shared/NOTICE', b'changed', outputs)
        with self.assertRaises(ValueError): catalog.write_once(root, '../escape', b'bad', outputs)


if __name__ == '__main__': unittest.main()
