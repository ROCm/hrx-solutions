"""Recipe bytes, independent archive oracles, and descriptor-selected bundles."""
from dataclasses import replace
import io
import json
from pathlib import Path
import tarfile
import tempfile
import stat
import unittest
from unittest.mock import patch
import zipfile

from binding.python.hem_common import HemError
from binding.python.models import recipes
from binding.python.models.recipe_types import (
    BundleFile, BundleManifest, PipelineDescriptor, ProtocolDescriptor, TextPrograms,
)
from build_tools.model_tools import package_families
from build_tools.validation import recipe_inventory as inventory

REPO = Path(__file__).resolve().parents[2]


class RecipePackagingTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.authored = {path: (REPO / 'loom-pipeline' / path).read_bytes()
                         for path in inventory.EXPECTED_RECIPES}

    def wheel(self, members):
        path = self.root / 'fixture.whl'
        with zipfile.ZipFile(path, 'w') as archive:
            for name, content in members:
                archive.writestr('hrx_solutions/loom-pipeline/' + name, content)
        return zipfile.ZipFile(path)

    def source(self, members, symlink=False):
        path = self.root / 'fixture.tar'
        with tarfile.open(path, 'w') as archive:
            for name, content in members:
                info = tarfile.TarInfo('package/loom-pipeline/' + name)
                info.size = len(content)
                if symlink:
                    info.type = tarfile.SYMTYPE
                    info.linkname = 'outside'
                archive.addfile(info, io.BytesIO(content))
        return tarfile.open(path)

    def test_real_three_recipes_have_independent_ids_coordinates_and_source_bytes(self):
        with self.wheel(self.authored.items()) as archive:
            inventory.validate_wheel(archive, REPO, recipes.parse)
        with self.source(self.authored.items()) as archive:
            inventory.validate_sdist(archive, REPO, recipes.parse)
        package = self.root / 'installed'
        for name, content in self.authored.items():
            path = package / 'loom-pipeline' / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        inventory.validate_installed(package, recipes.parse, REPO)
        selected = next(iter(self.authored))
        (package / 'loom-pipeline' / selected).write_bytes(b'{}')
        with self.assertRaises(HemError):
            inventory.validate_installed(package, recipes.parse, REPO)

    def test_archive_inventory_rejects_missing_duplicate_swapped_changed_and_unregistered(self):
        original = list(self.authored.items())
        changed = json.loads(original[0][1])
        changed['artifacts'][0]['source']['revision'] = '0' * 40
        mutations = {
            'missing': original[1:],
            'duplicate': original + original[:1],
            'swapped': [(original[0][0], original[1][1]), (original[1][0], original[0][1]), original[2]],
            'pin': [(original[0][0], json.dumps(changed).encode()), *original[1:]],
            'unregistered': original + [('llm/other/recipes/unknown.json', original[0][1])],
            'bytes': [(original[0][0], original[0][1] + b'\n'), *original[1:]],
        }
        for name, members in mutations.items():
            for create, validate in ((self.wheel, inventory.validate_wheel), (self.source, inventory.validate_sdist)):
                with self.subTest(case=name, archive=create.__name__), create(members) as archive:
                    with self.assertRaises(RuntimeError):
                        validate(archive, REPO, recipes.parse)
        with self.source(original, symlink=True) as archive, self.assertRaisesRegex(RuntimeError, 'regular'):
            inventory.validate_sdist(archive, REPO, recipes.parse)

    def test_wheel_recipes_cannot_be_symlinks(self):
        path = self.root / 'symlink.whl'
        with zipfile.ZipFile(path, 'w') as archive:
            for name, content in self.authored.items():
                info = zipfile.ZipInfo('hrx_solutions/loom-pipeline/' + name)
                info.external_attr = (stat.S_IFLNK | 0o777) << 16
                archive.writestr(info, content)
        with zipfile.ZipFile(path) as archive, self.assertRaisesRegex(RuntimeError, 'regular'):
            inventory.validate_wheel(archive, REPO, recipes.parse)

    def test_source_parity_compares_asr_recipe_bytes_and_inventory(self):
        from build_tools.validation.check_installed_cpu import source_parity
        name = 'recognition/moonshine/streaming_tiny/recipes/moonshine-streaming-tiny.json'
        source = self.root / 'source/loom-pipeline' / name
        target = self.root / 'package/loom-pipeline' / name
        source.parent.mkdir(parents=True)
        target.parent.mkdir(parents=True)
        source.write_bytes(self.authored[name])
        target.write_bytes(self.authored[name])
        checked = source_parity(self.root / 'source', self.root / 'package')
        self.assertIn('loom-pipeline/' + name, checked)
        target.write_bytes(self.authored[name] + b'\n')
        with self.assertRaisesRegex(AssertionError, 'installed authored recipe mismatch'):
            source_parity(self.root / 'source', self.root / 'package')
        target.unlink()
        with self.assertRaisesRegex(AssertionError, 'installed recipe inventory mismatch'):
            source_parity(self.root / 'source', self.root / 'package')

    def bindings(self):
        descriptors = {}
        for path, content in self.authored.items():
            recipe = recipes.parse(content)
            catalog = str(Path(path).parent.parent)
            files = [BundleFile('loom-pipeline/' + path, 'authored recipe'),
                     BundleFile('loom-pipeline/' + catalog + '/pipeline.loombc', 'program')]
            descriptors[recipe.pipeline] = PipelineDescriptor(
                recipe.pipeline, recipe.modality, {},
                ProtocolDescriptor('test', 'test', '/', 'json', (), ()), (), None,
                TextPrograms('pipeline.loombc', 'pipeline.loombc', ('pipeline.loombc',)),
                BundleManifest(catalog, tuple(files)))
        class Binding:
            def __init__(self, descriptor): self.descriptor = descriptor
            def describe(self): return self.descriptor
            def validate_artifacts(self, artifacts): self.artifacts = artifacts
        return {key: Binding(value) for key, value in descriptors.items()}

    def test_real_authored_selection_drives_bundle_union_and_exact_staging(self):
        bindings = self.bindings()
        with patch.object(package_families.pipelines, 'get', side_effect=bindings.__getitem__):
            bundle = package_families.recipe_bundle(REPO)
        self.assertEqual(len(bundle), 6)
        def producer(root, outputs, repository, decoder):
            result = []
            for name in (*bundle, 'loom-pipeline/llm/catalog-build.json', 'loom-pipeline/unused.json'):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b'fixture:' + name.encode())
                result.append(path)
            return result
        prepared = (Path('decoder'), ((producer, {}, None),), bundle)
        package = self.root / 'package'
        outputs = package_families.stage(package, prepared, REPO)
        self.assertEqual({p.relative_to(package).as_posix() for p in outputs},
                         set(bundle) | {'loom-pipeline/llm/catalog-build.json'})
        self.assertFalse((package / 'loom-pipeline/unused.json').exists())
        package_families.validate_bundle(package, prepared)
        (package / bundle[0]).unlink()
        with self.assertRaises(FileNotFoundError):
            package_families.validate_bundle(package, prepared)

    def test_bundle_requires_each_product_and_rejects_duplicate_producer_files(self):
        required = ('loom-pipeline/llm/required.loombc',)
        def producer(root, outputs, repository, decoder):
            path = root / 'loom-pipeline/llm/catalog-build.json'
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'compiler metadata')
            return [path]
        products = ((producer, {}, None),)
        with self.assertRaisesRegex(ValueError, 'missing compiled/catalog products'):
            package_families.stage(self.root / 'package', (Path('decoder'), products, required), REPO)
        with self.assertRaisesRegex(ValueError, 'Duplicate compiled/catalog product'):
            package_families.stage(self.root / 'package', (Path('decoder'), products * 2, required), REPO)

    def test_bundle_rejects_unsafe_duplicate_omitted_and_unregistered_files(self):
        for mutation in ('unsafe', 'duplicate', 'omitted', 'unregistered'):
            bindings = self.bindings()
            binding = bindings['qwen-08b']
            descriptor = binding.describe()
            files = descriptor.bundle.files
            if mutation == 'unsafe': files += (BundleFile('../escape', 'bad'),)
            elif mutation == 'duplicate': files += files[:1]
            elif mutation == 'omitted': files = files[1:]
            else: files += (BundleFile('loom-pipeline/llm/recipes/other.json', 'bad'),)
            binding.descriptor = replace(descriptor, bundle=replace(descriptor.bundle, files=files))
            with self.subTest(mutation=mutation), patch.object(package_families.pipelines, 'get', side_effect=bindings.__getitem__):
                with self.assertRaises(ValueError):
                    package_families.recipe_bundle(REPO)


if __name__ == '__main__': unittest.main()
