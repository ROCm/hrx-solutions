"""Catalog closure and artifact contracts before switching wheel consumers."""
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
import subprocess

from build_tools.model_tools import loom_catalog as catalog

REPOSITORY = Path(__file__).resolve().parents[2]


class CatalogTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.pipeline = self.root / "loom-pipeline"
        shutil.copytree(REPOSITORY / "loom-pipeline", self.pipeline, symlinks=True)
        self.inventory = self.pipeline / "catalog_sources.bzl"
        self.sources = catalog.validate_catalog(self.inventory)
        self.decoder_validation = patch.object(catalog, 'validate_bytecode_artifact').start()
        self.addCleanup(patch.stopall)
        self.manifest = self.pipeline / "llm/qwen/qwen_08b/sources.txt"

    def products(self):
        outputs = {}
        for source, variant in catalog.variants(self.inventory).items():
            path = self.root / "bazel-out/config/bin/loom-pipeline/bytecode" / variant["output"]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"LOOMfixture")
            outputs[source] = path
        return outputs

    def test_inventory_matches_both_complete_authored_manifests(self):
        self.assertEqual(len(self.sources), 62)
        for architecture in catalog.ARCHITECTURES:
            manifest = self.pipeline / "llm/qwen" / architecture / "sources.txt"
            entries = catalog.manifest_entries(manifest, self.pipeline / "llm")
            self.assertEqual(len(entries), 58)
            self.assertEqual(sum(Path(p).name in catalog.ROLES for p in entries), 3)
        self.assertTrue((self.manifest.parent / "control.loom").is_symlink())

    def test_inventory_drift_and_duplicates_fail(self):
        original = self.inventory.read_text()
        self.inventory.write_text(original.replace("'qwen/continue.loom',", '', 1))
        with self.assertRaisesRegex(ValueError, "inventory/manifest/profile drift"):
            catalog.validate_catalog(self.inventory)
        self.inventory.write_text(original.replace('[', '["qwen/continue.loom",', 1))
        with self.assertRaisesRegex(ValueError, "Duplicate source identity"):
            catalog.validate_catalog(self.inventory)

    def test_inventory_accepts_only_one_literal_variant_assignment(self):
        original = self.inventory.read_text()
        invalid = [
            "",
            original + "\nEXTRA = []\n",
            original + "\n" + original,
            original.replace("CATALOG_VARIANTS =", "OTHER ="),
            original.replace("CATALOG_VARIANTS =", "CATALOG_VARIANTS = OTHER ="),
            original.replace("CATALOG_VARIANTS =", "CATALOG_VARIANTS: dict ="),
            "CATALOG_VARIANTS = dict()",
            "CATALOG_VARIANTS = {'qwen_08b': dict()}",
            "CATALOG_VARIANTS = {'qwen_08b': {}, 'qwen_08b': {}}",
            "CATALOG_VARIANTS = {[]: {}}",
            "CATALOG_VARIANTS = []",
            "CATALOG_VARIANTS = {'unsupported': {}}",
            original.replace("'identifier':", "'identifier': 'hidden', 'identifier':", 1),
        ]
        for text in invalid:
            with self.subTest(text=text[:100]):
                self.inventory.write_text(text)
                with self.assertRaises(ValueError):
                    catalog.variants(self.inventory)

    def test_variant_schema_and_source_paths_remain_strict(self):
        original = catalog.variants(self.inventory)
        invalid = []
        for key, value in [('identifier', 1), ('identifier', ''),
                           ('output', '../pipeline.loombc'), ('sources', 'module.loom'),
                           ('sources', [1]), ('extra', [])]:
            variant = dict(original['qwen_08b'], **{key: value})
            invalid.append({'qwen_08b': variant})
        invalid.extend([{'qwen_08b': []}, {'qwen_08b': {}}])
        for source in ('../escape.loom', '/absolute.loom', 'a/../module.loom',
                       'a\\module.loom', 'module.loombc'):
            invalid.append({'qwen_08b': dict(original['qwen_08b'], sources=[source])})
        for value in invalid:
            with self.subTest(value=value):
                self.inventory.write_text('CATALOG_VARIANTS = ' + repr(value))
                with self.assertRaises(ValueError):
                    catalog.catalog_sources(self.inventory)

    def test_inventory_preserves_source_order_and_profile_contract(self):
        value = catalog.variants(self.inventory)
        sources = value['qwen_08b']['sources']
        sources.reverse()
        self.inventory.write_text('CATALOG_VARIANTS = ' + repr(value))
        self.assertEqual(catalog.catalog_sources(self.inventory),
                         tuple(dict.fromkeys(source for variant in value.values()
                                             for source in variant["sources"])))
        with self.assertRaisesRegex(ValueError, 'inventory/manifest/profile drift'):
            catalog.validate_catalog(self.inventory)
        sources.reverse()
        value['qwen_08b']['identifier'] = 'other-profile'
        self.inventory.write_text('CATALOG_VARIANTS = ' + repr(value))
        with self.assertRaisesRegex(ValueError, 'inventory/manifest/profile drift'):
            catalog.validate_catalog(self.inventory)

    def test_source_archive_flattened_role_symlinks_preserve_complete_inventory(self):
        for architecture in catalog.ARCHITECTURES:
            path = self.pipeline / 'llm/qwen' / architecture / 'control.loom'
            content = path.read_bytes()
            path.unlink()
            path.write_bytes(content)
        self.assertEqual(catalog.validate_catalog(self.inventory), self.sources)
        (self.pipeline / 'llm/qwen/undeclared.loom').write_text('func.def public @omitted() {}')
        with self.assertRaisesRegex(ValueError, 'Undeclared authored'):
            catalog.validate_catalog(self.inventory)

    def test_new_unreferenced_authored_module_cannot_escape_inventory(self):
        (self.pipeline / 'llm/unlisted.loom').write_text('func.def public @omitted() {}')
        with self.assertRaisesRegex(ValueError, 'Undeclared authored'):
            catalog.validate_catalog(self.inventory)

    def test_role_and_normalized_duplicate_drift_fail(self):
        original = self.manifest.read_text()
        for text, message in [('\n'.join(line for line in original.splitlines() if line != 'prepare.loom')+'\n', 'prepare.loom role'),
                              (original+'./config.loom\n', 'Duplicate normalized'),
                              (original+'../control.loom\n', 'control.loom role')]:
            with self.subTest(message=message):
                self.manifest.write_text(text)
                with self.assertRaisesRegex(ValueError, message):
                    catalog.validate_catalog(self.inventory)

    def test_missing_or_escaped_symlink_input_fails(self):
        path = self.manifest.parent / 'control.loom'
        path.unlink()
        outside = self.root / 'outside.loom'
        outside.write_text('outside')
        path.symlink_to(outside)
        with self.assertRaises(ValueError):
            catalog.validate_catalog(self.inventory)
        outside.unlink()
        with self.assertRaises(FileNotFoundError):
            catalog.validate_catalog(self.inventory)

    def test_validation_resolves_bazel_sandbox_inventory_symlink(self):
        sandbox = self.root / 'sandbox'
        sandbox.mkdir()
        inventory = sandbox / 'catalog_sources.bzl'
        inventory.symlink_to(self.inventory)
        self.assertEqual(catalog.validate_catalog(inventory), self.sources)

    def test_prepare_changes_only_exact_known_locator(self):
        for architecture in catalog.ARCHITECTURES:
            source = (self.pipeline / 'llm/qwen' / architecture / 'prepare.loom').read_text()
            transformed = catalog.preparation_bytecode_source(source)
            self.assertEqual(transformed.replace(b'pipeline.loombc'.hex(), b'weights.loom'.hex()), source)
            for drift in (source.replace(b'weights.loom'.hex(), b'other.loom'.hex()),
                          source+'\nglobal.rodata.def @policy = bytes("00")\n'):
                with self.assertRaisesRegex(ValueError, 'exactly one expected'):
                    catalog.preparation_bytecode_source(drift)

    def test_discovery_uses_queried_outputs_and_rejects_incomplete_or_extra(self):
        outputs = self.products()
        queried = [p.relative_to(self.root) for p in outputs.values()]
        self.assertEqual(catalog.discover_bytecode_outputs(queried, self.inventory, self.root), outputs)
        for paths, message in [(queried[:-1], 'Missing required'), (queried+queried[:1], 'duplicate'),
                               (queried+[Path('unexpected.loombc')], 'Unexpected')]:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                catalog.discover_bytecode_outputs(paths, self.inventory, self.root)
        next(iter(outputs.values())).write_bytes(b'bad')
        with self.assertRaisesRegex(ValueError, 'invalid required'):
            catalog.discover_bytecode_outputs(queried, self.inventory, self.root)

    def test_query_uses_exact_bazel_outputs_with_caller_build_environment(self):
        outputs = self.products()
        queried = "\n".join(str(p.relative_to(self.root)) for p in outputs.values())
        with patch.object(catalog.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0, stdout=queried)) as run:
            found = catalog.query_bytecode_outputs(self.root / 'bazel', self.root,
                                                  build_options=('--config=opt', '--disk_cache='),
                                                  environment={'BUILD': 'identity'})
        self.assertEqual(found, outputs)
        self.assertEqual(run.call_args.args[0], [str(self.root / 'bazel'), 'cquery', '--config=opt', '--disk_cache=', '--output=files', catalog.CATALOG_TARGET])
        self.assertTrue(run.call_args.kwargs['check'])
        self.assertEqual(run.call_args.kwargs['env'], {'BUILD': 'identity'})

    def test_staging_has_clean_allowlist_regular_outputs_and_manifest_closure(self):
        outputs = self.products()
        source = 'qwen_08b'
        output = outputs[source]
        output.unlink()
        output.symlink_to(outputs['qwen_27b'])
        destination = self.root / 'staged'
        destination.mkdir()
        (destination / 'stale.loom').write_text('stale')
        (destination / 'stale.loombc').write_bytes(b'LOOMstale')
        staged = catalog.stage_bytecode_catalog(destination, outputs, self.inventory, decoder=Path("fixture-decoder"))
        self.assertEqual(len(staged), 8)  #two complete modules, descriptors/profiles, license, provenance.
        self.assertFalse(list(destination.rglob('*.loom')))
        self.assertTrue(all(not p.is_symlink() for p in staged))
        for architecture in catalog.ARCHITECTURES:
            directory = destination / 'qwen' / architecture
            self.assertTrue((directory / 'pipeline.loombc').is_file())
            self.assertTrue((directory / 'pipeline.json').is_file())
            self.assertFalse((directory / 'sources.txt').exists())
        self.assertFalse((destination / 'README.md').exists())
        self.assertFalse((destination / 'stale.loombc').exists())

    def test_header_preserving_corrupt_artifact_cannot_be_blessed_or_reuse_stale_stage(self):
        outputs = self.products()
        destination = self.root / 'staged'
        catalog.stage_bytecode_catalog(destination, outputs, self.inventory, decoder=Path('fixture-decoder'))
        outputs['qwen_08b'].write_bytes(b'LOOMtruncated')
        self.decoder_validation.side_effect = subprocess.CalledProcessError(1, ['pinned-decoder'])
        with self.assertRaises(subprocess.CalledProcessError):
            catalog.stage_bytecode_catalog(destination, outputs, self.inventory, decoder=Path('fixture-decoder'))
        self.assertFalse(list(destination.rglob('*.loombc')))

    def test_missing_current_artifact_cannot_reuse_staged_snapshot(self):
        outputs = self.products()
        destination = self.root / 'staged'
        catalog.stage_bytecode_catalog(destination, outputs, self.inventory, decoder=Path("fixture-decoder"))
        del outputs["qwen_08b"]
        with self.assertRaisesRegex(ValueError, 'output inventory'):
            catalog.stage_bytecode_catalog(destination, outputs, self.inventory, decoder=Path("fixture-decoder"))
        self.assertFalse(list(destination.rglob('*.loombc')))


if __name__ == '__main__':
    unittest.main()
