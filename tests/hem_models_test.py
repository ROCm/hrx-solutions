"""Common acquisition contracts with actual authored recipes and controlled HF I/O."""
from contextlib import redirect_stderr
from dataclasses import replace
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from binding.python.control import hem_state
from binding.python.models import hem_models as models, model_download, recipes, pipelines
from binding.python.models.recipe_types import (GeneratedArtifact, GeneratedSource, HFSource,
    PreparedModel, PreparationRequirement, GeneratedOutput, BundleManifest)
from binding.python.hem_common import Deadline, HemError
from tests.gguf_fixtures import write_gguf


class Binding:
    """Faithful acquisition-only binding; concrete policy is tested by its owner."""
    def __init__(self, pipeline, modality, artifacts):
        self.id, self.modality, self.artifacts = pipeline, modality, artifacts
        outputs = tuple(GeneratedOutput(a.role, a.format, a.path) for a in artifacts
                        if isinstance(a.source, GeneratedSource))
        self.preparation = PreparationRequirement((), outputs) if outputs else None
        self.events = []
        self.catalog = None

    def describe(self):
        return SimpleNamespace(id=self.id, modality=self.modality, preparation=self.preparation,
                               bundle=BundleManifest(self.catalog, ()))

    def validate_artifacts(self, artifacts):
        if artifacts != self.artifacts:
            raise HemError('unsupported artifact tuple')

    def preflight(self, plan, deadline):
        self.events.append('preflight')
        deadline.remaining()

    def prepare(self, plan, inputs, attempt, deadline):
        self.events.append('prepare')
        deadline.remaining()
        assert set(inputs) == {a.id for a in plan.inputs}
        result = []
        for a in plan.generated:
            path = attempt / 'output' / a.path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'generated')
            result.append(GeneratedArtifact(a.role, a.format, a.path, path))
        return tuple(result)


class AcquisitionFixture:
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.cache = self.root / 'cache'
        self.environment = patch.dict(os.environ, {'HF_HUB_CACHE': str(self.cache), 'HF_HUB_OFFLINE': '0'})
        self.environment.start(); self.addCleanup(self.environment.stop)
        self.registry = hem_state.Registry(self.root / 'state')
        source = Path(__file__).resolve().parents[1] / 'loom-pipeline'
        self.declarations = {p.stem: recipes.parse(p.read_bytes()) for p in source.glob('**/recipes/*.json')}
        self.bindings = {r.pipeline: Binding(r.pipeline, r.modality, r.artifacts)
                         for r in self.declarations.values()}
        for path in source.glob('**/recipes/*.json'):
            recipe = recipes.parse(path.read_bytes())
            self.bindings[recipe.pipeline].catalog = str(path.parent.parent.relative_to(source))
        self.selector = patch.object(pipelines, 'get', side_effect=lambda key: self.bindings[key])
        self.selector.start(); self.addCleanup(self.selector.stop)
        self.recipe = self.declarations['qwen35-08b-q5-k-xl']
        self.network = patch.object(model_download, '_run_hf', side_effect=AssertionError('unexpected network'))
        self.fetch = self.network.start(); self.addCleanup(self.network.stop)
        self.output = redirect_stderr(io.StringIO())
        self.output.__enter__(); self.addCleanup(self.output.__exit__, None, None, None)
        self.gguf = self.root / 'input.gguf'
        write_gguf(self.gguf, tensors=[('tiny', [1], 0, 0)])
        self.tokenizer = self.root / 'tokenizer.json'
        self.tokenizer.write_text('{"model":{"fixture":true}}')

    def cached(self, recipe=None):
        for a in (recipe or self.recipe).artifacts:
            if isinstance(a.source, HFSource):
                path = model_download._cache_path(a.source)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b'opaque HF input')

    def acquire(self, name='tiny', **kwargs):
        kwargs.setdefault('pipeline', self.recipe.pipeline)
        return models.import_model(self.registry, name, ggufs=[self.gguf], tokenizer=self.tokenizer, **kwargs)

    def download(self, name='downloaded', **kwargs):
        return models.download(self.registry, self.recipe.id, model_id=name, **kwargs)


class HemModelsTest(AcquisitionFixture, unittest.TestCase):
    def test_owned_copy_and_immutable_retry(self):
        row = self.acquire()
        target = Path(row['directory']) / 'model.gguf'
        original = target.read_bytes()
        self.gguf.write_bytes(b'changed source')
        self.acquire()
        self.assertFalse(target.is_symlink())
        self.assertEqual(target.read_bytes(), original)
        self.assertEqual(models.show(self.registry, 'tiny')['state'], 'ready')
        self.assertEqual(len(self.registry.list_models()), 1)

    def test_alias_ownership_and_independent_tokenizers(self):
        first = self.acquire('first', alias=True)
        other = self.root / 'other.json'; other.write_text('{}')
        second = models.import_model(self.registry, 'second', ggufs=[self.gguf], tokenizer=other,
                                     pipeline=self.recipe.pipeline, alias=True)
        for row, token in ((first, self.tokenizer), (second, other)):
            path = Path(row['directory']) / 'tokenizer.json'
            self.assertTrue(path.is_symlink()); self.assertEqual(path.resolve(), token)
        self.assertEqual(len(models.inventory(self.registry)), 2)

    def test_conflicting_id_preserves_original(self):
        self.acquire()
        before = self.registry.get_model('tiny')
        other = self.root / 'other.gguf'; other.write_bytes(b'other')
        with self.assertRaisesRegex(HemError, 'different mappings'):
            models.import_model(self.registry, 'tiny', ggufs=[other], tokenizer=self.tokenizer,
                                pipeline=self.recipe.pipeline)
        self.assertEqual(before, self.registry.get_model('tiny'))

    def test_missing_alias_inspectable_and_revalidation_fails(self):
        self.acquire(alias=True); self.gguf.unlink()
        self.assertEqual(models.show(self.registry, 'tiny')['state'], 'missing')
        with self.assertRaisesRegex(HemError, 'missing'):
            models.validate(self.registry, 'tiny')
        self.assertFalse(self.registry.get_model('tiny')['validation']['valid'])

    def test_incomplete_copy_retry_reuses_completed_members(self):
        copy = models.shutil.copyfileobj
        def fail_second(source, destination, **kwargs):
            if source.name == str(self.tokenizer):
                raise OSError('interrupted')
            return copy(source, destination, **kwargs)
        with patch.object(models.shutil, 'copyfileobj', side_effect=fail_second):
            with self.assertRaisesRegex(HemError, 'interrupted'):
                self.acquire()
        row = self.registry.get_model('tiny'); stage = Path(row['pending']['path'])
        inode = (stage / 'model.gguf').stat().st_ino
        self.acquire()
        self.assertEqual((Path(row['directory']) / 'model.gguf').stat().st_ino, inode)

    def test_atomic_partial_copy_is_not_reused(self):
        def fail(source, destination, **kwargs):
            destination.write(b'incomplete'); raise OSError('interrupted')
        with patch.object(models.shutil, 'copyfileobj', side_effect=fail):
            with self.assertRaises(HemError): self.acquire()
        row = self.registry.get_model('tiny'); stage = Path(row['pending']['path'])
        self.assertFalse((stage / 'model.gguf').exists())
        self.acquire()
        self.assertEqual((Path(row['directory']) / 'model.gguf').read_bytes(), self.gguf.read_bytes())

    def test_crash_after_rename_resumes_without_adopting_orphan(self):
        update = self.registry.update_model
        def crash(model_id, changes, **kwargs):
            if changes.get('pending') == {'stage': 'qualification'}: raise RuntimeError('crash')
            return update(model_id, changes, **kwargs)
        with patch.object(self.registry, 'update_model', side_effect=crash):
            with self.assertRaisesRegex(RuntimeError, 'crash'): self.acquire()
        row = self.registry.get_model('tiny')
        self.assertFalse(Path(row['pending']['path']).exists())
        self.acquire()
        self.assertEqual(models.show(self.registry, 'tiny')['state'], 'ready')

    def test_changed_assets_are_unvalidated_and_rechecked(self):
        self.acquire(alias=True)
        self.gguf.write_bytes(b'changed')
        self.assertEqual(models.show(self.registry, 'tiny')['state'], 'unvalidated')
        # No model architecture gate: accessible opaque bytes remain allowed.
        models.validate(self.registry, 'tiny')
        self.assertEqual(models.show(self.registry, 'tiny')['state'], 'ready')

    def test_reference_mapping_cannot_be_requalified_after_retargeting(self):
        row = self.acquire(alias=True)
        target = Path(row['directory']) / 'model.gguf'
        other = self.root / 'other.gguf'; other.write_bytes(self.gguf.read_bytes())
        target.unlink(); target.symlink_to(other)
        with self.assertRaisesRegex(HemError, 'mapping changed'):
            models.model_files(self.registry, 'tiny')
        self.assertFalse(models.show(self.registry, 'tiny')['accessible'])
        target.unlink(); target.symlink_to(self.gguf)
        models.validate(self.registry, 'tiny')
        self.assertEqual(models.show(self.registry, 'tiny')['state'], 'ready')

    def test_drift_during_presence_check_is_rejected(self):
        self.acquire()
        evidence = models._evidence
        def drift(files):
            value = evidence(files)
            (files.directory / 'model.gguf').write_bytes(b'new')
            return value
        with patch.object(models, '_evidence', side_effect=drift):
            with self.assertRaisesRegex(HemError, 'changed during'):
                models.validate(self.registry, 'tiny')

    def test_all_three_authored_recipes_drive_the_same_acquisition(self):
        for recipe in self.declarations.values():
            self.cached(recipe)
            result = models.download(self.registry, recipe.id, model_id=recipe.id, offline=True)
            self.assertEqual(result.recipe_id, recipe.id)
            self.assertEqual(result.pipeline, recipe.pipeline)
            prepared = models.model_files(self.registry, recipe.id)
            self.assertEqual(prepared.recipe_identity, recipe.identity)
            self.assertEqual([(a.id, a.logical_path) for a in prepared.files],
                             [(a.id, a.path) for a in recipe.artifacts])
            expected = ['preflight', 'prepare'] if self.bindings[recipe.pipeline].preparation else []
            self.assertEqual(self.bindings[recipe.pipeline].events, expected)
        self.fetch.assert_not_called()

    def test_offline_preserves_record_and_retry_uses_cache(self):
        with self.assertRaisesRegex(HemError, 'Offline cache is missing'): self.download(offline=True)
        self.assertEqual(models.show(self.registry, 'downloaded')['state'], 'incomplete')
        self.cached(); answer = self.download(offline=True)
        self.assertEqual(answer.state, 'acquired'); self.fetch.assert_not_called()
        missing = model_download._cache_path(self.recipe.artifacts[0].source); missing.unlink()
        self.assertEqual(models.show(self.registry, 'downloaded')['state'], 'missing')
        self.fetch.side_effect = lambda command: missing.write_bytes(b'restored')
        self.download(); self.fetch.assert_called_once()

    def test_dryrun_never_initializes_state_or_executes_hf(self):
        with patch.object(hem_state, 'Registry', side_effect=AssertionError('state')):
            answer = models.download(None, self.recipe.id, model_id='preview', dry_run=True)
        self.assertEqual(answer.state, 'preview')
        self.assertEqual([a.logical_path for a in answer.artifacts], ['model.gguf', 'tokenizer.json'])
        self.fetch.assert_not_called()
        self.assertFalse(self.cache.exists())

    def test_download_passes_exact_pinned_command_and_preserves_partial_cache(self):
        calls = []
        def fetch(command):
            calls.append(command)
            item = next(a for a in self.recipe.artifacts if a.source.filename == command[3])
            path = model_download._cache_path(item.source); path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'opaque')
            if len(calls) == 2: raise models.ModelImportError('interrupted')
        self.fetch.side_effect = fetch
        with self.assertRaisesRegex(HemError, 'interrupted'): self.download()
        self.assertEqual(calls[0], model_download.command(self.recipe.artifacts[0].source))
        self.fetch.reset_mock(); self.download(); self.fetch.assert_not_called()
        files = models.model_files(self.registry, 'downloaded')
        self.assertIsInstance(files, PreparedModel)
        for item, declaration in zip(files.files, self.recipe.artifacts):
            self.assertTrue(item.path.is_symlink())
            self.assertEqual(os.readlink(item.path), str(model_download._cache_path(declaration.source)))

    def test_consumer_rechecks_even_saved_valid_and_recipe_identity(self):
        self.cached(); self.download()
        files = models.model_files(self.registry, 'downloaded')
        self.assertEqual(files.recipe_identity, self.recipe.identity)
        changed = replace(self.recipe, identity='0' * 64)
        with patch.object(recipes, 'load', return_value=changed):
            with self.assertRaisesRegex(HemError, 'declaration'): models.model_files(self.registry, 'downloaded')
            with self.assertRaisesRegex(HemError, 'different mappings'): self.download()
        with self.assertRaisesRegex(HemError, 'pipeline'):
            models.model_files(self.registry, 'downloaded', pipeline='other')

    def test_unsupported_admission_precedes_cache_registry_and_process(self):
        with patch.object(self.bindings[self.recipe.pipeline], 'validate_artifacts', side_effect=HemError('unsupported')), \
             patch.object(model_download, '_cache_path', side_effect=AssertionError('cache')):
            with self.assertRaisesRegex(HemError, 'unsupported'): self.download()
        self.assertEqual(self.registry.list_models(), [])

    def test_explicit_destinations_and_repeated_roles_never_use_basename(self):
        data = models._declaration(self.recipe)
        data['artifacts'][0]['path'] = 'nested/runtime.gguf'
        extra = dict(data['artifacts'][0], id='second', path='other/second.gguf',
                     source=dict(data['artifacts'][0]['source'], filename='upstream/second.gguf'))
        data['artifacts'].insert(1, extra)
        recipe = recipes.parse(json.dumps(data).encode())
        self.bindings[recipe.pipeline] = Binding(recipe.pipeline, recipe.modality, recipe.artifacts)
        self.cached(recipe)
        with patch.object(recipes, 'load', return_value=recipe): self.download()
        row = self.registry.get_model('downloaded')
        root = Path(row['directory'])
        self.assertTrue((root / 'nested/runtime.gguf').is_symlink())
        self.assertTrue((root / 'other/second.gguf').is_symlink())
        self.assertFalse((root / self.recipe.artifacts[0].source.filename).exists())

    def test_bookkeeping_collisions_rejected_before_state(self):
        for path in ('.loom-acquisition.json', '.model.gguf.partial'):
            data = models._declaration(self.recipe)
            data['artifacts'][1]['path'] = path
            recipe = recipes.parse(json.dumps(data).encode())
            binding = Binding(recipe.pipeline, recipe.modality, recipe.artifacts)
            with patch.object(recipes, 'load', return_value=recipe), patch.object(pipelines, 'get', return_value=binding):
                with self.assertRaisesRegex(HemError, 'bookkeeping'): self.download()
        self.assertEqual(self.registry.list_models(), [])

    def test_explicit_import_ignores_unrelated_broken_bindings(self):
        selected = self.recipe.pipeline
        calls = []
        def only_selected(key):
            calls.append(key)
            if key != selected:
                raise HemError('unrelated binding unavailable')
            return self.bindings[key]
        with patch.object(pipelines, 'get', side_effect=only_selected):
            self.acquire()
            directory = self.root / 'complete'; directory.mkdir()
            for artifact in self.recipe.artifacts:
                (directory / artifact.path).write_bytes(b'local input')
            row = models.import_assets(self.registry, 'complete', directory=directory, pipeline=selected)
            self.assertTrue(row['validation']['valid'])
            self.assertEqual(models.model_files(self.registry, 'complete').pipeline, selected)
        self.assertTrue(calls)
        self.assertEqual(set(calls), {selected})

    def test_complete_support_discovery_still_rejects_unrelated_broken_binding(self):
        def broken(key):
            if key != self.recipe.pipeline:
                raise HemError('unrelated binding unavailable')
            return self.bindings[key]
        with patch.object(pipelines, 'get', side_effect=broken):
            with self.assertRaisesRegex(HemError, 'unrelated binding unavailable'):
                models.available()

    def test_explicit_import_rejects_broken_selected_binding_before_state(self):
        with patch.object(pipelines, 'get', side_effect=HemError('selected binding unavailable')):
            with self.assertRaisesRegex(HemError, 'selected binding unavailable'):
                self.acquire()
            with self.assertRaisesRegex(HemError, 'selected binding unavailable'):
                models.import_assets(self.registry, 'complete', directory=self.root, pipeline=self.recipe.pipeline)
        self.assertEqual(self.registry.list_models(), [])

    def test_ambiguous_import_requests_existing_pipeline(self):
        with self.assertRaisesRegex(HemError, 'pipeline'):
            models.import_model(self.registry, 'tiny', ggufs=[self.gguf], tokenizer=self.tokenizer)

    def test_invalid_fragment_format_retains_copied_assets_for_inspection(self):
        other = self.root / 'duplicate.gguf'; other.write_bytes(self.gguf.read_bytes())
        with self.assertRaisesRegex(HemError, 'Invalid fragment format'):
            models.import_model(self.registry, 'split', ggufs=[self.gguf, other],
                                tokenizer=self.tokenizer, pipeline=self.recipe.pipeline)
        row = self.registry.get_model('split')
        self.assertEqual(row['pending']['stage'], 'staging')
        self.assertTrue((Path(row['pending']['path']) / 'model-00002-of-00002.gguf').is_file())
        self.assertFalse(Path(row['directory']).exists())

    def test_bad_fragment_mapping_retained_for_inspection(self):
        with self.assertRaises(HemError):
            models.import_model(self.registry, 'split', ggufs=[self.gguf, self.gguf],
                                tokenizer=self.tokenizer, pipeline=self.recipe.pipeline)
        self.assertEqual(self.registry.list_models(), [])

    def test_fragment_index_is_scoped_and_has_native_suffix(self):
        files = []
        for i in range(2):
            path = self.root / f'fragment-{i}.gguf'
            write_gguf(path, tensors=[(f'tensor-{i}', [1], 0, 0)])
            files.append(path)
        row = models.import_model(self.registry, 'split', ggufs=files, tokenizer=self.tokenizer,
                                  pipeline=self.recipe.pipeline)
        prepared = models.model_files(self.registry, 'split')
        index = next(a for a in prepared.files if a.role == 'weights')
        self.assertEqual(index.path.suffixes[-2:], ['.index', '.json'])
        self.assertFalse(index.path.is_symlink())
        self.assertTrue(json.loads(index.path.read_text())['weight_map'])
        self.assertEqual(models.show(self.registry, 'split')['state'], 'ready')
