"""Real ASR construction and fixed Moonshine binding, without model downloads."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from binding.python.hem_common import Deadline, HemError, InvalidInput
from binding.python.models import model_download, moonshine_preparation as helper
from binding.python.models import pipelines, pipelines_moonshine as moonshine, recipes
from binding.python.models.modalities import asr
from binding.python.models.recipe_types import (
    AcquisitionPlan, GeneratedArtifact, GeneratedSource, HFSource, PreparedArtifact,
    PreparedModel, RecognitionPrograms,
)


REPOSITORY = Path(__file__).resolve().parents[1]
CATALOG = REPOSITORY / 'loom-pipeline' / moonshine.CATALOG
RECIPE = CATALOG / 'recipes/moonshine-streaming-tiny.json'


class CatalogLookupTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.packaged = self.root / 'packaged'
        self.catalog = self.packaged / moonshine.CATALOG
        self.catalog.parent.mkdir(parents=True)
        roots = patch.object(pipelines, 'catalog_roots', return_value=(self.packaged,))
        roots.start(); self.addCleanup(roots.stop)
        for module, name in ((helper, 'preflight'), (helper, 'prepare'),
                             (model_download, 'cache_root')):
            unused = patch.object(module, name, side_effect=AssertionError('catalog query must stay pure'))
            unused.start(); self.addCleanup(unused.stop)

    def test_public_lookup_rejects_escaping_catalog_with_domain_error(self):
        outside = self.root / 'outside'
        outside.mkdir()
        self.catalog.symlink_to(outside, target_is_directory=True)
        with self.assertRaisesRegex(HemError, 'Moonshine.*catalog') as caught:
            pipelines.get(moonshine.PIPELINE)
        self.assertIsInstance(caught.exception.__cause__, ValueError)

    def test_public_lookup_wraps_catalog_resolution_failures(self):
        self.catalog.mkdir()
        for error in (PermissionError('unreadable catalog'), RuntimeError('symlink loop')):
            with self.subTest(error=error), patch.object(Path, 'resolve', side_effect=error):
                with self.assertRaisesRegex(HemError, 'Moonshine.*catalog') as caught:
                    pipelines.get(moonshine.PIPELINE)
                self.assertIs(caught.exception.__cause__, error)


class ASRPipelineTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.catalog = self.root / 'catalog'
        shutil.copytree(CATALOG, self.catalog)
        # Packaging owns this deliberate metadata cutover. Exercise its agreed
        # program-only form without changing sibling-owned source declarations.
        document = json.loads((self.catalog / 'catalog.json').read_text())
        document.pop('download_recipe', None)
        document['definition'].pop('pending_requests', None)
        document['definition'].pop('options', None)
        (self.catalog / 'catalog.json').write_text(json.dumps(document))
        selected = patch.object(moonshine, '_catalog_directory', return_value=self.catalog)
        selected.start(); self.addCleanup(selected.stop)
        self.recipe = recipes.parse(RECIPE.read_bytes())
        self.binding = moonshine.get(self.recipe.pipeline)
        self.descriptor = self.binding.describe()
        self.plan = self.plan_for(self.recipe.artifacts)

    def plan_for(self, artifacts):
        return AcquisitionPlan('another-recipe-id', self.recipe.pipeline,
                               tuple(item for item in artifacts if isinstance(item.source, HFSource)),
                               tuple(item for item in artifacts if isinstance(item.source, GeneratedSource)),
                               self.descriptor.preparation)

    def prepared(self):
        directory = self.root / 'prepared'
        return PreparedModel('user-name', self.recipe.id, self.recipe.identity,
                             self.recipe.pipeline, directory,
                             tuple(PreparedArtifact(item.id, item.role, item.format, item.path,
                                                    directory / item.path,
                                                    'generated-owned' if isinstance(item.source, GeneratedSource)
                                                    else 'hf-reference')
                                   for item in self.recipe.artifacts))

    def test_actual_recipe_descriptor_and_source_bundle(self):
        self.binding.validate_artifacts(self.recipe.artifacts)
        descriptor = asr.describe(self.binding)
        self.assertEqual(descriptor.protocol.operation, 'transcription')
        self.assertEqual(descriptor.protocol.request_encoding, 'multipart')
        self.assertEqual(descriptor.protocol.endpoint, '/v1/audio/transcriptions')
        self.assertEqual(descriptor.parameters['pending_requests'].value, 4)
        self.assertEqual(descriptor.parameters['options'].value, {})
        self.assertEqual(descriptor.telemetry.raw_events, 'unverified')
        self.assertEqual(descriptor.telemetry.metrics, ())
        with self.assertRaises(TypeError):
            descriptor.parameters['pending_requests'] = None
        paths = [entry.path for entry in descriptor.bundle.files]
        self.assertEqual(len(paths), len(set(paths)))
        self.assertIn('loom-pipeline/' + moonshine.CATALOG + '/recipes/moonshine-streaming-tiny.json', paths)
        self.assertIn('_native/share/loom-serve/preparers/moonshine/prepare_assets.py', paths)
        self.assertIn('_native/share/loom-serve/preparers/moonshine/provenance.json', paths)
        prefix = 'loom-pipeline/' + moonshine.CATALOG + '/'
        self.assertEqual({name[len(prefix):] for name in paths if name.startswith(prefix)}, {
            'prepare.loombc', 'control.loombc', 'providers.loombc', 'weights.loombc',
            'artifacts.json', 'catalog.json', 'sources.json', 'LICENSE.moonshine',
            'LICENSE-APACHE-2.0', 'recipes/moonshine-streaming-tiny.json',
        })
        self.assertTrue(descriptor.programs.prepare.endswith('.loom'))

    def test_selected_description_and_recipe_lookup_never_preflight_or_read_cache(self):
        with patch.object(helper, 'preflight', side_effect=AssertionError('not pure')), \
             patch.object(helper, 'prepare', side_effect=AssertionError('not pure')), \
             patch.object(model_download, 'cache_root', side_effect=AssertionError('not pure')):
            self.assertEqual(asr.describe(self.binding), self.descriptor)
            self.assertEqual(recipes.load(self.recipe.id), self.recipe)

    def test_fixed_assertions_exact_types_and_unknown_keys(self):
        for settings in ({}, {'pending_requests': 4}, {'options': {}}, dict(self.recipe.configuration)):
            asr.validate(settings, self.descriptor)
        cases = (
            ({'pending_requests': True}, 'expected integer'),
            ({'pending_requests': 4.0}, 'expected integer'),
            ({'pending_requests': '4'}, 'expected integer'),
            ({'pending_requests': 5}, 'expected fixed value 4'),
            ({'options': None}, 'expected object'),
            ({'options': {'language': 'en'}}, 'expected fixed value'),
            ({'options': {'x': float('nan')}}, 'finite JSON'),
            ({'rows': 4}, 'unsupported for modality asr'),
            ({'temperature': 0}, 'unsupported for modality asr'),
        )
        for settings, message in cases:
            with self.subTest(settings=settings), self.assertRaisesRegex(InvalidInput, message):
                asr.validate(settings, self.descriptor)

    def test_definition_is_recognition_and_uses_effective_settings_and_asset_directory(self):
        prepared = self.prepared()
        document = asr.service_definition(self.recipe.configuration, prepared, self.descriptor.programs)
        self.assertEqual(document, dict(kind='recognition', prepare='prepare.loom', control='control.loom',
                                       sources=list(self.descriptor.programs.sources),
                                       assets=str(prepared.directory), pending_requests=4, options={}))
        document['options']['mutated'] = True
        self.assertEqual(self.recipe.configuration['options'], {})
        with self.assertRaisesRegex(InvalidInput, 'effective'):
            asr.service_definition({}, prepared, self.descriptor.programs)

    def test_unknown_log_event_preserves_json_without_metrics(self):
        event = {'event': 'unknown', 'text': 'line\nnext', 'counters': {'audio_frames': 100}}
        view = asr.interpret(event, self.descriptor)
        self.assertEqual(json.loads(view.text), event)
        self.assertNotIn('\n', view.text)
        self.assertFalse(view.recognized)
        self.assertEqual(view.metrics, ())

    def test_fixed_inputs_and_generated_paths_refuse_unsupported_changes(self):
        original = self.recipe.artifacts
        changes = (
            replace(original[0], source=replace(original[0].source, revision='0' * 40)),
            replace(original[0], source=replace(original[0].source, filename='another.safetensors')),
            replace(original[0], source=replace(original[0].source, repo_id='other/model')),
            replace(original[0], format='gguf'),
            replace(original[0], path='nested/model.safetensors'),
            replace(original[0], role='checkpoint'),
            replace(original[0], id=original[1].id),
        )
        for changed in changes:
            with self.subTest(changed=changed), self.assertRaisesRegex(InvalidInput, 'unsupported-preparation'):
                self.binding.validate_artifacts((changed, *original[1:]))
        with self.assertRaisesRegex(InvalidInput, 'unsupported-preparation'):
            self.binding.validate_artifacts(original[:-1])
        with self.assertRaisesRegex(InvalidInput, 'unsupported-preparation'):
            self.binding.validate_artifacts((*original[:-1], replace(original[-1], path='other.safetensors')))

    def test_plan_preflight_is_explicit_and_bad_inputs_fail_before_helper(self):
        deadline = Deadline.after()
        with patch.object(helper, 'preflight') as preflight:
            self.binding.preflight(self.plan, deadline)
            preflight.assert_called_once_with(deadline=deadline)
            preflight.reset_mock()
            bad = replace(self.plan, inputs=(replace(self.plan.inputs[0], format='gguf'), *self.plan.inputs[1:]))
            with self.assertRaisesRegex(InvalidInput, 'unsupported-preparation'):
                self.binding.preflight(bad, deadline)
            preflight.assert_not_called()

    def test_prepare_uses_explicit_artifact_ids_not_recipe_name(self):
        artifacts = tuple(replace(item, id='custom-' + str(index))
                          for index, item in enumerate(self.recipe.artifacts))
        plan = self.plan_for(artifacts)
        resolved = {item.id: self.root / item.path for item in plan.inputs}
        attempt = self.root / 'attempt'
        deadline = Deadline.after()
        expected = (GeneratedArtifact('rotary', 'safetensors', 'rotary.safetensors', attempt / 'output/rotary.safetensors'),)
        with patch.object(helper, 'prepare', return_value=expected) as prepare:
            self.assertEqual(self.binding.prepare(plan, resolved, attempt, deadline), expected)
            prepare.assert_called_once_with({item.path: resolved[item.id] for item in plan.inputs}, attempt, deadline)
            with self.assertRaisesRegex(InvalidInput, 'resolved Moonshine inputs'):
                self.binding.prepare(plan, {}, attempt, deadline)

    def compiled_catalog(self):
        for path in self.catalog.glob('*.loom'):
            path.unlink()
        document = json.loads((self.catalog / 'catalog.json').read_text())
        document['definition'].update(prepare='prepare.loombc', control='control.loombc', sources=['providers.loombc'])
        (self.catalog / 'catalog.json').write_text(json.dumps(document))
        names = ('prepare.loombc', 'control.loombc', 'providers.loombc', 'weights.loombc')
        for name in names:
            (self.catalog / name).write_bytes(b'LOOM fixture ' + name.encode())
        manifest = dict(format=1, policy='weights.loombc',
                        files={name: hashlib.sha256((self.catalog / name).read_bytes()).hexdigest() for name in names},
                        roots={name: {} for name in names})
        (self.catalog / 'artifacts.json').write_text(json.dumps(manifest))

    def test_compiled_program_integrity_and_full_inventory(self):
        source_bundle = self.descriptor.bundle
        self.compiled_catalog()
        descriptor = self.binding.describe()
        self.assertEqual(descriptor.bundle, source_bundle)
        self.assertEqual(descriptor.programs, RecognitionPrograms('prepare.loombc', 'control.loombc', ('providers.loombc',)))
        (self.catalog / 'control.loombc').write_bytes(b'LOOM tampered')
        with self.assertRaisesRegex(HemError, 'integrity mismatch'):
            self.binding.describe()

    def test_compiled_manifest_required_and_extra_executable_refused(self):
        self.compiled_catalog()
        (self.catalog / 'extra.loombc').write_bytes(b'LOOM extra')
        with self.assertRaisesRegex(HemError, 'Undeclared'):
            self.binding.describe()
        (self.catalog / 'extra.loombc').unlink()
        (self.catalog / 'artifacts.json').unlink()
        with self.assertRaisesRegex(HemError, 'integrity metadata'):
            self.binding.describe()

    def test_catalog_refuses_old_policy_fields_and_program_escape(self):
        path = self.catalog / 'catalog.json'
        document = json.loads(path.read_text())
        document['definition']['pending_requests'] = 4
        path.write_text(json.dumps(document))
        with self.assertRaisesRegex(HemError, 'only recognition program'):
            self.binding.describe()
        del document['definition']['pending_requests']
        document['definition']['prepare'] = '../escape.loom'
        path.write_text(json.dumps(document))
        with self.assertRaisesRegex(HemError, 'normalized relative'):
            self.binding.describe()

    def test_native_manifest_requires_exact_boolean(self):
        path = self.catalog / 'catalog.json'
        document = json.loads(path.read_text())
        document['assets'][0]['required'] = 1
        path.write_text(json.dumps(document))
        with self.assertRaisesRegex(HemError, 'asset manifest'):
            self.binding.describe()


class MoonshineHelperContractTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.inputs = {}
        for name in helper.INPUT_FILES:
            path = self.root / 'cache' / name
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(b'opaque immutable input')
            self.inputs[name] = path
        # The acquisition owner supplies this typed shared locator primitive.
        # This fixture checks the exact source handed to that peer interface;
        # real combined cache lookup is a later composition gate.
        def locate(source):
            self.assertIsInstance(source, HFSource)
            self.assertEqual(source.repo_id, helper.INPUT_REPOSITORY)
            self.assertEqual(source.revision, helper.INPUT_REVISION)
            return self.inputs[source.filename]
        cached = patch.object(model_download, '_cache_path', side_effect=locate)
        cached.start(); self.addCleanup(cached.stop)

    def attempt(self, name):
        path = self.root / name
        path.mkdir(mode=0o700)
        return path

    def complete(self, arguments, *, missing=None, alias=False, **kwargs):
        output = Path(arguments[arguments.index('--output') + 1])
        self.assertFalse(output.exists())
        self.assertEqual(arguments[-1], '--offline')
        output.mkdir()
        for name in (*helper.INPUT_FILES, 'rotary.safetensors', 'provenance.json', 'model.json'):
            if name == missing:
                continue
            if name == 'rotary.safetensors' and alias:
                (output / name).symlink_to(self.inputs['model.safetensors'])
            else:
                (output / name).write_bytes(b'opaque completion, not a launch document')
        return 0

    def test_explicit_inputs_path_attempt_and_generated_tuple(self):
        attempt = self.attempt('one')
        with patch.object(helper, '_run', side_effect=self.complete):
            result = helper.prepare(self.inputs, attempt, Deadline.after())
        self.assertEqual(result, (GeneratedArtifact('rotary', 'safetensors', 'rotary.safetensors',
                                                    attempt / 'output/rotary.safetensors'),))
        self.assertTrue(json.loads((attempt / 'receipt.json').read_text())['completed'])
        self.assertTrue((attempt / 'stdout.log').is_file())
        self.assertTrue((attempt / 'stderr.log').is_file())

    def test_failed_completion_retains_evidence_and_new_attempt_can_succeed(self):
        first, second = self.attempt('one'), self.attempt('two')
        with patch.object(helper, '_run', side_effect=lambda *args, **kwargs: self.complete(*args, missing='model.json', **kwargs)):
            with self.assertRaisesRegex(HemError, 'did not complete'):
                helper.prepare(self.inputs, first, Deadline.after())
        self.assertFalse(json.loads((first / 'receipt.json').read_text())['completed'])
        with patch.object(helper, '_run', side_effect=self.complete):
            self.assertEqual(helper.prepare(self.inputs, second, Deadline.after())[0].file.parent,
                             second / 'output')

    def test_generated_alias_refused_without_completion_proof(self):
        attempt = self.attempt('one')
        with patch.object(helper, '_run', side_effect=lambda *args, **kwargs: self.complete(*args, alias=True, **kwargs)):
            with self.assertRaisesRegex(HemError, 'aliased'):
                helper.prepare(self.inputs, attempt, Deadline.after())
        self.assertFalse(json.loads((attempt / 'receipt.json').read_text())['completed'])

    def test_missing_or_foreign_resolved_input_refuses_before_spawn(self):
        for name, inputs in (('one', {}), ('two', dict(self.inputs, **{'model.safetensors': self.root / 'foreign'}))):
            attempt = self.attempt(name)
            with self.subTest(name=name), patch.object(helper, '_run') as run:
                with self.assertRaisesRegex(HemError, 'pinned'):
                    helper.prepare(inputs, attempt, Deadline.after())
                run.assert_not_called()

    def test_preflight_uses_supplied_budget_capped_at_existing_limit(self):
        class Budget:
            def remaining(self):
                return 2.5
        with patch.object(helper, 'TOOL_FILES', {}), patch.object(helper, '_run', return_value=0) as run:
            helper.preflight(deadline=Budget())
        self.assertEqual(run.call_args.kwargs['timeout'], 2.5)


if __name__ == '__main__':
    unittest.main()
