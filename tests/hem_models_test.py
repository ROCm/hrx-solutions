"""Actual format3 recipe/executable policy with controlled cache and helper effects."""
from contextlib import redirect_stderr
from copy import deepcopy
import io
import json
import os
import shutil
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from binding.python.control import hem_state
from binding.python.models import hem_models as models, model_download, recipes, pipelines
from binding.python.models.preparation import moonshine as preparation
from binding.python.hem_common import Deadline, HemError
from tests.gguf_fixtures import write_gguf


class AcquisitionFixture:
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name); self.cache = self.root / 'cache'
        self.environment = patch.dict(os.environ, {'HF_HUB_CACHE': str(self.cache), 'HF_HUB_OFFLINE': '0'})
        self.environment.start(); self.addCleanup(self.environment.stop)
        self.registry = hem_state.Registry(self.root / 'state')
        self.declarations = {r['id']: r for r, _ in recipes.list()}
        self.recipe = self.declarations['qwen35-08b-q5-k-xl']
        self.network = patch.object(model_download, '_run_hf', side_effect=AssertionError('unexpected network'))
        self.fetch = self.network.start(); self.addCleanup(self.network.stop)
        self.events = []
        preflight = patch.object(preparation, 'preflight', side_effect=lambda deadline: self.events.append('preflight'))
        preflight.start(); self.addCleanup(preflight.stop)
        def prepare(inputs, attempt, deadline):
            self.events.append('prepare'); deadline.remaining()
            self.assertEqual(set(inputs), {'model.safetensors','tokenizer.json','config.json'})
            path = attempt / 'output/rotary.safetensors'; path.parent.mkdir(); path.write_bytes(b'generated')
            return {'rotary.safetensors': path}
        generate = patch.object(preparation, 'prepare', side_effect=prepare)
        generate.start(); self.addCleanup(generate.stop)
        self.output = redirect_stderr(io.StringIO())
        self.output.__enter__(); self.addCleanup(self.output.__exit__, None, None, None)
        self.gguf = self.root / 'input.gguf'; write_gguf(self.gguf, tensors=[('tiny',[1],0,0)])
        self.tokenizer = self.root / 'tokenizer.json'; self.tokenizer.write_text('{"model":{"fixture":true}}')

    def cached(self, recipe=None):
        for item in (recipe or self.recipe)['files'].values():
            if item['source']['kind'] == 'huggingface':
                path = model_download._cache_path(item['source'])
                path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(b'opaque HF input')

    def acquire(self, name='tiny', **kwargs):
        kwargs.setdefault('recipe_id', self.recipe['id'])
        return models.import_model(self.registry, name, ggufs=[self.gguf], tokenizer=self.tokenizer, **kwargs)

    def download(self, name='downloaded', **kwargs):
        return models.download(self.registry, self.recipe['id'], model_id=name, **kwargs)

    def inspect(self, name, executable=None, modality=None):
        with self.registry.lock('model', name):
            return models.qualify(self.registry, name, executable=executable, modality=modality)


class HemModelsTest(AcquisitionFixture, unittest.TestCase):
    def test_broken_programs_fail_before_model_effects_and_passive_reads_do_not_write(self):
        self.acquire('existing')
        source = next(root for root in pipelines.catalog_roots() if root.is_dir())
        target = self.root/'broken-programs'; shutil.copytree(source,target)
        catalog = target/'llm/qwen/qwen_08b'
        declaration = json.loads((catalog/'executable.json').read_text())
        program = target/declaration['programs']['root']/declaration['programs']['control']
        program.unlink()
        prepared = self.root/'prepared'; prepared.mkdir()
        for item in self.recipe['files'].values(): (prepared/item['path']).write_bytes(b'local')
        with patch.object(pipelines,'catalog_roots',return_value=(target,)), \
             patch.object(self.registry,'prepare_model',side_effect=AssertionError('effect before admission')):
            for effect in (lambda:self.download(),
                           lambda:models.download(None,self.recipe['id'],model_id='preview',dry_run=True),
                           lambda:self.acquire('new'),
                           lambda:models.import_assets(self.registry,'new',directory=prepared,recipe_id=self.recipe['id']),
                           models.available):
                with self.subTest(effect=effect), self.assertRaises(HemError): effect()
            before = self.registry.get_model('existing')['validation']
            self.assertFalse(models.show(self.registry,'existing')['accessible'])
            self.assertFalse(models.inventory(self.registry)[0]['accessible'])
            self.assertEqual(self.registry.get_model('existing')['validation'],before)
            with self.assertRaises(HemError): models.validate(self.registry,'existing')
            self.assertFalse(self.registry.get_model('existing')['validation']['valid'])
        self.assertEqual(len(self.registry.list_models()),1)
        self.fetch.assert_not_called(); self.assertEqual(self.events,[])

    def test_program_admission_is_once_per_operation_under_original_deadline(self):
        budget = Deadline.after(30); self.registry.deadline = budget
        with patch.object(pipelines,'programs',wraps=pipelines.programs) as programs:
            self.cached(); self.download()
            self.assertEqual(programs.call_count,1)
            self.assertIs(programs.call_args.args[2],budget)
            self.inspect('downloaded')
            self.assertEqual(programs.call_count,2)
            self.acquire('local')
            programs.reset_mock()
            self.assertEqual(len(models.inventory(self.registry)),2)
            self.assertEqual(programs.call_count,1)
            self.assertIs(programs.call_args.args[2],budget)
        expired = Deadline(0,clock=lambda:1)
        with patch.object(self.registry,'prepare_model',side_effect=AssertionError('reservation')):
            with self.assertRaises(HemError):
                models.download(self.registry,self.recipe['id'],model_id='expired',deadline=expired)

    def test_actual_lifecycle_qualify_keywords_and_original_deadline(self):
        from binding.python.control.hem_instances import InstanceManager
        from binding.python.models import hem_config
        self.acquire()
        manager = InstanceManager(self.registry, processes=object())
        reference = self.recipe['executable']
        for assertions in ({'executable': None, 'modality': None},
                           {'executable': reference, 'modality': 'chat'}):
            budget = Deadline.after(30)
            with patch.object(pipelines, 'programs', wraps=pipelines.programs) as programs, \
                 patch.object(hem_config, 'resolve', side_effect=RuntimeError('qualified; stop before native')) as resolve:
                with self.assertRaisesRegex(RuntimeError, 'qualified; stop before native'):
                    manager.start({'id': 'unreached'}, 'tiny', budget, **assertions)
                self.assertEqual(programs.call_count, 1)
                self.assertIs(programs.call_args.args[2], budget)
                resolve.assert_called_once()
            paths, _, _ = models.inspect_record(self.registry.get_model('tiny'), **assertions)
            self.assertEqual(set(paths), {'weights', 'tokenizer'})
        for assertions in ({'executable': 'llm/qwen/qwen_27b/executable.json', 'modality': None},
                           {'executable': None, 'modality': 'asr'}):
            with patch.object(self.registry, 'record_validation', side_effect=AssertionError('validation effect')) as writes, \
                 patch.object(hem_config, 'resolve', side_effect=AssertionError('resolve')) as resolve:
                with self.assertRaises(HemError):
                    manager.start({'id': 'unreached'}, 'tiny', Deadline.after(30), **assertions)
                writes.assert_not_called(); resolve.assert_not_called()
            self.assertEqual(self.registry.get_model('tiny')['validation']['valid'], True)

    def test_real_association_drift_persists_failure_with_or_without_matching_assertions(self):
        from binding.python.models.preparation import moonshine_contract as operation
        source = next(root for root in pipelines.catalog_roots() if root.is_dir())
        original_chat = self.recipe
        for kind in ('recipe', 'executable', 'operation', 'modality'):
            target = self.root / ('drift-' + kind); shutil.copytree(source, target)
            recipe = (self.declarations['moonshine-streaming-tiny'] if kind == 'operation' else original_chat)
            with patch.object(pipelines, 'catalog_roots', return_value=(target,)):
                prepared = self.root / ('prepared-' + kind); prepared.mkdir()
                for item in recipe['files'].values(): (prepared / item['path']).write_bytes(b'prepared fixture')
                names = [kind + '-plain', kind + '-asserted']
                for name in names:
                    models.import_assets(self.registry, name, directory=prepared, recipe_id=recipe['id'])
                contract = operation.load_operation()
                if kind == 'recipe':
                    path = next(target.glob(f"**/recipes/{recipe['id']}.json"))
                    path.write_bytes(path.read_bytes() + b'\n')
                elif kind == 'executable':
                    path = target / recipe['executable']; path.write_bytes(path.read_bytes() + b'\n')
                elif kind == 'modality':
                    path = target / recipe['executable']; declaration = json.loads(path.read_bytes())
                    declaration['modality'] = 'asr'; declaration['native_contract'] = 'recognition-v1'
                    declaration['serving'] = {'pending_requests': [1, 8],
                                              'defaults': {'pending_requests': 2, 'options': {}}}
                    for purpose, asset in declaration['assets'].items():
                        asset['path'] = recipe['files'][purpose]['path']
                    path.write_text(json.dumps(declaration))
                    pipelines.load(recipe['executable'])  # Valid metadata drift, no native ASR claim.
                else:
                    contract['tool_revision'] = '0' * 40
                operation_path = self.root / ('operation-' + kind + '.json')
                operation_path.write_text(json.dumps(contract))
                modality = 'asr' if kind == 'operation' else 'chat'
                with patch.object(operation, 'operation_path', return_value=operation_path):
                    for name, assertions in zip(names, ({}, {'executable': recipe['executable'], 'modality': modality})):
                        with self.subTest(kind=kind, assertions=assertions), \
                             self.registry.lock('model', name), self.assertRaisesRegex(HemError, 'association'):
                            models.qualify(self.registry, name, **assertions)
                        record = self.registry.get_model(name)
                        self.assertFalse(record['validation']['valid'])
                        self.assertEqual(record['pending'], {'stage': 'published-invalid'})
                        self.assertIsNone(record['pipeline'])
        self.assertEqual(self.events, []); self.fetch.assert_not_called()

    def test_actual_instance_export_uses_canonical_qualification_selectors(self):
        from binding.python import hem_cli
        from binding.python.cli import hem_instance_cli
        self.acquire()
        for index, options in enumerate(([], ['--executable', self.recipe['executable'], '--modality', 'chat'])):
            output = self.root / f'export-{index}.json'
            args = hem_cli.parse_args(['instance', 'export', 'tiny', '--output', str(output), *options])
            budget = Deadline.after(30); self.registry.deadline = budget
            events = []
            actual_programs = pipelines.programs
            def admit(*values):
                events.append('programs'); return actual_programs(*values)
            def registry(**values):
                events.append('Registry'); self.assertIs(values['deadline'], budget); return self.registry
            with patch.object(hem_instance_cli, 'Registry', side_effect=registry), \
                 patch.object(pipelines, 'programs', side_effect=admit) as programs:
                result = hem_instance_cli.run(args, budget)
            self.assertTrue(result['completed'])
            self.assertTrue(output.is_file())
            # Explicit program preflight precedes Registry; qualification and
            # checked interpretation each retain their own fresh admission.
            expected = (['programs'] if options else []) + ['Registry', 'programs', 'programs']
            self.assertEqual(events, expected)
            self.assertEqual(programs.call_count, len(expected) - 1)
            self.assertTrue(all(call.args[2] is budget for call in programs.call_args_list))
        output = self.root / 'wrong-export.json'
        args = hem_cli.parse_args(['instance', 'export', 'tiny', '--output', str(output), '--modality', 'asr'])
        with patch.object(hem_instance_cli, 'Registry', return_value=self.registry), \
             patch.object(self.registry, 'record_validation', side_effect=AssertionError('validation write')) as writes:
            with self.assertRaises(HemError): hem_instance_cli.run(args, Deadline.after(30))
            writes.assert_not_called()
        self.assertFalse(output.exists())

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
                                     recipe_id=self.recipe['id'], alias=True)
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
                                recipe_id=self.recipe['id'])
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
            self.inspect('tiny')
        self.assertFalse(models.show(self.registry, 'tiny')['accessible'])
        target.unlink(); target.symlink_to(self.gguf)
        models.validate(self.registry, 'tiny')
        self.assertEqual(models.show(self.registry, 'tiny')['state'], 'ready')

    def test_drift_during_presence_check_is_rejected(self):
        self.acquire()
        evidence = models._evidence
        def drift(files):
            value = evidence(files)
            files[0].write_bytes(b'new')
            return value
        with patch.object(models, '_evidence', side_effect=drift):
            with self.assertRaisesRegex(HemError, 'changed during'):
                models.validate(self.registry, 'tiny')

    def test_all_three_authored_recipes_drive_the_same_acquisition(self):
        for recipe in self.declarations.values():
            self.cached(recipe)
            result = models.download(self.registry, recipe['id'], model_id=recipe['id'], offline=True)
            self.assertEqual(result['provenance']['recipe_id'], recipe['id'])
            self.assertEqual(result['pipeline'], recipe['executable'])
            record, paths, dependencies, evidence = self.inspect(recipe['id'])
            self.assertEqual(record['provenance']['recipe_identity'], recipes.load(recipe['id'])[1])
            self.assertEqual(set(paths), set(recipe['files']))
            self.assertEqual([a['logical_path'] for a in record['assets']], [a['path'] for a in recipe['files'].values()])
        self.assertEqual(self.events, ['preflight','prepare'])
        self.fetch.assert_not_called()

    def test_offline_preserves_record_and_retry_uses_cache(self):
        with self.assertRaisesRegex(HemError, 'Offline cache is missing'): self.download(offline=True)
        self.assertEqual(models.show(self.registry, 'downloaded')['state'], 'incomplete')
        self.cached(); answer = self.download(offline=True)
        self.assertEqual(answer['state'], 'ready'); self.fetch.assert_not_called()
        missing = model_download._cache_path(self.recipe['files']['weights']['source']); missing.unlink()
        self.assertEqual(models.show(self.registry, 'downloaded')['state'], 'missing')
        self.fetch.side_effect = lambda command: missing.write_bytes(b'restored')
        self.download(); self.fetch.assert_called_once()

    def test_dryrun_never_initializes_state_or_executes_hf(self):
        with patch.object(hem_state, 'Registry', side_effect=AssertionError('state')):
            answer = models.download(None, self.recipe['id'], model_id='preview', dry_run=True)
        self.assertEqual(answer['state'], 'preview')
        self.assertEqual([a['logical_path'] for a in answer['artifacts']], ['model.gguf','tokenizer.json'])
        self.fetch.assert_not_called(); self.assertFalse(self.cache.exists())

    def test_download_passes_exact_pinned_command_and_preserves_partial_cache(self):
        calls = []
        def fetch(command):
            calls.append(command)
            item = next(a for a in self.recipe['files'].values() if a['source']['filename'] == command[3])
            path = model_download._cache_path(item['source']); path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'opaque')
            if len(calls) == 2: raise models.ModelImportError('interrupted')
        self.fetch.side_effect = fetch
        with self.assertRaisesRegex(HemError, 'interrupted'): self.download()
        self.assertEqual(calls[0], model_download.command(self.recipe['files']['weights']['source']))
        self.fetch.reset_mock(); self.download(); self.fetch.assert_not_called()
        _, paths, _, _ = self.inspect('downloaded')
        for purpose, declaration in self.recipe['files'].items():
            self.assertTrue(paths[purpose].is_symlink())
            self.assertEqual(os.readlink(paths[purpose]), str(model_download._cache_path(declaration['source'])))

    def test_consumer_rechecks_even_saved_valid_and_recipe_identity(self):
        self.cached(); self.download()
        with patch.object(recipes, 'load', return_value=(deepcopy(self.recipe), '0'*64)):
            with self.assertRaisesRegex(HemError, 'association'): self.inspect('downloaded')
            with self.assertRaisesRegex(HemError, 'different mappings'): self.download()
        with self.assertRaisesRegex(HemError, 'association'):
            self.inspect('downloaded', executable='other')

    def test_executable_byte_identity_drift_is_not_requalified_or_overwritten(self):
        source = next(root for root in pipelines.catalog_roots() if root.is_dir())
        target = self.root / 'identity-catalog'; shutil.copytree(source, target)
        with patch.object(pipelines, 'catalog_roots', return_value=(target,)):
            self.acquire()
            before = self.registry.get_model('tiny')
            manifest = target / self.recipe['executable']
            manifest.write_bytes(manifest.read_bytes() + b'\n')
            self.assertFalse(models.show(self.registry, 'tiny')['accessible'])
            self.assertEqual(before, self.registry.get_model('tiny'))
            with self.assertRaisesRegex(HemError, 'association'):
                self.inspect('tiny')
            with self.assertRaisesRegex(HemError, 'different mappings'):
                self.acquire()
            self.assertEqual(self.registry.get_model('tiny')['provenance'], before['provenance'])

    def test_operation_snapshot_drift_cannot_requalify_prepared_association(self):
        from binding.python.models.preparation import moonshine_contract as operation
        recipe = self.declarations['moonshine-streaming-tiny']
        prepared = self.root / 'prepared-operation'; prepared.mkdir()
        for item in recipe['files'].values(): (prepared / item['path']).write_bytes(b'prepared')
        models.import_assets(self.registry, 'speech', directory=prepared, recipe_id=recipe['id'])
        before = self.registry.get_model('speech')
        contract = operation.load_operation(); contract['tool_revision'] = '0' * 40
        path = self.root / 'operation.json'; path.write_text(json.dumps(contract))
        with patch.object(operation, 'operation_path', return_value=path):
            self.assertFalse(models.show(self.registry, 'speech')['accessible'])
            self.assertEqual(before, self.registry.get_model('speech'))
            with self.assertRaisesRegex(HemError, 'association'):
                models.inspect_record(before)
            with self.assertRaisesRegex(HemError, 'different mappings'):
                models.import_assets(self.registry, 'speech', directory=prepared, recipe_id=recipe['id'])
        self.assertEqual(self.events, []); self.fetch.assert_not_called()

    def test_declaration_only_chat_and_nonpreparing_asr_onboarding_acquisition(self):
        """Catalog DATA plus tiny cached bytes; no native/model inference claim."""
        from binding.python.models.preparation import moonshine_contract as operation
        source = next(root for root in pipelines.catalog_roots() if root.is_dir())
        target = self.root / 'onboarding-catalog'; shutil.copytree(source, target)
        cases = []
        for original in (self.recipe, self.declarations['moonshine-streaming-tiny']):
            recipe = deepcopy(original)
            manifest = json.loads((target / original['executable']).read_text())
            manifest['id'] += '-prepared-fixture'
            manifest['preparation'] = None
            reference = str(Path(original['executable']).parent / 'fixture/executable.json')
            recipe['executable'] = reference; recipe['id'] += '-fixture'
            for purpose, item in recipe['files'].items():
                if item['source']['kind'] == 'generated':
                    item['source'] = dict(kind='huggingface', repo_id='fixture/prepared',
                                          revision='0' * 40, filename=item['path'])
            catalog = (target / reference).parent; (catalog / 'recipes').mkdir(parents=True)
            (target / reference).write_text(json.dumps(manifest))
            (catalog / 'recipes' / (recipe['id'] + '.json')).write_text(json.dumps(recipe))
            self.cached(recipe); cases.append((recipe, manifest))
        with patch.object(pipelines, 'catalog_roots', return_value=(target,)), \
             patch.object(operation, 'operation_path', side_effect=AssertionError('no operation selected')):
            for recipe, manifest in cases:
                row = models.download(self.registry, recipe['id'], model_id=recipe['id'], offline=True)
                self.assertEqual(row['state'], 'ready')
                self.assertEqual(row['executable'], recipe['executable'])
                self.assertEqual(row['executable_id'], manifest['id'])
                self.assertEqual(row['modality'], manifest['modality'])
                self.assertIsNone(row['provenance']['preparation'])
        self.assertEqual(self.events, []); self.fetch.assert_not_called()

    def test_nonpreparing_selection_never_reads_unrelated_operation_contract(self):
        from binding.python.models.preparation import moonshine_contract as operation
        with patch.object(operation, 'operation_path', side_effect=AssertionError('unselected operation')):
            self.acquire()
            self.cached(); self.download()
            self.assertTrue(models.show(self.registry, 'tiny')['accessible'])
        self.assertEqual(self.events, []); self.fetch.assert_not_called()

    def test_unsupported_admission_precedes_cache_registry_and_process(self):
        with patch.object(pipelines, 'validate_files', side_effect=HemError('unsupported')), \
             patch.object(model_download, '_cache_path', side_effect=AssertionError('cache')):
            with self.assertRaisesRegex(HemError, 'unsupported'): self.download()
        self.assertEqual(self.registry.list_models(), [])

    def test_explicit_destinations_never_use_basename(self):
        data = deepcopy(self.recipe); data['files']['weights']['path'] = 'nested/runtime.gguf'
        data['files']['tokenizer']['path'] = 'other/tokens.json'
        recipes_root = self.root/'catalog'
        source = next(root for root in pipelines.catalog_roots() if root.is_dir())
        shutil.copytree(source, recipes_root)
        selected = next(recipes_root.glob(f"**/recipes/{data['id']}.json"))
        selected.write_text(json.dumps(data))
        self.cached(data)
        with patch.object(pipelines, 'catalog_roots', return_value=(recipes_root,)):
            self.download()
        root = Path(self.registry.get_model('downloaded')['directory'])
        self.assertTrue((root/'nested/runtime.gguf').is_symlink())
        self.assertTrue((root/'other/tokens.json').is_symlink())
        self.assertFalse((root/self.recipe['files']['weights']['source']['filename']).exists())

    def test_bookkeeping_collisions_rejected_before_state(self):
        for paths in (['.loom-acquisition.json'], ['model.gguf','.model.gguf.partial']):
            with self.assertRaisesRegex(HemError, 'bookkeeping'): models._admit_paths(paths)
        self.assertEqual(self.registry.list_models(), [])

    def test_explicit_import_ignores_unrelated_broken_bindings(self):
        selected = self.recipe['executable']; calls = []; load = pipelines.load
        def only_selected(key):
            calls.append(key)
            if key != selected: raise HemError('unrelated binding unavailable')
            return load(key)
        with patch.object(pipelines, 'load', side_effect=only_selected):
            self.acquire()
            directory = self.root/'complete'; directory.mkdir()
            for artifact in self.recipe['files'].values(): (directory/artifact['path']).write_bytes(b'local input')
            row = models.import_assets(self.registry, 'complete', directory=directory, recipe_id=self.recipe['id'])
            self.assertTrue(row['validation']['valid'])
            self.assertEqual(self.inspect('complete')[0]['pipeline'], selected)
        self.assertTrue(calls); self.assertEqual(set(calls), {selected})

    def test_selected_public_imports_ignore_malformed_unrelated_recipes(self):
        source = next(root for root in pipelines.catalog_roots() if root.is_dir())
        target = self.root/'selected-catalog'
        shutil.copytree(source, target)
        for original in source.glob('**/recipes/*.json'):
            copy = target/original.relative_to(source); copy.parent.mkdir(parents=True,exist_ok=True)
            copy.write_bytes(original.read_bytes())
            if original.stem != self.recipe['id']:
                copy.write_text('{broken unrelated JSON')
        prepared = self.root/'complete'; prepared.mkdir()
        for item in self.recipe['files'].values(): (prepared/item['path']).write_bytes(b'local')
        with patch.object(pipelines,'catalog_roots',return_value=(target,)):
            self.assertEqual(self.acquire('selected')['state'],'ready')
            row = models.import_assets(self.registry,'prepared',directory=prepared,recipe_id=self.recipe['id'])
            self.assertEqual(row['state'],'ready')
            with self.assertRaises(HemError): models.available()
        self.assertEqual(self.events,[]); self.fetch.assert_not_called()

    def test_selected_asr_prepared_import_ignores_malformed_unrelated_recipe(self):
        recipe = self.declarations['moonshine-streaming-tiny']
        source = next(root for root in pipelines.catalog_roots() if root.is_dir())
        target = self.root/'selected-asr-catalog'
        shutil.copytree(source, target)
        for original in source.glob('**/recipes/*.json'):
            copy = target/original.relative_to(source); copy.parent.mkdir(parents=True,exist_ok=True)
            copy.write_bytes(original.read_bytes() if original.stem == recipe['id'] else b'[]')
        prepared = self.root/'complete-asr'; prepared.mkdir()
        for item in recipe['files'].values(): (prepared/item['path']).write_bytes(b'local')
        with patch.object(pipelines,'catalog_roots',return_value=(target,)):
            row = models.import_assets(self.registry,'asr',directory=prepared,recipe_id=recipe['id'])
            self.assertEqual(row['state'],'ready')
            with self.assertRaises(HemError): models.available()
        self.assertEqual(self.events,[]); self.fetch.assert_not_called()

    def test_complete_support_discovery_still_rejects_unrelated_broken_binding(self):
        load = pipelines.load
        def broken(key):
            if key != self.recipe['executable']: raise HemError('unrelated binding unavailable')
            return load(key)
        with patch.object(pipelines, 'load', side_effect=broken):
            with self.assertRaisesRegex(HemError, 'unrelated binding unavailable'): models.available()

    def test_explicit_import_rejects_broken_selected_binding_before_state(self):
        with patch.object(pipelines, 'load', side_effect=HemError('selected binding unavailable')):
            with self.assertRaisesRegex(HemError, 'selected binding unavailable'):
                self.acquire()
            with self.assertRaisesRegex(HemError, 'selected binding unavailable'):
                models.import_assets(self.registry, 'complete', directory=self.root, recipe_id=self.recipe['id'])
        self.assertEqual(self.registry.list_models(), [])

    def test_import_requires_explicit_recipe(self):
        with self.assertRaisesRegex(HemError, 'recipe'):
            models.import_model(self.registry, 'tiny', ggufs=[self.gguf], tokenizer=self.tokenizer)

    def test_invalid_fragment_format_retains_copied_assets_for_inspection(self):
        other = self.root / 'duplicate.gguf'; other.write_bytes(self.gguf.read_bytes())
        with self.assertRaisesRegex(HemError, 'Invalid fragment format'):
            models.import_model(self.registry, 'split', ggufs=[self.gguf, other],
                                tokenizer=self.tokenizer, recipe_id=self.recipe['id'])
        row = self.registry.get_model('split')
        self.assertEqual(row['pending']['stage'], 'staging')
        self.assertTrue((Path(row['pending']['path']) / 'model-00002-of-00002.gguf').is_file())
        self.assertFalse(Path(row['directory']).exists())

    def test_bad_fragment_mapping_retained_for_inspection(self):
        with self.assertRaises(HemError):
            models.import_model(self.registry, 'split', ggufs=[self.gguf, self.gguf],
                                tokenizer=self.tokenizer, recipe_id=self.recipe['id'])
        self.assertEqual(self.registry.list_models(), [])

    def test_fragment_index_is_scoped_and_has_native_suffix(self):
        files = []
        for i in range(2):
            path = self.root / f'fragment-{i}.gguf'
            write_gguf(path, tensors=[(f'tensor-{i}', [1], 0, 0)])
            files.append(path)
        row = models.import_model(self.registry, 'split', ggufs=files, tokenizer=self.tokenizer,
                                  recipe_id=self.recipe['id'])
        _, paths, dependencies, _ = self.inspect('split')
        index = paths['weights']
        self.assertEqual(len(dependencies), 4)
        self.assertEqual(index.suffixes[-2:], ['.index', '.json'])
        self.assertFalse(index.is_symlink())
        self.assertTrue(json.loads(index.read_text())['weight_map'])
        self.assertEqual(models.show(self.registry, 'split')['state'], 'ready')
