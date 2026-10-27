"""Direct configuration boundary; policy doubles do not qualify native execution."""
from contextlib import contextmanager
from dataclasses import replace
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

from binding.python.models import hem_config, hem_models, pipelines, recipes, modalities
from binding.python.models.modalities import chat
from binding.python.control.hem_instances import InstanceManager
from binding.python.cli import hem_instance_cli
from binding.python.hem_common import Deadline, DeadlineExpired, HemError, InvalidInput


class PlainConfigurationTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.catalog_root = self.root / 'loom-pipeline'
        self.catalog = self.catalog_root / 'examples/chat'
        self.catalog.mkdir(parents=True)
        (self.catalog / 'program.loom').write_text('program')
        self.directory = self.root / 'model'
        self.directory.mkdir()
        self.paths = {}
        assets = []
        for purpose, name in [('weights', 'model.gguf'), ('tokenizer', 'tokenizer.json')]:
            path = self.directory / name
            path.write_bytes(b'fixture; format admission belongs to inspect_record')
            self.paths[purpose] = path
            assets.append(dict(id=purpose, role=purpose, logical_path=name))
        self.dependencies = tuple(self.paths.values())
        self.evidence = self.stats(self.dependencies)
        self.reference = 'examples/chat/executable.json'
        self.serving = {'bounds':{'rows':[1,16], 'context_capacity':[1,262144],
                                  'pool_capacity':[1,4194304], 'prefill_capacity':[1,262144]},
                        'mtp_depth_max':3, 'defaults':dict(rows=2,context_capacity=8192,
                            pool_capacity=32768,prefill_capacity=128,mtp=True,mtp_depth=3)}
        self.executable = dict(id='example-chat', modality='chat', native_contract='text-v1',
                               serving=self.serving, assets={}, preparation=None)
        self.document = dict(format=3, id='fixture', executable=self.reference, name='Fixture',
                             files={key:dict(path=path.name) for key,path in self.paths.items()},
                             configuration={'rows':4})
        self.record = dict(id='local', directory=str(self.directory), pipeline=self.reference, assets=assets,
            provenance=dict(recipe_id='fixture', recipe_identity='a'*64, declaration=self.document,
                pipeline=self.reference, executable=dict(reference=self.reference,id='example-chat',
                    identity='b'*64, declaration=deepcopy(self.executable))))
        self.policy = Mock(wraps=chat)
        self.policy.PROTOCOL = chat.PROTOCOL
        self.policy.definition.return_value = dict(kind='text', rows=4, weights=str(self.paths['weights']))
        self.programs = Mock(return_value=dict(prepare='program.loom',control='program.loom',sources=['program.loom']))
        for target, attribute, value in [(recipes,'load',(self.document,'a'*64)),
                (pipelines,'load',(self.executable,self.catalog,'b'*64)),
                (pipelines,'list',((self.reference,self.executable,self.catalog,'b'*64),)),
                (pipelines,'catalog_roots',(self.catalog_root,)), (modalities,'get',self.policy)]:
            patcher = patch.object(target,attribute,return_value=value,create=True)
            mocked = patcher.start()
            if target is pipelines and attribute == 'load': self.reader_load = mocked
            self.addCleanup(patcher.stop)
        for name, value in [('programs',self.programs), ('validate_files',Mock(return_value=None))]:
            patcher = patch.object(pipelines,name,value,create=True)
            patcher.start();self.addCleanup(patcher.stop)
        self.deadline = Deadline.after(20)

    def stats(self, paths):
        result = []
        for path in paths:
            st = path.stat()
            result.append(dict(path=str(path), device=st.st_dev, inode=st.st_ino,
                               size=st.st_size, mtime_ns=st.st_mtime_ns, ctime_ns=st.st_ctime_ns))
        return result

    def resolve(self, **kwargs):
        return hem_config.resolve(self.record, self.paths, self.dependencies, self.evidence,
                                  deadline=self.deadline, **kwargs)

    def test_support_programs_consume_the_callers_deadline(self):
        self.programs.return_value = {'prepare':'program.loom', 'control':'program.loom', 'sources':[]}
        self.policy.parameters.return_value = {}
        self.policy.PROTOCOL = {'operation':'chat'}
        document = hem_config.executable_info(self.reference, deadline=self.deadline)
        self.programs.assert_called_once_with(self.executable, self.catalog, self.deadline)
        self.assertEqual(document['programs']['providers'], [])
        self.assertNotIn('bundle', document)
        self.programs.reset_mock()
        with self.assertRaises(DeadlineExpired):
            hem_config.executable_info(self.reference, deadline=Deadline(0, clock=lambda:1))
        self.programs.assert_not_called()

    def test_each_layer_is_admitted_before_final_validation_and_definition(self):
        config = self.root / 'config.json'
        config.write_text('{"rows": 5}')
        answer = self.resolve(config_path=config, overrides={'rows': 6})
        self.assertEqual(answer.settings['rows'], 6)
        self.assertEqual(answer.sources['rows'], 'override')
        self.assertEqual([call.args[1] for call in self.policy.validate_layer.call_args_list],
                         [self.serving['defaults'], {'rows':4}, {'rows':5}, {'rows':6}])
        self.policy.finalize.assert_called_once()
        self.assertEqual(self.policy.finalize.call_args.args[1]['rows'], 6)
        self.programs.assert_called_once_with(self.executable, self.catalog, self.deadline)
        self.assertEqual(answer.catalog_relative_path, Path('examples/chat'))
        with self.assertRaises(TypeError): answer.definition['kind'] = 'changed'
        self.policy.definition.return_value['rows'] = 99
        self.assertEqual(answer.definition['rows'], 4)

    def test_bad_lower_layer_is_not_repaired_by_override(self):
        def check(_pipeline, values):
            if values.get('rows') is False:
                raise InvalidInput('rows: expected integer')
        self.policy.validate_layer.side_effect = check
        self.document['configuration']['rows'] = False
        with self.assertRaises(InvalidInput): self.resolve(overrides={'rows':4})
        self.policy.finalize.assert_not_called()
        self.policy.definition.assert_not_called()

    def test_mtp_winner_rank_removes_inherited_depth_but_rejects_explicit_conflict(self):
        self.document['configuration'].update(mtp=True, mtp_depth=3)
        answer = self.resolve(overrides={'mtp':False})
        self.assertNotIn('mtp_depth', answer.settings)
        self.assertEqual(answer.sources['mtp_depth'], {'state':'inactive','by':'override'})
        with self.assertRaises(InvalidInput): self.resolve(overrides={'mtp':False,'mtp_depth':3})
        self.document['configuration']['mtp'] = False
        answer = self.resolve(overrides={'mtp':True})
        self.assertEqual(answer.settings['mtp_depth'], 3)

    def test_identity_failure_precedes_policy_and_no_reselection_at_emission(self):
        self.record['provenance']['recipe_identity'] = 'b'*64
        with self.assertRaisesRegex(InvalidInput, 'new ID'): self.resolve()
        self.policy.definition.assert_not_called()
        self.record['provenance']['recipe_identity'] = 'a'*64
        answer = self.resolve()
        with patch.object(pipelines, 'load', side_effect=AssertionError('reselection')):
            document = hem_config.deployment_document(answer, 'instance', catalog=self.catalog)
        self.assertEqual(document['admission'], 'drained')

    def test_executable_association_is_current_before_configuration(self):
        original = deepcopy(self.record['provenance']['executable'])
        for key in ('reference','id','identity','declaration'):
            self.record['provenance']['executable'][key] = 'changed'
            with self.subTest(key=key), self.assertRaisesRegex(InvalidInput,'new ID'):
                self.resolve()
            self.record['provenance']['executable'] = deepcopy(original)
        self.policy.definition.assert_not_called()

    def test_no_preparation_rejects_a_saved_operation_association(self):
        self.record['provenance']['preparation'] = {'id':'unexpected','declaration':{}}
        with self.assertRaisesRegex(InvalidInput,'preparation association'):
            self.resolve()
        self.policy.definition.assert_not_called()

    def test_full_root_drift_and_foreign_catalog_rejected(self):
        other = self.catalog_root / 'other.json'
        other.write_text('{}')
        def mutate(*_args):
            other.write_text('{"changed":true}')
            return {'kind':'text'}
        self.policy.definition.side_effect = mutate
        with self.assertRaisesRegex(InvalidInput, 'changed'): self.resolve()
        self.policy.definition.side_effect = None
        self.reader_load.return_value = (self.executable,self.directory,'b'*64)
        with self.assertRaisesRegex(InvalidInput, 'outside'): self.resolve()

    def test_file_stat_drift_during_definition_is_rejected(self):
        def mutate(*_args):
            self.paths['weights'].write_bytes(b'changed')
            return {'kind':'text'}
        self.policy.definition.side_effect = mutate
        with self.assertRaisesRegex(InvalidInput, 'changed'): self.resolve()

    def test_split_index_complete_dependencies_are_retained(self):
        index = self.directory / 'model.safetensors.index.json'
        index.write_text('{}')
        self.paths['weights'] = index
        self.record['assets'][0]['logical_path'] = index.name
        self.dependencies = (*self.dependencies, index)
        self.evidence = self.stats(self.dependencies)
        answer = self.resolve()
        self.assertEqual(answer.dependencies, self.dependencies)
        self.assertIn(self.directory/'model.gguf', answer.dependencies)
        self.assertEqual(self.policy.definition.call_args.args[1]['weights'], index)

    def test_missing_evidence_and_wrong_purpose_mapping_fail(self):
        self.evidence.pop()
        with self.assertRaisesRegex(InvalidInput, 'cover'): self.resolve()
        self.evidence = self.stats(self.dependencies)
        self.paths['weights'] = self.paths['tokenizer']
        with self.assertRaisesRegex(InvalidInput, 'mapping'): self.resolve()

    def test_original_deadline_and_checked_root_apply_to_export(self):
        answer = self.resolve()
        self.assertIs(self.programs.call_args.args[2], self.deadline)
        (self.catalog_root/'unrelated').write_text('changed')
        with self.assertRaisesRegex(InvalidInput, 'differs'):
            hem_config.deployment_document(answer, 'instance', catalog=self.catalog, deadline=self.deadline)
        with self.assertRaises(DeadlineExpired):
            hem_config.resolve(self.record, self.paths, self.dependencies, self.evidence, deadline=Deadline(0, clock=lambda:1))

    def test_atomic_export_retains_previous_file_on_rejection(self):
        output = self.root/'export.json'
        output.write_text('previous')
        with self.assertRaises(InvalidInput):
            hem_config.export_deployment(self.record, self.paths, self.dependencies, [], output, 'name')
        self.assertEqual(output.read_text(), 'previous')
        hem_config.export_deployment(self.record,self.paths,self.dependencies,self.evidence,output,'name')
        self.assertEqual(json.loads(output.read_text())['definition']['kind'], 'text')
        self.assertEqual(list(self.root.glob('.export.json.*')), [])

    def test_export_qualifies_once_inside_original_model_lock(self):
        from binding.python import hem_cli
        active = []
        @contextmanager
        def lock(kind, name, *, deadline):
            self.assertIs(deadline,self.deadline)
            active.append((kind,name))
            try: yield
            finally: active.pop()
        registry = Mock()
        registry.lock.side_effect = lock
        def qualify(*_args, **_kwargs):
            self.assertEqual(active,[('model','local')])
            return self.record,self.paths,self.dependencies,self.evidence
        args = hem_cli.parse_args(['instance','export','local','--output',str(self.root/'output.json')])
        with patch.object(hem_instance_cli,'Registry',return_value=registry), \
             patch.object(hem_models,'qualify',create=True,side_effect=qualify) as qualification:
            hem_instance_cli.run(args,self.deadline)
        qualification.assert_called_once_with(registry,'local',executable=None,modality=None)
        self.assertEqual(active,[])

    def test_start_keeps_model_lock_through_resolution_and_server_lock(self):
        events, held = [], []
        @contextmanager
        def lock(kind, name, *, deadline):
            self.assertIs(deadline, self.deadline)
            events.append(kind)
            if kind == 'server':
                self.assertEqual(held, ['model'])
                self.assertEqual(events, ['model', 'qualify', 'definition', 'server'])
                raise InvalidInput('bounded stop before native lifecycle')
            held.append(kind)
            try: yield
            finally: held.pop()
        registry = Mock()
        registry.lock.side_effect = lock
        def qualify(*args, **kwargs):
            self.assertEqual(held, ['model'])
            events.append('qualify')
            return self.record, self.paths, self.dependencies, self.evidence
        def definition(*args):
            self.assertEqual(held, ['model'])
            events.append('definition')
            return {'kind':'text'}
        self.policy.definition.side_effect = definition
        manager = InstanceManager(registry, processes=Mock())
        with patch.object(hem_models, 'qualify', create=True, side_effect=qualify) as inspected:
            with self.assertRaisesRegex(InvalidInput, 'bounded stop'):
                manager.start({'id':'server'}, 'local', self.deadline)
        inspected.assert_called_once_with(registry, 'local', executable=None, modality=None)
        self.assertEqual(held, [])

    def test_retained_catalog_uses_flattened_identity_and_rejects_escape(self):
        answer = self.resolve()
        launch = self.root/'launch'
        retained_root = launch/'loom-pipeline'
        retained = retained_root/answer.catalog_relative_path
        retained.mkdir(parents=True)
        manager = InstanceManager(Mock(),processes=SimpleNamespace(package_root=self.root))
        server = dict(build=dict(directory=str(launch)))
        self.assertEqual(manager._retained_catalog(server,answer,retained_root),retained)
        with self.assertRaises(HemError):
            manager._retained_catalog(server,replace(answer,catalog_relative_path=Path('elsewhere')),retained_root)
