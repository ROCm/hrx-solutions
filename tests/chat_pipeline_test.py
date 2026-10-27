"""Actual CHAT declarations and model-independent source/compiled admission."""
from copy import deepcopy
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import MappingProxyType
import unittest
from unittest.mock import patch

from binding.python.hem_common import Deadline, DeadlineExpired, HemError, InvalidInput
from binding.python.control.hem_state import Registry
from binding.python.models import hem_models, model_download, modalities, pipelines, recipes
from binding.python.models.modalities import chat
from tests.recipe_config_fixture import SOURCE


class ChatFunctionsTest(unittest.TestCase):
    def serving(self, identifier='qwen38-27b-q5-k-xl'):
        recipe, _ = recipes.load(identifier)
        executable, catalog, _ = pipelines.load(recipe['executable'])
        return recipe, executable, catalog

    def test_actual_configurations_and_programs_drive_unchanged_native_fields(self):
        paths={'weights':Path('/hf/custom.gguf'),'tokenizer':Path('/hf/custom.json')}
        for identifier, depth in (('qwen35-08b-q5-k-xl',0),('qwen38-27b-q5-k-xl',3)):
            recipe, executable, catalog=self.serving(identifier)
            serving=executable['serving'];settings=chat.defaults(serving)
            self.assertEqual(settings,recipe['configuration'])
            chat.validate_layer(serving,settings);chat.finalize(serving,settings,{}, {})
            programs=pipelines.programs(executable,catalog,Deadline.after())
            document=chat.definition(settings,paths,Path('/local'),programs)
            self.assertEqual(document,{'kind':'text',**programs,'weights':'/hf/custom.gguf',
                'tokenizer':'/hf/custom.json','rows':4,'context_capacity':8192,'pool_capacity':32768,
                'prefill_capacity':128,'draft_depth':depth,'options':{}})
            for invalid in ({'tokenizer':paths['tokenizer']},{**paths,'weights':Path('relative.gguf')}):
                with self.assertRaises(InvalidInput):chat.definition(settings,invalid,Path('/local'),programs)

    def test_every_current_bound_type_unknown_and_unsupported_field_is_checked(self):
        for identifier in ('qwen35-08b-q5-k-xl','qwen38-27b-q5-k-xl'):
            _,executable,_=self.serving(identifier);serving=executable['serving']
            for values in ({'rows':True},{'rows':1.0},{'rows':0},{'context_capacity':None},
                           {'mtp':0},{'unknown':1},None,[],{'mtp_depth':True}):
                with self.subTest(identifier=identifier,values=values),self.assertRaises(InvalidInput):
                    chat.validate_layer(serving,values)
            for field,interval in serving['bounds'].items():
                for value in (interval[0]-1,interval[1]+1):
                    with self.assertRaises(InvalidInput):chat.validate_layer(serving,{field:value})
        _,executable,_=self.serving('qwen35-08b-q5-k-xl')
        for value in ({'mtp':True},{'mtp_depth':1}):
            with self.assertRaises(InvalidInput):chat.validate_layer(executable['serving'],value)

    def test_final_completeness_cross_rules_and_explicit_index_path(self):
        _,executable,_=self.serving();serving=executable['serving'];defaults=chat.defaults(serving)
        for changes in ({'prefill_capacity':3},{'context_capacity':64}):
            values={**defaults,**changes}
            with self.assertRaises(InvalidInput):chat.finalize(serving,values,{}, {})
        for key in defaults:
            values=dict(defaults);values.pop(key)
            with self.subTest(key=key),self.assertRaises(InvalidInput):chat.finalize(serving,values,{}, {})
        chat.validate_layer(serving,{'prefill_capacity':2,'mtp_depth':3})
        values=dict(defaults,mtp=False)
        with self.assertRaises(InvalidInput):chat.finalize(serving,values,{'mtp':'file'},
                                                               {'mtp':2,'mtp_depth':2})
        values.pop('mtp_depth');chat.finalize(serving,values,{}, {})
        document=chat.definition(values,{'weights':Path('/models/model.safetensors.index.json'),
            'tokenizer':Path('/models/tokenizer.json')},Path('/models'),
            {'prepare':'p','control':'c','sources':['first','next']})
        self.assertEqual(document['weights'],'/models/model.safetensors.index.json')
        self.assertEqual(document['sources'],['first','next'])
        self.assertEqual(document['draft_depth'],0)

    def test_defaults_support_and_logs_are_pure_unshared_values(self):
        _,executable,_=self.serving();serving=executable['serving']
        values=chat.defaults(serving);chat.defaults(serving)['rows']=999
        self.assertEqual(chat.defaults(serving),values)
        chat.parameters(serving)['rows']['constraints']['maximum']=999
        self.assertEqual(chat.parameters(serving)['rows']['constraints']['maximum'],16)
        chat.interpret({})
        with ExitStack() as stack:
            for target in ('builtins.open','pathlib.Path.open','subprocess.run','socket.create_connection'):
                stack.enter_context(patch(target,side_effect=AssertionError('policy I/O')))
            chat.defaults(serving);chat.parameters(serving);chat.validate_layer(serving,values)
            chat.finalize(serving,values,{}, {})
            event=MappingProxyType({'unknown':(MappingProxyType({'text':'hi'}),)})
            self.assertEqual(json.loads(chat.interpret(event)),{'unknown':[{'text':'hi'}]})
        self.assertEqual(chat.PROTOCOL['endpoint'],'/v1/chat/completions')

    def test_modality_import_does_not_select_or_import_any_model(self):
        self.assertIs(modalities.get('chat'),chat)
        for unknown in ('qwen','moonshine','transcription',None):
            with self.assertRaises(InvalidInput):modalities.get(unknown)
        code=('import sys; from binding.python.models import modalities; '
              'modalities.get("chat"); '
              'assert not any(n.endswith(("pipelines_qwen","pipelines_moonshine","modalities.asr")) for n in sys.modules)')
        subprocess.run([sys.executable,'-c',code],check=True,capture_output=True,text=True)

    def test_actual_cached_acquisition_and_both_native_definitions_without_network_or_helper(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);registry=Registry(root/'state')
            with patch.dict(os.environ,{'HF_HUB_CACHE':str(root/'hf')}), \
                 patch.object(model_download,'_run_hf',side_effect=AssertionError('network')), \
                 patch('binding.python.models.preparation.moonshine.preflight',side_effect=AssertionError('helper')), \
                 patch('binding.python.models.preparation.moonshine.prepare',side_effect=AssertionError('helper')):
                for identifier,depth in (('qwen35-08b-q5-k-xl',0),('qwen38-27b-q5-k-xl',3)):
                    recipe,executable,catalog=self.serving(identifier);cached={}
                    for purpose,item in recipe['files'].items():
                        path=model_download._cache_path(item['source']);path.parent.mkdir(parents=True,exist_ok=True)
                        path.write_bytes(b'opaque presence fixture; no inference proof');cached[purpose]=path
                    stats={k:p.stat() for k,p in cached.items()}
                    self.assertEqual(hem_models.download(registry,identifier,model_id=identifier,offline=True)['state'],'ready')
                    record=registry.get_model(identifier);paths,dependencies,evidence=hem_models.inspect_record(record)
                    self.assertEqual(set(paths.values()),set(dependencies));self.assertEqual(evidence,record['validation']['evidence'])
                    for purpose,path in paths.items():
                        self.assertEqual(os.readlink(path),str(cached[purpose]));self.assertEqual(cached[purpose].stat(),stats[purpose])
                    programs=pipelines.programs(executable,catalog,Deadline.after())
                    document=chat.definition(recipe['configuration'],paths,Path(record['directory']),programs)
                    self.assertEqual(document['draft_depth'],depth)
                    self.assertEqual(document['weights'],str(paths['weights']))
                    self.assertEqual(document['tokenizer'],str(paths['tokenizer']))

    def test_declared_asset_policy_preserves_custom_paths_and_rejects_unbound_generation(self):
        recipe,executable,_=self.serving('qwen35-08b-q5-k-xl');files=deepcopy(recipe['files'])
        files['weights']['path']='custom/nested.gguf';pipelines.validate_files(executable,files)
        for purpose in files:
            changed=deepcopy(files);changed[purpose]['source']={'kind':'generated'}
            with self.assertRaises(HemError):pipelines.validate_files(executable,changed)
        for changed in ({},{**files,'extra':files['weights']},
                        {**files,'tokenizer':dict(files['tokenizer'],path='tokenizer.bin')}):
            with self.assertRaises(HemError):pipelines.validate_files(executable,changed)


class ProgramAdmissionTest(unittest.TestCase):
    def setUp(self):
        temporary=tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup)
        self.root=Path(temporary.name)/'loom-pipeline';shutil.copytree(SOURCE,self.root)
        patcher=patch.object(pipelines,'catalog_roots',return_value=(self.root,));patcher.start();self.addCleanup(patcher.stop)
        self.reference='llm/qwen/qwen_08b/executable.json'

    def selected(self):
        return pipelines.load(self.reference)

    def compiled(self,reference=None):
        source,catalog,identity=pipelines.load(reference or self.reference)
        value=deepcopy(source);value['id']='compiled-fixture-'+value['modality']
        relative='compiled/'+value['id'];root=self.root/relative;root.mkdir(parents=True)
        names=['pipeline.loombc']+(['weights.loombc'] if value['programs']['support'] else [])
        for name in names:(root/name).write_bytes(b'LOOM synthetic integrity fixture '+name.encode())
        value['programs']=dict(root=relative,encoding='bytecode',prepare=names[0],control=names[0],
            sources=[names[0]],support=names[1:],files={n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in names})
        value['provenance']=dict(source_declaration_sha256=identity,source_files=source['programs']['files'])
        self.reference=relative+'/executable.json';(root/'executable.json').write_text(json.dumps(value))
        return root,value

    def test_real_source_missing_changed_duplicate_escape_and_expired_reads(self):
        value,catalog,_=self.selected();name=next(iter(value['programs']['files']));path=catalog/name;before=path.read_bytes()
        pipelines.programs(value,catalog,Deadline.after())
        for content in (None,b'changed'):
            if content is None:path.unlink()
            else:path.write_bytes(content)
            with self.assertRaises(HemError):pipelines.programs(value,catalog,Deadline.after())
            path.write_bytes(before)
        for role,invalid in (('prepare','../outside.loom'),('sources',[value['programs']['sources'][0]]*2)):
            changed=deepcopy(value);changed['programs'][role]=invalid
            with self.assertRaises(HemError):pipelines.parse(json.dumps(changed).encode())
        with patch.object(Path,'open',side_effect=AssertionError('expired I/O')),self.assertRaises(DeadlineExpired):
            pipelines.programs(value,catalog,Deadline(0,clock=lambda:1))

    def test_compiled_singleton_and_required_support_hash_inventory(self):
        root,value=self.compiled('recognition/moonshine/streaming_tiny/executable.json')
        selected,catalog,_=self.selected();programs=pipelines.programs(selected,catalog,Deadline.after())
        self.assertEqual(programs['sources'],['pipeline.loombc'])
        (root/'weights.loombc').write_bytes(b'LOOM changed')
        with self.assertRaises(HemError):pipelines.programs(selected,catalog,Deadline.after())

    def test_compiled_extra_executable_hash_and_encoding_provenance_are_required(self):
        root,value=self.compiled();selected,catalog,_=self.selected()
        (root/'extra.loom').write_text('undeclared')
        with self.assertRaises(HemError):pipelines.programs(selected,catalog,Deadline.after())
        (root/'extra.loom').unlink();(root/'pipeline.loombc').write_bytes(b'wrong encoding')
        selected['programs']['files']['pipeline.loombc']=hashlib.sha256((root/'pipeline.loombc').read_bytes()).hexdigest()
        with self.assertRaises(HemError):pipelines.programs(selected,catalog,Deadline.after())
        value['provenance']=None
        with self.assertRaises(HemError):pipelines.parse(json.dumps(value).encode())

    def test_closed_declaration_rejects_obsolete_authorities_and_wrong_asset_types(self):
        value,_,_=self.selected()
        for field in ('download_recipe','pending_requests','options','definition'):
            changed=deepcopy(value);changed[field]={}
            with self.assertRaises(HemError):pipelines.parse(json.dumps(changed).encode())
        changed=deepcopy(value);changed['assets']['weights']['path']=True
        with self.assertRaises(HemError):pipelines.parse(json.dumps(changed).encode())
        (self.root/self.reference).unlink()
        with self.assertRaises(HemError):self.selected()  # No sources.txt/catalog.json fallback.

    def test_manifest_and_program_symlinks_cannot_escape_and_errors_stay_domain_errors(self):
        value,catalog,_=self.selected();name=next(iter(value['programs']['files']));path=catalog/name
        outside=self.root.parent/'outside.loom';outside.write_bytes(path.read_bytes());path.unlink();path.symlink_to(outside)
        with self.assertRaises(HemError):pipelines.programs(value,catalog,Deadline.after())
        for error in (PermissionError('unreadable'),RuntimeError('symlink loop')):
            with patch.object(Path,'resolve',side_effect=error),self.assertRaises(HemError):self.selected()
