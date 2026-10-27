"""ASR contract and exact preparation data without model-name policy imports."""
from copy import deepcopy
from pathlib import Path
from types import MappingProxyType
import json
import unittest
from unittest.mock import patch

from binding.python.hem_common import Deadline, HemError, InvalidInput
from binding.python.models import pipelines, recipes, model_download
from binding.python.models.modalities import asr


class ASRFunctionsTest(unittest.TestCase):
    def setUp(self):
        self.recipe,_=recipes.load('moonshine-streaming-tiny')
        self.executable,self.catalog,_=pipelines.load(self.recipe['executable'])
        self.serving=self.executable['serving']

    def test_fixed_types_partial_layers_completeness_and_default_ownership(self):
        asr.validate_layer(self.serving,{'pending_requests':4})
        asr.validate_layer(self.serving,{'options':MappingProxyType({})})
        for values in ({'pending_requests':True},{'pending_requests':4.0},{'pending_requests':8},
                       {'options':None},{'options':[]},{'options':{'temperature':0}},{'rows':4}):
            with self.subTest(values=values),self.assertRaises(InvalidInput):asr.validate_layer(self.serving,values)
        with self.assertRaises(InvalidInput):asr.finalize(self.serving,{'pending_requests':4},{},{})
        settings=asr.defaults(self.serving);asr.validate_layer(self.serving,settings);asr.finalize(self.serving,settings,{},{})
        asr.defaults(self.serving)['options']['mutated']=1
        self.assertEqual(asr.defaults(self.serving),{'pending_requests':4,'options':{}})

    def test_native_definition_protocol_support_and_immutable_raw_log_view(self):
        settings=asr.defaults(self.serving)
        programs={'prepare':'p.loombc','control':'c.loombc','sources':['providers.loombc']}
        self.assertEqual(asr.definition(settings,{},Path('/models/speech'),programs),
                         {'kind':'recognition',**programs,'assets':'/models/speech','pending_requests':4,'options':{}})
        self.assertEqual(asr.parameters(self.serving)['options']['value'],{})
        self.assertEqual(asr.PROTOCOL['request_encoding'],'multipart')
        self.assertEqual(asr.PROTOCOL['endpoint'],'/v1/audio/transcriptions')
        with self.assertRaises(InvalidInput):asr.definition(settings,{},Path('relative'),programs)
        with self.assertRaises(InvalidInput):asr.validate_contract({'model':'anything'})
        self.assertEqual(json.loads(asr.interpret(MappingProxyType({'unknown':(MappingProxyType({'value':'raw'}),)}))),
                         {'unknown':[{'value':'raw'}]})

    def test_current_source_program_query_has_no_cache_or_helper_effects(self):
        with patch('binding.python.models.preparation.moonshine.prepare',side_effect=AssertionError('helper')), \
             patch.object(model_download,'cache_root',side_effect=AssertionError('cache')):
            programs=pipelines.programs(self.executable,self.catalog,Deadline.after())
        self.assertTrue(programs['prepare'].endswith('.loom'))
        self.assertEqual(set(programs),{'prepare','control','sources'})
        self.assertEqual(self.recipe['configuration'],asr.defaults(self.serving))

    def test_exact_operation_tuple_rejects_pins_names_destinations_extra_and_missing_before_helper(self):
        files=self.recipe['files'];pipelines.validate_files(self.executable,files)
        with patch('binding.python.models.preparation.moonshine.preflight') as preflight, \
             patch('binding.python.models.preparation.moonshine.prepare') as prepare:
            for field,value in (('revision','0'*40),('filename','other.json'),('repo_id','other/repo')):
                changed=deepcopy(files);changed['weights']['source'][field]=value
                with self.subTest(field=field),self.assertRaises(HemError):pipelines.validate_files(self.executable,changed)
            for purpose in files:
                changed=deepcopy(files);changed[purpose]['path']='nested/'+changed[purpose]['path']
                with self.subTest(purpose=purpose),self.assertRaises(HemError):pipelines.validate_files(self.executable,changed)
            for changed in ({k:v for k,v in files.items() if k!='rotary'}, {**files,'extra':files['weights']},
                            {**files,'rotary':{'path':'rotary.safetensors','source':files['weights']['source']}}):
                with self.assertRaises(HemError):pipelines.validate_files(self.executable,changed)
        preflight.assert_not_called();prepare.assert_not_called()
        # Support copies must not rewrite the actual admitted declaration.
        from binding.python.models import hem_config
        view=hem_config.executable_info(self.recipe['executable'])
        self.assertEqual(set(view['assets']),{'weights','tokenizer','configuration','rotary'})
        self.assertEqual(view['preparation']['outputs'],{'rotary.safetensors':'rotary'})
        view['assets']['weights']['path']='wrong'
        pipelines.validate_files(self.executable,files)
