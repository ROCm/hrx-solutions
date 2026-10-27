"""Closed modality semantics on declared facts, without model-name dispatch."""
import unittest
from binding.python.hem_common import InvalidInput
from binding.python.models.modalities import chat, asr


def text_serving(depth=0):
    return {'bounds': {'rows':[1,16], 'context_capacity':[1,262144],
                       'pool_capacity':[1,4194304], 'prefill_capacity':[1,262144]},
            'mtp_depth_max':depth,
            'defaults':dict(rows=4,context_capacity=8192,pool_capacity=32768,prefill_capacity=128,
                            mtp=bool(depth),**({'mtp_depth':3} if depth else {}))}


def recognition_serving():
    return {'pending_requests':[4,4], 'defaults':{'pending_requests':4,'options':{}}}


class ModalityConfigurationTest(unittest.TestCase):
    def test_declared_defaults_are_copied_and_current_native_values_preserved(self):
        for module, serving in ((chat,text_serving()),(chat,text_serving(3)),(asr,recognition_serving())):
            with self.subTest(module=module.__name__,serving=serving):
                module.validate_contract(serving)
                values=module.defaults(serving)
                sources={key:'default' for key in values};ranks={key:0 for key in values}
                module.finalize(serving,values,sources,ranks)
                native=module.definition(values,{'weights':'/model/model.gguf','tokenizer':'/model/tokenizer.json'},
                                         '/model',{'prepare':'prepare.loom','control':'control.loom','sources':['weights.loom']})
                if module is chat:
                    self.assertEqual(native['draft_depth'],serving['mtp_depth_max'])
                    self.assertEqual(native['rows'],4)
                    self.assertEqual(native['context_capacity'],8192)
                else:
                    self.assertEqual(native['pending_requests'],4)
                    values['options']['not_shared']=True
                    self.assertEqual(serving['defaults']['options'],{})

    def test_invalid_contracts_and_invalid_defaults_fail_before_execution(self):
        mutations=[lambda s:s.update(extra=1), lambda s:s['bounds'].update(other=[1,2]),
                   lambda s:s['bounds'].update(rows=[True,16]),lambda s:s['bounds'].update(rows=[2,1]),
                   lambda s:s.update(mtp_depth_max=-1),lambda s:s.update(mtp_depth_max=True),
                   lambda s:s['defaults'].update(rows='4'),lambda s:s['defaults'].update(mtp=True),
                   lambda s:s['defaults'].update(prefill_capacity=9000),lambda s:s['defaults'].pop('mtp')]
        for mutate in mutations:
            serving=text_serving();mutate(serving)
            with self.subTest(serving=serving),self.assertRaises(InvalidInput):chat.validate_contract(serving)
        for interval in ([True,4],[4,3],[0,4],[1,2.0],[],[1,2,3]):
            serving=recognition_serving();serving['pending_requests']=interval
            with self.subTest(interval=interval),self.assertRaises(InvalidInput):asr.validate_contract(serving)
        for values in ({'pending_requests':8,'options':{}},{'pending_requests':4,'options':{'x':1}},
                       {'pending_requests':4},{'pending_requests':False,'options':{}}):
            serving=recognition_serving();serving['defaults']=values
            with self.subTest(values=values),self.assertRaises(InvalidInput):asr.validate_contract(serving)

    def test_each_field_keeps_exact_types_unknown_and_capability_rejections(self):
        for value in (True,1.0,'1',None,0,17):
            with self.subTest(value=value),self.assertRaises(InvalidInput):chat.validate_layer(text_serving(),{'rows':value})
        for layer in ({'mtp':True},{'mtp_depth':None},{'mtp_depth':1},{'unknown':1}):
            with self.subTest(layer=layer),self.assertRaises(InvalidInput):chat.validate_layer(text_serving(),layer)
        for layer in ({'pending_requests':8},{'pending_requests':True},{'options':{'x':1}},{'options':None},{'rows':1}):
            with self.subTest(layer=layer),self.assertRaises(InvalidInput):asr.validate_layer(recognition_serving(),layer)

    def test_chat_rank_rule_is_modality_owned_and_keeps_source_attribution(self):
        serving=text_serving(3);values=chat.defaults(serving)
        values['mtp']=False
        sources={key:'recipe' for key in values};sources['mtp']='override'
        ranks={key:1 for key in values};ranks['mtp']=3
        chat.finalize(serving,values,sources,ranks)
        self.assertNotIn('mtp_depth',values)
        self.assertEqual(sources['mtp_depth'],{'state':'inactive','by':'override'})
        for depth_rank in (2,3):
            values=chat.defaults(serving);values['mtp']=False
            with self.subTest(rank=depth_rank),self.assertRaises(InvalidInput):
                chat.finalize(serving,values,{'mtp':'file','mtp_depth':'override'},
                              {'mtp':2,'mtp_depth':depth_rank})
        values=chat.defaults(serving)
        chat.finalize(serving,values,{}, {})
        self.assertEqual(values['mtp_depth'],3)

    def test_cross_rules_apply_only_after_individual_layers(self):
        serving=text_serving(3)
        chat.validate_layer(serving,{'context_capacity':64})
        values=chat.defaults(serving);values['context_capacity']=64
        with self.assertRaises(InvalidInput):chat.finalize(serving,values,{}, {})
        values['prefill_capacity']=32
        chat.finalize(serving,values,{}, {})
        values['prefill_capacity']=3
        with self.assertRaises(InvalidInput):chat.finalize(serving,values,{}, {})

    def test_two_new_same_native_contracts_need_only_different_data(self):
        text=text_serving();text['bounds']['rows']=[2,8];text['defaults']['rows']=2
        recognition={'pending_requests':[1,8],'defaults':{'pending_requests':2,'options':{}}}
        for module,serving,layer in ((chat,text,{'rows':8}),(asr,recognition,{'pending_requests':8})):
            module.validate_contract(serving)
            module.validate_layer(serving,layer)
            values=module.defaults(serving);values.update(layer)
            module.finalize(serving,values,{}, {})
            support=module.parameters(serving)
            field=next(iter(layer))
            self.assertEqual(support[field]['state'],'configurable')
            self.assertEqual(support[field]['constraints']['maximum'],8)
        self.assertEqual(chat.parameters(text)['mtp']['value'],False)
        self.assertEqual(chat.parameters(text)['mtp_depth']['state'],'unsupported')
        self.assertEqual(asr.parameters(recognition_serving())['pending_requests']['value'],4)

    def test_supported_but_default_off_depth_can_be_reactivated(self):
        serving=text_serving(3);serving['defaults']['mtp']=False
        chat.validate_contract(serving)
        values=chat.defaults(serving);values['mtp']=True
        chat.finalize(serving,values,{'mtp':'override','mtp_depth':'default'},
                      {'mtp':3,'mtp_depth':0})
        self.assertEqual(values['mtp_depth'],3)
        values=chat.defaults(serving)
        chat.finalize(serving,values,{'mtp':'default','mtp_depth':'default'},
                      {'mtp':0,'mtp_depth':0})
        self.assertNotIn('mtp_depth',values)

    def test_fixed_chat_capacity_is_declared_not_a_model_name_condition(self):
        serving=text_serving();serving['bounds']['rows']=[4,4]
        chat.validate_contract(serving)
        self.assertEqual(chat.parameters(serving)['rows'],{'type':'integer','value':4,'state':'fixed'})
        with self.assertRaises(InvalidInput):chat.validate_layer(serving,{'rows':3})
