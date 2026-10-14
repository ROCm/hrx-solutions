from dataclasses import replace
import hashlib
import json
from pathlib import Path
import re
import tempfile
import unittest

from binding.python.gguf import ArrayInfo, GGML_TYPES, GGUF, Metadata, Tensor, tensor_size
from binding.python.qwen_validation import compare_gguf, compare_tokenizer, expected_tensors
from tests.gguf_fixtures import PROFILE_PATH, profile


def matching_model(contract=None):
    contract = contract or profile()
    metadata = {}
    for key, value in contract['metadata'].items():
        if isinstance(value,list):
            metadata[key] = Metadata(9,ArrayInfo(4,len(value),tuple(value)))
        else:
            kind = 8 if isinstance(value,str) else 6 if isinstance(value,float) else 4
            metadata[key] = Metadata(kind,value)
    metadata.update({key:Metadata(9,ArrayInfo(value['type'],value['count'])) for key,value in contract['metadata_arrays'].items()})
    types = {value[0]:key for key,value in GGML_TYPES.items()}
    tensors = {name:Tensor(name,tuple(spec['shape']),types[spec['type']],0,tensor_size(tuple(spec['shape']),types[spec['type']])) for name,spec in expected_tensors(contract).items()}
    return GGUF(3,metadata,tensors,0,0,0)


class QwenComparisonTest(unittest.TestCase):
    def setUp(self):
        self.profile=profile()
        self.model=matching_model(self.profile)

    def test_positive_and_advisory_fields(self):
        self.assertEqual(len(self.model.tensors),866)
        self.assertEqual(compare_gguf(self.model,self.profile),())
        self.model.metadata.update({'general.name':Metadata(8,'Fine-tune'), 'general.file_type':Metadata(4,999), 'custom.description':Metadata(8,'unused')})
        self.model.tensors['aux.weight']=Tensor('aux.weight',(1,),0,0,4)
        self.assertEqual(compare_gguf(self.model,self.profile),())

    def test_architecture_and_same_byte_count_wrong_shape_or_type(self):
        self.model.metadata['general.architecture']=Metadata(8,'qwen3')
        tensor=self.model.tensors['blk.0.ssm_a']
        self.model.tensors[tensor.name]=replace(tensor,shape=(24,2),ggml_type=26)
        problems='\n'.join(compare_gguf(self.model,self.profile))
        self.assertIn("expected 'qwen35'",problems)
        self.assertIn('expected shape [48]',problems)
        self.assertIn('expected F32, found I32',problems)

    def test_metadata_type_missing_arrays_and_layer_pattern(self):
        self.model.metadata['qwen35.embedding_length']=Metadata(6,5120.0)
        del self.model.metadata['qwen35.ssm.conv_kernel']
        self.model.metadata['tokenizer.ggml.tokens']=Metadata(9,ArrayInfo(8,248044))
        self.model.metadata['qwen35.attention.recurrent_layers']=Metadata(9,ArrayInfo(7,65,(False,)*65))
        problems='\n'.join(compare_gguf(self.model,self.profile))
        for text in ('embedding_length','Missing required metadata: qwen35.ssm.conv_kernel','tokenizer.ggml.tokens','recurrent_layers'):
            self.assertIn(text,problems)

    def test_float_tolerance_scaling_and_unknown_geometry(self):
        self.model.metadata['qwen35.attention.layer_norm_rms_epsilon']=Metadata(6,1e-6)
        self.model.metadata['qwen35.rope.scaling.type']=Metadata(8,'linear')
        self.model.metadata['qwen35.rope.scaling.factor']=Metadata(6,1.0)
        self.assertEqual(compare_gguf(self.model,self.profile),())
        self.model.metadata['qwen35.rope.scaling.factor']=Metadata(6,2.0)
        self.model.metadata['qwen35.attention.sliding_window']=Metadata(4,1024)
        problems='\n'.join(compare_gguf(self.model,self.profile))
        self.assertIn('scaling.factor',problems)
        self.assertIn('Unsupported Qwen configuration',problems)

    def test_missing_mtp_and_extra_block(self):
        del self.model.tensors['blk.64.nextn.eh_proj.weight']
        self.model.tensors['blk.65.attn_norm.weight']=Tensor('blk.65.attn_norm.weight',(5120,),0,0,20480)
        problems='\n'.join(compare_gguf(self.model,self.profile))
        self.assertIn('MTP: missing required tensor blk.64.nextn.eh_proj.weight',problems)
        self.assertIn('Unexpected model-block tensor',problems)

    def test_tensor_contract_matches_catalog_bindings(self):
        # A stale profile must fail when native consumers change names/views.
        catalog=PROFILE_PATH.parent
        patterns = [(catalog/'programs/qwen38/model_decode.loom',None),
                    (catalog/'programs/qwen38/output_head_decode.loom',None),
                    (catalog/'programs/qwen38/gdn_layer_decode.loom',0),
                    (catalog/'programs/qwen38/attention_layer_decode.loom',3),
                    (catalog/'programs/qwen38/attention_layer_decode.loom',64),
                    (catalog/'mtp.loom',None)]
        bound=set()
        regex=r'command.parameter [^,]+, "([^"]+)"(?:\[%layer\])? : view<(\d+)x(i8|f32)>'
        for source,layer in patterns:
            for name,count,kind in re.findall(regex,source.read_text()):
                name=name.replace('{}',str(layer)) if layer is not None else name
                bound.add(name)
                tensor=self.model.tensors[name]
                self.assertEqual(tensor.size_bytes,int(count)*(4 if kind=='f32' else 1),name)
        representative={name for name in self.model.tensors if not name.startswith('blk.') or name.startswith(('blk.0.','blk.3.','blk.64.'))}
        self.assertEqual(bound,representative)

    def test_tokenizer_config_ignores_formatting_but_detects_changed_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'tokenizer.json'
            data={'model':{'type':'BPE','vocab':{'a':0}},'added_tokens':[]}
            self.profile['tokenizer_canonical_sha256']=hashlib.sha256(json.dumps(data,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
            path.write_text(json.dumps(data,indent=4))
            self.assertEqual(compare_tokenizer(path,self.profile),())
            data['model']['vocab']['a']=1
            path.write_text(json.dumps(data))
            self.assertIn('Tokenizer configuration differs',compare_tokenizer(path,self.profile)[0])
            path.write_text('{')
            self.assertIn('not valid UTF-8 JSON',compare_tokenizer(path,self.profile)[0])
            path.write_text('[]')
            self.assertIn('model object',compare_tokenizer(path,self.profile)[0])


if __name__ == '__main__':
    unittest.main()
