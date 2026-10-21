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


def selected_binding_source(text, config):
    """Resolve static recipe branches for the existing source-binding check.

    Runtime-dependent branches remain visible. This checks source contracts;
    the existing model witness validates actual compiler materialization.
    """
    values = {}

    def visit(lines):
        selected = []
        i = 0
        while i < len(lines):
            line = lines[i]
            branch = re.search(r'scf.if (%\w+).*\{\s*$', line)
            if branch and branch[1] in values:
                indent = line[:len(line)-len(line.lstrip())]
                end = i + 1
                alternate = None
                while end < len(lines):
                    if lines[end] == indent + '} else {':
                        alternate = end
                    elif lines[end] == indent + '}':
                        break
                    end += 1
                if end == len(lines):
                    raise AssertionError('Unclosed static branch: ' + line)
                if values[branch[1]]:
                    body = lines[i+1:alternate if alternate is not None else end]
                else:
                    body = lines[alternate+1:end] if alternate is not None else []
                resolved = visit(body)
                selected.extend(resolved)
                result = re.match(r'\s*(%\w+(?:, %\w+)*) = scf.if', line)
                if result:
                    yields = [re.match(r'\s*scf.yield (%\w+(?:, %\w+)*) :', item) for item in resolved]
                    operands = next((match[1].split(', ') for match in reversed(yields) if match), None)
                    if operands is None:
                        raise AssertionError('Missing static branch yield: ' + line)
                    for name, operand in zip(result[1].split(', '), operands):
                        if operand in values:
                            values[name] = values[operand]
                i = end + 1
                continue
            fact = re.search(r'(%\w+) = config.get @([\w.]+) : (?:index|i32)', line)
            constant = re.search(r'(%\w+) = (?:index|scalar).constant (\d+|true|false) :', line)
            comparison = re.search(r'(%\w+) = index.cmp (eq|ne), (%\w+), (%\w+) : index', line)
            if fact and fact[2] in config:
                values[fact[1]] = int(config[fact[2]])
            elif constant:
                values[constant[1]] = {'true': 1, 'false': 0}.get(constant[2], constant[2])
                values[constant[1]] = int(values[constant[1]])
            elif comparison and comparison[3] in values and comparison[4] in values:
                equal = values[comparison[3]] == values[comparison[4]]
                values[comparison[1]] = equal if comparison[2] == 'eq' else not equal
            selected.append(line)
            i += 1
        return selected

    return '\n'.join(visit(text.splitlines())), values


class QwenComparisonTest(unittest.TestCase):
    def test_mixed_08b_profile_and_no_mtp_contract(self):
        contract = profile('qwen_08b')
        model = matching_model(contract)
        self.assertEqual(len(model.tensors), 320)
        self.assertEqual(model.tensors['blk.8.ffn_gate.weight'].type_name, 'Q4_K')
        self.assertEqual(model.tensors['blk.15.attn_q.weight'].type_name, 'Q6_K')
        self.assertNotIn('output.weight', model.tensors)
        self.assertEqual(compare_gguf(model, contract), ())
        changed = dict(model.tensors)
        changed['blk.8.ffn_gate.weight'] = replace(changed['blk.8.ffn_gate.weight'], ggml_type=13)
        self.assertIn('expected Q4_K, found Q5_K', '\n'.join(compare_gguf(replace(model, tensors=changed), contract)))
        changed = dict(model.tensors)
        changed['blk.24.nextn.enorm.weight'] = changed['output_norm.weight']
        changed['output.weight'] = changed['token_embd.weight']
        problems = '\n'.join(compare_gguf(replace(model, tensors=changed), contract))
        self.assertIn('Unexpected model-block tensor', problems)
        self.assertIn('Unexpected untied output.weight', problems)

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
        catalog=PROFILE_PATH.parent.parent
        config=dict(re.findall(r'config.def @([\w.]+) = (\d+) : (?:index|i32)', (PROFILE_PATH.parent/'config.loom').read_text()))
        patterns = [(catalog/'programs/qwen38/model_decode.loom',None),
                    (catalog/'programs/qwen38/embedding.loom',None),
                    (catalog/'programs/qwen38/ffn.loom',0),
                    (catalog/'programs/qwen38/ffn.loom',3),
                    (catalog/'programs/qwen38/ffn.loom',64),
                    (catalog/'programs/qwen38/output_head_decode.loom',None),
                    (catalog/'programs/qwen38/gdn_layer_decode.loom',0),
                    (catalog/'programs/qwen38/attention_layer_decode.loom',3),
                    (catalog/'programs/qwen38/attention_layer_decode.loom',64),
                    (catalog/'mtp.loom',None)]
        bound=set()
        regex=r'command.parameter [^,]+, "([^"]+)"(?:\[([^]]+)\])? : view<(\d+|\[%\w+\])x(i8|f32)>'
        for source,layer in patterns:
            text,values=selected_binding_source(source.read_text(),config)
            for name,substitution,count,kind in re.findall(regex,text):
                if '{}' in name:
                    value=layer if substitution=='%layer' else values[substitution]
                    name=name.replace('{}',str(value))
                count=values[count[1:-1]] if count.startswith('[') else int(count)
                bound.add(name)
                tensor=self.model.tensors[name]
                self.assertEqual(tensor.size_bytes,count*(4 if kind=='f32' else 1),name)
        representative={name for name in self.model.tensors if not name.startswith('blk.') or name.startswith(('blk.0.','blk.3.','blk.64.'))}
        self.assertEqual(bound,representative)
        # The architecture's cold VM companions intentionally retain static
        # geometry. Keep their existing contract aligned with compiled shape.
        properties={key.removeprefix('qwen.model.'):int(value) for key,value in config.items() if key.startswith('qwen.model.')}
        for property_name,metadata_key in [('hidden_size','embedding_length'),('ffn_size','feed_forward_length'),('attention_heads','attention.head_count'),('kv_heads','attention.head_count_kv'),('head_size','attention.key_length'),('gdn_key_heads','ssm.group_count'),('gdn_value_heads','ssm.time_step_rank'),('gdn_head_size','ssm.state_size'),('gdn_convolution_taps','ssm.conv_kernel'),('rope_size','rope.dimension_count')]:
            self.assertEqual(properties[property_name],self.profile['metadata']['qwen35.'+metadata_key],property_name)
        self.assertEqual(properties['ffn_size'],self.profile['metadata']['qwen35.feed_forward_length'])
        self.assertEqual(properties['base_layers'],self.profile['base_layer_count'])
        self.assertEqual(properties['mtp_layer'],self.profile['mtp_layer'])
        self.assertEqual(properties['group_count']*4,properties['base_layers'])
        self.assertEqual(properties['gdn_layers'],properties['group_count']*3)
        prepare=(PROFILE_PATH.parent/'prepare.loom').read_text()
        storage={name:int(value) for name,value in re.findall(r'(%\w+) = scalar.constant (\d+) : i64',prepare)}
        for variable,property_name in [('%gdn_bytes','gdn_state_bytes'),('%target_token_bytes','target_kv_token_bytes'),('%draft_token_bytes','draft_kv_token_bytes'),('%carry_stride','hidden_row_bytes'),('%residual_bytes','hidden_capacity_bytes'),('%plane_token_bytes','kv_plane_token_bytes'),('%target_planes','target_kv_planes')]:
            self.assertEqual(storage[variable],properties[property_name],variable)
        self.assertIn(f"scalar.constant {properties['ffn_gate_weight_bytes']} : i64",(PROFILE_PATH.parent/'weights.loom').read_text())

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


class QwenNativePolicyContractTest(unittest.TestCase):
    def test_manifests_supply_pinned_text_policy_abi(self):
        # Public policy bindings required by HRX61417137 text control/chat and
        # parameter preparation. This protects source/runtime ABI, not model data.
        # Keep this catalog packaging guard in sync when reviewing an HRX pin.
        expected = {
            'encode_epoch': (('buffer',) * 4 + ('i32',) * 8 + ('buffer',) * 2, ('i32',)),
            'publish_epoch': (('buffer',) * 2 + ('i32',) * 6 + ('buffer',) * 4, ()),
            'step': (('buffer',) + ('i32',) * 3 + ('hal.buffer',) * 6, ('i64', 'i32')),
            'epoch': (('buffer',) + ('i32',) * 10 + ('hal.buffer',) * 12, ('i64',)),
            'chat_image_limits': ((), ('i64',) * 5),
            'chat_message': (('buffer',) * 6 + ('i64',) * 4, ('buffer', 'i64', 'buffer')),
            'prepare_input': (('buffer',) * 2 + ('i64', 'i32', 'i32', 'i64'), ('buffer', 'i64', 'buffer')),
            'prepare_weight': (('buffer',), ('i64',)),
            'request_defaults': (('buffer',), ('buffer',)),
            'request_options': (('buffer', 'buffer', 'i64'), ('buffer',)),
            'bind_request': (('buffer', 'i32', 'buffer'), ()),
            'output_next': (('buffer', 'i64', 'i64', 'i32'), ('i64',) * 4),
            'complete_text': (('buffer',) * 5 + ('i64',) * 2, ('buffer', 'i32')),
            'chat_end': (('i64',), ('buffer', 'i64')),
        }
        root = Path(__file__).resolve().parents[1] / 'loom-pipeline/llm/qwen'
        for architecture in ('qwen_27b', 'qwen_08b'):
            directory = root / architecture
            sources = '\n'.join((directory / entry.strip()).read_text()
                                for entry in (directory / 'sources.txt').read_text().splitlines()
                                if entry.strip() and not entry.startswith('#'))
            for name, (arguments, results) in expected.items():
                with self.subTest(architecture=architecture, hook=name):
                    signatures = re.findall(
                        r'func.def public @' + name + r'\(([^)]*)\)(?: -> \(([^)]*)\))? \{', sources)
                    self.assertEqual(len(signatures), 1, 'missing or duplicate native policy hook')
                    args, returns = signatures[0]
                    self.assertEqual(tuple(re.findall(r': ([\w.]+)', args)), arguments)
                    self.assertEqual(tuple(part.strip() for part in returns.split(',') if part.strip()), results)


if __name__ == '__main__':
    unittest.main()
