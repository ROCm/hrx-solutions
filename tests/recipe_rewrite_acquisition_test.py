"""Rough plain-contract acquisition checks; no network/helper/service execution.

Actual format3 reader and executable declarations are composed; helper effects are controlled.
Real Registry/copy/alias/index/receipt publication code runs on tiny temp files.
"""
from contextlib import redirect_stderr
from copy import deepcopy
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from binding.python.control.hem_state import Registry
from binding.python.models import hem_models as models, model_download, recipes, pipelines
from binding.python.models.preparation import moonshine as moonshine_preparation
from binding.python.cli import hem_model_cli
from binding.python.hem_common import Deadline, HemError
from tests.gguf_fixtures import write_gguf


class AcquisitionDraftTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        env = patch.dict(os.environ, {'HF_HUB_CACHE': str(self.root/'hf'), 'HF_HUB_OFFLINE': '0'})
        env.start(); self.addCleanup(env.stop)
        self.registry = Registry(self.root/'state')
        self.documents, self.shas = {}, {}
        source = Path(__file__).resolve().parents[1]/'loom-pipeline'
        for path in source.glob('**/recipes/*.json'):
            doc,sha=recipes.parse(path.read_bytes())
            self.documents[doc['id']] = doc
            self.shas[doc['id']] = sha
        network=patch.object(model_download,'_run_hf',side_effect=AssertionError('network forbidden'))
        network.start(); self.addCleanup(network.stop)
        self.qwen=self.documents['qwen35-08b-q5-k-xl']
        self.asr=self.documents['moonshine-streaming-tiny']
        self.events=[]
        self.preparer=moonshine_preparation
        preflight=patch.object(self.preparer,'preflight',side_effect=lambda deadline:self.events.append('preflight'))
        preflight.start(); self.addCleanup(preflight.stop)
        def prepare(inputs,attempt,deadline):
            self.events.append('prepare'); deadline.remaining()
            self.assertEqual(set(inputs),{'model.safetensors','tokenizer.json','config.json'})
            result=attempt/'output/rotary.safetensors'; result.parent.mkdir(); result.write_bytes(b'rotary')
            return {'rotary.safetensors':result}
        generate=patch.object(self.preparer,'prepare',side_effect=prepare)
        generate.start(); self.addCleanup(generate.stop)
        self.gguf=self.root/'input.gguf'; write_gguf(self.gguf,tensors=[('one',[1],0,0)])
        self.token=self.root/'tokenizer.json'; self.token.write_text('{}')
        self.stderr=redirect_stderr(io.StringIO()); self.stderr.__enter__()
        self.addCleanup(self.stderr.__exit__,None,None,None)

    def cache(self,doc):
        for item in doc['files'].values():
            if item['source']['kind']=='huggingface':
                path=model_download._cache_path(item['source']); path.parent.mkdir(parents=True,exist_ok=True)
                path.write_bytes(b'opaque cached input')

    def download(self,doc=None,name='model',**kwargs):
        doc=doc or self.qwen
        return models.download(self.registry,doc['id'],model_id=name,offline=True,**kwargs)

    def imported(self,alias=False):
        return models.import_model(self.registry,'local',ggufs=[self.gguf],tokenizer=self.token,
                                   alias=alias,recipe_id=self.qwen['id'])

    def test_all_three_recipes_and_plain_inspection_handoff(self):
        for doc in self.documents.values():
            if doc['id'] == 'pocket-tts-alba':
                # Pocket has its own helper/acquisition/retry coverage.
                continue
            self.cache(doc); view=self.download(doc,doc['id'])
            self.assertEqual(view['state'],'ready')
            record=self.registry.get_model(doc['id']); original=deepcopy(record)
            paths,deps,evidence=models.inspect_record(record)
            self.assertEqual(set(paths),set(doc['files']))
            self.assertEqual(set(deps),set(paths.values()))
            self.assertEqual(record,original)
            self.assertEqual(evidence,record['validation']['evidence'])
            self.assertEqual([a['prepared_ownership'] for a in record['assets']],
                             ['hf-reference' if f['source']['kind']=='huggingface' else 'generated-owned'
                              for f in doc['files'].values()])
        self.assertEqual(self.events,['preflight','prepare'])

    def test_preview_cli_no_registry_and_exact_commands(self):
        args=SimpleNamespace(command='download',recipe=self.asr['id'],id='preview',executable=None,modality=None,
                             offline=True,dry_run=True)
        with patch.object(hem_model_cli,'Registry',side_effect=AssertionError('Registry')):
            result=hem_model_cli.run(args,Deadline.after())
        self.assertEqual(result['state'],'preview'); self.assertTrue(result['completed'])
        self.assertEqual(result['artifacts'][-1]['ownership'],'generated-owned')
        self.assertEqual(self.events,[])
        self.assertFalse((self.root/'hf').exists())
        self.assertIn('hf download moonshine-ai/moonshine-streaming-tiny',self.stderr._new_target.getvalue())

    def test_passive_show_no_validation_writes_qualify_success_failure_writes(self):
        self.imported(alias=True)
        with patch.object(self.registry,'record_validation',wraps=self.registry.record_validation) as write:
            self.assertEqual(models.show(self.registry,'local')['state'],'ready'); write.assert_not_called()
            with self.registry.lock('model','local'):
                record,paths,deps,evidence=models.qualify(self.registry,'local')
            self.assertEqual(write.call_count,1)
            self.assertEqual(record['validation']['evidence'],evidence)
            self.gguf.unlink()
            self.assertEqual(models.show(self.registry,'local')['state'],'missing')
            self.assertEqual(write.call_count,1)
            with self.registry.lock('model','local'),self.assertRaises(HemError):
                models.qualify(self.registry,'local')
            self.assertEqual(write.call_count,2)
            self.assertFalse(self.registry.get_model('local')['validation']['valid'])

    def test_pending_staging_rejected_without_validation_write(self):
        with self.assertRaisesRegex(HemError,'Offline cache'): self.download()
        with patch.object(self.registry,'record_validation',side_effect=AssertionError('write')):
            with self.registry.lock('model','model'),self.assertRaisesRegex(HemError,'incomplete'):
                models.qualify(self.registry,'model')
            shown=models.show(self.registry,'model')
        self.assertEqual(shown['state'],'incomplete'); self.assertFalse(shown['accessible'])

    def test_import_copy_retry_and_alias_retarget_guard(self):
        row=self.imported(); target=Path(row['directory'])/'model.gguf'; before=target.read_bytes()
        self.gguf.write_bytes(b'changed'); self.imported()
        self.assertEqual(target.read_bytes(),before)
        alias=models.import_model(self.registry,'alias',ggufs=[self.gguf],tokenizer=self.token,
                                  alias=True,recipe_id=self.qwen['id'])
        path=Path(alias['directory'])/'model.gguf'; path.unlink(); path.symlink_to(self.token)
        with self.assertRaisesRegex(HemError,'mapping changed'): models.inspect_record(alias)

    def test_split_index_and_complete_dependency_list(self):
        second=self.root/'second.gguf'; write_gguf(second,tensors=[('two',[1],0,0)])
        row=models.import_model(self.registry,'split',ggufs=[self.gguf,second],tokenizer=self.token,
                                alias=True,recipe_id=self.qwen['id'])
        paths,deps,evidence=models.inspect_record(row)
        self.assertTrue(str(paths['weights']).endswith('.safetensors.index.json'))
        self.assertEqual(len(deps),4)
        self.assertEqual(set(json.loads(paths['weights'].read_text())['weight_map']),{'one','two'})

    def test_complete_prepared_import_no_helper_or_hf(self):
        root=self.root/'complete'; root.mkdir()
        for item in self.asr['files'].values(): (root/item['path']).write_bytes(b'caller prepared')
        row=models.import_assets(self.registry,'prepared',directory=root,recipe_id=self.asr['id'])
        self.assertEqual(row['state'],'ready'); self.assertEqual(self.events,[])
        self.assertTrue(all(a['prepared_ownership']=='caller-owned' for a in row['assets']))

    def test_preflight_failure_precedes_fetch_and_fresh_prepare_budget(self):
        module=self.preparer
        with patch.object(module,'preflight',side_effect=HemError('tool missing')):
            with self.assertRaisesRegex(HemError,'tool missing'): self.download(self.asr)
        clock=[0.0]; run=module.prepare
        def preflight(deadline): self.assertEqual(deadline.remaining(),30); clock[0]+=25
        def fetch(command):
            clock[0]+=400
            source=next(f['source'] for f in self.asr['files'].values()
                        if f['source'].get('filename')==command[3])
            path=model_download._cache_path(source); path.parent.mkdir(parents=True,exist_ok=True); path.write_bytes(b'input')
        def prepare(inputs,attempt,deadline):
            self.assertEqual(deadline.remaining(),300); return run(inputs,attempt,deadline)
        with patch.object(module,'preflight',side_effect=preflight),patch.object(module,'prepare',side_effect=prepare), \
             patch.object(model_download,'_run_hf',side_effect=fetch),patch.object(models,'Deadline') as deadlines:
            deadlines.after.side_effect=lambda seconds=300:Deadline(clock[0]+seconds,clock=lambda:clock[0])
            models.download(self.registry,self.asr['id'],model_id='model',offline=False)

    def test_generated_bad_result_and_receipt_identity(self):
        self.cache(self.asr); module=self.preparer
        with patch.object(module,'prepare',return_value={'rotary.safetensors':self.gguf}):
            with self.assertRaisesRegex(HemError,'Incomplete preparation'): self.download(self.asr)
        self.download(self.asr)
        row=self.registry.get_model('model'); root=Path(row['directory'])
        receipt=root/models._GENERATED_RECEIPT; original=receipt.read_bytes()
        (root/'rotary.safetensors').write_bytes(b'drift')
        with self.assertRaisesRegex(HemError,'identity is unproven'): self.download(self.asr)
        self.assertEqual(receipt.read_bytes(),original)

    def test_rename_recovery_keeps_original_generated_receipt(self):
        self.cache(self.asr); update=self.registry.update_model
        def crash(model_id,changes,**kwargs):
            if changes.get('pending')=={'stage':'qualification'}: raise RuntimeError('after rename')
            return update(model_id,changes,**kwargs)
        with patch.object(self.registry,'update_model',side_effect=crash):
            with self.assertRaises(RuntimeError): self.download(self.asr)
        row=self.registry.get_model('model'); root=Path(row['directory'])
        proof=(root/models._GENERATED_RECEIPT).read_bytes(); self.download(self.asr)
        self.assertEqual((root/models._GENERATED_RECEIPT).read_bytes(),proof)
        self.assertEqual(self.events,['preflight','prepare'])

    def test_same_id_identity_conflict_and_cli_avoids_post_success_show(self):
        self.cache(self.qwen); self.download()
        with patch.object(recipes,'load',return_value=(deepcopy(self.qwen),'f'*64)):
            with self.assertRaisesRegex(HemError,'different mappings'): self.download()
        args=SimpleNamespace(command='download',recipe=self.qwen['id'],id='second',executable=None,modality=None,
                             offline=True,dry_run=False)
        with patch.object(hem_model_cli,'Registry',return_value=self.registry), \
             patch.object(models,'show',side_effect=AssertionError('second inspection')):
            result=hem_model_cli.run(args,Deadline.after())
        self.assertEqual(result['state'],'ready')
