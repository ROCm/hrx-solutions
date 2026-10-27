"""Data-only compiler/packaging onboarding with explicit structural program fixtures."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import pprint
import tempfile
import unittest
from unittest.mock import patch

from binding.python.models import pipelines, recipes
from build_tools.model_tools import loom_catalog as compiler, package_families as packaging



class UniformCatalogTest(unittest.TestCase):
    def setUp(self):
        temporary=tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup)
        self.root=Path(temporary.name);self.repository=self.root/'source'
        self.catalogs=self.repository/'loom-pipeline';self.catalogs.mkdir(parents=True)
        self.builds={};self.binary={};self.decoded={};self.queried=[]
        self.decoder=self.root/'decoder';self.decoder.touch()
        patch.object(compiler.subprocess,'run',side_effect=self.decode).start()
        self.addCleanup(patch.stopall)
        (self.catalogs/'NOTICE').write_text('authored notice\n')

    def decode(self,arguments,**kwargs):
        self.assertTrue(kwargs['check'])
        Path(arguments[-1].removeprefix('--output=')).write_text(self.decoded[Path(arguments[1])])

    def add(self,folder,identifier,modality,*,recipe=True):
        directory=self.catalogs/folder;directory.mkdir(parents=True)
        names=['prepare.loom','control.loom','provider.loom','policy.loom']
        policy='global.rodata.def @policy = bytes("'+b'policy.loom'.hex()+'")\n'
        for index,name in enumerate(names):
            (directory/name).write_text('func.def public @root_'+str(index)+'()\n'+(policy if index==0 else ''))
        hashes={name:compiler.digest((directory/name).read_bytes()) for name in names}
        if modality=='chat':
            serving=dict(bounds={key:[1,1024] for key in ('rows','context_capacity','pool_capacity','prefill_capacity')},
                         mtp_depth_max=0,defaults=dict(rows=2,context_capacity=64,pool_capacity=128,prefill_capacity=16,mtp=False))
            assets={'weights':{'format':'gguf','path':None},'tokenizer':{'format':'json','path':None}}
            groups={'engine.loombc':dict(sources=[folder+'/'+name for name in names],
                policy={'source':folder+'/prepare.loom','from':'policy.loom','to':'engine.loombc'},exact_configs=True)}
            roles=dict(prepare='engine.loombc',control='engine.loombc',sources=['engine.loombc'],support=[])
            contract='text-v1'
        else:
            serving=dict(pending_requests=[1,8],defaults={'pending_requests':2,'options':{}})
            assets={'features':{'format':'json','path':'features.json'}}
            groups={output:dict(sources=[folder+'/'+name],
                policy=({'source':folder+'/'+name,'from':'policy.loom','to':'policy.loombc'} if name=='prepare.loom' else None),
                exact_configs=True) for name,output in zip(names,['bootstrap.loombc','loop.loombc','providers.loombc','policy.loombc'])}
            roles=dict(prepare='bootstrap.loombc',control='loop.loombc',sources=['providers.loombc'],support=['policy.loombc'])
            contract='recognition-v1'
        source=dict(format=1,id=identifier,modality=modality,native_contract=contract,serving=serving,assets=assets,
            programs=dict(root=folder,encoding='source',prepare='prepare.loom',control='control.loom',sources=['provider.loom'],support=['policy.loom'],files=hashes),
            preparation=None,provenance=None)
        reference=folder+'/executable.json'
        (directory/'executable.json').write_text(json.dumps(source,indent=2)+'\n')
        build=dict(root=folder,groups=groups,roles=roles,notices=['NOTICE'],helper=None)
        self.builds[reference]=build
        for output,group in groups.items():
            artifact=self.root/'bazel-out/bin/loom-pipeline/bytecode'/folder/output
            artifact.parent.mkdir(parents=True,exist_ok=True);artifact.write_bytes(b'LOOMfixture:'+identifier.encode()+output.encode())
            self.binary[folder+'/'+output]=artifact
            texts=[]
            for name in group['sources']:
                text=(self.catalogs/name).read_text()
                if group['policy'] and name==group['policy']['source']:
                    text=compiler.preparation_bytecode_source(text,group['policy']['from'],group['policy']['to'])
                texts.append(text)
            self.decoded[artifact]='\n'.join(texts)
        if recipe:self.add_recipe(reference,identifier+'-preset',source)
        self.save()
        return reference,source

    def add_recipe(self,reference,identifier,source):
        directory=self.catalogs/Path(reference).parent/'recipes';directory.mkdir(exist_ok=True)
        files={purpose:dict(path=entry['path'] or ('model.gguf' if purpose=='weights' else 'tokenizer.json'),
            source=dict(kind='huggingface',repo_id='fixture/assets',revision='a'*40,
                        filename=entry['path'] or ('model.gguf' if purpose=='weights' else 'tokenizer.json')))
               for purpose,entry in source['assets'].items()}
        document=dict(format=3,id=identifier,name=identifier,executable=reference,files=files,configuration={})
        path=directory/(identifier+'.json');path.write_text(json.dumps(document)+'\n');return path

    def save(self):
        (self.catalogs/'catalog_sources.bzl').write_text('CATALOG_BUILDS = '+pprint.pformat(self.builds,sort_dicts=False)+'\n')

    def query(self,target):
        self.queried.append(target)
        return tuple(self.binary.values()) if target==compiler.CATALOG_TARGET else (self.decoder,)

    def stage(self,**kwargs):
        prepared=packaging.prepare(self.query,self.repository,**kwargs)
        package=self.root/('package-'+str(len(list(self.root.glob('package-*')))))
        outputs=packaging.stage(package,prepared,self.repository)
        return package,outputs,prepared

    def test_new_chat_asr_and_existing_recipe_are_data_only_source_to_compiled(self):
        source_paths=[Path(compiler.__file__),Path(packaging.__file__)]
        before={str(p):compiler.digest(p.read_bytes()) for p in source_paths}
        first,chat=self.add('catalogs/unseen-text','unseen-text-v9','chat')
        self.stage(selected=(first,))
        second,asr=self.add('audio/unseen-recognizer','unseen-recognition-v6','asr')
        extra=self.add_recipe(first,'different-quantization',chat)
        package,outputs,prepared=self.stage()
        self.assertEqual({str(p):compiler.digest(p.read_bytes()) for p in source_paths},before)
        self.assertEqual(self.queried,[compiler.CATALOG_TARGET,compiler.DECODER_TARGET]*2)
        self.assertEqual(len(outputs),len(set(outputs)))
        self.assertEqual(set(outputs),{p for p in package.rglob('*') if p.is_file()})
        self.assertEqual((package/'loom-pipeline'/extra.relative_to(self.catalogs)).read_bytes(),extra.read_bytes())
        for reference,source in ((first,chat),(second,asr)):
            compiled=json.loads((package/'loom-pipeline'/reference).read_bytes())
            for field in ('id','modality','native_contract','serving','assets','preparation'):
                self.assertEqual(compiled[field],source[field])
            self.assertEqual(compiled['programs']['encoding'],'bytecode')
            self.assertEqual(compiled['provenance'],dict(source_declaration_sha256=compiler.digest((self.catalogs/reference).read_bytes()),
                                                       source_files=source['programs']['files']))
            selected=package/'loom-pipeline'/compiled['programs']['root']
            self.assertEqual({p.relative_to(selected).as_posix() for p in selected.rglob('*.loombc')},set(compiled['programs']['files']))
            self.assertTrue((package/'loom-pipeline'/Path(reference).parent/'build.json').is_file())
        self.assertFalse(list(package.rglob('*.loom')))
        self.assertFalse(list(package.rglob('*.gguf')))
        self.assertFalse(list(package.rglob('*.safetensors')))
        self.assertEqual(packaging.stage_helpers(package/'_native',prepared,self.root/'absent-hrx',repository=self.repository),[])
        self.assertEqual(sum(p.name=='NOTICE' for p in outputs),1)

    def test_executable_without_recipe_is_still_a_selected_build_output(self):
        reference,_=self.add('empty-recipes/catalog','program-without-recipe','chat',recipe=False)
        package,_,_=self.stage()
        self.assertTrue((package/'loom-pipeline'/reference).is_file())
        self.assertFalse(list(package.glob('**/recipes/*.json')))

    def test_selected_catalog_ignores_unrelated_metadata_but_global_rejects(self):
        good,_=self.add('alpha/native','alpha-native','chat')
        bad,_=self.add('beta/native','beta-native','asr')
        (self.catalogs/bad).write_text('{malformed')
        package,_,_=self.stage(selected=(good,))
        self.assertFalse((package/'loom-pipeline'/bad).exists())
        with self.assertRaises(ValueError):self.stage()

    def test_source_hash_build_closure_and_missing_products_fail_before_stage(self):
        reference,_=self.add('proof/native','proof-native','chat')
        self.binary.clear()
        with self.assertRaisesRegex(ValueError,'Missing selected compiler'):self.stage()
        self.assertFalse(list(self.root.glob('package-*')))
        (self.catalogs/'proof/native/control.loom').write_text('changed source')
        with self.assertRaisesRegex(ValueError,'source hash'):self.stage()
        self.assertFalse(list(self.root.glob('package-*')))

    def test_wrong_decoder_closure_and_policy_are_not_blessed_by_hashes(self):
        self.add('checks/native','checks-native','chat')
        artifact=next(iter(self.decoded))
        original=self.decoded[artifact]
        self.decoded[artifact]=original.replace('public @root_1','public @wrong')
        with self.assertRaisesRegex(ValueError,'closure'):self.stage()
        self.decoded[artifact]=original.replace(b'engine.loombc'.hex(),b'policy.loom'.hex())
        with self.assertRaisesRegex(ValueError,'policy'):self.stage()


if __name__=='__main__':unittest.main()
