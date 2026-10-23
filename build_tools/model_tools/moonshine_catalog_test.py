"""Selected source ownership, compiled closure and archive integrity contracts."""
from copy import deepcopy
import io
import json
from pathlib import Path
import shutil
import tarfile
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from build_tools.model_tools import moonshine_catalog as catalog, loom_catalog
from binding.python.models import curated_catalog
from binding.python.hem_common import InvalidInput

REPOSITORY = Path(__file__).resolve().parents[2]


class MoonshineCatalogTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / 'loom-pipeline' / catalog.RELATIVE
        self.source.parent.mkdir(parents=True)
        shutil.copytree(REPOSITORY/'loom-pipeline'/catalog.RELATIVE,self.source)
        shutil.copyfile(REPOSITORY/'loom-pipeline/moonshine_sources.bzl',self.root/'loom-pipeline/moonshine_sources.bzl')
        self.groups = catalog.validate_sources(self.root)

    def products(self):
        paths = {}
        for name in catalog.ARTIFACTS:
            path = self.root / 'build' / name; path.parent.mkdir(exist_ok=True)
            path.write_bytes(b'LOOMfixture'); paths[name] = path
        return paths

    def stage(self):
        def decoded(path,sources,root,decoder,policy=False):
            return catalog.roots('\n'.join((root/source).read_text() for source in sources))
        destination = self.root/'installed'
        with patch.object(catalog,'validate_artifact',side_effect=decoded):
            catalog.stage(destination,self.products(),self.root,decoder=Path('fixture'))
        return destination

    def test_bound_query_retains_four_artifact_parser_and_rejections(self):
        outputs={}
        for name in catalog.ARTIFACTS:
            path=self.root/'bazel-bin'/catalog.RELATIVE/name
            path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'LOOMfixture')
            outputs[name]=path
        def query(target):
            self.assertEqual(target,catalog.TARGET)
            return tuple(outputs.values())
        self.assertEqual(catalog.query_outputs(query),outputs)
        paths=tuple(outputs.values())
        for invalid in (paths[:-1],paths+paths[:1],paths+(self.root/'extra.loombc',)):
            with self.assertRaisesRegex(ValueError,'Missing|Unexpected|duplicate'):
                catalog.query_outputs(lambda target:invalid)
        paths[0].write_bytes(b'bad')
        with self.assertRaisesRegex(ValueError,'invalid Moonshine'):
            catalog.query_outputs(query)

    def test_exact_four_artifacts_and_physically_owned_sixteen_inputs(self):
        self.assertEqual(len({source for group in self.groups.values() for source in group}),16)
        self.assertEqual(set(self.groups),set(catalog.ARTIFACTS))
        self.assertEqual(catalog.roots((self.source/'reset.loom').read_text())['commands'],['reset'])
        self.assertEqual(catalog.roots((self.source/'reset.loom').read_text())['configs'],['moonshine.threads'])
        (self.source/'encoder.loom').write_text('changed numerical program')
        with self.assertRaisesRegex(ValueError,'provenance'):
            catalog.validate_sources(self.root)

    def test_unknown_or_unowned_sources_and_mapping_are_rejected(self):
        (self.source/'extra.loom').write_text('extra')
        with self.assertRaisesRegex(ValueError,'Undeclared'):
            catalog.validate_sources(self.root)
        (self.source/'extra.loom').unlink()
        external=self.root/'external.loom';shutil.copyfile(self.source/'encoder.loom',external)
        (self.source/'encoder.loom').unlink();(self.source/'encoder.loom').symlink_to(external)
        with self.assertRaisesRegex(ValueError,'physically'):
            catalog.validate_sources(self.root)

    def test_decoded_function_command_config_closure_and_policy_projection(self):
        def decode(arguments,**kwargs):
            Path(arguments[-1].removeprefix('--output=')).write_text(decoded)
        artifact=self.products()['providers.loombc']
        inputs='\n'.join((self.source/name).read_text() for name in self.groups['providers.loombc'])
        decoded=inputs
        with patch.object(catalog.subprocess,'run',side_effect=decode):
            expected=catalog.validate_artifact(artifact,self.groups['providers.loombc'],self.source,Path('fixture'))
            self.assertEqual(expected['commands'],['decode','encode','reset'])
            decoded=inputs.replace('command.program.def public @encode','command.program.def @encode',1)
            with self.assertRaisesRegex(ValueError,'closure'):
                catalog.validate_artifact(artifact,self.groups['providers.loombc'],self.source,Path('fixture'))
            decoded=loom_catalog.preparation_bytecode_source((self.source/'prepare.loom').read_text(),'weights.loombc')
            catalog.validate_artifact(self.products()['prepare.loombc'],['prepare.loom'],self.source,Path('fixture'),True)
            decoded=decoded.replace(b'weights.loombc'.hex(),b'weights.loom'.hex())
            with self.assertRaisesRegex(ValueError,'policy'):
                catalog.validate_artifact(self.products()['prepare.loombc'],['prepare.loom'],self.source,Path('fixture'),True)

    def test_installed_inventory_policy_integrity_and_missing_metadata(self):
        installed=self.stage()
        entry=curated_catalog.load_catalog('moonshine-streaming-tiny',directory=installed)
        self.assertEqual(entry.definition['sources'],['providers.loombc'])
        self.assertFalse(list(installed.glob('*.loom')))
        self.assertEqual(set(path.name for path in installed.iterdir()),set((*catalog.ARTIFACTS,*catalog.DATA,'artifacts.json')))
        policy=installed/'weights.loombc';original=policy.read_bytes();policy.write_bytes(b'LOOMtampered')
        with self.assertRaisesRegex(InvalidInput,'integrity'):
            curated_catalog.load_catalog('moonshine-streaming-tiny',directory=installed)
        policy.write_bytes(original);(installed/'artifacts.json').unlink()
        with self.assertRaisesRegex(InvalidInput,'requires artifact'):
            curated_catalog.load_catalog('moonshine-streaming-tiny',directory=installed)

    def test_wheel_allowlist_missing_extra_tampered_and_wrong_native_mapping(self):
        installed=self.stage()
        original={path.name:path.read_bytes() for path in installed.iterdir()}
        cases=[None,'missing-policy','extra-program','tampered','native-source-reference','notice']
        for case in cases:
            members=dict(original)
            if case=='missing-policy':members.pop('weights.loombc')
            elif case=='extra-program':members['hidden.loombc']=b'LOOMextra'
            elif case=='tampered':members['weights.loombc']=b'LOOMwrong'
            elif case=='native-source-reference':
                document=json.loads(members['catalog.json']);document['definition']['control']='control.loom';members['catalog.json']=json.dumps(document).encode()
            elif case=='notice':members['LICENSE.moonshine']=b'wrong'
            path=self.root/'test.whl'
            with zipfile.ZipFile(path,'w') as archive:
                for name,content in members.items():archive.writestr('hrx_solutions/loom-pipeline/'+catalog.RELATIVE+'/'+name,content)
            with zipfile.ZipFile(path) as archive:
                if case is None:catalog.validate_archive(archive,self.root)
                else:
                    with self.subTest(case=case),self.assertRaises(ValueError):catalog.validate_archive(archive,self.root)

    def test_source_archive_owns_inputs_and_excludes_payloads(self):
        def archive(extra=None,missing=None):
            path=self.root/'source.tar.gz'
            with tarfile.open(path,'w:gz') as tar:
                for source in self.source.iterdir():
                    if source.name!=missing:tar.add(source,arcname='package/loom-pipeline/'+catalog.RELATIVE+'/'+source.name)
                if extra:
                    info=tarfile.TarInfo('package/'+extra);info.size=1;tar.addfile(info,io.BytesIO(b'x'))
            return tarfile.open(path)
        with archive() as tar:catalog.validate_source_archive(tar,self.root)
        with archive(missing='encoder.loom') as tar,self.assertRaisesRegex(ValueError,'missing'):
            catalog.validate_source_archive(tar,self.root)
        with archive(extra='recording.wav') as tar,self.assertRaisesRegex(ValueError,'payload'):
            catalog.validate_source_archive(tar,self.root)

    def test_recognition_source_abi_and_framing_bounds(self):
        prepare=(self.source/'prepare.loom').read_text(); control=(self.source/'control.loom').read_text()
        signature='-> ('+', '.join(['i64']*14+['i32']+['buffer']*4)+')'
        self.assertIn(signature,prepare)
        self.assertIn('%event_capacity = scalar.constant 1 : i64',prepare)
        self.assertIn('%outputs = scalar.constant 1 : i32',prepare)
        self.assertIn('%event_bytes = scalar.constant 16444 : i64',prepare)
        self.assertIn('%temperature: f32, %outputs: i32) -> (buffer, buffer)',control)
        self.assertIn('%outputs_ok = scalar.cmpi eq, %outputs, %transcript_output : i32',control)
        self.assertIn('func.call @transcript(%text, %length, %revision, %end, %final)',control)
        self.assertIn('%prefix_length = index.constant 12 : offset',control)
        self.assertIn('view.store %one, %header[0]',control)
        self.assertIn('view.store %zero, %header[1]',control)
        self.assertIn('view.store %length32, %header[2]',control)
        self.assertIn('func.return %count, %packet, %event_bytes, %complete : i64, buffer, i64, i32',control)
        # Empty reset/no-event paths remain genuinely empty, not a count-zero packet.
        self.assertIn('scf.yield %zero64, %empty, %zero : i64, buffer, i32',control)
        self.assertIn('%packet_length = buffer.length %packet',control)


if __name__=='__main__':unittest.main()
