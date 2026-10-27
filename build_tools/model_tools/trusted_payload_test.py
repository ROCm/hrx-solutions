"""Selected trusted payload staging; synthetic tools do not qualify real preparation."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from build_tools.model_tools import loom_catalog as producer
from binding.python.models.preparation.moonshine_contract import load_operation


class TrustedPayloadTest(unittest.TestCase):
    def setUp(self):
        temporary=tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root=Path(temporary.name)
        self.hrx=self.root/'hrx'
        self.tools=self.hrx/'experimental/loom_serve/models/moonshine'
        self.tools.mkdir(parents=True)
        self.payload={'prepare_assets.py': b'# trusted tool fixture\n','model.json':b'{}\n'}
        self.operation=deepcopy(load_operation())
        self.operation['tool_files']={name:hashlib.sha256(data).hexdigest() for name,data in self.payload.items()}
        for name,data in self.payload.items():(self.tools/name).write_bytes(data)
        (self.hrx/'LICENSE').write_bytes(b'license fixture')
        self.contract=self.root/'operation.json'
        self.contract.write_text(json.dumps(self.operation,indent=2)+'\n')
        self.native=self.root/'package/_native'
        self.helper=dict(contract='operation.json',upstream='experimental/loom_serve/models/moonshine',
                         destination='_native/share/loom-serve/preparers/moonshine')

    def stage(self, native):
        outputs={}
        producer.stage_trusted_payload(native.parent,self.root,self.hrx,self.helper,self.operation['id'],outputs)
        return list(outputs.values())

    def test_selected_stage_includes_exact_operation_tools_license_and_provenance(self):
        files=self.stage(self.native)
        self.assertEqual({p.name for p in files},{'operation.json','prepare_assets.py','model.json','LICENSE','provenance.json'})
        self.assertTrue(all(p.is_file() and not p.is_symlink() for p in files))
        destination=self.native/'share/loom-serve/preparers/moonshine'
        self.assertEqual((destination/'operation.json').read_bytes(),self.contract.read_bytes())
        proof=json.loads((destination/'provenance.json').read_text())
        self.assertEqual(proof['files'],self.operation['tool_files'])
        self.assertEqual(proof['revision'],self.operation['tool_revision'])

    def test_changed_tool_or_contract_is_rejected(self):
        (self.tools/'prepare_assets.py').write_bytes(b'changed tool')
        with self.assertRaisesRegex(ValueError,'selected operation'):self.stage(self.native)
        self.contract.write_bytes(b'{}')
        with self.assertRaisesRegex(ValueError,'contract'):self.stage(self.root/'other-native')

    def test_import_does_not_read_unselected_operation(self):
        code=("from unittest.mock import patch; "
              "p=patch('binding.python.models.preparation.moonshine_contract.load_operation',side_effect=AssertionError('unselected operation')); "
              "p.start(); import build_tools.model_tools.package_families")
        subprocess.run([sys.executable,'-c',code],check=True,capture_output=True,text=True)


if __name__=='__main__':unittest.main()
