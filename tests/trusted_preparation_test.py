"""Pure operation admission: recipe declarations cannot extend the trusted helper."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch

from binding.python.hem_common import HemError
from binding.python.models.preparation import moonshine_contract as identity
from binding.python.models import preparation


class TrustedPreparationTest(unittest.TestCase):
    def setUp(self):
        self.operation = identity.load_operation()
        self.files = {}
        self.assets = {}
        inputs = {}
        for index, (name, source) in enumerate(self.operation['inputs'].items()):
            purpose = 'input' + str(index)
            inputs[name] = purpose
            self.files[purpose] = {'path': name, 'source': deepcopy(source)}
            self.assets[purpose] = {'format': Path(name).suffix[1:], 'path': name}
        outputs = {}
        for name, output in self.operation['outputs'].items():
            purpose = output['purpose']; outputs[name] = purpose
            self.files[purpose] = {'path': name, 'source': {'kind': 'generated'}}
            self.assets[purpose] = {'format': output['format'], 'path': name}
        self.binding = {'operation': self.operation['id'], 'inputs': inputs, 'outputs': outputs}

    def test_current_exact_contract_and_nonpreparing_admission(self):
        self.assertEqual(preparation.validate_preparation(self.binding, self.files, self.assets), self.operation)
        self.assertIsNone(preparation.validate_preparation(None, {'weights': {'source': {'kind': 'huggingface'}}}, {}))
        with self.assertRaisesRegex(HemError, 'require'):
            preparation.validate_preparation(None, self.files, self.assets)

    def test_repin_filename_and_invented_output_cannot_extend_helper(self):
        for field, value in [('revision', '0' * 40), ('repo_id', 'other/model'), ('filename', 'other.json')]:
            files = deepcopy(self.files)
            files[next(iter(self.binding['inputs'].values()))]['source'][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(HemError, 'pinned'):
                preparation.validate_preparation(self.binding, files, self.assets)
        for mutation in ('operation', 'inputs', 'outputs'):
            binding = deepcopy(self.binding)
            if mutation == 'operation': binding[mutation] = 'arbitrary-import.module'
            else: binding[mutation]['extra'] = 'rotary'
            with self.subTest(mutation=mutation), self.assertRaises(HemError):
                preparation.validate_preparation(binding, self.files, self.assets)

    def test_generated_output_destination_format_and_coverage_are_checked(self):
        files = deepcopy(self.files); files['rotary']['path'] = 'elsewhere.safetensors'
        with self.assertRaises(HemError): preparation.validate_preparation(self.binding, files, self.assets)
        assets = deepcopy(self.assets); assets['rotary']['format'] = 'json'
        with self.assertRaises(HemError): preparation.validate_preparation(self.binding, self.files, assets)
        files = deepcopy(self.files); files['extra'] = {'path': 'extra.json', 'source': {'kind': 'generated'}}
        with self.assertRaises(HemError): preparation.validate_preparation(self.binding, files, self.assets)

    def test_missing_or_malformed_unselected_operation_does_not_block_module_import(self):
        source = Path(identity.__file__).read_text()
        with tempfile.TemporaryDirectory() as temporary:
            package = Path(temporary) / 'binding/python'
            native = package / '_native'; native.mkdir(parents=True)
            operation = native / identity.RELATIVE / 'operation.json'
            for malformed in (False, True):
                if malformed:
                    operation.parent.mkdir(parents=True); operation.write_text('{broken')
                module = ModuleType('binding.python.models.isolated_identity')
                module.__package__ = 'binding.python.models.preparation'
                module.__file__ = str(package / 'models/preparation/moonshine_contract.py')
                exec(compile(source, module.__file__, 'exec'), module.__dict__)
                self.assertIsNone(preparation.validate_preparation(None, {}, {}))
                with self.assertRaises(HemError): module.load_operation()

    def test_trusted_document_rejects_duplicates_bad_identity_and_hashes(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'operation.json'
            with patch.object(identity, 'operation_path', return_value=path):
                for field, value in [('format', True), ('tool_revision', '../x'), ('id', 'other'),
                                     ('tool_files', {'prepare_assets.py': 'x'})]:
                    document = deepcopy(self.operation); document[field] = value
                    path.write_text(json.dumps(document))
                    with self.subTest(field=field), self.assertRaises(HemError): identity.load_operation()
                path.write_text('{"format":1,"format":1}')
                with self.assertRaisesRegex(HemError, 'duplicate'): identity.load_operation()
                path.unlink(); target = Path(temporary) / 'target'; target.write_text(json.dumps(self.operation))
                path.symlink_to(target)
                with self.assertRaisesRegex(HemError, 'alias'): identity.load_operation()


if __name__ == '__main__':
    unittest.main()
