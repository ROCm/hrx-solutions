"""Focused contract/implementation separation; caller outcome proof is separate."""
from copy import deepcopy
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from binding.python.hem_common import Deadline, HemError
from binding.python.models import preparation
from binding.python.models.preparation import moonshine_contract


class PreparationBoundaryTest(unittest.TestCase):
    def test_admission_never_imports_converter(self):
        code = '''
import sys
from binding.python.models import preparation
assert preparation.validate_binding(None, {}) is None
assert preparation.validate_preparation(None, {}, {}) is None
assert not any(name.endswith('.moonshine_contract') for name in sys.modules)
contract = preparation.load_operation('moonshine-assets-v1')
assert contract['id'] == 'moonshine-assets-v1'
assert not any(name.endswith('.preparation.moonshine') for name in sys.modules)
assert 'huggingface_hub' not in sys.modules and 'safetensors' not in sys.modules
'''
        subprocess.run([sys.executable, '-c', code], check=True)

    def fixture(self):
        operation = preparation.load_operation('moonshine-assets-v1')
        inputs = {name: 'input' + str(index) for index, name in enumerate(operation['inputs'])}
        outputs = {name: value['purpose'] for name, value in operation['outputs'].items()}
        files = {inputs[name]: {'path': name, 'source': deepcopy(source)}
                 for name, source in operation['inputs'].items()}
        assets = {purpose: {'path': name, 'format': name.rsplit('.', 1)[1]}
                  for name, purpose in inputs.items()}
        for name, purpose in outputs.items():
            files[purpose] = {'path': name, 'source': {'kind': 'generated'}}
            assets[purpose] = {'path': name, 'format': operation['outputs'][name]['format']}
        return operation, {'operation': operation['id'], 'inputs': inputs, 'outputs': outputs}, files, assets

    def test_real_contract_bindings_pins_and_completeness(self):
        operation, binding, files, assets = self.fixture()
        self.assertEqual(preparation.validate_binding(binding, assets), operation)
        self.assertEqual(preparation.validate_preparation(binding, files, assets), operation)
        changed = deepcopy(files)
        changed[next(iter(binding['inputs'].values()))]['source']['revision'] = '0' * 40
        with self.assertRaisesRegex(HemError, 'pinned'):
            preparation.validate_preparation(binding, changed, assets)
        changed = deepcopy(files)
        changed['extra'] = {'path': 'extra.json', 'source': {'kind': 'generated'}}
        with self.assertRaisesRegex(HemError, 'every generated'):
            preparation.validate_preparation(binding, changed, assets)

    def test_invalid_unknown_and_unsafe_bindings_fail_closed(self):
        _, binding, files, assets = self.fixture()
        for operation in ('untrusted-code', [], None):
            with self.subTest(operation=operation), self.assertRaises(HemError):
                preparation.load_operation(operation)
        for mutation in ('duplicate', 'unsafe', 'missing', 'output'):
            changed = deepcopy(binding)
            if mutation == 'duplicate':
                names = list(changed['inputs'])
                changed['inputs'][names[1]] = changed['inputs'][names[0]]
            elif mutation == 'unsafe': changed['inputs']['../escape'] = 'rotary'
            elif mutation == 'missing': changed['inputs'].pop(next(iter(changed['inputs'])))
            else: changed['outputs']['rotary.safetensors'] = next(iter(changed['inputs'].values()))
            with self.subTest(mutation=mutation), self.assertRaises(HemError):
                preparation.validate_preparation(changed, files, assets)
        with self.assertRaises(HemError): preparation.validate_preparation(None, files, assets)

    def test_distinct_implementation_dispatch_stays_inside_boundary(self):
        calls = []
        contract = {'id': 'test-only-transform'}
        executor = SimpleNamespace(preflight=lambda deadline: calls.append(('preflight', deadline)),
                                   prepare=lambda inputs, attempt, deadline: (inputs, attempt, deadline))
        with patch.dict(preparation._IMPLEMENTATIONS,
                        {'test-only-transform': (lambda: contract, lambda: executor)}):
            self.assertIs(preparation.load_operation('test-only-transform'), contract)
            deadline = Deadline.after()
            preparation.preflight('test-only-transform', deadline)
            self.assertEqual(calls, [('preflight', deadline)])
            self.assertEqual(preparation.prepare('test-only-transform', {'source': 'path'}, 'scratch', deadline),
                             ({'source': 'path'}, 'scratch', deadline))

    def test_selected_contract_errors_remain_domain_errors(self):
        with patch.object(moonshine_contract, 'operation_path', side_effect=OSError('missing')):
            with self.assertRaisesRegex(HemError, 'operation contract'):
                preparation.load_operation('moonshine-assets-v1')


if __name__ == '__main__':
    unittest.main()
