import copy
import unittest
from tests.native_lifecycle_fixture import activated, eager_start, exact


def status(**updates):
    loom = dict(incarnation=1, registration='ready', parameter_operation='idle',
                parameters={'active': True}, status={'admission': 'drained'})
    loom.update(updates)
    return {'data': [{'id': 'sample', 'loom': loom}]}


class NativeLifecycleFixtureTest(unittest.TestCase):
    def test_readiness_requires_completed_activation_and_closed_admission(self):
        self.assertTrue(activated(status(), 'sample', 1))
        for change in [dict(registration='registered'),
                       dict(parameter_operation='activating'),
                       dict(parameters={'active': False}),
                       dict(status={'admission': 'open'})]:
            with self.subTest(change=change):
                self.assertFalse(activated(status(**change), 'sample', 1))
        with self.assertRaises(AssertionError):
            activated(status(error='failed'), 'sample', 1)

    def test_replacement_does_not_keep_original_incarnation_alive(self):
        self.assertIsNone(exact(status(), 'sample', 2))

    def test_failed_activation_observation_does_not_open_admission(self):
        calls = []
        metadata = status(registration='registered', parameters={'active': False})
        def request(path, body=None):
            calls.append(path)
            return (200, copy.deepcopy(metadata)) if path.endswith('status') else (202, {})
        def observe(predicate):
            raise TimeoutError('activation deadline')
        with self.assertRaises(TimeoutError):
            eager_start(request, observe, {'name': 'sample'})
        self.assertEqual(calls, ['/v1/models/register', '/v1/status', '/v1/models/activate'])

    def test_metadata_only_registration_can_omit_parameters(self):
        calls = []
        metadata = status(registration='registered')
        del metadata['data'][0]['loom']['parameters']
        opened = status(status={'admission': 'open'})
        def request(path, body=None):
            calls.append(path)
            if path == '/v1/status':
                return 200, opened if '/v1/models/undrain' in calls else metadata
            return (200 if path == '/v1/models/undrain' else 202), {}
        def observe(predicate):
            self.assertTrue(predicate(status()))
            return status()
        original, ready = eager_start(request, observe, {'name': 'sample'})
        self.assertEqual(original, 1)
        self.assertEqual(ready, status())


if __name__ == '__main__':
    unittest.main()
