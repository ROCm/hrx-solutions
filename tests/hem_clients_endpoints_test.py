"""Client-independent endpoint ownership/publication and lifecycle regression."""
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

PACKAGE = os.environ.get('HEM_TEST_PACKAGE', 'binding.python')
clients = importlib.import_module(PACKAGE + '.integrations.hem_clients_endpoints')
common = importlib.import_module(PACKAGE + '.hem_common')
contracts = importlib.import_module(PACKAGE + '.integrations.pi.hem_pi_contracts')
state = importlib.import_module(PACKAGE + '.integrations.pi.hem_pi_state')
Registry = importlib.import_module(PACKAGE + '.hem_state').Registry
from tests import hem_server_cli_test as fixtures


class ProjectionTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.registry = Registry(Path(self.temp.name) / 'state')
        self.path = self.registry.root / clients.ENDPOINT_PATH
        self.registry.create_server('a.b', saved_settings=dict(device='task', listen='[::1]:12345'))

    def ensure(self):
        return clients.ensure_endpoints(self.registry, common.Deadline.after())

    def publish(self):
        return clients.publish_endpoints(self.registry, common.Deadline.after())

    def test_opt_in_without_pi_or_target_registry_idempotent_complete_snapshot(self):
        before = set(self.registry.root.rglob('*'))
        self.assertIsNone(clients.read_endpoints(self.registry.root))
        self.assertFalse(self.publish()['changed'])
        self.assertEqual(before, set(self.registry.root.rglob('*')))
        self.assertIsNone(state.get_target(self.registry))
        self.assertTrue(self.ensure()['changed'])
        document = clients.read_endpoints(self.registry.root)
        self.assertEqual(set(document), {'schema_version', 'owner', 'hem_root', 'servers'})
        self.assertEqual(document['servers'][0]['base_url'], 'http://[::1]:12345/v1')
        stamp = self.path.stat().st_mtime_ns
        self.assertFalse(self.ensure()['changed'])
        self.assertEqual(self.path.stat().st_mtime_ns, stamp)
        self.registry.create_server('a-b', saved_settings=dict(device='task', listen='127.0.0.1:12346'))
        self.assertTrue(self.publish()['changed'])
        document = clients.read_endpoints(self.registry.root)
        self.assertEqual([r['server'] for r in document['servers']], ['a-b', 'a.b'])
        self.assertEqual(len({r['provider'] for r in document['servers']}), 2)
        self.assertIsNone(state.get_target(self.registry))
        self.assertFalse((self.registry.root / 'integrations').exists())
        with self.registry._transaction() as connection:
            self.assertEqual({r[0] for r in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")},
                             {'models', 'servers', 'instances', 'pi_target'})

    def test_known_prior_owned_projection_is_passive_until_explicit_configure(self):
        NATIVE_PROFILE, PREVIOUS_NATIVE_PROFILE = contracts.NATIVE_PROFILE, contracts.PREVIOUS_NATIVE_PROFILE
        self.ensure()
        prior = clients.read_endpoints(self.registry.root)
        prior['servers'][0]['native_profile'] = PREVIOUS_NATIVE_PROFILE
        data = clients._encoded(prior)
        self.path.write_bytes(data)
        stamp = self.path.stat().st_mtime_ns
        self.assertEqual(clients.read_endpoints(self.registry.root), prior)
        self.assertEqual(self.path.read_bytes(), data)
        self.assertEqual(self.path.stat().st_mtime_ns, stamp)
        with self.assertRaisesRegex(common.InvalidInput, 'explicit native-profile refresh'):
            self.publish()
        self.assertEqual(self.path.read_bytes(), data)
        # An incidental publication failure reports repair without changing the
        # completed native operation or its ownership/lifecycle result.
        payload = dict(state='running', completed=True, accepted=True, instance='chat')
        outcome = clients.update_result(self.registry, common.Deadline.after(), payload)
        self.assertEqual((outcome['state'], outcome['completed'], outcome['accepted']), ('running', True, True))
        self.assertEqual(outcome['clients_update'], 'failed')
        self.assertIn('explicit native-profile refresh', outcome['clients_error'])
        self.assertEqual(self.path.read_bytes(), data)
        self.assertTrue(self.ensure()['changed'])
        self.assertEqual(clients.read_endpoints(self.registry.root)['servers'][0]['native_profile'], NATIVE_PROFILE)

    def test_prior_projection_still_rejects_unknown_or_mixed_profiles(self):
        NATIVE_PROFILE, PREVIOUS_NATIVE_PROFILE = contracts.NATIVE_PROFILE, contracts.PREVIOUS_NATIVE_PROFILE
        self.ensure()
        prior = clients.read_endpoints(self.registry.root)
        for profile in ('unknown', PREVIOUS_NATIVE_PROFILE):
            invalid = json.loads(json.dumps(prior))
            invalid['servers'][0]['native_profile'] = profile
            if profile == PREVIOUS_NATIVE_PROFILE:
                row = dict(invalid['servers'][0], server='z', provider=clients.provider_id('z'), native_profile=NATIVE_PROFILE)
                invalid['servers'].append(row)
            data = clients._encoded(invalid)
            self.path.write_bytes(data)
            with self.assertRaises(common.InvalidInput): self.ensure()
            self.assertEqual(self.path.read_bytes(), data)

    def test_corrupt_foreign_and_unknown_never_adopted_even_configure(self):
        self.ensure()
        valid = clients.read_endpoints(self.registry.root)
        variants = [b'not-json', b'['*2000+b']'*2000, b'{\"schema_version\":1,\"schema_version\":2}', json.dumps({**valid, 'owner': 'foreign'}).encode(),
                    json.dumps({**valid, 'hem_root': '/elsewhere'}).encode(),
                    json.dumps({**valid, 'schema_version': True}).encode(),
                    json.dumps({**valid, 'models': []}).encode()]
        for content in variants:
            self.path.write_bytes(content)
            for action in (self.publish, self.ensure):
                with self.assertRaises(common.InvalidInput):
                    action()
                self.assertEqual(self.path.read_bytes(), content)
        # The documented repair does not silently delete user data: move aside
        # after inspection, then explicit configure recreates from server records.
        retained = self.path.with_suffix('.conflict')
        self.path.rename(retained)
        self.assertTrue(self.ensure()['changed'])
        self.assertEqual(clients.read_endpoints(self.registry.root), valid)
        self.assertTrue(retained.exists())

    def test_strict_rows_origins_order_identity_and_size(self):
        self.ensure()
        valid = clients.read_endpoints(self.registry.root)
        row = valid['servers'][0]
        for origin in ('http://localhost:3/v1', 'https://127.0.0.1:3/v1', 'http://127.0.0.1:0/v1',
                       'http://127.0.0.1:3/v1?x', 'http://0.0.0.0:3/v1', 'http://user@127.0.0.1:3/v1'):
            with self.assertRaises(common.InvalidInput):
                clients.validate_endpoints({**valid, 'servers': [{**row, 'base_url': origin}]}, self.registry.root)
        for rows in ([row, row], [{**row, 'provider': 'wrong'}], [{**row, 'native_profile': 'unknown'}],
                     [{**row, 'models': []}]):
            with self.assertRaises(common.InvalidInput):
                clients.validate_endpoints({**valid, 'servers': rows}, self.registry.root)
        before = self.path.read_bytes()
        with patch.object(self.registry, 'list_servers', return_value=[dict(id='x'*450000, saved_settings=dict(listen='127.0.0.1:3'))]):
            with self.assertRaises(common.InvalidInput):
                self.publish()
        self.assertEqual(self.path.read_bytes(), before)

    def test_symlinks_directory_modes_compare_replace_conflict_retained(self):
        self.ensure()
        original = self.path.read_bytes()
        self.path.unlink()
        outside = Path(self.temp.name) / 'foreign'
        outside.write_bytes(original)
        self.path.symlink_to(outside)
        with self.assertRaises(common.InvalidInput):
            self.publish()
        self.assertEqual(outside.read_bytes(), original)
        self.path.unlink()
        self.path.write_bytes(original)
        self.path.chmod(0o600)
        self.path.parent.chmod(0o777)
        with self.assertRaises(common.InvalidInput):
            self.publish()
        self.path.parent.chmod(0o700)
        actual = clients.atomic_replace
        def conflict(path, data, **kwargs):
            path.write_bytes(b'user edit')
            return actual(path, data, **kwargs)
        self.registry.create_server('two', saved_settings=dict(device='task', listen='127.0.0.1:3'))
        with patch.object(clients, 'atomic_replace', side_effect=conflict):
            with self.assertRaises(common.InvalidInput):
                self.publish()
        self.assertEqual(self.path.read_bytes(), b'user edit')

    def test_shared_integration_lock_concurrency_timeout_and_no_reentrancy(self):
        code = (f'from {PACKAGE}.integrations.hem_clients_endpoints import ensure_endpoints; '
                f'from {PACKAGE}.hem_common import Deadline; from {PACKAGE}.hem_state import Registry; '
                'import sys; r=Registry(sys.argv[1],deadline=Deadline.after(2)); ensure_endpoints(r,Deadline.after(2))')
        children = [subprocess.Popen([sys.executable, '-c', code, str(self.registry.root)],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE) for _ in range(3)]
        try:
            for child in children:
                out, err = child.communicate(timeout=5)
                self.assertEqual(child.returncode, 0, err)
        finally:
            for child in children:
                if child.poll() is None:
                    child.terminate()
                child.communicate(timeout=2)
        self.assertEqual(len(clients.read_endpoints(self.registry.root)['servers']), 1)
        with state.integration_lock(self.registry.root, common.Deadline.after()):
            with self.assertRaisesRegex(common.HemError, 'not reentrant'):
                self.ensure()
            result = subprocess.run([sys.executable, '-c', code.replace('after(2)', 'after(.1)'), str(self.registry.root)],
                                    capture_output=True, timeout=2)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(b'deadline', result.stderr)

    def test_secondary_failure_preserves_result_and_pi_independent(self):
        pi = importlib.import_module(PACKAGE + '.integrations.pi.hem_pi_endpoints')
        self.ensure()
        self.path.write_bytes(b'corrupt')
        payload = pi.update_result(self.registry, common.Deadline.after(), common.result('running', completed=True))
        self.assertEqual(payload['state'], 'running')
        self.assertTrue(payload['completed'])
        self.assertEqual(payload['clients_update'], 'failed')
        self.assertNotIn('pi_update', payload)
        self.assertEqual(self.path.read_bytes(), b'corrupt')
        with patch.object(pi, 'publish_endpoints', side_effect=common.HemError('Pi issue')):
            self.path.unlink()
            self.ensure()
            payload = pi.update_result(self.registry, common.Deadline.after(), common.result('stopped', completed=True))
        self.assertEqual(payload['pi_update'], 'failed')
        self.assertEqual(payload['clients_update'], 'unchanged')
        self.assertTrue(payload['completed'])


class LifecycleTest(fixtures.ServerCliTest):
    """CPU native process fixture proves real CLI publication hook semantics."""
    def test_start_status_reconcile_shutdown_only_root_projection(self):
        registry = Registry(self.state)
        clients.ensure_endpoints(registry, common.Deadline.after())
        self.assertEqual(clients.read_endpoints(self.state)['servers'], [])
        first, _ = self.start('one')
        self.assertEqual(first['clients_update'], 'updated')
        self.assertEqual(clients.read_endpoints(self.state)['servers'][0]['base_url'], first['endpoint'] + '/v1')
        for args in [('status', 'one'), ('reconcile', 'one'), ('shutdown', 'one')]:
            payload, _ = self.invoke(*args)
            self.assertTrue(payload['completed'])
            self.assertEqual(payload['clients_update'], 'unchanged')
        self.assertIsNone(state.get_target(Registry(self.state)))
        self.assertEqual(len(clients.read_endpoints(self.state)['servers']), 1)

    def test_bad_projection_does_not_turn_native_success_into_retry(self):
        registry = Registry(self.state)
        clients.ensure_endpoints(registry, common.Deadline.after())
        (self.state / clients.ENDPOINT_PATH).write_bytes(b'bad')
        first, stderr = self.start('one')
        self.assertTrue(first['completed'])
        self.assertEqual(first['clients_update'], 'failed')
        self.assertIn('native result unchanged', stderr)
        payload, _ = self.invoke('shutdown', 'one')
        self.assertTrue(payload['completed'])
        self.assertEqual(payload['clients_update'], 'failed')


if __name__ == '__main__':
    unittest.main()
