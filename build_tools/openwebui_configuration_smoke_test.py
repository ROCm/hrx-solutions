"""Gate failures must retain evidence and clean all owned resources."""
import unittest
from unittest import mock
import types
from pathlib import Path
import subprocess
import tempfile

import openwebui_configuration_smoke as smoke


class CleanupTest(unittest.TestCase):
    def gate(self):
        return types.SimpleNamespace(setup=mock.Mock(), attempts={'a': {}, 'b': {}},
            cleanup_attempt=mock.Mock(), save=mock.Mock(), results={})

    def test_stream_failure_still_stops_webui_and_all_native_attempts(self):
        gate = self.gate()
        check = mock.Mock()
        check.run.side_effect = AssertionError('stream missing')
        with mock.patch.object(smoke, 'ConfigurationGate', return_value=check):
            with self.assertRaisesRegex(AssertionError, 'stream missing'):
                smoke.qualify(gate)
        check.stop.assert_called_once()
        self.assertEqual(gate.cleanup_attempt.call_args_list, [mock.call('a'), mock.call('b')])
        self.assertFalse(gate.results['success'])
        gate.save.assert_called_once()

    def test_original_child_cleanup_failure_cannot_publish_complete(self):
        gate = self.gate()
        original = RuntimeError('bounded launch failed')
        original.cleanup_unresolved = 'sensitive detail must not be retained'
        gate.setup.side_effect = original
        with mock.patch.object(smoke, 'ConfigurationGate'), mock.patch.object(smoke, 'write_cleanup') as publish:
            with self.assertRaisesRegex(RuntimeError, 'bounded launch failed'):
                smoke.qualify(gate)
        self.assertEqual(gate.cleanup_attempt.call_count, 2)
        self.assertFalse(gate.results['success'])
        self.assertEqual(gate.results['cleanup_errors'], ['owned synchronous child cleanup unresolved'])
        publish.assert_called_once_with('unresolved', attempts=['a', 'b'],
            errors=['owned synchronous child cleanup unresolved'])
        self.assertNotIn(original.cleanup_unresolved, str(gate.results))
        self.assertNotIn(original.cleanup_unresolved, str(publish.call_args))

    def test_cleanup_failure_cannot_claim_success_or_skip_other_cleanup(self):
        gate = self.gate()
        check = mock.Mock()
        check.stop.side_effect = RuntimeError('owned service still running')
        gate.cleanup_attempt.side_effect = [RuntimeError('first native cleanup'), None]
        with mock.patch.object(smoke, 'ConfigurationGate', return_value=check):
            with self.assertRaisesRegex(RuntimeError, 'cleanup unresolved'):
                smoke.qualify(gate)
        self.assertEqual(gate.cleanup_attempt.call_count, 2)
        self.assertFalse(gate.results['success'])
        self.assertEqual(gate.results['cleanup_errors'], ['RuntimeError', 'RuntimeError'])

    def container_check(self, root):
        check = object.__new__(smoke.ConfigurationGate)
        check.gate = types.SimpleNamespace(args=types.SimpleNamespace(webui_python=None), save=mock.Mock())
        check.root = Path(root)
        check.settings = {'PORT': '18001'}
        check.name, check.container_owner = 'unique-owned-name', 'private-owner-label'
        check.container, check.child = False, None
        check.evidence = {'cleanup': []}
        return check

    def test_lost_docker_ack_cleans_only_labelled_id_after_attempt_was_recorded(self):
        with tempfile.TemporaryDirectory() as root:
            check = self.container_check(root)
            owned = 'a' * 64
            replies = [subprocess.CompletedProcess([], 0, '', ''), TimeoutError('lost acknowledgement'),
                       subprocess.CompletedProcess([], 0, owned + '\n', ''),
                       subprocess.CompletedProcess([], 0, '', ''), subprocess.CompletedProcess([], 0, '', '')]
            with mock.patch.object(smoke, 'run_child', side_effect=replies) as run:
                with self.assertRaisesRegex(TimeoutError, 'lost acknowledgement'):
                    check.start()
                self.assertTrue(check.container)
                with mock.patch.object(smoke.socket, 'socket') as sock:
                    sock.return_value.__enter__.return_value.connect_ex.return_value = 1
                    check.stop()
            calls = [call.args[0] for call in run.call_args_list]
            self.assertIn('hrx.openwebui-gate-owner=private-owner-label', calls[1])
            self.assertEqual(calls[3], ['docker', 'rm', '-f', owned])
            self.assertIn('label=hrx.openwebui-gate-owner=private-owner-label', calls[2])
            self.assertFalse(check.container)

    def test_preexisting_same_name_is_not_adopted_or_removed(self):
        with tempfile.TemporaryDirectory() as root:
            check = self.container_check(root)
            with mock.patch.object(smoke, 'run_child', return_value=subprocess.CompletedProcess([], 0, 'b'*64, '')) as run:
                with self.assertRaisesRegex(AssertionError, 'already exists'):
                    check.start()
            self.assertFalse(check.container)
            self.assertEqual(run.call_count, 1)

    def test_setup_failure_remains_original_and_records_cleanup(self):
        gate = self.gate()
        gate.setup.side_effect = ValueError('wheel mismatch')
        with mock.patch.object(smoke, 'ConfigurationGate') as factory:
            with self.assertRaisesRegex(ValueError, 'wheel mismatch'):
                smoke.qualify(gate)
        factory.assert_not_called()
        self.assertFalse(gate.results['success'])
        self.assertEqual(gate.cleanup_attempt.call_count, 2)


if __name__ == '__main__':
    unittest.main()
