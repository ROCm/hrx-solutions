"""Public WebUI command contract, safe equivalents and common failure envelope."""
from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from binding.python import hem_cli, hem_openwebui as owner
from binding.python.hem_common import DeadlineExpired, HemError, InvalidInput, UncertainOutcome


class Response(dict):
    exit_code = 0


class WebUICliTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name) / 'state'
        self.target = 'http://127.0.0.1:3000'
        self.token = 'admin-super-secret-token'
        self.addCleanup(patch.stopall)
        patch.dict(os.environ, {'LOOM_SERVE_DIR': str(self.root),
            'OPENWEBUI_URL': self.target, 'OPENWEBUI_TOKEN': self.token}).start()
        self.coordinator = Mock(side_effect=self.coordinate)
        patch.object(owner, '_configure_owned', self.coordinator).start()

    def coordinate(self, root, target, token, deadline, *, check=False, disable=False, display=None, stt_server=None, stt_instance=None, client_factory=None):
        self.assertEqual(root, self.root)
        self.assertEqual(token, self.token)
        self.assertLessEqual(deadline.remaining(), 10)
        display('GET', '/api/version')
        display('GET', '/openai/config')
        if not check and not disable:
            display('POST', '/openai/config/update', {'operation': 'configure', 'before': [], 'after': []})
        return Response(state='absent' if disable else 'configured', completed=True,
            target_url=target, mode='check' if check else 'disable' if disable else 'configure',
            changed=not check and not disable, configuration='absent' if disable else 'configured',
            service='reachable', compatibility='qualified', model_availability='unprobed',
            prerequisites={'persistence': 'operator-required/unverified',
                'user_info_forwarding_disabled': 'operator-required/unverified'}, pending=False, findings=[])

    def invoke(self, *flags, expected=0, json_output=True):
        output, errors = io.StringIO(), io.StringIO()
        argv = ['openwebui', 'configure', *flags] + (['--json'] if json_output else [])
        with redirect_stdout(output), redirect_stderr(errors):
            code = hem_cli.main(argv)
        self.assertEqual(code, expected, output.getvalue() + errors.getvalue())
        self.assertNotIn(self.token, output.getvalue() + errors.getvalue())
        if json_output:
            lines = output.getvalue().splitlines()
            self.assertEqual(len(lines), 1)
            return json.loads(lines[0]), errors.getvalue()
        return output.getvalue(), errors.getvalue()

    def test_stt_selector_pair_forwarded_and_rejected_combinations_are_passive(self):
        self.invoke('--stt-server', 'local', '--stt-instance', 'dictation')
        self.assertEqual(self.coordinator.call_args.kwargs['stt_server'], 'local')
        self.assertEqual(self.coordinator.call_args.kwargs['stt_instance'], 'dictation')
        self.coordinator.reset_mock()
        for flags in (('--stt-server', 'local'), ('--stt-instance', 'dictation'),
                      ('--disable', '--stt-server', 'local', '--stt-instance', 'dictation')):
            self.invoke(*flags, expected=2)
        self.coordinator.assert_not_called()

    def test_modes_share_budget_and_stdout_stderr_contract(self):
        for flags, mode in [((), 'configure'), (('--check',), 'check'), (('--disable',), 'disable')]:
            with self.subTest(mode=mode):
                response, errors = self.invoke(*flags)
                self.assertEqual(response['mode'], mode)
                self.assertEqual(response['model_availability'], 'unprobed')
                self.assertIn('${OPENWEBUI_TOKEN}', errors)
                self.assertIn('WebUI API request', errors)
                self.assertIn('global user-info forwarding must be disabled', response['guidance'])
                self.assertIn('Export/backups are your responsibility', errors)
                if mode == 'configure':
                    self.assertIn('REDACTED, NON-REPLAYABLE', errors)
                    self.assertIn('preserved provider keys omitted', errors)
                    self.assertIn('Owned connection intent:', errors)
                else:
                    self.assertNotIn('--data-binary', errors)
        self.assertFalse(self.root.exists())

    def test_explicit_url_overrides_environment_and_canonicalizes_ipv6(self):
        response, _ = self.invoke('--url', 'http://[0:0:0:0:0:0:0:1]:3001/')
        self.assertEqual(response['target_url'], 'http://[::1]:3001')
        with patch.dict(os.environ, {'OPENWEBUI_URL': 'invalid'}):
            self.invoke('--url', self.target)

    def test_missing_and_unsafe_target_do_not_call_coordinator_or_echo_input(self):
        with patch.dict(os.environ, {}, clear=True):
            self.invoke(expected=2)
        for value in ['', 'http://localhost:3000', 'http://127.0.0.1', 'https://127.0.0.1:3000',
                      'http://127.0.0.1:3000/subpath', 'http://127.0.0.1:3000?token=foreign-key',
                      'http://user:foreign-key@127.0.0.1:3000']:
            response, errors = self.invoke('--url', value, expected=2)
            self.assertNotIn('foreign-key', json.dumps(response) + errors)
        self.coordinator.assert_not_called()

    def test_missing_token_and_relative_root_no_initialization(self):
        with patch.dict(os.environ, {'OPENWEBUI_TOKEN': ''}):
            self.invoke(expected=2)
        with patch.dict(os.environ, {'LOOM_SERVE_DIR': 'relative'}):
            self.invoke(expected=2)
        self.coordinator.assert_not_called()
        self.assertFalse(self.root.exists())

    def test_grammar_rejects_extra_commands_flags_abbreviations_and_mode_conflict(self):
        for flags in [('--check', '--disable'), ('--chec',), ('--token', 'foreign-key'),
                      ('--timeout', 'nan'), ('--timeout', '0')]:
            response, errors = self.invoke(*flags, expected=2)
            self.assertNotIn('foreign-key', json.dumps(response) + errors)
        self.coordinator.assert_not_called()

    def test_common_failure_categories_keep_single_envelope(self):
        for error in [HemError('WebUI unavailable'), InvalidInput('WebUI version unsupported'),
                      DeadlineExpired('WebUI deadline expired', accepted=False),
                      UncertainOutcome('WebUI update unresolved', accepted=None)]:
            with self.subTest(code=error.exit_code):
                self.coordinator.side_effect = error
                response, errors = self.invoke(expected=error.exit_code)
                self.assertFalse(response['completed'])
                self.assertEqual(response['accepted'], error.accepted)
                self.assertEqual(response['state'], 'uncertain' if error.exit_code == 4 else 'failed')

    def test_malformed_argument_and_boundary_errors_cannot_echo_admin_token(self):
        self.invoke('--timeout', self.token, expected=2)
        self.coordinator.side_effect = HemError('Rejected credential ' + self.token)
        response, _ = self.invoke(expected=1)
        self.assertIn('<redacted>', response['error'])

    def test_result_exit_category_and_human_rendering_remain_distinct(self):
        response = self.coordinate(self.root, self.target, self.token,
            __import__('binding.python.hem_common', fromlist=['Deadline']).Deadline.after(),
            display=lambda *args: None)
        response.update(configuration='partial', service='unauthorized', completed=False)
        response.exit_code = 1
        self.coordinator.side_effect = None
        self.coordinator.return_value = response
        result, _ = self.invoke('--disable', expected=1)
        self.assertEqual(result['configuration'], 'partial')
        output, _ = self.invoke('--disable', expected=1, json_output=False)
        self.assertIn('Open WebUI integration:', output)
        self.assertIn('unprobed', output)
        self.assertIn('unauthorized', output)

    def test_help_is_passive_without_environment_or_client_binary(self):
        with patch.dict(os.environ, {}, clear=True):
            for argv in [['openwebui'], ['openwebui', '--help'], ['openwebui', 'configure', '--help']]:
                output = io.StringIO()
                with redirect_stdout(output):
                    try:
                        code = hem_cli.main(argv)
                    except SystemExit as error:
                        code = error.code
                self.assertEqual(code, 0)
                self.assertIn('Open WebUI', output.getvalue())
        self.coordinator.assert_not_called()


if __name__ == '__main__':
    unittest.main()
