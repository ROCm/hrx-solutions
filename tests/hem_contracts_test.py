"""Hem grammar and shared boundaries; no models, native services or downloads."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from binding.python import hem_cli, hem_display
from binding.python.hem_arguments import select_server
from binding.python.hem_common import (
    Deadline, DeadlineExpired, HemError, InvalidInput, UncertainOutcome, result,
)


class GrammarTest(unittest.TestCase):
    def parse(self, text):
        return hem_cli.parse_args(shlex.split(text))

    def test_complete_four_type_inventory(self):
        examples = {
            'model': ['available', 'download recipe --id qwen08',
                      'import qwen08 --gguf a.gguf b.gguf --tokenizer tokenizer.json --alias',
                      'validate qwen08', 'list', 'show qwen08'],
            'pipeline': ['list', 'show qwen-08b'],
            'server': ['start --name local --device amdgpu --listen 127.0.0.1:8080',
                       'start local', 'list', 'show local', 'status local',
                       'logs local --follow', 'reconcile local', 'shutdown local --abort'],
            'instance': ['start qwen08 --name chat --server local',
                         'stop chat --url http://127.0.0.1:8080 --no-wait',
                         'list', 'show chat', 'status chat', 'logs chat --follow',
                         'export qwen08 --output model.json'],
        }
        self.assertEqual(set(examples), set(hem_cli.DOMAINS))
        for domain, commands in examples.items():
            for command in commands:
                with self.subTest(domain=domain, command=command):
                    args = self.parse(domain + ' ' + command)
                    self.assertEqual(args.resource, domain)
                    self.assertEqual(args.command, command.split()[0])

    def test_no_aliases_shortcuts_or_incomplete_invocations(self):
        for text in ('models list', 'start qwen08', 'instance activate chat',
                     'model download recipe', 'model import qwen08 --gguf w',
                     'instance export qwen08', 'server start',
                     'server start local --name other', 'server start local --device task',
                     'instance start qwen08 --no-wait', 'server shutdown local --no-wait',
                     'instance list --serv local',
                     'instance list --server local --url http://localhost:8080'):
            with self.subTest(text=text), self.assertRaises(InvalidInput):
                self.parse(text)

    def test_optional_configuration_preserves_provenance(self):
        args = self.parse('instance start qwen08')
        self.assertIsNone(args.mtp)
        self.assertIsNone(args.mtp_depth)
        self.assertIsNone(args.context_capacity)
        self.assertIsNone(args.name)
        self.assertEqual(args.timeout, 10)
        args = self.parse('instance start qwen27 --no-mtp --mtp-depth 3 --context-capacity 4096')
        # Resolution, including file-vs-flag contradictions, belongs to hem_config.
        self.assertFalse(args.mtp)
        self.assertEqual(args.mtp_depth, 3)
        self.assertEqual(args.context_capacity, 4096)

    def test_timeout_and_names_reject_unsafe_values(self):
        for value in ('0', '-1', 'nan', 'inf', 'oops'):
            with self.subTest(value=value), self.assertRaises(InvalidInput):
                self.parse('server start local --timeout ' + value)
        for value in ('../outside', '.', '..', '/absolute'):
            with self.subTest(value=value), self.assertRaises(InvalidInput):
                self.parse('model show ' + value)

    def test_selection_is_explicit_or_sole_configured_server(self):
        self.assertEqual(select_server(None, ['local']), 'local')
        self.assertEqual(select_server('second', ['local', 'second']), 'second')
        for selected, configured in ((None, []), (None, ['a', 'b']), ('unknown', ['a'])):
            with self.assertRaises(InvalidInput):
                select_server(selected, configured)

    def test_json_works_at_every_grammar_level(self):
        for text in ('--json model list', 'model --json list', 'model list --json'):
            self.assertTrue(self.parse(text).json)


class DeadlineAndResultTest(unittest.TestCase):
    def invoke(self, argv, *, response=None, error=None):
        output, diagnostics = io.StringIO(), io.StringIO()
        with patch.object(hem_cli.hem_instance_cli, 'run', return_value=response,
                          side_effect=error) as run:
            with redirect_stdout(output), redirect_stderr(diagnostics):
                code = hem_cli.main(argv)
        return code, output.getvalue(), diagnostics.getvalue(), run

    def test_failure_preserves_acceptance_through_dispatch(self):
        cases = (
            (DeadlineExpired, True, 3),
            (UncertainOutcome, None, 4),
            (HemError, False, 1),
        )
        for error_type, accepted, expected in cases:
            with self.subTest(accepted=accepted, error_type=error_type):
                error = error_type('native operation incomplete', accepted=accepted)
                code, output, diagnostics, _ = self.invoke(
                    ['instance', 'stop', 'chat', '--server', 'local', '--json'],
                    error=error)
                data = json.loads(output)
                self.assertEqual(code, expected)
                self.assertIs(data['accepted'], accepted)
                self.assertFalse(data['completed'])
                self.assertEqual(data['instance'], 'chat')
                self.assertIn('native operation incomplete', diagnostics)
                if accepted is not None:
                    self.assertIn('accepted: ' + str(accepted).lower(), diagnostics)

    def test_message_only_errors_leave_acceptance_unknown(self):
        code, output, diagnostics, _ = self.invoke(
            ['instance', 'stop', 'chat', '--json'], error=UncertainOutcome('response lost'))
        self.assertEqual(code, 4)
        self.assertIsNone(json.loads(output)['accepted'])
        self.assertIn('accepted: unknown', diagnostics)

    def test_shared_budget_does_not_reset_between_phases(self):
        now = [100.0]
        deadline = Deadline.after(clock=lambda: now[0])
        self.assertEqual(deadline.remaining(), 10)
        now[0] += 4
        self.assertEqual(deadline.remaining(), 6)
        now[0] += 6
        with self.assertRaises(DeadlineExpired):
            deadline.remaining()
        # No late external action: code is not reached after the guard.
        calls = []
        with self.assertRaises(DeadlineExpired):
            deadline.remaining()
            calls.append('submit')
        self.assertEqual(calls, [])

    def test_default_and_override_are_passed_as_one_budget(self):
        for suffix, expected in (([], 10), (['--timeout', '2.5'], 2.5)):
            code, _, _, run = self.invoke(['instance', 'start', 'qwen08', *suffix],
                                          response=result('ready', completed=True))
            self.assertEqual(code, 0)
            deadline = run.call_args.args[1]
            self.assertGreater(deadline.remaining(), expected - 0.1)
            self.assertLessEqual(deadline.remaining(), expected)

    def test_json_result_and_diagnostics_are_separate(self):
        def action(args, deadline):
            hem_display.emit('curl http://localhost:8080/v1/status')
            return result('ready', completed=True, instance='chat')
        out, err = io.StringIO(), io.StringIO()
        with patch.object(hem_cli.hem_instance_cli, 'run', side_effect=action):
            with redirect_stdout(out), redirect_stderr(err):
                self.assertEqual(hem_cli.main(['instance', 'status', 'chat', '--json']), 0)
        self.assertEqual(json.loads(out.getvalue()),
                         {'instance': 'chat', 'state': 'ready', 'completed': True})
        self.assertIn('curl ', err.getvalue())

    def test_known_pending_timeout_and_uncertain_timeout_differ(self):
        uncertainty = UncertainOutcome('undrain response lost; inspect admission')
        uncertainty.__cause__ = TimeoutError('socket timed out')
        for error, expected in ((HemError('native failed'), 1),
                                (InvalidInput('invalid config'), 2),
                                (DeadlineExpired('accepted retirement still pending'), 3),
                                (uncertainty, 4)):
            with self.subTest(expected=expected):
                code, out, err, _ = self.invoke(
                    ['instance', 'stop', 'chat', '--json'], error=error)
                self.assertEqual(code, expected)
                data = json.loads(out)
                self.assertFalse(data['completed'])
                self.assertEqual(data['error'], str(error))
                self.assertIn(str(error), err)
                self.assertEqual(data['state'], 'uncertain' if expected == 4 else 'failed')

    def test_parse_errors_are_honest(self):
        for argv in (['instances', 'list', '--json'],
                     ['instance', 'list', '--server', 'local', '--url', 'http://127.0.0.1:8080', '--json'],
                     ['instance', 'start', 'qwen08', '--timeout', '0', '--json']):
            out, err = io.StringIO(), io.StringIO()
            with redirect_stdout(out), redirect_stderr(err):
                self.assertEqual(hem_cli.main(argv), 2)
            self.assertFalse(json.loads(out.getvalue())['completed'])
            self.assertTrue(err.getvalue())

    def test_help_subprocess_does_not_create_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            import os
            environment = dict(os.environ, LOOM_SERVE_DIR=str(Path(temporary) / 'state'))
            process = subprocess.run([sys.executable, '-m', 'binding.python.hem_cli',
                                      'server', 'shutdown', '--help'],
                                     env=environment, capture_output=True, text=True, check=False)
            self.assertEqual(process.returncode, 0)
            self.assertEqual(process.stderr, '')
            self.assertIn('--abort', process.stdout)
            self.assertFalse(Path(environment['LOOM_SERVE_DIR']).exists())


class EquivalentTest(unittest.TestCase):
    def test_native_arguments_and_environment_round_trip(self):
        argv = ['loom-serve', '--device=amdgpu', '--listen=127.0.0.1:8080',
                '--tokenizer', '/models/tokenizer.json',
                '/path with spaces/$(touch bad);`echo bad`']
        self.assertEqual(shlex.split(hem_display.command(argv)), argv)
        rendered = hem_display.command(argv, environment={'LOOM_SERVE_DIR': '/state with spaces'})
        self.assertEqual(shlex.split(rendered), ['env', 'LOOM_SERVE_DIR=/state with spaces', *argv])

    def test_credentials_redacted_without_changing_execution_inputs(self):
        argv = ['curl', 'https://alice:pword@localhost:8080/v1/status?token=tokenvalue',
                '-H', 'Authorization: Bearer headersecret', '--api-key=flagsecret',
                '--password', 'passwordsecret', '-u', 'user:password',
                '--user=inlineuser:inlinepassword', '--oauth2-bearer', 'bearersecret',
                '-uattached:attachedsecret', '--header=Authorization: Bearer attachedheader',
                'customsecret']
        before = list(argv)
        rendered = hem_display.command(argv, environment={'HF_TOKEN': 'envsecret'},
                                       secrets=['customsecret'])
        for secret in ('alice', 'pword', 'tokenvalue', 'headersecret', 'flagsecret',
                       'passwordsecret', 'user:password', 'inlineuser', 'inlinepassword',
                       'bearersecret', 'attachedsecret', 'attachedheader', 'customsecret', 'envsecret'):
            self.assertNotIn(secret, rendered)
        self.assertIn('credentials redacted', rendered)
        self.assertEqual(argv, before)

    def test_http_body_is_inspectable_and_exact(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'body with spaces.json'
            body = b'{"model":"chat-small","mode":"drain"}\n'
            path.write_bytes(body)
            rendered = hem_display.http('POST', 'http://localhost:8080/v1/models/unregister',
                                        body_path=path, body=body)
            argv = shlex.split(rendered)
            self.assertEqual(argv[argv.index('--data-binary') + 1], '@' + str(path))
            self.assertEqual(path.read_bytes(), body)
            with self.assertRaises(InvalidInput):
                hem_display.http('POST', 'http://localhost', body_path=path, body=b'{}')
            with self.assertRaises(InvalidInput):
                hem_display.http('POST', 'http://localhost', body=body)
            path.unlink()
            with self.assertRaises(InvalidInput):
                hem_display.http('POST', 'http://localhost', body_path=path, body=body)


if __name__ == '__main__':
    unittest.main()
