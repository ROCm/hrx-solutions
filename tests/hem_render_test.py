"""Human command fixtures, stream contracts and replay-safe equivalent display."""
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import shlex
import tempfile
import unittest
from unittest.mock import patch

from binding.python import hem_cli, hem_display
from binding.python.cli import hem_help, hem_render, hem_server_cli
from binding.python.models import hem_config, hem_models
from binding.python.hem_common import HemError, DeadlineExpired, UncertainOutcome
from binding.python.cli.hem_server_cli import ObservationResult


SETTINGS = {'rows': 4, 'context_capacity': 8192, 'pool_capacity': 32768,
            'prefill_capacity': 128, 'mtp': False, 'mtp_depth': None}
MODEL = dict(model='qwen08', id='qwen08', state='ready', completed=True,
             pipeline='qwen-08b', accessible=True, directory='/models/qwen08',
             weights='/models/qwen08/model.gguf', tokenizer='/models/qwen08/tokenizer.json',
             check_kind='acquisition-presence',
             artifacts=[dict(id='weights',role='weights',format='gguf',logical_path='model.gguf',path='/models/qwen08/model.gguf',ownership='caller-reference')],
             assets=[dict(logical_path='model.gguf', ownership='reference', source_path='/external/weights.gguf')],
             validation=dict(valid=True, evidence=[dict(inode=999)]),
             provenance=dict(kind='import'), pending=None, instances=[])
NATIVE = {'id': 'chat-small', 'loom': {'incarnation': 17, 'registration': 'ready',
          'parameters': {'primary': {'active': True}}, 'parameter_operation': 'idle', 'status': {'admission': 'open'}}}
INSTANCE = dict(state='observed', completed=True, instance='chat-small', model='qwen08',
                server='local', endpoint='http://127.0.0.1:8080', observed_at=0,
                incarnation=17, native=NATIVE, settings=SETTINGS, pending=None,
                document='/state/registration.json', dependencies=['/models/qwen08/model.gguf'])
SERVER = dict(state='running', completed=True, server='local', process_health='verified',
              http_health='healthy', observed_at=0,
              record=dict(id='local', endpoint='http://127.0.0.1:8080',
                          saved_settings=dict(device='amdgpu', listen='127.0.0.1:8080')),
              native=dict(server={'worker_count': 4}, data=[NATIVE]))


def fixtures():
    result = {}
    for command in ('show', 'download', 'import', 'validate'):
        result['model', command] = MODEL
    result['model', 'list'] = dict(items=[MODEL])
    result['model', 'available'] = dict(items=[dict(id='qwen35-08b-q5-k-xl', name='Qwen 0.8B', pipeline='qwen-08b')])
    info = hem_config.pipeline_info('qwen-08b')
    result['pipeline', 'show'] = info
    result['pipeline', 'list'] = dict(items=[info])
    for command in ('status', 'reconcile'):
        result['server', command] = SERVER
    result['server', 'list'] = dict(items=[SERVER])
    result['server', 'show'] = dict(server='local', saved_settings=SERVER['record']['saved_settings'])
    result['server', 'start'] = dict(SERVER, endpoint='http://127.0.0.1:8080')
    result['server', 'shutdown'] = dict(server='local', state='stopped', completed=True)
    for command in ('show', 'status'):
        result['instance', command] = INSTANCE
    result['instance', 'list'] = dict(items=[INSTANCE], server='local', endpoint='http://127.0.0.1:8080', server_health='running')
    result['instance', 'start'] = dict(INSTANCE, state='running', pipeline='qwen-08b')
    result['instance', 'stop'] = dict(instance='chat-small', state='stopped', completed=True)
    result['instance', 'export'] = dict(instance='chat-small', model='qwen08', pipeline='qwen-08b', output='/exports/register.json', settings=SETTINGS)
    for resource in ('server', 'instance'):
        result[resource, 'logs'] = dict(logs='native log line\n', followed=False)
    result['pi', 'configure'] = dict(state='configured', configured=True, changed=False, completed=True, compatibility=dict(state='compatible'), next_action='loom-control pi configure --check')
    return result


class RendererTest(unittest.TestCase):
    def invoke(self, argv, data, *, error=None):
        output, diagnostics = io.StringIO(), io.StringIO()
        resource = next(word for word in argv if word in hem_cli.DOMAINS)
        def run(args, deadline):
            hem_display.emit(hem_display.command(['curl', '--request', 'GET', 'http://127.0.0.1:8080/v1/status']))
            if error:
                raise error
            return data
        with patch.object(hem_cli.DOMAINS[resource], 'run', side_effect=run):
            with redirect_stdout(output), redirect_stderr(diagnostics):
                code = hem_cli.main(argv)
        return code, output.getvalue(), diagnostics.getvalue()

    def test_all_resource_presentations_and_json_payloads(self):
        cases = fixtures()
        self.assertEqual(len(cases), 23)
        for (resource, command), data in cases.items():
            example = hem_help.ACTIONS[resource][command][1].splitlines()[0]
            argv = shlex.split(example)
            args = hem_cli.parse_args(argv)
            before = deepcopy(data)
            with self.subTest(resource=resource, command=command):
                rendered = hem_render.render(args, data, width=80)
                self.assertTrue(rendered)
                self.assertNotIn('"completed"', rendered)
                self.assertNotIn("{'", rendered)
                self.assertNotIn('inode', rendered)
                self.assertEqual(data, before)
                code, output, diagnostics = self.invoke(argv, data)
                self.assertEqual(code, 0)
                self.assertTrue(output)
                self.assertIn('# Equivalent command', diagnostics)
                self.assertIn('curl ', diagnostics)
                code, output, diagnostics = self.invoke(argv + ['--json'], data)
                self.assertEqual(code, 0)
                self.assertEqual(json.loads(output), data)
                self.assertIn('curl ', diagnostics)

    def test_available_displays_actual_authored_recipe_names(self):
        items = hem_models.available()
        self.assertEqual(len(items), 3)
        args = hem_cli.parse_args(['model', 'available'])
        for width in (80, 240):
            with self.subTest(width=width):
                output = hem_render.render(args, dict(items=items), width=width)
                self.assertIn('Name', output)
                self.assertNotIn('Description', output)
                self.assertNotIn('Unknown', output)
                for item in items:
                    self.assertIn(item['name'], output)
                    self.assertIn(item['id'], output)
                    self.assertIn(item['pipeline'], output)

    def test_empty_and_unavailable_inventories(self):
        for argv, phrase in ((['model', 'list'], 'No local models'),
                             (['server', 'list'], 'No servers'),
                             (['instance', 'list'], 'No instances')):
            code, output, _ = self.invoke(argv, dict(items=[]))
            self.assertEqual(code, 0)
            self.assertIn(phrase, output)
            self.assertIn('loom-control ', output)
        for health in ('unknown', 'unhealthy'):
            args = hem_cli.parse_args(['instance', 'list'])
            output = hem_render.render(args, dict(items=[], server_health=health))
            self.assertIn('inventory unavailable', output)
            self.assertNotIn('No instances', output)

    def test_invalid_assets_pending_and_external_mismatch_are_prominent(self):
        broken = dict(MODEL, state='missing', error='Alias target disappeared',
                      validation=dict(valid=False, evidence={'error': 'Wrong tokenizer'}),
                      pending=dict(stage='published-invalid'))
        _, output, _ = self.invoke(['model', 'list'], dict(items=[broken]))
        for phrase in ('missing', 'Alias target disappeared', 'Wrong tokenizer', 'Pending'):
            self.assertIn(phrase, output)
        external = dict(INSTANCE, model=None, settings=None, tracked_submission=dict(incarnation=16),
                        document=None, dependencies=None, pending=dict(action='stop', accepted=None))
        _, output, _ = self.invoke(['instance', 'show', 'chat-small'], external)
        for phrase in ('incarnation mismatch', 'Unknown', 'Pending', 'accepted: Unknown'):
            self.assertIn(phrase, output)
        self.assertNotIn('qwen08', output)
        bad = dict(INSTANCE, state='unhealthy', native={'loom': {'error': {'code': 4, 'message': 'Device failed'}}})
        _, output, _ = self.invoke(['instance', 'status', 'chat-small'], bad)
        self.assertIn('Device failed', output)
        self.assertIn('Native error', output)

    def test_dry_run_no_wait_and_structural_readiness(self):
        _, output, _ = self.invoke(['model', 'download', 'recipe', '--id', 'qwen08'],
                                  dict(state='preview', model_id='qwen08', recipe_id='recipe', artifacts=[], offline=True))
        self.assertIn('no files acquired', output)
        self.assertIn('no model record created', output)
        self.assertIn('Yes', output)
        _, output, _ = self.invoke(['instance', 'stop', 'chat-small', '--url', 'http://127.0.0.1:8080', '--no-wait'],
                                  dict(state='stopping', completed=False, accepted=True, instance='chat-small'))
        self.assertIn('retirement is not yet confirmed', ' '.join(output.split()))
        self.assertIn('--url http://127.0.0.1:8080', output)
        _, output, _ = self.invoke(['model', 'validate', 'qwen08'], MODEL)
        self.assertIn('Prepared asset checks do not prove native inference readiness', ' '.join(output.split()))
        self.assertNotIn('ready to accept requests', output)
        _, output, _ = self.invoke(['instance', 'export', 'qwen08', '--output', 'r.json'], fixtures()['instance', 'export'])
        self.assertIn('Replace SERVER:PORT', ' '.join(output.split()))

    def test_long_values_survive_narrow_and_redirected_layouts(self):
        identifier = 'long-model-' * 15
        path = '/path with spaces/' + 'segment/' * 25 + 'weights.gguf'
        for width in (35, 80):
            args = hem_cli.parse_args(['model', 'list'])
            output = hem_render.render(args, dict(items=[dict(MODEL, model=identifier)]), width=width)
            self.assertIn(identifier, output)
            output = hem_render.render(hem_cli.parse_args(['model', 'show', 'qwen08']), dict(MODEL, artifacts=[dict(id='weights', role='weights', path=path)]), width=width)
            self.assertIn(path, output)
        output = hem_render.render(hem_cli.parse_args(['instance', 'status', 'chat']), dict(INSTANCE, incarnation=None,
                                  native={'loom': {'parameters': {'active': False}}}), width=35)
        self.assertIn('No', output)
        self.assertIn('Unknown', output)
        args = hem_cli.parse_args(['instance', 'start', identifier, '--url', 'http://127.0.0.1:8080'])
        output = hem_render.render(args, dict(INSTANCE, instance=identifier, state='running'), width=35)
        self.assertIn('Inspect: loom-control instance status ' + identifier + ' --url http://127.0.0.1:8080', output)

    def test_exit_codes_are_preserved_for_human_results_and_errors(self):
        for state, code in (('running', 0), ('unhealthy', 1), ('unknown', 4)):
            data = ObservationResult(dict(SERVER, state=state))
            actual, output, _ = self.invoke(['server', 'status', 'local'], data)
            self.assertEqual(actual, code)
            self.assertIn(state, output)
        for error, expected in ((HemError('failed', accepted=False), 1),
                                (DeadlineExpired('still pending', accepted=True), 3),
                                (UncertainOutcome('outcome unknown'), 4)):
            code, output, errors = self.invoke(['instance', 'stop', 'chat'], None, error=error)
            self.assertEqual(code, expected)
            self.assertEqual(output, '')
            self.assertIn(str(error), errors)

    def test_all_path_fields_stay_complete_in_narrow_and_redirected_output(self):
        path = '/external model directory/fragment file with spaces.gguf'
        document = '/external model directory/deployment document with spaces.json'
        cases = [
            (['model', 'show', 'qwen08'], dict(MODEL, artifacts=[dict(id='part', role='weights-shard', path=path)]), [path]),
            (['model', 'import', 'qwen08', '--gguf', 'w.gguf', '--tokenizer', 't.json'],
             dict(MODEL, assets=[dict(ownership='reference', source_path=path)]), [path]),
            (['instance', 'show', 'chat'], dict(INSTANCE, document=document, dependencies=[path]), [document, path]),
            (['instance', 'export', 'qwen08', '--output', 'r.json'],
             dict(fixtures()['instance', 'export'], output=document), [document]),
        ]
        for width in (35, 80):
            for argv, data, paths in cases:
                with self.subTest(width=width, argv=argv):
                    output = hem_render.render(hem_cli.parse_args(argv), data, width=width)
                    for value in paths:
                        self.assertIn(value, output)

    def test_logs_are_raw_and_follow_has_no_duplicate_final_snapshot(self):
        for resource in ('server', 'instance'):
            for followed in (False, True):
                snapshot = '{"event":"log","message":"diagnostic"}\n'
                def run(args, deadline):
                    if followed:
                        print(snapshot, end='', file=__import__('sys').stderr)
                    return dict(logs=snapshot, followed=followed)
                for json_mode in (False, True):
                    argv = [resource, 'logs', 'chat'] + (['--json'] if json_mode else [])
                    output, errors = io.StringIO(), io.StringIO()
                    with patch.object(hem_cli.DOMAINS[resource], 'run', side_effect=run):
                        with redirect_stdout(output), redirect_stderr(errors):
                            self.assertEqual(hem_cli.main(argv), 0)
                    if json_mode:
                        self.assertEqual(json.loads(output.getvalue())['logs'], snapshot)
                    else:
                        self.assertEqual(output.getvalue(), '' if followed else snapshot)
                    self.assertEqual(errors.getvalue(), snapshot if followed else '')

    def test_server_follow_flushes_only_new_final_bytes_in_human_mode(self):
        for json_mode in (False, True):
            errors = io.StringIO()
            with patch.object(hem_server_cli, 'read_logs', side_effect=[b'first\n', b'first\nlast\n']):
                with redirect_stderr(errors):
                    result = hem_server_cli._logs(None, None, dict(id='local'), True, json_output=json_mode)
            self.assertEqual(result['logs'], 'first\nlast\n')
            self.assertEqual(errors.getvalue(), 'first\n' if json_mode else 'first\nlast\n')

    def test_rendering_never_contacts_models_registry_or_native_services(self):
        data = fixtures()
        with patch('pathlib.Path.read_text', side_effect=AssertionError('renderer read a file')), \
             patch('pathlib.Path.read_bytes', side_effect=AssertionError('renderer read a file')), \
             patch('socket.socket', side_effect=AssertionError('renderer contacted network')):
            for (resource, command), response in data.items():
                args = hem_cli.parse_args(shlex.split(hem_help.ACTIONS[resource][command][1].splitlines()[0]))
                hem_render.render(args, response, width=80)


class FormattedEquivalentTest(unittest.TestCase):
    def replay(self, command):
        output = io.StringIO()
        hem_display.emit(command, stream=output)
        value = output.getvalue()
        return value, shlex.split(value.replace('\\\n', ''), comments=True)

    def test_multiline_preserves_argv_environment_and_legacy_json_value(self):
        argv = ['loom-serve', '--device=amdgpu', '/file with spaces/$(echo bad);`uname`', "a'b", 'line\nbreak']
        command = hem_display.command(argv, environment={'LOOM_SERVE_DIR': '/state with spaces'})
        display, replay = self.replay(command)
        self.assertIn('\\\n', display)
        self.assertEqual(replay, ['env', 'LOOM_SERVE_DIR=/state with spaces', *argv])
        self.assertEqual(json.loads(json.dumps(command)), shlex.join(['env', 'LOOM_SERVE_DIR=/state with spaces', *argv]))
        self.assertEqual(argv[2], '/file with spaces/$(echo bad);`uname`')

    def test_http_body_header_transport_and_redaction(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "registration ' with spaces.json"
            body = b'{"utf8":"\xc3\xa9","model":"chat"}\n'
            path.write_bytes(body)
            command = hem_display.http('POST', 'http://localhost:8080/v1/models/register',
                                       body_path=path, body=body, headers={'X-Loom-Model': 'chat', 'X-Loom-Cursor': '1:2:0'})
            display, argv = self.replay(command)
            self.assertIn('# Native HTTP request', display)
            self.assertEqual(argv[argv.index('--data-binary') + 1], '@' + str(path))
            self.assertIn('X-Loom-Model: chat', argv)
            self.assertIn('X-Loom-Cursor: 1:2:0', argv)
            self.assertEqual(path.read_bytes(), body)
        command = hem_display.command(['curl', 'https://name:secret@example.com/?token=private',
                                       '--header', 'Authorization: Bearer protected'])
        display, argv = self.replay(command)
        self.assertIn('supply credentials to replay', display)
        for secret in ('name:secret', 'private', 'protected'):
            self.assertNotIn(secret, display)
        self.assertIn('Authorization: <redacted>', argv)

    def test_http_label_cannot_rewrite_literal_body_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / '# Equivalent command.json'
            body = b'{"model":"chat"}\n'
            path.write_bytes(body)
            command = hem_display.http('POST', 'http://localhost:8080/v1/models/register',
                                       body_path=path, body=body)
            display, argv = self.replay(command)
            self.assertIn('# Native HTTP request', display)
            self.assertEqual(argv[argv.index('--data-binary') + 1], '@' + str(path))
            self.assertEqual(path.read_bytes(), body)


if __name__ == '__main__':
    unittest.main()
