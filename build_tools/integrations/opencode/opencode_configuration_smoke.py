"""Real installed-wheel generated OpenCode configuration acceptance.

Uses the existing model Gate's public commands and bounded ownership cleanup.
No protocol fixture, generated-file patching, vendor tool/session qualification,
background discovery or full-window request is involved.
"""
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import uuid


BINARY_SHA256 = '2df936a20cf74af5706020342164628f9a31978c50c2b0ad306b42b614fe245d'


from build_tools.integrations.common import provider, snapshot


class ConfigurationGate:
    def __init__(self, gate):
        self.gate = gate
        self.endpoints = gate.state_path('a') / 'clients/endpoints.json'
        self.runtime = gate.args.opencode_runtime.resolve()
        self.client = self.runtime / 'bin/opencode'
        self.target = gate.runtime_root / 'opencode-config'
        self.project = gate.runtime_root / 'opencode-project'
        self.child = None
        self.identity = None
        self.log = None
        self.password = uuid.uuid4().hex
        self.evidence = dict(commands=[], metadata=[], requests=[], client_cleanup=[])
        gate.results['opencode_configuration'] = self.evidence
        self.env = dict(os.environ, LOOM_SERVE_DIR=str(gate.state_path('a')),
                        OPENCODE_CONFIG_DIR=str(self.target), OPENCODE_PASSWORD=self.password,
                        OPENCODE_CONFIG_PROJECT_DISABLE='1', OPENCODE_DISABLE_MODELS_FETCH='1',
                        XDG_DATA_HOME=str(gate.runtime_root / 'opencode-data'),
                        XDG_CACHE_HOME=str(gate.runtime_root / 'opencode-cache'),
                        XDG_STATE_HOME=str(gate.runtime_root / 'opencode-state'),
                        PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1')
        for key in ('PYTHONPATH', 'PYTHONHOME', 'OPENCODE_CONFIG', 'OPENCODE_CONFIG_CONTENT',
                    'OPENCODE_SERVER', 'OPENCODE_SERVER_URL'):
            self.env.pop(key, None)
        self.env['PATH'] = str(self.client.parent) + os.pathsep + str(gate.states['a'].parent) + os.pathsep + self.env.get('PATH', '')

    def save(self):
        self.gate.save()

    def run_client(self, *args, timeout=15):
        began = time.monotonic()
        result = subprocess.run([str(self.client), *args], cwd=self.project, env=self.env,
                                capture_output=True, text=True, timeout=timeout)
        number = len(self.evidence['commands'])
        for suffix, text in (('stdout', result.stdout), ('stderr', result.stderr)):
            (self.gate.root / f'opencode-{number:02}.{suffix}').write_text(text.replace(self.password, '[REDACTED]'))
        record = dict(args=args, exit=result.returncode, seconds=time.monotonic() - began)
        self.evidence['commands'].append(record)
        self.save()
        assert result.returncode == 0, (record, result.stdout[-2000:], result.stderr[-2000:])
        return result.stdout

    def command(self, *args, **kwargs):
        return self.gate.command('a', 'opencode', 'configure', '--opencode-dir', str(self.target),
                                 *args, env_overrides=self.env, **kwargs)

    def start_client(self):
        assert self.child is None
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
        self.url = f'http://127.0.0.1:{port}'
        self.port = port
        index = len(self.evidence['client_cleanup'])
        self.log = (self.gate.root / f'opencode-server-{index}.stderr').open('w')
        self.identity = None
        self.child = subprocess.Popen([str(self.client), 'serve', '--hostname', '127.0.0.1', '--port', str(port)],
            cwd=self.project, env=self.env, stdin=subprocess.DEVNULL, stdout=self.log, stderr=self.log)
        # Save process identity immediately, before readiness can fail.
        from hrx_solutions.control.hem_process import identity
        self.identity = identity(self.child.pid)
        end = time.monotonic() + 10
        while time.monotonic() < end:
            assert self.child.poll() is None, 'owned OpenCode exited during startup'
            with socket.socket() as sock:
                if sock.connect_ex(('127.0.0.1', port)) == 0:
                    return
            time.sleep(.05)
        raise TimeoutError('owned OpenCode startup exceeded 10 seconds')

    def stop_client(self):
        if self.child is None:
            return
        child, saved = self.child, self.identity
        # Always reap the exact Popen child, even if capturing procfs identity
        # failed immediately after spawn. Never discover cleanup PIDs by port.
        forced = False
        try:
            if child.poll() is None:
                child.terminate()
            try:
                code = child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                forced = True
                child.kill()
                code = child.wait(timeout=5)
        finally:
            self.log.close()
        self.child = None
        if saved is not None:
            from hrx_solutions.control.hem_process import _confirmed_exit
            assert _confirmed_exit(saved), saved
        with socket.socket() as sock:
            assert sock.connect_ex(('127.0.0.1', self.port)) != 0, 'owned OpenCode port remains open'
        self.evidence['client_cleanup'].append(dict(pid=child.pid, process=saved, exit=code,
            processes_exited=True, identity_verified=saved is not None, port_closed=True))
        self.save()
        assert not forced, 'OpenCode required forced termination'

    def cleanup_client(self):
        original = sys.exc_info()[1]
        try:
            self.stop_client()
        except Exception as error:
            self.evidence['cleanup_error'] = repr(error)
            self.save()
            if original is None:
                raise
            print('OpenCode cleanup unresolved: ' + str(error), file=sys.stderr)

    def reload(self):
        self.run_client('reload', '--server', self.url)

    def models(self, expected):
        payload = json.loads(self.run_client('api', '--server', self.url, 'model.list'))
        rows = [row for row in payload['data'] if row['providerID'].startswith('loom-hem-')]
        assert {row['providerID'] + '/' + row['id'] for row in rows} == set(expected), rows
        for row in rows:
            native = expected[row['providerID'] + '/' + row['id']]
            assert row['modelID'] == row['id'], row
            assert row['settings']['baseURL'] == native['base_url'], row
            assert not row['settings'].get('apiKey'), row
            assert row['limit'] == native['limit'], row
            assert row['capabilities'] == dict(tools=False, input=['text'], output=['text']), row
            assert row['package'] == '@opencode/ai/providers/openai-compatible', row
        self.evidence['metadata'].append(rows)
        self.save()
        return rows

    def native(self, namespace, name):
        server, endpoint, _, _ = self.gate.servers[namespace]
        documents = {}
        for path in ('status', 'models'):
            with self.gate.opener.open(endpoint + '/v1/' + path, timeout=5) as response:
                documents[path] = json.load(response)
        card = next(row for row in documents['models']['data'] if row['id'] == name)
        status = next(row for row in documents['status']['data'] if row['id'] == name)
        assert card['loom']['incarnation'] == status['loom']['incarnation']
        assert status['loom']['registration'] == 'ready'
        assert status['loom']['parameters']['primary']['active'] and status['loom']['status']['admission'] == 'open'
        assert card['input_modalities'] == ['text'] and card['output_modalities'] == ['text']
        record = dict(base_url=endpoint + '/v1', limit=dict(context=card['context_length'],
            output=min(card['context_length'], card['limits']['max_output_tokens'])), native=documents)
        assert record['limit']['context'] == 32768, record
        self.evidence.setdefault('native', {})[provider(server) + '/' + name] = record
        self.save()
        return record

    def activate(self, namespace, name):
        answer = self.gate.command(namespace, 'instance', 'start', 'small-a', '--name', name,
            '--server', self.gate.servers[namespace][0], '--pipeline', 'qwen-08b',
            '--context-capacity', '32768', '--pool-capacity', '32768', '--rows', '4', '--prefill-capacity', '128')
        assert answer['completed'] and answer['accepted'], answer
        return provider(self.gate.servers[namespace][0]) + '/' + name, self.native(namespace, name)

    def restart_native(self, namespace, previous):
        name, endpoint, _, port = previous
        record = self.gate.inspect_server(namespace, name)['record']
        assert record is not None and record['process'] is None, record
        self.gate.attempts[namespace] = dict(name=name, port=port, token=None, process=None)
        answer = self.gate.command(namespace, 'server', 'start', name)
        self.gate.attempts[namespace].update(token=answer['record']['launch_token'], process=answer['process'])
        assert answer['endpoint'] == endpoint, answer
        self.gate.servers[namespace] = (name, endpoint, answer['process'], port)

    def request(self, model):
        text = self.run_client('run', '--server', self.url, '--model', model, '--format', 'json',
                               'Say hello in two words. Do not use tools.', timeout=60)
        events = [json.loads(line) for line in text.splitlines() if line.startswith('{')]
        assert not any(event.get('type') == 'error' for event in events), events
        assert any(event.get('type') == 'text' and event.get('part', {}).get('text', '').strip() for event in events), events
        self.evidence['requests'].append(dict(selected=model, events=events))
        self.save()

    def run(self):
        gate = self.gate
        for directory in (self.target, self.project):
            directory.mkdir(mode=0o700)
        assert hashlib.sha256(self.client.read_bytes()).hexdigest() == BINARY_SHA256
        assert (self.runtime / 'package-lock.json').read_bytes() == (gate.args.source / 'build_tools/integrations/opencode/runtime/package-lock.json').read_bytes()
        self.evidence['runtime'] = dict(version=self.run_client('--version').strip(),
            executable=str(self.client.resolve()), binary_sha256=BINARY_SHA256,
            lock_sha256=hashlib.sha256((self.runtime / 'package-lock.json').read_bytes()).hexdigest())
        assert self.evidence['runtime']['version'] == 'opencode v2.0.31'
        # Unrelated user policy/credentials are immutable inputs to installation.
        original = '{\n  // Unrelated settings and default provider remain user-owned.\n  "model": "unrelated/provider",\n}\n'
        config = self.target / 'opencode.jsonc'
        config.write_text(original)
        config.chmod(0o600)
        credentials = self.target / 'user-credentials.txt'
        credentials.write_text('synthetic non-secret user-owned fixture\n')
        credentials.chmod(0o600)
        preserved = snapshot(self.target)
        gate.states['c'] = gate.states['a']
        gate.namespace_roots['c'] = 'a'
        gate.start_server('c', 'server-opencode-sibling')
        expected = dict([self.activate('a', 'oc-qwen-a'), self.activate('c', 'oc-qwen-b')])
        configured = self.command()
        assert configured['configured'], configured
        assert 'policy_conflicts' not in configured and 'policy_warnings' not in configured, configured
        assert 'host policy is unassessed' in configured['guidance'], configured
        installed = self.target / 'plugins/loom-control-opencode'
        from hrx_solutions.integrations.opencode.hem_opencode import bundled_assets
        source = gate.args.source / 'binding/python/opencode_plugin'
        for name, data in bundled_assets().items():
            assert data == (source.parent / name if name == "native_protocol.ts" else source / name).read_bytes() == (installed / name).read_bytes(), name
        before = snapshot(self.target)
        repeated = self.command()
        assert repeated['changed'] is False and snapshot(self.target) == before
        checked = self.command('--check')
        assert checked['configured'] and snapshot(self.target) == before
        assert 'policy_conflicts' not in checked and 'policy_warnings' not in checked, checked
        self.start_client()
        try:
            self.models(expected)
            for model in expected:
                self.request(model)
            gate.command('a', 'instance', 'stop', 'oc-qwen-a', '--server', gate.servers['a'][0])
            # Captured registry is intentionally unchanged before explicit reload.
            self.models(expected)
            self.reload()
            only_b = {key: value for key, value in expected.items() if key.endswith('/oc-qwen-b')}
            self.models(only_b)
            expected.update([self.activate('a', 'oc-qwen-a')])
            self.reload()
            self.models(expected)
            # Both native server and client restart without configuration writes.
            previous_c = gate.servers['c']
            gate.shutdown('c')
            self.restart_native('c', previous_c)
            expected.update([self.activate('c', 'oc-qwen-b')])
            self.stop_client()
            self.start_client()
            self.models(expected)
            # A newly saved public server appears through lifecycle projection,
            # without another configure invocation or a new private lifecycle API.
            gate.states['d'] = gate.states['a']
            gate.namespace_roots['d'] = 'a'
            gate.start_server('d', 'server-opencode-new')
            expected.update([self.activate('d', 'oc-qwen-new')])
            self.reload()
            self.models(expected)
            self.request(next(key for key in expected if key.endswith('/oc-qwen-new')))
            gate.shutdown('d')
            gate.shutdown('c')
            previous_a = gate.servers['a']
            gate.shutdown('a')
            self.reload()
            self.models({})
            assert snapshot(self.target) == before, 'refresh/restart/lifecycle wrote owned configuration'
        finally:
            self.cleanup_client()
        # Passive --check remains write-free without the client. Compatibility
        # is unqualified while ownership/configuration remain inspectable.
        self.env['PATH'] = '/no-client'
        checked = self.command('--check', expected=1)
        assert checked['configured'] and checked['runtime_compatibility']['state'] == 'unqualified'
        assert snapshot(self.target) == before
        self.evidence['absent_client_check'] = checked
        # Foreign/edited files are preserved (source tests cover destructive
        # conflict variants); ordinary owned disable removes only our plugin.
        disabled = self.command('--disable')
        assert disabled['state'] == 'disabled' and not installed.exists(), disabled
        assert all(snapshot(self.target)[name] == value for name, value in preserved.items())
        assert self.endpoints.is_file()
        self.evidence['idempotent_passive_preserved'] = True
        self.evidence['success'] = True
        # Outer CI gate still checks its normal empty-server shutdown path.
        self.restart_native('a', previous_a)
        self.save()


def qualify(gate):
    check = ConfigurationGate(gate)
    try:
        check.run()
    finally:
        check.cleanup_client()
