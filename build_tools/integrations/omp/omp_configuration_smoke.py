"""Real installed-wheel OMP generated-configuration acceptance; no fake endpoint.

Only public configure writes the adapter/YAML. Genuine OMP startup, explicit
refresh and short CLI requests exercise those unchanged generated files.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from build_tools.integrations.common import provider, snapshot


def configuration_snapshot(target):
    """OMP's own SQLite runtime stores are not toolkit configuration.

    Keep every other target file, including unexpected configuration additions,
    in the byte+mtime comparison. Exact released database basenames only. The
    mixed auth/usage store is not certified here; passive checks compare all
    bytes including these databases and never execute the client.
    """
    caches = {'agent.db', 'models.db', 'skill-descriptions.db'}
    return {name:value for name,value in snapshot(target).items()
            if name not in caches and not any(name == cache+suffix for cache in caches for suffix in ('-wal','-shm'))}


class ConfigurationGate:
    def __init__(self, gate):
        self.gate = gate
        self.endpoints = gate.state_path('a') / 'clients/endpoints.json'
        self.runtime = gate.args.omp_runtime.resolve()
        self.bun = self.runtime / 'runtime/node_modules/.bin/bun'
        self.package = self.runtime / 'client/node_modules/@oh-my-pi/pi-coding-agent'
        self.client = self.package / 'src/cli.ts'
        self.target = gate.runtime_root / 'omp-agent'
        self.project = gate.runtime_root / 'omp-project'
        self.child = None
        self.identity = None
        self.evidence = dict(commands=[], metadata=[], requests=[], client_cleanup=[])
        gate.results['omp_configuration'] = self.evidence
        self.env = dict(os.environ, LOOM_SERVE_DIR=str(gate.state_path('a')),
            PI_CODING_AGENT_DIR=str(self.target), OMP_PROFILE='', PI_PROFILE='',
            XDG_DATA_HOME=str(gate.runtime_root / 'omp-data'), XDG_CACHE_HOME=str(gate.runtime_root / 'omp-cache'),
            XDG_STATE_HOME=str(gate.runtime_root / 'omp-state'), LOOM_OMP_TEST_PACKAGE=str(self.package),
            PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1')
        for key in ('PYTHONPATH', 'PYTHONHOME', 'BUN_OPTIONS', 'NODE_OPTIONS', 'NODE_PATH'):
            self.env.pop(key, None)
        self.env['PATH'] = str(self.bun.parent) + os.pathsep + str(self.runtime / 'client/node_modules/.bin') + os.pathsep + self.env.get('PATH', '')

    def save(self):
        self.gate.save()

    def stop_client(self):
        if self.child is None:
            return
        child, saved = self.child, self.identity
        forced = False
        if child.poll() is None:
            child.terminate()
        try:
            code = child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            forced = True
            child.kill()
            code = child.wait(timeout=5)
        self.child = None
        if saved is not None:
            from hrx_solutions.hem_process import _confirmed_exit
            assert _confirmed_exit(saved), saved
        self.evidence['client_cleanup'].append(dict(pid=child.pid, process=saved, exit=code,
            processes_exited=True, identity_verified=saved is not None, forced=forced))
        self.save()
        assert not forced, 'OMP required forced termination'

    def cleanup_client(self):
        original = sys.exc_info()[1]
        try:
            self.stop_client()
        except Exception as error:
            self.evidence['cleanup_error'] = repr(error)
            self.save()
            if original is None:
                raise
            print('OMP cleanup unresolved: ' + str(error), file=sys.stderr)

    def run_client(self, *args, timeout=15, probe=False):
        began = time.monotonic()
        entry = self.gate.args.source / 'tests/omp_configuration_probe.mjs' if probe else self.client
        self.identity = None
        self.child = subprocess.Popen([str(self.bun), str(entry), *args], cwd=self.project,
            env=self.env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            from hrx_solutions.hem_process import identity
            self.identity = identity(self.child.pid)
            stdout, stderr = self.child.communicate(timeout=timeout)
            code = self.child.returncode
        finally:
            self.cleanup_client()
        number = len(self.evidence['commands'])
        for suffix, text in (('stdout', stdout), ('stderr', stderr)):
            (self.gate.root / f'omp-{number:02}.{suffix}').write_text(text)
        record = dict(args=args, probe=probe, exit=code, seconds=time.monotonic()-began)
        self.evidence['commands'].append(record)
        self.save()
        assert code == 0, (record, stdout[-2000:], stderr[-2000:])
        return stdout

    def command(self, *args, **kwargs):
        return self.gate.command('a', 'omp', 'configure', '--omp-dir', str(self.target),
            *args, env_overrides=self.env, **kwargs)

    def models(self, expected):
        # Explicit public refresh proves the user-facing action; a new host
        # then proves the documented restart consumes current production data.
        listing = json.loads(self.run_client('models', 'refresh', '--json'))
        rows = [row for row in listing['models'] if row['provider'].startswith('loom-hem-')]
        assert {row['provider']+'/'+row['id'] for row in rows} == set(expected), rows
        normalized = json.loads(self.run_client(probe=True))['models']
        assert {row['provider']+'/'+row['id'] for row in normalized} == set(expected), normalized
        for row in normalized:
            native = expected[row['provider']+'/'+row['id']]
            assert row['baseUrl'] == native['base_url'], row
            assert row['contextWindow'] == native['limit']['context'], row
            assert row['maxTokens'] == native['limit']['output'], row
            assert row['supportsTools'] is False and row['reasoning'] is False and row['input'] == ['text'], row
        self.evidence['metadata'].append(dict(cli=rows, normalized=normalized))
        self.save()

    def native(self, namespace, name):
        server, endpoint, _, _ = self.gate.servers[namespace]
        documents = {}
        for path in ('status', 'models'):
            with self.gate.opener.open(endpoint+'/v1/'+path, timeout=5) as response:
                documents[path] = json.load(response)
        card = next(row for row in documents['models']['data'] if row['id'] == name)
        status = next(row for row in documents['status']['data'] if row['id'] == name)
        assert card['loom']['incarnation'] == status['loom']['incarnation']
        assert status['loom']['registration'] == 'ready' and status['loom']['parameters']['primary']['active']
        assert status['loom']['status']['admission'] == 'open'
        assert card['input_modalities'] == ['text'] and card['output_modalities'] == ['text']
        record = dict(base_url=endpoint+'/v1', limit=dict(context=card['context_length'],
            output=min(card['context_length'], card['limits']['max_output_tokens'])), native=documents)
        assert record['limit']['context'] == 32768
        self.evidence.setdefault('native', {})[provider(server)+'/'+name] = record
        self.save()
        return record

    def activate(self, namespace, name):
        answer = self.gate.command(namespace, 'instance', 'start', 'small-a', '--name', name,
            '--server', self.gate.servers[namespace][0], '--pipeline', 'qwen-08b',
            '--context-capacity', '32768', '--pool-capacity', '32768', '--rows', '4', '--prefill-capacity', '128')
        assert answer['completed'] and answer['accepted'], answer
        return provider(self.gate.servers[namespace][0])+'/'+name, self.native(namespace, name)

    def restart_native(self, namespace, previous):
        name, endpoint, _, port = previous
        record = self.gate.inspect_server(namespace, name)['record']
        assert record is not None and record['process'] is None
        self.gate.attempts[namespace] = dict(name=name, port=port, token=None, process=None)
        answer = self.gate.command(namespace, 'server', 'start', name)
        self.gate.attempts[namespace].update(token=answer['record']['launch_token'], process=answer['process'])
        assert answer['endpoint'] == endpoint
        self.gate.servers[namespace] = name, endpoint, answer['process'], port

    def request(self, model):
        text = self.run_client('--print', '--no-session', '--model', model,
            'Reply with exactly Hello! and nothing else. Do not use tools.', timeout=90)
        assert text.strip() and not text.lstrip().startswith(('Error:', 'error:')), text
        self.evidence['requests'].append(dict(selected=model, response=text.strip()))
        self.save()

    def run(self):
        gate = self.gate
        for directory in (self.target, self.project, Path(self.env['XDG_DATA_HOME']), Path(self.env['XDG_CACHE_HOME']), Path(self.env['XDG_STATE_HOME'])):
            directory.mkdir(mode=0o700)
        from hrx_solutions.hem_omp import BUN_SHA256
        assert hashlib.sha256(self.bun.read_bytes()).hexdigest() == BUN_SHA256
        for directory, names in [('client', ('package.json','bun.lock')), ('runtime', ('package.json','package-lock.json'))]:
            for name in names:
                assert (self.runtime/directory/name).read_bytes() == (gate.args.source/'build_tools/integrations/omp/runtime'/directory/name).read_bytes()
        self.evidence['runtime'] = dict(version=self.run_client('--version').strip(),
            executable=str(self.client.resolve()), bun=str(self.bun.resolve()), bun_sha256=BUN_SHA256)
        assert self.evidence['runtime']['version'] == 'omp/18.11.2'
        gate.states['e'] = gate.states['a']; gate.namespace_roots['e'] = 'a'
        gate.start_server('e', 'server-omp-sibling')
        expected = dict([self.activate('a','omp-qwen-a'), self.activate('e','omp-qwen-b')])
        # Borrow one semantically equivalent user stanza; it must never be owned.
        borrowed = provider(gate.servers['e'][0])
        original = '# Keep user comments and provider order.\nproviders:\n  unrelated:\n    apiKey: synthetic-not-a-secret\n  '+borrowed+':\n    auth: none\n'
        config = self.target/'models.yml'; config.write_text(original); config.chmod(0o600)
        settings = self.target/'config.yml'; settings.write_text('# User default remains untouched.\nmodel: unrelated/default\n'); settings.chmod(0o600)
        assert self.command()['configured']
        installed = self.target/'extensions/loom-control-omp'
        from hrx_solutions.hem_omp import bundled_assets
        for name,data in bundled_assets().items():
            assert data == (installed/name).read_bytes() == (gate.args.source/'binding/python'/('native_protocol.ts' if name=='native_protocol.ts' else 'omp_extension/'+name)).read_bytes()
        before = configuration_snapshot(self.target)
        assert self.command()['changed'] is False and configuration_snapshot(self.target) == before
        passive_before = snapshot(self.target)
        assert self.command('--check')['configured'] and snapshot(self.target) == passive_before
        self.models(expected)
        for model in expected: self.request(model)
        gate.command('a','instance','stop','omp-qwen-a','--server',gate.servers['a'][0])
        self.models({key:value for key,value in expected.items() if key.endswith('/omp-qwen-b')})
        expected.update([self.activate('a','omp-qwen-a')]); self.models(expected)
        previous_e = gate.servers['e']; gate.shutdown('e'); self.restart_native('e',previous_e)
        expected.update([self.activate('e','omp-qwen-b')]); self.models(expected)
        assert configuration_snapshot(self.target) == before, 'runtime refreshed configuration'
        gate.states['f'] = gate.states['a']; gate.namespace_roots['f'] = 'a'
        gate.start_server('f','server-omp-new')
        new_model,new_native = self.activate('f','omp-qwen-new')
        passive_before = snapshot(self.target)
        missing = self.command('--check',expected=(1,2))
        assert not missing['configured'] and 'configure' in json.dumps(missing).lower(), missing
        assert snapshot(self.target) == passive_before
        assert configuration_snapshot(self.target) == before
        self.models(expected)
        assert self.command()['configured']
        expected[new_model] = new_native
        updated = configuration_snapshot(self.target)
        assert settings.read_text() == '# User default remains untouched.\nmodel: unrelated/default\n'
        assert 'apiKey: synthetic-not-a-secret' in config.read_text()
        assert self.command()['changed'] is False and configuration_snapshot(self.target) == updated
        self.models(expected); self.request(new_model)
        gate.shutdown('f')
        self.models({key:value for key,value in expected.items() if key != new_model})
        gate.shutdown('e'); previous_a = gate.servers['a']; gate.shutdown('a'); self.models({})
        assert configuration_snapshot(self.target) == updated
        # Passive inspection and disable must not execute OMP/Bun.
        self.env['PATH'] = '/no-client'
        passive_before = snapshot(self.target)
        checked = self.command('--check',expected=1)
        assert checked['configured'] and checked['runtime_compatibility']['state'] == 'unqualified'
        assert snapshot(self.target) == passive_before
        self.evidence['absent_client_check'] = checked
        disabled = self.command('--disable')
        assert disabled['state'] == 'disabled' and not installed.exists()
        assert config.read_text() == original
        assert self.endpoints.is_file()
        self.evidence['idempotent_passive_preserved'] = True
        self.evidence['success'] = True
        self.restart_native('a',previous_a)
        self.save()


def qualify(gate):
    check = ConfigurationGate(gate)
    try:
        check.run()
    finally:
        check.cleanup_client()
