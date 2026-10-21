#!/usr/bin/env python3
"""Qualify public generated WebUI wiring against real isolated native models.

Run outside the checkout in an installed-wheel venv. The caller holds the GPU
lock. Docker supplies the pinned host-network runtime and a bounded passive
observer; --webui-python plus --webui-backend select a native isolated runtime.
No existing user WebUI, configuration, or process is adopted.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import secrets
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from build_tools.validation.hem_model_smoke import Cancelled, Gate, run_child, write_cleanup

IMAGE = 'ghcr.io/open-webui/open-webui@sha256:a537bbe1e6b428820f52ed4a029d7fff97062e24d5cf32c365fba36fd0542def'
REVISION = '91ea4e3421d1d8069f13d4a463fcb5986d94ea60'


def port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


class ConfigurationGate:
    def __init__(self, gate):
        self.gate = gate
        self.root = gate.runtime_root / 'webui'
        self.root.mkdir(mode=0o700)
        (self.root / 'data').mkdir(mode=0o700)
        self.url = 'http://127.0.0.1:' + str(port())
        self.name = 'loom-webui-gate-' + uuid.uuid4().hex[:12]
        self.container_owner = uuid.uuid4().hex
        self.child = None
        self.process = None
        self.container = False
        self.token = None
        self.password = secrets.token_urlsafe(32)
        self.foreign_key = secrets.token_urlsafe(32)
        self.evidence = dict(management=[], streams=[], cleanup=[], image=IMAGE,
                             source_revision=REVISION,
                             topology='native' if gate.args.webui_python else 'linux-host-network')
        gate.results['openwebui_configuration'] = self.evidence
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        self.settings = dict(WEBUI_ADMIN_EMAIL='gate@example.invalid', WEBUI_ADMIN_PASSWORD=self.password,
            WEBUI_SECRET_KEY=secrets.token_urlsafe(32), HOST='127.0.0.1', PORT=self.url.rsplit(':', 1)[1],
            DATA_DIR=str(self.root / 'data'), OFFLINE_MODE='true', HF_HUB_OFFLINE='1',
            ENABLE_OLLAMA_API='false', ENABLE_OPENAI_API='true', OPENAI_API_BASE_URLS='', OPENAI_API_KEYS='',
            ENABLE_PERSISTENT_CONFIG='true', RESET_CONFIG_ON_START='false',
            ENABLE_FORWARD_USER_INFO_HEADERS='false', ENABLE_SIGNUP='false',
            RAG_EMBEDDING_ENGINE='', RAG_EMBEDDING_MODEL='', SCARF_NO_ANALYTICS='true', DO_NOT_TRACK='true')
        labels = json.loads(subprocess.check_output(['docker', 'image', 'inspect', '--format',
            '{{json .Config.Labels}}', IMAGE], text=True))
        assert labels['org.opencontainers.image.revision'] == REVISION
        self.evidence['image_revision_verified'] = True
        if gate.args.webui_python:
            code = ('import hashlib,json,pathlib; root=pathlib.Path("/app/backend/open_webui"); '
                    'print(json.dumps({str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() '
                    'for p in root.rglob("*.py")}))')
            expected = json.loads(subprocess.check_output(['docker', 'run', '--rm', '--entrypoint',
                'python', IMAGE, '-c', code], text=True, timeout=20))
            source = gate.args.webui_backend / 'open_webui'
            actual = {str(path.relative_to(source)): hashlib.sha256(path.read_bytes()).hexdigest()
                      for path in source.rglob('*.py')}
            assert actual == expected, 'native WebUI source differs from exact pinned image'
            self.evidence['native_source_sha256'] = actual

    def request(self, path, body=None, stream=False, expected=200):
        headers = {'Content-Type': 'application/json'}
        if self.token:
            headers['Authorization'] = 'Bearer ' + self.token
        began = time.monotonic()
        request = urllib.request.Request(self.url + path, headers=headers,
            data=None if body is None else json.dumps(body).encode())
        try:
            with self.opener.open(request, timeout=30 if stream else 10) as response:
                status, data = response.status, response.read(2*1024*1024)
        except urllib.error.HTTPError as error:
            status, data = error.code, error.read(2*1024*1024)
        self.evidence['management'].append(dict(path=path, method='GET' if body is None else 'POST',
            status=status, seconds=time.monotonic()-began, bytes=len(data)))
        self.gate.save()
        assert status == expected, (path, status)  # Bodies can contain credentials.
        return data if stream else json.loads(data)

    def start(self):
        if self.gate.args.webui_python:
            env = dict(os.environ, **self.settings, PYTHONNOUSERSITE='1')
            env.pop('PYTHONPATH', None)
            env.pop('PYTHONHOME', None)
            identity = json.loads(subprocess.check_output([str(self.gate.args.webui_python), '-c',
                'import sys,site,json,uvicorn; print(json.dumps(dict(version=sys.version,prefix=sys.prefix,base_prefix=sys.base_prefix,user_site=site.ENABLE_USER_SITE,uvicorn=uvicorn.__file__)))'],
                env=env, cwd=self.root, text=True))
            assert identity['prefix'] != identity['base_prefix'] and not identity['user_site']
            assert Path(identity['uvicorn']).is_relative_to(Path(identity['prefix']))
            self.evidence['native_runtime'] = identity
            log = (self.root / 'native.log').open('ab')
            try:
                self.child = subprocess.Popen([str(self.gate.args.webui_python), '-m', 'uvicorn',
                    'open_webui.main:app', '--host', '127.0.0.1', '--port', self.settings['PORT']],
                    env=env, cwd=self.gate.args.webui_backend, stdout=log, stderr=log)
                from hrx_solutions.hem_process import identity
                self.process = identity(self.child.pid)
            finally:
                log.close()
        else:
            assert not self.container_ids(owned=False), 'generated WebUI container name already exists'
            envfile = self.root / 'runtime.env'
            envfile.write_text(''.join(name + '=' + value + '\n' for name, value in self.settings.items()))
            envfile.chmod(0o600)
            # Record ownership intent before submission: Docker may create the
            # labelled container even if the CLI acknowledgement is lost.
            self.container = True
            result = run_child(['docker', 'run', '-d', '--name', self.name,
                '--label', 'hrx.openwebui-gate-owner=' + self.container_owner, '--network', 'host',
                '--env-file', str(envfile), '-v', str(self.root)+':'+str(self.root), IMAGE],
                capture_output=True, text=True, timeout=30)
            assert result.returncode == 0, 'owned WebUI container failed to start'
        # Startup dependency/database migrations are outside the 10s management budget.
        end = time.monotonic() + 90
        while time.monotonic() < end:
            if self.child:
                assert self.child.poll() is None, 'owned native WebUI exited during startup; see private runtime log'
            try:
                version = self.request('/api/version')
                assert version['version'] == '0.12.3'
                self.evidence['version'] = version
                return
            except (urllib.error.URLError, ConnectionError, TimeoutError):
                time.sleep(.5)
        raise TimeoutError('owned WebUI startup exceeded 90s')

    def container_ids(self, owned=True):
        argv = ['docker', 'container', 'ls', '-a', '--no-trunc', '--filter',
                'name=^/' + self.name + '$', '--format', '{{.ID}}']
        if owned:
            argv.extend(['--filter', 'label=hrx.openwebui-gate-owner=' + self.container_owner])
        result = run_child(argv, capture_output=True, text=True, timeout=10)
        assert result.returncode == 0, 'cannot inspect owned WebUI container cleanup'
        ids = result.stdout.split()
        assert len(ids) <= 1 and all(len(value) == 64 and all(c in '0123456789abcdef' for c in value) for value in ids)
        return ids

    def stop(self):
        if self.child:
            child = self.child
            forced = False
            if child.poll() is None:
                child.terminate()
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                forced = True
                child.kill()
                child.wait(timeout=5)
            self.child = None
            if self.process:
                from hrx_solutions.hem_process import _confirmed_exit
                assert _confirmed_exit(self.process)
            self.evidence['cleanup'].append(dict(pid=child.pid, process=self.process, exited=True, forced=forced))
            assert not forced, 'owned native WebUI required forced termination'
        if self.container:
            ids = self.container_ids()
            for container_id in ids:
                result = run_child(['docker', 'rm', '-f', container_id], capture_output=True, timeout=20)
                assert result.returncode == 0, 'owned WebUI container cleanup failed'
            assert not self.container_ids(), 'owned WebUI container remains after cleanup'
            self.container = False
            self.evidence['cleanup'].append(dict(container=self.name, owned_ids=ids, exited=True))
        with socket.socket() as sock:
            assert sock.connect_ex(('127.0.0.1', int(self.settings['PORT']))) != 0, 'owned WebUI port remains open'
        self.gate.save()

    def configure(self, *args, expected=0, token=None):
        index = len(self.gate.results['commands'])
        result = self.gate.command('a', 'openwebui', 'configure', '--url', self.url, *args,
            expected=expected, env_overrides=dict(OPENWEBUI_TOKEN=self.token if token is None else token))
        displayed = (self.gate.root / f'{index:03}-a.stderr').read_text()
        assert self.token not in displayed and self.foreign_key not in displayed
        if '--check' in args or result.get('changed') is False:
            assert '/openai/config/update' not in displayed, 'unexpected update in passive/no-op command'
        self.evidence.setdefault('configure_results', []).append(result)
        self.gate.save()
        return result

    def models(self, expected):
        rows = self.request('/api/models?refresh=true')['data']
        actual = {row['id'] for row in rows if row['id'].startswith('loom-hem-')}
        assert actual == set(expected), (actual, set(expected))
        self.evidence.setdefault('inventories', []).append(sorted(actual))
        self.gate.save()

    def activate(self, namespace):
        from hrx_solutions.hem_pi_contracts import provider_id
        server = self.gate.servers[namespace][0]
        answer = self.gate.command(namespace, 'instance', 'start', 'small', '--name', 'shared-08b',
            '--server', server, '--pipeline', 'qwen-08b')
        assert answer['completed'] and answer['accepted']
        return provider_id(server) + '.shared-08b'

    def stream(self, namespace, target):
        number = len(self.evidence['streams'])
        output = self.root / ('wire-' + str(number) + '.json')
        observer = 'loom-webui-observer-' + uuid.uuid4().hex[:12]
        script = self.gate.args.source / 'build_tools/integrations/openwebui/openwebui_wire_capture.py'
        native_port = self.gate.servers[namespace][3]
        child = subprocess.Popen(['docker', 'run', '--rm', '--name', observer, '--network', 'host',
            '--cap-drop', 'ALL', '--cap-add', 'NET_RAW', '--cap-add', 'DAC_OVERRIDE', '-v', str(script)+':/capture.py:ro',
            '-v', str(self.root)+':/capture', '--entrypoint', 'python', IMAGE,
            '/capture.py', str(native_port), '/capture/'+output.name],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            end = time.monotonic()+10
            while not output.with_suffix('.ready').exists():
                assert child.poll() is None, 'owned passive observer exited'
                assert time.monotonic() < end, 'owned passive observer startup timed out'
                time.sleep(.05)
            raw = self.request('/api/chat/completions', dict(model=target,
                messages=[dict(role='user', content='Say hello briefly.')], stream=True, max_tokens=24), stream=True)
            frames = [line[6:] for line in raw.decode().splitlines() if line.startswith('data: ')]
            assert frames and frames[-1] == '[DONE]'
            events = [json.loads(frame) for frame in frames[:-1]]
            assert not any('error' in event for event in events)
            assert any(choice.get('delta', {}).get('content', '').strip() for event in events for choice in event.get('choices', []))
            assert child.wait(timeout=5) == 0
            wire = json.loads(output.read_text())
            assert wire['model'] == 'shared-08b' and wire['port'] == native_port
            assert wire['stream'] and wire['max_tokens'] == 24
            assert not any(name in ('authorization', 'cookie') or name.startswith(('x-openwebui-', 'x-user-', 'x-chat-')) for name in wire['header_names'])
            self.evidence['streams'].append(dict(selected=target, native=wire, sse_done=True, content_present=True))
            self.gate.save()
        finally:
            # The exact owned disposable observer is the only cleanup target.
            run_child(['docker', 'rm', '-f', observer], capture_output=True, timeout=10)
            child.wait(timeout=5)

    def preserved(self):
        config = self.request('/openai/config')
        assert config['OPENAI_API_BASE_URLS'][0] == 'http://127.0.0.1:9/v1'
        assert config['OPENAI_API_KEYS'][0] == self.foreign_key
        assert config['OPENAI_API_CONFIGS']['0'] == self.foreign
        assert self.request('/api/v1/models/model?id=foreign-profile') == self.profile
        return config

    def run(self):
        gate = self.gate
        self.start()
        login = self.request('/api/v1/auths/signin', dict(email=self.settings['WEBUI_ADMIN_EMAIL'], password=self.password))
        assert login['role'] == 'admin'
        self.token = login['token']
        login = None
        self.foreign = dict(enable=False, auth_type='bearer', prefix_id='foreign-fixture', model_ids=['unused'],
                            connection_type='external', extra_fixture='preserve')
        self.request('/openai/config/update', dict(ENABLE_OPENAI_API=True,
            OPENAI_API_BASE_URLS=['http://127.0.0.1:9/v1'], OPENAI_API_KEYS=[self.foreign_key], OPENAI_API_CONFIGS={'0': self.foreign}))
        self.request('/api/v1/models/create', dict(id='foreign-profile', base_model_id='unrelated',
            name='Unrelated profile', meta={}, params={'temperature': .123}, access_grants=[]))
        self.profile = self.request('/api/v1/models/model?id=foreign-profile')
        default_config = self.request('/api/v1/configs/models')
        gate.import_model('a', 'small', gate.args.model08, 'Qwen3.5-0.8B-UD-Q5_K_XL.gguf', 'qwen-08b')
        targets = {}
        for namespace, name in [('a', 'webui-first'), ('c', 'webui-second')]:
            gate.states[namespace] = gate.states['a']
            gate.namespace_roots[namespace] = 'a'
            gate.start_server(namespace, name)
            targets[namespace] = self.activate(namespace)
        configured = self.configure()
        assert configured['completed'] and configured['changed']
        initial = self.preserved()
        repeated = self.configure()
        assert repeated['completed'] and not repeated['changed']
        owner = gate.state_path('a') / 'clients/openwebui' / (hashlib.sha256(self.url.encode()).hexdigest() + '.json')
        before_check = owner.read_bytes(), owner.stat().st_mtime_ns
        checked = self.configure('--check')
        assert checked['completed'] and not checked['changed'] and checked['model_availability'] == 'unprobed'
        assert (owner.read_bytes(), owner.stat().st_mtime_ns) == before_check
        assert self.preserved() == initial
        self.models(targets.values())
        for namespace, target in targets.items():
            self.stream(namespace, target)
        gate.command('a', 'instance', 'stop', 'shared-08b', '--server', gate.servers['a'][0])
        self.models([targets['c']])
        self.stream('c', targets['c'])
        gate.command('c', 'instance', 'stop', 'shared-08b', '--server', gate.servers['c'][0])
        self.models([])
        for namespace in targets:
            assert self.activate(namespace) == targets[namespace]
        self.models(targets.values())
        previous = gate.servers['a']
        gate.shutdown('a')
        name, endpoint, _, native_port = previous
        gate.attempts['a'] = dict(name=name, port=native_port, token=None, process=None)
        answer = gate.command('a', 'server', 'start', name)
        gate.attempts['a'].update(token=answer['record']['launch_token'], process=answer['process'])
        gate.servers['a'] = (name, endpoint, answer['process'], native_port)
        assert self.activate('a') == targets['a']
        self.stop()
        self.start()
        assert self.preserved() == initial
        self.models(targets.values())
        self.stream('a', targets['a'])
        gate.states['d'] = gate.states['a']
        gate.namespace_roots['d'] = 'a'
        gate.start_server('d', 'webui-new')
        targets['d'] = self.activate('d')
        configured = self.configure()
        assert configured['completed'] and configured['changed']
        self.models(targets.values())
        self.stream('d', targets['d'])
        assert self.request('/api/v1/configs/models') == default_config
        # Deliberately drift one owned fixture only after all unchanged generated wiring proof.
        config = self.preserved()
        edited = copy.deepcopy(config)
        edited['OPENAI_API_CONFIGS']['1']['extra_owned_edit'] = 'preserve'
        self.request('/openai/config/update', edited)
        assert not self.configure('--check', expected=2)['completed']
        partial = self.configure('--disable', expected=1)
        assert not partial['completed']
        remaining = self.preserved()
        assert len(remaining['OPENAI_API_BASE_URLS']) == 2
        assert remaining['OPENAI_API_CONFIGS']['1']['extra_owned_edit'] == 'preserve'
        assert not self.configure('--disable', expected=1, token='invalid-owned-fixture')['completed']
        self.stop()
        assert not self.configure('--disable', expected=1)['completed']
        self.start()
        # Operator explicitly restores its test edit; toolkit then proves ordinary cleanup.
        remaining = self.preserved()
        del remaining['OPENAI_API_CONFIGS']['1']['extra_owned_edit']
        self.request('/openai/config/update', remaining)
        disabled = self.configure('--disable')
        assert disabled['completed']
        assert len(self.preserved()['OPENAI_API_BASE_URLS']) == 1
        assert self.request('/api/v1/configs/models') == default_config
        assert not self.configure('--disable')['changed']
        self.evidence['success'] = True
        gate.save()


def qualify(gate):
    """Run and always verify cleanup; cleanup failures cannot leave success true."""
    check = None
    original = None
    cleanup_errors = []
    try:
        gate.setup()
        check = ConfigurationGate(gate)
        check.run()
    except BaseException as error:
        original = error
        gate.results['error'] = type(error).__name__ + ': ' + str(error)
        raise
    finally:
        # run_child records a surviving owned child session on the original
        # exception. Never let later native/container cleanup erase that fact.
        if getattr(original, 'cleanup_unresolved', None):
            cleanup_errors.append('owned synchronous child cleanup unresolved')
        try:
            if check:
                check.stop()
        except BaseException as error:
            cleanup_errors.append(type(error).__name__)
        for namespace in list(gate.attempts):
            try:
                gate.cleanup_attempt(namespace)
            except BaseException as error:
                cleanup_errors.append(type(error).__name__)
        gate.results['cleanup_errors'] = cleanup_errors
        gate.results['success'] = original is None and not cleanup_errors
        gate.save()
        write_cleanup('unresolved' if cleanup_errors else 'complete', attempts=list(gate.attempts),
                      errors=cleanup_errors)
        if cleanup_errors and original is None:
            raise RuntimeError('owned WebUI/native cleanup unresolved: ' + ', '.join(cleanup_errors))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--wheel', type=Path, required=True)
    parser.add_argument('--wheel-sha256', required=True)
    parser.add_argument('--model08', type=Path, required=True)
    parser.add_argument('--webui-python', type=Path)
    parser.add_argument('--webui-backend', type=Path)
    args = parser.parse_args()
    if bool(args.webui_python) != bool(args.webui_backend):
        parser.error('native runtime requires both --webui-python and --webui-backend')
    if args.run.exists() and any(args.run.iterdir()):
        parser.error('--run must be empty')
    args.run.mkdir(parents=True, exist_ok=True)
    args.profile, args.acquire08 = 'ci08', False
    args.native_provenance = 'Caller-supplied candidate wheel; retained wheel hash and installed payload identity'
    gate = Gate(args)
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    def cancelled(signum, frame):
        for sig in previous:
            signal.signal(sig, signal.SIG_IGN)
        raise Cancelled(signum)
    for sig in previous:
        signal.signal(sig, cancelled)
    try:
        qualify(gate)
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


if __name__ == '__main__':
    try:
        main()
    except Cancelled as error:
        raise SystemExit(128 + error.signum)
