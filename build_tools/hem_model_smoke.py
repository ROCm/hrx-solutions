#!/usr/bin/env python3
"""Qualify an installed Loom Control wheel with immutable local Qwen fixtures.

Run with a standalone venv interpreter, outside the source checkout. The caller
must hold the host GPU lock. Acquisition is explicit; HOME and credentials stay intact.
"""
import argparse
import hashlib
import json
import os
import re
from pathlib import Path
import site
import shlex
import sqlite3
import socket
import signal
import subprocess
import sys
import time
import urllib.request
import zipfile

from local_ci_runtime import validate_stage


class Cancelled(BaseException):
    def __init__(self, signum):
        self.signum = signum
        super().__init__(f'cancelled by signal {signum}')


def write_cleanup(status, **details):
    destination = os.environ.get('HRX_LOCAL_CI_CLEANUP_FILE')
    if destination:
        path = Path(destination)
        temporary = path.with_name(path.name + '.tmp')
        temporary.write_text(json.dumps(dict(status=status, **details)) + '\n')
        temporary.replace(path)


def session_live(group):
    for path in Path('/proc').glob('[0-9]*/stat'):
        try:
            fields = path.read_text().rsplit(')', 1)[1].split()
            if int(fields[2]) == group and fields[0] != 'Z':
                return True
        except (FileNotFoundError, ProcessLookupError):
            pass
    return False


def run_child(argv, *, capture_output=False, timeout=None, check=False, **kwargs):
    """Own synchronous CLI/download/setup children without touching native sessions."""
    if capture_output:
        kwargs.update(stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    child = None
    old_mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT, signal.SIGTERM})
    try:
        child = subprocess.Popen(argv, start_new_session=True,
                                 preexec_fn=lambda: signal.pthread_sigmask(signal.SIG_SETMASK, old_mask),
                                 **kwargs)
        signal.pthread_sigmask(signal.SIG_SETMASK, old_mask)
        stdout, stderr = child.communicate(timeout=timeout)
    except BaseException as original:
        signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT, signal.SIGTERM})
        try:
            if child is not None:
                try:
                    os.killpg(child.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    child.communicate(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.communicate(timeout=5)
                # The CLI can exit before a downloader it spawned. Prove its
                # separate session is empty; native launches use other sessions.
                end = time.monotonic() + 5
                while session_live(child.pid):
                    try:
                        os.killpg(child.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    if time.monotonic() >= end:
                        raise RuntimeError('owned synchronous child session remains live')
                    time.sleep(.01)
        except Exception as cleanup:
            original.cleanup_unresolved = repr(cleanup)
        raise
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, old_mask)
    result = subprocess.CompletedProcess(argv, child.returncode, stdout, stderr)
    if check:
        result.check_returncode()
    return result


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Gate:
    def __init__(self, args):
        self.args, self.root = args, args.run.resolve()
        internal_root = os.environ.get('HRX_LOCAL_CI_RUNTIME_ROOT')
        self.runtime_root = validate_stage(Path(internal_root), self.root) if internal_root else self.root
        self.results = {'commands': [], 'queries': [], 'cleanup': [], 'success': False,
                        'runtime_root': str(self.runtime_root)}
        self.states = {}
        self.servers = {}
        self.attempts = {}
        self.namespace_roots = {}
        self.streams = []
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def save(self):
        (self.root / 'results.json').write_text(json.dumps(self.results, indent=2) + '\n')

    def state_path(self, namespace):
        return self.runtime_root / self.namespace_roots.get(namespace, namespace) / 'state'

    def verify_empty_state(self, namespace):
        with sqlite3.connect(self.state_path(namespace) / 'control.sqlite3') as connection:
            tables = [row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")]
            assert set(tables) == {'models', 'servers', 'instances', 'pi_target'}, tables
            assert connection.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'
            counts = {table: connection.execute('SELECT COUNT(*) FROM ' + table).fetchone()[0] for table in tables}
            assert counts['instances'] == 0, counts
            assert counts['pi_target'] == 0, counts
            return counts

    def command(self, namespace, *args, expected=0, command_timeout=15, env_overrides=None):
        env = dict(os.environ, LOOM_SERVE_DIR=str(self.state_path(namespace)), PYTHONNOUSERSITE='1')
        env.update(env_overrides or {})
        env.pop('PYTHONPATH', None)
        env.pop('PYTHONHOME', None)
        python = self.states[namespace]
        # Downloads spawn `hf`: select this namespace's installed CLI before
        # inherited tools, preserving the caller's HOME/HF cache/credentials.
        env['PATH'] = str(python.parent) + os.pathsep + env.get('PATH', '')
        start = time.monotonic()
        process = run_child([str(python.parent / 'loom-control'), *args, '--json'],
                                 env=env, cwd=self.root, capture_output=True, text=True, timeout=command_timeout)
        number = len(self.results['commands'])
        (self.root / f'{number:03}-{namespace}.stderr').write_text(process.stderr)
        (self.root / f'{number:03}-{namespace}.stdout').write_text(process.stdout)
        item = dict(namespace=namespace, args=args, seconds=time.monotonic()-start, exit=process.returncode)
        self.results['commands'].append(item)
        self.save()
        codes = expected if isinstance(expected, tuple) else (expected,)
        assert process.returncode in codes, (item, f'see {self.root / f"{number:03}-{namespace}.stderr"} and matching stdout')
        payload = json.loads(process.stdout)
        if args[:2] == ('instance', 'start') and process.returncode == 0:
            # Human equivalents use shell continuations; inspect logical commands.
            commands = process.stderr.replace('\\\n', '').splitlines()
            operations = [line.split('/v1/models/')[1].split()[0] for line in commands if '--request POST' in line and '/v1/models/' in line]
            assert operations == ['register', 'activate', 'undrain'], operations
            register_line = next(line for line in commands if '--request POST' in line and '/v1/models/register' in line)
            argv = shlex.split(register_line)
            document = json.loads(Path(argv[argv.index('--data-binary') + 1].removeprefix('@')).read_text())
            assert document['admission'] == 'drained', document
            item['native_order'] = operations
        self.save()
        return payload

    def setup(self):
        assert sys.prefix != sys.base_prefix and not site.ENABLE_USER_SITE
        assert 'PYTHONPATH' not in os.environ and 'PYTHONHOME' not in os.environ
        assert not Path.cwd().is_relative_to(self.args.source.resolve())
        assert digest(self.args.wheel) == self.args.wheel_sha256
        # Reuse the installed contract authority: catalog-build metadata maps
        # authored source inputs to packaged bytecode instead of source files.
        import hrx_solutions
        from check_installed_cpu import source_parity, verify_wheel_payload
        package = Path(hrx_solutions.__file__).resolve().parent
        package.relative_to(Path(sys.prefix).resolve())
        verify_wheel_payload(self.args.wheel, package)
        self.results['source_parity'] = source_parity(self.args.source, package)
        namespaces = ('a',) if self.args.profile == 'ci08' else ('a', 'b')
        for namespace in namespaces:
            directory = self.root / namespace
            directory.mkdir()
            (self.runtime_root / namespace).mkdir(mode=0o700, exist_ok=True)
            venv = directory / 'venv'
            run_child(['python3', '-m', 'venv', str(venv)], check=True)
            python = venv / 'bin/python'
            run_child([str(python), '-m', 'pip', 'install', '--disable-pip-version-check', str(self.args.wheel),
                            *(['huggingface_hub[cli]==0.34.4'] if self.args.acquire08 else [])],
                           check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            self.states[namespace] = python
            env = dict(os.environ, LOOM_SERVE_DIR=str(self.state_path(namespace)), PYTHONNOUSERSITE='1')
            check = subprocess.check_output([str(python), '-c',
                'import hrx_solutions,sys,site,json; print(json.dumps(dict(package=hrx_solutions.__file__,prefix=sys.prefix,user_site=site.ENABLE_USER_SITE)))'], env=env, cwd=self.root, text=True)
            identity = json.loads(check)
            assert Path(identity['package']).is_relative_to(venv) and not identity['user_site']
            package = Path(identity['package']).parent
            native = package / '_native/bin/loom-serve'
            assert native.is_file(), 'native-bearing wheel required'
            with zipfile.ZipFile(self.args.wheel) as archive:
                installed = {}
                for name in archive.namelist():
                    relative = name.removeprefix('hrx_solutions/')
                    if name.startswith('hrx_solutions/') and not name.endswith('/') and (relative.startswith('_native/') or relative.startswith('loom-pipeline/') or relative.startswith('hem_')):
                        assert (package / relative).read_bytes() == archive.read(name), name
                        installed[relative] = digest(package / relative)
            self.results[namespace] = dict(identity=identity, installed_parity=installed, native_sha256=digest(native))
        self.results['source_commit'] = subprocess.check_output(['git', '-C', str(self.args.source), 'rev-parse', 'HEAD'], text=True).strip()
        self.results['wheel_sha256'] = digest(self.args.wheel)
        self.results['native_provenance'] = self.args.native_provenance
        self.results['profile'] = self.args.profile
        self.save()

    def start_server(self, namespace, name):
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
        # Namespaces are fresh, but verify the name is unused before claiming a
        # launch attempt. Never adopt an existing service for failure cleanup.
        assert self.inspect_server(namespace, name)['record'] is None, name
        self.attempts[namespace] = dict(name=name, port=port, token=None, process=None)
        record = self.command(namespace, 'server', 'start', '--name', name, '--device', 'amdgpu', '--listen', f'127.0.0.1:{port}')
        self.attempts[namespace].update(token=record['record']['launch_token'], process=record['process'])
        self.servers[namespace] = (name, record['endpoint'], record['process'], port)
        assert self.command(namespace, 'instance', 'list', '--server', name)['items'] == []
        return record

    def import_model(self, namespace, name, path, filename, pipeline):
        assert (path / filename).is_file() and (path / 'tokenizer.json').is_file()
        answer = self.command(namespace, 'model', 'import', name, '--gguf', str(path / filename), '--tokenizer', str(path / 'tokenizer.json'), '--alias', '--pipeline', pipeline)
        assert answer['completed'] and answer['validation']['valid']
        assert Path(answer['weights']).resolve() == (path / filename).resolve()
        assert Path(answer['tokenizer']).resolve() == (path / 'tokenizer.json').resolve()
        assert Path(answer['directory']).name == name
        assert all(asset['ownership'] == 'reference' for asset in answer['assets'])
        return answer

    def start_instance(self, namespace, model, pipeline):
        name = self.servers[namespace][0]
        answer = self.command(namespace, 'instance', 'start', model, '--server', name, '--pipeline', pipeline)
        assert answer['completed'] and answer['accepted']
        status = self.command(namespace, 'instance', 'status', model, '--server', name)
        loom = status['native']['loom']
        assert loom['registration'] == 'ready' and loom.get('error') is None, loom
        assert loom['parameter_operation'] == 'idle' and loom['parameters']['active'], loom
        assert loom['status']['admission'] == 'open', loom
        return answer

    def query(self, namespace, model):
        endpoint = self.servers[namespace][1]
        body = dict(model=model, messages=[dict(role='user', content='Reply with exactly Hello! and nothing else.')], temperature=0, max_tokens=32, stream=True)
        began = time.monotonic()
        req = urllib.request.Request(endpoint.rstrip('/') + '/v1/chat/completions', data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
        with self.opener.open(req, timeout=30) as response:
            raw = response.read().decode()
        frames = [line[6:] for line in raw.splitlines() if line.startswith('data: ')]
        assert frames[-1] == '[DONE]', raw
        events = [json.loads(frame) for frame in frames[:-1]]
        assert not any('error' in event for event in events), events
        text = ''.join(choice.get('delta', {}).get('content', '') for event in events for choice in event.get('choices', []))
        assert text.strip() == 'Hello!', text
        assert any(choice.get('finish_reason') == 'stop' for event in events for choice in event.get('choices', []))
        self.results['queries'].append(dict(namespace=namespace, model=model, answer=text, seconds=time.monotonic()-began))
        (self.root / f'query-{namespace}-{len(self.results["queries"])}.sse').write_text(raw)
        self.save()

    def long_query(self, namespace, model):
        endpoint = self.servers[namespace][1]
        body = dict(model=model, messages=[dict(role='user', content='Count every integer from 1 through 10000, without skipping any. Start immediately and keep counting.')], temperature=0, max_tokens=16 if model == 'large' else 2048, stream=True)
        path = self.root / f'long-query-{namespace}-{len(self.streams)}.sse'
        log = self.root / f'long-query-{namespace}-{len(self.streams)}.stderr'
        with path.open('wb') as output, log.open('wb') as errors:
            child = subprocess.Popen(['curl', '--noproxy', '*', '--silent', '--show-error', '--max-time', '60', '--header', 'Content-Type: application/json', '--data-binary', json.dumps(body), endpoint + '/v1/chat/completions'], stdout=output, stderr=errors)
        self.streams.append(child)
        end = time.monotonic() + 5
        while time.monotonic() < end:
            with self.opener.open(endpoint + '/v1/status', timeout=2) as response:
                status = json.load(response)
            if any(item['loom'].get('status', {}).get('active_requests', 0) > 0 for item in status['data']):
                self.results.setdefault('inflight', []).append(dict(namespace=namespace, status=status, stream=str(path)))
                self.save()
                return child, path
            assert child.poll() is None, path.read_text()
            time.sleep(.01)
        raise AssertionError('No active native request observed')

    def shutdown(self, namespace, abort=False, timeout=None):
        name, endpoint, process, port = self.servers[namespace]
        self.command(namespace, 'server', 'logs', name)
        answer = self.command(namespace, 'server', 'shutdown', name, *(['--abort'] if abort else []), *(['--timeout', str(timeout)] if timeout is not None else []))
        pid = process['pid']
        try:
            assert answer['completed'] and answer['state'] == 'stopped', answer
            end = time.monotonic() + 10
            while True:
                remaining = end - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError('owned shutdown exit verification exceeded 10 seconds')
                if self.inspect_server(namespace, name, process, timeout=remaining)['exited']:
                    break
                time.sleep(min(.01, max(0, end - time.monotonic())))
            with socket.socket() as sock:
                assert sock.connect_ex(('127.0.0.1', port)) != 0, 'owned port still open'
        except Exception:
            self.shutdown_diagnostics(namespace, process)
            raise
        # absent describes recorded live processes, not their procfs entries.
        self.results['cleanup'].append(dict(namespace=namespace, pid=pid, absent=True,
                                            processes_exited=True, port_closed=True))
        del self.servers[namespace]
        self.attempts.pop(namespace, None)
        self.save()

    def shutdown_diagnostics(self, namespace, process):
        """Capture only bounded procfs identity evidence before recovery."""
        evidence = {}
        for role, saved in (('server', process), ('logger', process.get('logger'))):
            if not saved:
                continue
            item = dict(pid=saved['pid'], saved_boot_id=saved['boot_id'],
                        saved_start_ticks=saved['start_ticks'])
            try:
                item['boot_id'] = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
                with Path(f'/proc/{saved["pid"]}/stat').open() as stream:
                    fields = stream.read(4096).rsplit(')', 1)[1].split()
                item.update(state=fields[0], parent_pid=int(fields[1]), start_ticks=int(fields[19]))
            except FileNotFoundError:
                item['missing'] = True
            except (OSError, ValueError, IndexError) as error:
                item['error'] = str(error)[:512]
            evidence[role] = item
        try:
            (self.root / f'shutdown-{namespace}.json').write_text(json.dumps(evidence, indent=2) + '\n')
        except OSError as error:
            print(f'Could not save shutdown diagnostics: {error}', file=sys.stderr)

    def inspect_server(self, namespace, name, process=None, timeout=10):
        """Read one owned row and verify saved process identities, outside source."""
        env = dict(os.environ, LOOM_SERVE_DIR=str(self.state_path(namespace)), PYTHONNOUSERSITE='1')
        env.pop('PYTHONPATH', None)
        env.pop('PYTHONHOME', None)
        script = """
import json, os, sys
from pathlib import Path
from hrx_solutions.hem_state import Registry
from hrx_solutions.hem_process import _confirmed_exit
root = Path(os.environ['LOOM_SERVE_DIR'])
rows = Registry(root).list_servers() if root.exists() else []
record = next((row for row in rows if row['id'] == sys.argv[1]), None)
process = json.loads(sys.argv[2])
print(json.dumps(dict(record=record, exited=(not process or
    (_confirmed_exit(process) and (not process.get('logger') or _confirmed_exit(process['logger'])))))))
"""
        child = run_child([str(self.states[namespace]), '-c', script, name, json.dumps(process)],
                               env=env, cwd=self.root, capture_output=True, text=True, timeout=timeout, check=True)
        return json.loads(child.stdout)

    def cleanup_attempt(self, namespace):
        """Recover only this fresh namespace/name, through bounded public commands."""
        attempt = self.attempts[namespace]
        name = attempt['name']
        end = time.monotonic() + 10

        def remaining():
            seconds = end - time.monotonic()
            if seconds <= 0:
                raise TimeoutError('owned launch cleanup exceeded 10 seconds')
            return seconds

        while True:
            record = self.inspect_server(namespace, name, timeout=remaining())['record']
            if record is None:
                break
            token = record['launch_token']
            if attempt['token'] is not None and token not in (None, attempt['token']):
                raise RuntimeError('launch token changed; cleanup ownership unresolved')
            attempt['token'] = attempt['token'] or token
            attempt['process'] = record['process'] or attempt['process']
            if (record.get('pending') or {}).get('action') == 'start':
                self.command(namespace, 'server', 'reconcile', name, expected=(0, 1, 4), command_timeout=remaining())
                time.sleep(min(.01, remaining()))
                continue
            if record['process']:
                try:
                    self.command(namespace, 'server', 'logs', name, command_timeout=remaining())
                except Exception as error:
                    print(f'Could not collect owned server log: {error}', file=sys.stderr)
                self.command(namespace, 'server', 'shutdown', name, '--abort', '--timeout', str(remaining()), command_timeout=remaining())
            elif token is not None:
                raise RuntimeError('launch has no verified process; cleanup unresolved')
            break
        while not self.inspect_server(namespace, name, attempt['process'], timeout=remaining())['exited']:
            time.sleep(min(.01, remaining()))
        with socket.socket() as sock:
            assert sock.connect_ex(('127.0.0.1', attempt['port'])) != 0, 'owned port still open'
        self.results['cleanup'].append(dict(namespace=namespace, name=name, absent=True,
                                            processes_exited=True, port_closed=True))
        self.servers.pop(namespace, None)
        del self.attempts[namespace]

    def pi_configuration(self):
        """Exercise only our generated installation against two real model IDs.

        Explicit previously qualified capacities make Pi's ordinary budgets fit;
        they do not promote defaults or qualify a full-context request.
        """
        runtime = self.args.pi_runtime.resolve()
        node_bin = runtime / 'node-v22.22.1-linux-x64/bin'
        pi_bin = runtime / 'node_modules/.bin'
        target = self.runtime_root / 'pi-agent'
        project = self.runtime_root / 'pi-project'
        target.mkdir(mode=0o700)
        project.mkdir(mode=0o700)
        original = '{"quietStartup":true}\n'
        (target / 'settings.json').write_text(original)
        environment = dict(os.environ, LOOM_SERVE_DIR=str(self.state_path('a')),
                           PI_CODING_AGENT_DIR=str(target), PYTHONNOUSERSITE='1', PI_OFFLINE='1')
        environment.pop('PYTHONPATH', None)
        environment.pop('PYTHONHOME', None)
        environment['PATH'] = os.pathsep.join((str(node_bin), str(pi_bin), str(self.states['a'].parent), environment.get('PATH', '')))
        assert subprocess.check_output([str(node_bin / 'node'), '--version'], env=environment, text=True).strip() == 'v22.22.1'
        assert digest(runtime / 'package-lock.json') == digest(self.args.source / 'build_tools/pi_runtime/package-lock.json')
        metadata = json.loads((runtime / 'node_modules/@earendil-works/pi-coding-agent/package.json').read_text())
        assert metadata['version'] == '1.3.0'
        self.results['pi_runtime'] = dict(node='22.22.1', package=metadata['name'], version=metadata['version'],
                                          lock_sha256=digest(runtime / 'package-lock.json'))
        self.states['c'] = self.states['a']
        self.namespace_roots['c'] = 'a'
        self.start_server('c', 'server-pi-sibling')
        for namespace, name in (('a', 'pi-qwen-a'), ('c', 'pi-qwen-b')):
            answer = self.command(namespace, 'instance', 'start', 'small-a', '--name', name,
                                  '--server', self.servers[namespace][0], '--pipeline', 'qwen-08b',
                                  '--context-capacity', '262144', '--pool-capacity', '262144',
                                  '--rows', '4', '--prefill-capacity', '128')
            assert answer['completed'] and answer['accepted'], answer
        configure_environment = {key: environment[key] for key in ('PATH', 'PI_CODING_AGENT_DIR', 'PI_OFFLINE')}
        configured = self.command('a', 'pi', 'configure', '--pi-dir', str(target), env_overrides=configure_environment)
        assert {item['id'] for item in configured['available']} == {'pi-qwen-a', 'pi-qwen-b'}, configured
        import hrx_solutions
        assets = Path(hrx_solutions.__file__).parent / 'pi_extension'
        for source in assets.iterdir():
            if source.is_file():
                assert source.read_bytes() == (target / 'extensions/loom-hem' / source.name).read_bytes()
        self.results['pi_responses'] = []
        for item in configured['available']:
            argv = [str(pi_bin / 'pi'), '--provider', item['provider'], '--model', item['id'],
                    '--print', '--mode', 'json', '--no-session', '--no-tools', '--thinking', 'off',
                    '--system-prompt', 'Reply briefly to the user.', 'Say hello in two words.']
            started = time.monotonic()
            response = run_child(argv, cwd=project, env=environment, capture_output=True, text=True, timeout=30)
            prefix = 'pi-' + item['id']
            (self.root / (prefix + '.stdout')).write_text(response.stdout)
            (self.root / (prefix + '.stderr')).write_text(response.stderr)
            evidence = dict(selected=item, exit=response.returncode, seconds=time.monotonic() - started)
            self.results['pi_responses'].append(evidence)
            self.save()
            assert response.returncode == 0, prefix
            events = [json.loads(line) for line in response.stdout.splitlines() if line.startswith('{')]
            messages = [event['message'] for event in events if event.get('type') == 'message_end'
                        and event.get('message', {}).get('role') == 'assistant']
            assert messages, prefix
            message = messages[-1]
            assert message.get('stopReason') == 'stop', message
            assert message.get('model') == item['id'] and message.get('provider') == item['provider'], message
            assert any(part.get('type') == 'text' and part.get('text', '').strip() for part in message.get('content', [])), message
            evidence['assistant'] = message
            self.save()
        repeated = self.command('a', 'pi', 'configure', '--pi-dir', str(target), env_overrides=configure_environment)
        assert repeated['changed'] is False, repeated
        assert (target / 'settings.json').read_text() == original
        self.command('a', 'instance', 'stop', 'pi-qwen-a', '--server', self.servers['a'][0])
        checked = self.command('a', 'pi', 'configure', '--check', '--pi-dir', str(target), env_overrides=configure_environment)
        assert [item['id'] for item in checked['available']] == ['pi-qwen-b'], checked
        self.command('c', 'instance', 'stop', 'pi-qwen-b', '--server', self.servers['c'][0])
        self.shutdown('c')
        self.results['pi_configuration'] = 'generated configuration passed two short real requests; full-context/default promotion not qualified'
        self.save()

    def run_ci08(self):
        """Required GPU CI profile; full two-model qualification remains explicit."""
        self.start_server('a', 'server-a')
        if self.args.acquire08:
            answer = self.command('a', 'model', 'download', 'qwen35-08b-q5-k-xl',
                                  '--id', 'small-a', '--pipeline', 'qwen-08b', command_timeout=600)
            assert answer['completed'] and answer['validation']['valid'], answer
        else:
            self.import_model('a', 'small-a', self.args.model08,
                              'Qwen3.5-0.8B-UD-Q5_K_XL.gguf', 'qwen-08b')
        self.start_instance('a', 'small-a', 'qwen-08b')
        self.query('a', 'small-a')
        self.command('a', 'instance', 'stop', 'small-a', '--server', 'server-a')
        assert self.command('a', 'instance', 'list', '--server', 'server-a')['items'] == []
        if getattr(self.args, 'pi_runtime', None):
            self.pi_configuration()
        self.shutdown('a')
        with sqlite3.connect(self.state_path('a') / 'control.sqlite3') as connection:
            assert connection.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'
            assert connection.execute('SELECT COUNT(*) FROM instances').fetchone()[0] == 0
        self.results['success'] = True

    def artifacts(self):
        """Publish only bounded diagnostic files and current SQLite snapshots."""
        target = self.root / 'diagnostics'
        target.mkdir(exist_ok=True)
        def redact(text):
            for key in ('HF_TOKEN', 'HUGGING_FACE_HUB_TOKEN', 'HUGGINGFACE_HUB_TOKEN'):
                if os.environ.get(key):
                    text = text.replace(os.environ[key], '[REDACTED]')
            return re.sub(r'hf_[A-Za-z0-9]+', '[REDACTED]', text)
        def publish(name, text):
            (target / name).write_bytes(redact(text).encode()[-65536:])
        for path in self.root.iterdir():
            if path.is_file() and path.suffix in ('.stdout', '.stderr', '.sse', '.json'):
                with path.open('rb') as stream:
                    stream.seek(max(0, path.stat().st_size - 65536))
                    data = stream.read(65536).decode(errors='replace')
                publish(path.name, data)
        for namespace in ('a', 'b'):
            state = self.state_path(namespace)
            database = state / 'control.sqlite3'
            if not database.exists():
                continue
            with sqlite3.connect(database) as source:
                assert source.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'
                output = target / (namespace + '-control.sqlite3')
                with sqlite3.connect(output) as destination:
                    source.backup(destination)
            if output.stat().st_size > 1048576:
                output.unlink()
                raise AssertionError('unexpected registry growth')
            # Only our namespace is inspected; no HF cache/model payload is copied.
            for index, log in enumerate(state.rglob('process.log*')):
                if index >= 16:
                    break
                if log.is_file():
                    with log.open('rb') as stream:
                        stream.seek(max(0, log.stat().st_size - 65536))
                        text = stream.read(65536).decode(errors='replace')
                    publish(f'{namespace}-{index}-process.log', text)

    def run(self):
        previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
        def cancelled(signum, frame):
            # Ignore subsequent termination signals until all owned cleanup ends.
            for sig in (signal.SIGINT, signal.SIGTERM):
                signal.signal(sig, signal.SIG_IGN)
            raise Cancelled(signum)
        for sig in previous:
            signal.signal(sig, cancelled)
        try:
            self._run()
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)

    def _run(self):
        try:
            self.setup()
            if getattr(self.args, 'profile', 'full') == 'ci08':
                self.run_ci08()
                return
            for namespace in ('a', 'b'):
                self.start_server(namespace, 'server-' + namespace)
                self.import_model(namespace, 'small-' + namespace, self.args.model08,
                                  'Qwen3.5-0.8B-UD-Q5_K_XL.gguf', 'qwen-08b')
                self.start_instance(namespace, 'small-' + namespace, 'qwen-08b')
                self.query(namespace, 'small-' + namespace)
            # A third native server shares A's registry but keeps an independent native inventory.
            self.states['c'] = self.states['a']
            self.namespace_roots['c'] = 'a'
            self.start_server('c', 'server-sibling')
            self.start_instance('c', 'small-a', 'qwen-08b')
            self.query('c', 'small-a')
            self.shutdown('c')
            self.query('a', 'small-a')
            for namespace, expected in [('a', 'small-a'), ('b', 'small-b')]:
                inventory = self.command(namespace, 'model', 'list')['items']
                assert len(inventory) == 1 and inventory[0]['id'] == expected, inventory
            self.command('a', 'instance', 'stop', 'small-a', '--server', 'server-a')
            assert self.command('a', 'instance', 'list', '--server', 'server-a')['items'] == []
            self.shutdown('a')
            self.query('b', 'small-b')
            abort_stream, abort_path = self.long_query('b', 'small-b')
            self.shutdown('b', abort=True)
            abort_stream.wait(timeout=10)
            self.results['abort_stream_exit'] = abort_stream.returncode
            self.results['abort_stream_bytes'] = abort_path.stat().st_size
            self.start_server('a', 'server-large')
            self.import_model('a', 'large', self.args.model27,
                              'Qwen3.8-27B-UD-Q5_K_XL.gguf', 'qwen')
            self.start_instance('a', 'large', 'qwen')
            self.query('a', 'large')
            self.command('a', 'instance', 'stop', 'large', '--server', 'server-large')
            self.start_instance('a', 'large', 'qwen')
            graceful_stream, graceful_path = self.long_query('a', 'large')
            pending = self.command('a', 'server', 'shutdown', 'server-large', '--timeout', '0.05', expected=3)
            assert pending['accepted'] is True and not pending['completed'], pending
            assert Path(f'/proc/{self.servers["a"][2]["pid"]}').exists()
            # Retry uses the ordinary 10s lifecycle budget after the deliberately short first wait.
            self.shutdown('a')
            graceful_stream.wait(timeout=5)
            assert graceful_stream.returncode == 0
            frames = [line[6:] for line in graceful_path.read_text().splitlines() if line.startswith('data: ')]
            assert frames[-1] == '[DONE]', frames
            events = [json.loads(frame) for frame in frames[:-1]]
            assert not any('error' in event for event in events), events
            assert any(choice.get('finish_reason') in ('stop', 'length') for event in events for choice in event.get('choices', []))
            self.results['graceful_stream_completed'] = True
            self.results['state_tables'] = {}
            for namespace in ('a', 'b'):
                self.results['state_tables'][namespace] = self.verify_empty_state(namespace)
            self.results['success'] = True
        finally:
            original_error = sys.exc_info()[1]
            if original_error is not None:
                self.results['error'] = repr(original_error)
            for sig in (signal.SIGINT, signal.SIGTERM):
                signal.signal(sig, signal.SIG_IGN)
            cleanup_error = getattr(original_error, 'cleanup_unresolved', None)
            for namespace in list(self.attempts):
                try:
                    self.cleanup_attempt(namespace)
                except Exception as error:
                    cleanup_error = error
                    self.results['success'] = False
                    self.results['cleanup'].append(dict(namespace=namespace, error=repr(error)))
                    print(f'Cleanup unresolved for {self.attempts[namespace]["name"]} in {self.root}: {error}', file=sys.stderr)
            for child in self.streams:
                if child.poll() is None:
                    try:
                        child.terminate()
                        child.wait(timeout=5)
                    except Exception as error:
                        cleanup_error = cleanup_error or error
                        self.results['cleanup'].append(dict(child=child.pid, error=repr(error)))
            self.results['success'] = bool(self.results['success'] and original_error is None and cleanup_error is None)
            self.save()
            write_cleanup('unresolved' if cleanup_error else 'complete', attempts=list(self.attempts), error=repr(cleanup_error) if cleanup_error else None)
            try:
                self.artifacts()
            except Exception as error:
                if original_error is None and cleanup_error is None:
                    raise
                print(f'Diagnostic collection failed: {error}', file=sys.stderr)
            if cleanup_error is not None and original_error is None:
                raise cleanup_error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--wheel', required=True, type=Path)
    parser.add_argument('--wheel-sha256', required=True)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--profile', choices=('ci08', 'full'), default='full')
    parser.add_argument('--pi-runtime', type=Path, help='CI-only integrity-pinned runtime staged by stage-pi-runtime.sh')
    parser.add_argument('--model08', type=Path)
    parser.add_argument('--model27', type=Path)
    parser.add_argument('--acquire08', action='store_true', help='Acquire exact pinned 0.8B recipe with Loom Control (CI only)')
    parser.add_argument('--native-provenance', default='Wheel bundled native payload; see wheel build provenance')
    args = parser.parse_args()
    if args.acquire08 and args.profile != 'ci08':
        parser.error('--acquire08 is restricted to --profile ci08')
    if not args.acquire08 and args.model08 is None:
        parser.error('--model08 is required without --acquire08')
    if args.profile == 'full' and args.model27 is None:
        parser.error('--model27 is required for the resource-qualified full profile')
    if args.run.exists() and any(args.run.iterdir()):
        parser.error('--run must be a new or empty external directory')
    args.run.mkdir(parents=True, exist_ok=True)
    try:
        Gate(args).run()
    except Cancelled as error:
        raise SystemExit(128 + error.signum)


if __name__ == '__main__':
    main()
