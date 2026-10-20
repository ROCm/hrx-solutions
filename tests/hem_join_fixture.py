"""CPU-only public CLI/process/HTTP/SQLite join fixture.

The private package is a copy of the intended source checkout, not an installed
wheel qualification. Only the synthetic tokenizer fingerprint is substituted;
GGUF/profile validation, native bootstrap, ownership proof and CLI are real.
The ELF extends the component fixture with deterministic external barriers.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from binding.python import hem_process, hem_state
from binding.python.hem_common import Deadline, UncertainOutcome
from tests.gguf_fixtures import install_lifecycle_profile, lifecycle_profile, write_matching_gguf
from tests.hem_instances_test import NATIVE
from tests.hem_process_test import port, wait_exit


NATIVE_JOIN = NATIVE.replace('#include <unistd.h>', '''#include <unistd.h>
#include <limits.h>
static char control[PATH_MAX];
static void barrier(const char *name) {
 char path[PATH_MAX], seen[PATH_MAX];
 snprintf(path,sizeof(path),"%s/hold-%s",control,name);
 snprintf(seen,sizeof(seen),"%s/seen-%s",control,name);
 FILE *f=fopen(seen,"w"); if(f) { fputs("seen",f); fclose(f); }
 while(access(path,F_OK)==0) usleep(10000);
}
static void event(const char *request) {
 char path[PATH_MAX]; snprintf(path,sizeof(path),"%s/events",control);
 FILE *f=fopen(path,"a"); if(f) { const char *end=strstr(request,"\\r\\n");
   fprintf(f,"%.*s\\n",end?(int)(end-request):0,request); fclose(f); }
}''').replace('if(!port) return 5;', '''if(!port) return 5;
 for(int i=1;i<argc;i++) if(!strncmp(argv[i],"--device=",9)) snprintf(control,sizeof(control),"%s",argv[i]+9);
 barrier("listen"); int incarnation=16;''').replace('int code=200;', 'event(request); int code=200;').replace(
    'registered=1; active=0; open=0; code=202;',
    'registered=1; incarnation++; active=0; open=0; code=202; barrier("register");').replace(
    'registered=0; code=202;', 'registered=0; code=202; barrier("unregister");').replace(
    'if(!strncmp(request,"POST",4))', '''if(strstr(request,"POST /fixture/replace ")) { registered=1; active=1; open=1; incarnation++; }
   if(!strncmp(request,"POST",4))''').replace(
    '\\"incarnation\\":17', '\\"incarnation\\":%d').replace(
    'else if(registered && !active) strcpy(body,', 'else if(registered && !active) snprintf(body,sizeof(body),').replace(
    'worker_count\\":1}}");\n   else if(registered)',
    'worker_count\\":1}}",incarnation);\n   else if(registered)').replace(
    ',active?"true":"false",open?', ',incarnation,active?"true":"false",open?')


class JoinFixture:
    def __init__(self, root, binary):
        self.root = Path(root)
        self.state = self.root / 'state'
        self.package = self.root / 'hrx_solutions'
        checkout = Path(__file__).resolve().parents[1]
        self.package.mkdir()
        for source in (checkout / 'binding/python').glob('*.py'):
            shutil.copy2(source, self.package / source.name)
        shutil.copytree(checkout / 'loom-pipeline', self.package / 'loom-pipeline')
        install_lifecycle_profile(self.package / 'loom-pipeline')
        (self.package / '_native/bin').mkdir(parents=True)
        (self.package / '_native/lib').mkdir()
        shutil.copy2(binary, self.package / '_native/bin/loom-serve')
        (self.package / '_native/lib/libhsa-runtime64.so.1').write_bytes(b'CPU fixture only')
        self.environment = dict(os.environ, LOOM_SERVE_DIR=str(self.state), PYTHONNOUSERSITE='1')
        self.environment.pop('PYTHONPATH', None)
        self.environment.pop('PYTHONHOME', None)
        digest = hashlib.sha256(b'{"model":{"fixture":true}}').hexdigest()
        self.script = (
            'from hrx_solutions import hem_cli, hem_config; '
            'original=hem_config.load_profile; '
            f'hem_config.load_profile=lambda catalog: dict(original(catalog), tokenizer_canonical_sha256={digest!r}); '
            'raise SystemExit(hem_cli.main())')
        self.children = []
        self.owned = []
        self.controls = {}
        self.registry = hem_state.Registry(self.state)

    def spawn(self, *arguments, script=None):
        child = subprocess.Popen([sys.executable, '-c', script or self.script, *arguments, '--json'],
                                 cwd=self.root, env=self.environment, text=True,
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.children.append(child)
        return child

    def finish(self, child, expected=0):
        output, diagnostic = child.communicate(timeout=15)
        assert child.returncode == expected, (child.args, child.returncode, output, diagnostic)
        assert 'Traceback' not in diagnostic, diagnostic
        return json.loads(output), diagnostic

    def cli(self, *arguments, expected=0):
        return self.finish(self.spawn(*arguments), expected)[0]

    def control(self, name):
        if name not in self.controls:
            self.controls[name] = self.root / ('control-' + name)
            self.controls[name].mkdir()
        return self.controls[name]

    def server_command(self, name):
        return ('server', 'start', '--name', name, '--device', str(self.control(name)),
                '--listen', f'127.0.0.1:{port()}')

    def start_server(self, name):
        answer = self.cli(*self.server_command(name))
        self.remember(name)
        return answer

    def remember(self, name):
        record = self.registry.get_server(name)
        if record['process'] and record not in self.owned:
            self.owned.append(record)
        return record

    def acquire(self):
        weights, tokenizer = self.root / 'weights.gguf', self.root / 'tokenizer.json'
        write_matching_gguf(weights, contract=lifecycle_profile())
        tokenizer.write_text('{"model":{"fixture":true}}')
        return self.cli('model', 'import', 'tiny', '--gguf', str(weights), '--tokenizer', str(tokenizer),
                        '--alias', '--pipeline', 'qwen-08b')

    def start_instance(self, name):
        return self.cli('instance', 'start', 'tiny', '--name', 'chat', '--server', name,
                        '--pipeline', 'qwen-08b')

    def events(self, name):
        path = self.control(name) / 'events'
        return path.read_text().splitlines() if path.exists() else []

    def mutations(self, name):
        return [event for event in self.events(name) if event.startswith('POST')]

    def barrier(self, name, stage):
        (self.control(name) / ('hold-' + stage)).touch()
        (self.control(name) / ('seen-' + stage)).unlink(missing_ok=True)

    def wait_barrier(self, name, stage, child, seconds=15):
        marker = self.control(name) / ('seen-' + stage)
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if marker.exists():
                return
            if child.poll() is not None:
                try:
                    stdout, stderr = child.communicate(timeout=1)
                except subprocess.TimeoutExpired as error:
                    stdout, stderr = error.output, error.stderr
                raise AssertionError(
                    f'{stage} barrier was not reached; child exited {child.returncode}; '
                    f'stdout={stdout!r}; stderr={stderr!r}'
                )
            time.sleep(.01)
        raise AssertionError(f'{stage} barrier was not reached within {seconds}s')

    def release(self, name, stage):
        (self.control(name) / ('hold-' + stage)).unlink(missing_ok=True)

    @staticmethod
    def until(predicate, seconds=5):
        until = time.monotonic() + seconds
        while time.monotonic() < until:
            if predicate():
                return
            time.sleep(.01)
        raise AssertionError('fixture condition did not become true')

    def close(self):
        # Bootstrap barriers live at the fixture root; native HTTP barriers
        # live in each server's control directory. Release both before waiting.
        for control in (self.root, *self.controls.values()):
            for path in control.glob('hold-*'):
                path.unlink()
        for child in self.children:
            if child.poll() is None:
                child.kill()
            child.communicate(timeout=5)
        for row in self.registry.list_servers():
            if row['process']:
                self.owned.append(row)
        seen = set()
        for row in self.owned:
            identity = row['process']
            key = (identity['pid'], identity['start_ticks'])
            if key in seen:
                continue
            seen.add(key)
            deadline = Deadline.after(3)
            executables = [row['build']['executable']]
            if (row.get('pending') or {}).get('stage') == 'bootstrap':
                # A durable bootstrap can still be Python. Its original
                # recorded incarnation and executable are an authorized fixture
                # identity too; never substitute a fresh arbitrary procfs PID.
                executables.append(identity['executable'])
            while not hem_process._confirmed_exit(identity):
                deadline.remaining()
                for executable in executables:
                    try:
                        with hem_process.PinnedProcess(identity, executable, emit=lambda _: None) as pinned:
                            pinned.send(signal.SIGTERM, deadline)
                            pinned.wait(deadline)
                        break
                    except UncertainOutcome as error:
                        if hem_process._confirmed_exit(identity):
                            break
                        if not isinstance(error, hem_process._ExecutableMismatch):
                            raise
                        # The same bootstrap may exec between pinned attempts.
                        # Retry only the two pre-recorded executable identities.
                if not hem_process._confirmed_exit(identity):
                    time.sleep(min(.01, deadline.remaining()))
            assert hem_process._confirmed_exit(identity)
            if identity.get('logger'):
                wait_exit(identity['logger'])
