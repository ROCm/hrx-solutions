#!/usr/bin/env python3
"""Private signal-aware supervisor for the model-smoke shell entrypoint."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

wheel, run, scripts, source, profile, *model = sys.argv[1:]
run = Path(run)
child = None
cancelled = None
unresolved = None
helper_started = False

def publish(status, **details):
    if os.environ.get('HRX_LOCAL_CI_CLEANUP_FILE'):
        path = Path(os.environ['HRX_LOCAL_CI_CLEANUP_FILE'])
        temporary = path.with_name(path.name + '.tmp')
        temporary.write_text(json.dumps(dict(status=status, **details)) + '\n')
        temporary.replace(path)


def interrupt(signum, frame):
    global cancelled
    if cancelled is None:
        cancelled = signum


for sig in (signal.SIGINT, signal.SIGTERM):
    signal.signal(sig, interrupt)


def invoke(argv, *, helper=False, **kwargs):
    global child, unresolved, helper_started
    if cancelled:
        return 128 + cancelled
    child = subprocess.Popen(argv, start_new_session=True, **kwargs)
    helper_started = helper_started or helper
    deadline = None
    while child.poll() is None:
        if cancelled and deadline is None:
            if helper:
                child.send_signal(cancelled)
            else:
                os.killpg(child.pid, signal.SIGTERM)
            deadline = time.monotonic() + (40 if helper else 10)
        if deadline and time.monotonic() >= deadline:
            unresolved = 'owned helper did not settle' if helper else 'setup child did not settle'
            # Native ownership must remain with the cooperative helper; never
            # kill its group or declare its services absent on a CLI exit.
            if not helper:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=5)
                unresolved = None
            return 128 + cancelled
        time.sleep(.05)
    result = child.wait()
    if cancelled and not helper:
        deadline = time.monotonic() + 5
        while True:
            live = False
            for path in Path('/proc').glob('[0-9]*/stat'):
                try:
                    fields = path.read_text().rsplit(')', 1)[1].split()
                    live = live or (int(fields[2]) == child.pid and fields[0] != 'Z')
                except (FileNotFoundError, ProcessLookupError):
                    pass
            if not live:
                break
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            if time.monotonic() >= deadline:
                unresolved = 'owned setup child session remains live'
                break
            time.sleep(.01)
    return result


code = 1
try:
    code = invoke(['python3', '-m', 'venv', str(run / 'venv')])
    if code == 0:
        with (run / 'install.log').open('w') as log:
            code = invoke([str(run / 'venv/bin/python'), '-m', 'pip', 'install', '--disable-pip-version-check', wheel], stdout=log, stderr=subprocess.STDOUT)
    if code == 0 and profile == 'openwebui':
        sys.path.insert(0, source)
        from build_tools.ci_tools.local_ci_runtime import create
        from build_tools.integrations.openwebui.openwebui_configuration_smoke import IMAGE
        if not os.environ.get('HRX_LOCAL_CI_RUNTIME_ROOT'):
            owner = create(run)
            (run / 'runtime-owner.json').write_text(json.dumps(owner) + '\n')
            os.environ['HRX_LOCAL_CI_RUNTIME_ROOT'] = str(Path(owner['root']) / 'openwebui')
        with (run / 'webui-image.log').open('w') as log:
            code = invoke(['docker', 'image', 'inspect', IMAGE], stdout=log, stderr=subprocess.STDOUT)
            if code and not cancelled:
                code = invoke(['docker', 'pull', IMAGE], stdout=log, stderr=subprocess.STDOUT)
    if code == 0:
        os.chdir(run)
        import hashlib
        sha = hashlib.sha256(Path(wheel).read_bytes()).hexdigest()
        helper = 'build_tools.integrations.openwebui.openwebui_configuration_smoke' if profile == 'openwebui' else 'build_tools.validation.hem_model_smoke'
        qualification = [] if profile == 'openwebui' else ['--profile', 'ci08']
        code = invoke([str(run / 'venv/bin/python'), '-c',
                       'import runpy, sys; sys.path.insert(0, sys.argv.pop(1)); runpy.run_module(sys.argv.pop(1), run_name="__main__", alter_sys=True)', source, helper,
                       '--run', str(run / 'lifecycle'), '--wheel', wheel, '--wheel-sha256', sha,
                       '--source', source, *model, *qualification], helper=True)
finally:
    if unresolved:
        publish('unresolved', error=unresolved, child_pid=child.pid if child else None)
    elif not helper_started:
        publish('complete', detail='setup children reaped; no native launch attempted')
    # The helper publishes only after checking native/logger identities and port.
if code == 0:
    print(profile + ' real-model configuration/lifecycle qualification passed: ' + str(run))
sys.exit(128 + cancelled if cancelled else code)
