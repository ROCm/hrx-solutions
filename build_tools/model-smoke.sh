#!/usr/bin/env bash
# Copyright 2026 The HRX Authors
# Licensed under the MIT License.
# SPDX-License-Identifier: MIT
# Usage: bash build_tools/model-smoke.sh [--openwebui] WHEEL NEW_RUN_DIRECTORY [MODEL08_DIRECTORY]
# Without local fixtures, acquires the exact pinned 0.8B recipe. Requires gfx1151.
# HOME, HF credentials/cache, and unrelated services are preserved.
set -euo pipefail
umask 077
profile=ci08
if [[ ${1:-} == --openwebui ]]; then
  profile=openwebui
  shift
  if [[ $# != 3 ]]; then
    echo "Open WebUI qualification requires an explicit immutable MODEL08_DIRECTORY" >&2
    exit 2
  fi
fi
if [[ $# -lt 2 || $# -gt 3 || ! -f $1 || $1 != *.whl ]]; then
  echo "Usage: $0 WHEEL NEW_RUN_DIRECTORY [MODEL08_DIRECTORY]" >&2
  exit 2
fi
wheel=$(realpath -- "$1")
if [[ -e $2 && ( ! -d $2 || -n $(ls -A -- "$2") ) ]]; then
  echo "Run directory must be new or empty: $2" >&2
  exit 2
fi
scripts=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source_dir=$(dirname -- "$scripts")
model_args=(--acquire08)
if [[ $# == 3 ]]; then
  model_args=(--model08 "$(realpath -- "$3")")
fi
mkdir -p -- "$2"
run=$(realpath -- "$2")
chmod 700 -- "$run"
if [[ $run == "$source_dir" || $run == "$source_dir/"* ]]; then
  echo "Run directory must be outside the source checkout: $run" >&2
  exit 2
fi
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1
export LOOM_SERVE_DIR="$run/bootstrap-state"
# Replace the shell with a signal-aware supervisor so cancellation reaches a
# synchronous setup child or the helper's launch-token-aware finally path.
exec python3 - "$wheel" "$run" "$scripts" "$source_dir" "$profile" "${model_args[@]}" <<'PYTHON'
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
        helper = 'integrations/openwebui/openwebui_configuration_smoke.py' if profile == 'openwebui' else 'validation/hem_model_smoke.py'
        qualification = [] if profile == 'openwebui' else ['--profile', 'ci08']
        code = invoke([str(run / 'venv/bin/python'), str(Path(scripts) / helper),
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
PYTHON
