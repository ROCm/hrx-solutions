#!/usr/bin/env bash
# Copyright 2026 The HRX Authors
# Licensed under the MIT License.
# SPDX-License-Identifier: MIT
# Build and repair the wheel in the pinned manylinux container.
# Run from a clean checkout with Docker available. Outputs: build/wheel/wheelhouse/.
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)

if [[ ${1:-} == --fast ]]; then
  shift
  exec env -u PYTHONPATH -u PYTHONHOME -u PYTHONUSERBASE -u LD_PRELOAD -u LD_AUDIT -u LD_LIBRARY_PATH \
    PYTHONNOUSERSITE=1 python3 -E -s "$repo/build_tools/wheel_tools/host_wheel.py" "$@"
fi

cache_args=(--env HRX_BAZEL_CACHE_ENABLED=0)
if [[ -n "${HRX_BAZEL_CACHE_DIR:-}" && ${HRX_BAZEL_CACHE_ENABLED:-1} != 0 ]]; then
  if result=$(cd -- "$repo" && python3 -m build_tools.ci_tools.wheel_cache prepare --directory "$HRX_BAZEL_CACHE_DIR") &&
      [[ $result == *'"enabled": true'* ]]; then
    cache_args=(--volume "$HRX_BAZEL_CACHE_DIR:/ci-cache" --env HRX_BAZEL_CACHE_ENABLED=1)
  else
    echo "Optional Bazel cache unavailable; building without it: ${result:-probe failed}" >&2
  fi
fi
container=quay.io/pypa/manylinux_2_28_x86_64@sha256:1e7b8460ad8fb3fb0fbe1cd818845159ac896b4fd17452d56e892c291f3b97a8

# Python owns the Docker CLI and exact container identity across cancellation.
exec python3 - "$container" --user "$(id -u):$(id -g)" \
  "${cache_args[@]}" --env HRX_WHEEL_CONTAINER="$container" \
  --env GITHUB_RUN_ID="${GITHUB_RUN_ID:-local}" --env GITHUB_RUN_ATTEMPT="${GITHUB_RUN_ATTEMPT:-1}" \
  --env USER="$(id -un)" \
  --volume "$repo:/workspace" --workdir /workspace \
  --env HRX_RELEASE_TYPE="${HRX_RELEASE_TYPE:-nightly}" \
  --env HRX_RUN_ID="${HRX_RUN_ID:-}" \
  --env HRX_ARTIFACT_SET="${HRX_ARTIFACT_SET:-core}" \
  "$container" \
  bash .github/scripts/build_python_wheel.sh <<'PYTHON'
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

owner = uuid.uuid4().hex
name = 'hrx-wheel-' + owner
label = 'org.hrx.local-ci-owner'
cancelled = None
child = None
cleanup = {'status': 'unresolved', 'container': name, 'owner': owner}


def publish(variable, value):
    if os.environ.get(variable):
        path = Path(os.environ[variable])
        temporary = path.with_name(path.name + '.tmp')
        temporary.write_text(json.dumps(value) + '\n')
        temporary.replace(path)


def interrupt(signum, frame):
    global cancelled
    if cancelled is None:
        cancelled = signum


for sig in (signal.SIGINT, signal.SIGTERM):
    signal.signal(sig, interrupt)
publish('HRX_LOCAL_CI_CONTAINER_STATE', cleanup)
code = 1
creation_settled = False
try:
    # Finish create before acting on cancellation: the daemon may have accepted
    # the request before the CLI returned its ID. No early-CID ownership race.
    child = subprocess.Popen(['docker', 'create', '--name', name, '--label', label + '=' + owner, *sys.argv[2:]], stdout=subprocess.PIPE, text=True)
    deadline = None
    while child.poll() is None:
        if cancelled:
            deadline = deadline or time.monotonic() + 30
            if time.monotonic() >= deadline:
                child.kill()
                child.wait(timeout=5)
                raise RuntimeError('Docker create did not settle; ownership unresolved')
        time.sleep(.05)
    identifier = child.communicate()[0].strip()
    code = child.returncode
    creation_settled = True
    if code == 0 and not cancelled:
        publish('HRX_LOCAL_CI_CONTAINER_STATE', dict(cleanup, container_id=identifier))
        child = subprocess.Popen(['docker', 'start', '--attach', name])
        while child.poll() is None and not cancelled:
            time.sleep(.05)
        if not cancelled:
            code = child.wait()
except Exception as error:
    cleanup['operation_error'] = str(error)
    print('Wheel build command failed: ' + str(error), file=sys.stderr)
    code = code or 1
finally:
    try:
        found = subprocess.run(['docker', 'inspect', '--format', '{{json .Config.Labels}}', name], capture_output=True, text=True, timeout=10)
        if found.returncode == 0:
            if json.loads(found.stdout).get(label) != owner:
                raise RuntimeError('container ownership label mismatch')
            retired = subprocess.run(['docker', 'rm', '--force', name], capture_output=True, text=True, timeout=15)
            if retired.returncode:
                raise RuntimeError('owned container removal failed: ' + retired.stderr[:512])
        elif 'no such' not in found.stderr.lower():
            raise RuntimeError('container absence could not be verified: ' + found.stderr[:512])
        verified = subprocess.run(['docker', 'inspect', name], capture_output=True, text=True, timeout=10)
        if verified.returncode == 0 or 'no such' not in verified.stderr.lower():
            raise RuntimeError('owned container absence not confirmed')
        if child is not None and child.poll() is None:
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.terminate()
                child.wait(timeout=5)
        if not creation_settled:
            raise RuntimeError('Docker create request did not settle; late ownership remains unresolved')
        cleanup.update(status='complete', child_exit=child.returncode if child else None)
    except Exception as error:
        cleanup['error'] = str(error)
        print('Owned build cleanup unresolved: ' + str(error), file=sys.stderr)
        if code == 0:
            code = 1
    if child is not None and child.poll() is None:
        child.terminate()
        try:
            child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait(timeout=5)
    publish('HRX_LOCAL_CI_CLEANUP_FILE', cleanup)
    publish('HRX_LOCAL_CI_CONTAINER_STATE', cleanup)
sys.exit(128 + cancelled if cancelled else code)
PYTHON
