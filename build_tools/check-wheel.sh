#!/usr/bin/env bash
# Copyright 2026 The HRX Authors
# Licensed under the MIT License.
# SPDX-License-Identifier: MIT
# Install a wheel in a new venv; native lifecycle is opt-in on an AMD GPU host.
# Usage: bash build_tools/check-wheel.sh WHEEL NEW_VENV_DIRECTORY [--native-lifecycle]
set -euo pipefail
umask 077
if [[ $# -lt 2 || $# -gt 3 || ! -f $1 || $1 != *.whl || -e $2 || (${3:-} != "" && ${3:-} != --native-lifecycle) ]]; then
  echo "Usage: $0 WHEEL NEW_VENV_DIRECTORY [--native-lifecycle] (venv must not exist)" >&2
  exit 2
fi
wheel=$(realpath -- "$1")
venv=$(realpath -m -- "$2")
scripts=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# Preserve the outside-checkout cwd while exposing only source build helpers.
run_helper() {
  "$venv/bin/python" -c 'import runpy, sys; sys.path.insert(0, sys.argv.pop(1)); runpy.run_module(sys.argv.pop(1), run_name="__main__", alter_sys=True)' "$(dirname -- "$scripts")" "$@"
}
# Isolate Python imports and all control/runtime state without changing HOME,
# XDG, Hugging Face credentials or the ordinary user's daemon.
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1
python3 -m venv "$venv"
runtime="$venv"
if [[ -n ${HRX_LOCAL_CI_RUNTIME_ROOT:-} ]]; then
  runtime=$(run_helper build_tools.ci_tools.local_ci_runtime "$HRX_LOCAL_CI_RUNTIME_ROOT" "$venv")
fi
export LOOM_SERVE_DIR="$runtime/loom-state"
mkdir "$venv/wheel-check"
cd "$venv/wheel-check"
"$venv/bin/python" -m pip install --disable-pip-version-check --no-index --no-deps "$wheel"
run_helper build_tools.validation.check_installed_wheel --wheel "$wheel" "${@:3}"
mkdir -m 700 "$runtime/cpu-fixtures"
TMPDIR="$runtime/cpu-fixtures" LOOM_SERVE_DIR="$runtime/installed-cpu-state" run_helper build_tools.validation.check_installed_cpu \
  --source-root "$(dirname -- "$scripts")" --evidence "$venv/wheel-check/installed-cpu-evidence.json"
clean_env=(env -u HSA_OVERRIDE_GFX_VERSION -u IREE_HAL_AMDGPU_LIBHSA_PATH \
  -u IREE_HAL_AMDGPU_LIBAQLPROFILE_PATH -u LD_LIBRARY_PATH)
"${clean_env[@]}" "$venv/bin/loom-serve" --help
"${clean_env[@]}" "$venv/bin/loom-control" --help
