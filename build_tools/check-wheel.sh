#!/usr/bin/env bash
# Copyright 2026 The HRX Authors
# Licensed under the MIT License.
# SPDX-License-Identifier: MIT
# Install a wheel in a new venv; native lifecycle is opt-in on an AMD GPU host.
# Usage: bash build_tools/check-wheel.sh WHEEL NEW_VENV_DIRECTORY [--native-lifecycle]
set -euo pipefail
if [[ $# -lt 2 || $# -gt 3 || ! -f $1 || $1 != *.whl || -e $2 || (${3:-} != "" && ${3:-} != --native-lifecycle) ]]; then
  echo "Usage: $0 WHEEL NEW_VENV_DIRECTORY [--native-lifecycle] (venv must not exist)" >&2
  exit 2
fi
wheel=$(realpath -- "$1")
venv=$(realpath -m -- "$2")
scripts=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# Isolate Python imports and all control/runtime state without changing HOME,
# XDG, Hugging Face credentials or the ordinary user's daemon.
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1
export LOOM_SERVE_DIR="$venv/loom-state"
python3 -m venv "$venv"
mkdir "$venv/wheel-check"
cd "$venv/wheel-check"
"$venv/bin/python" -m pip install --disable-pip-version-check --no-index --no-deps "$wheel"
"$venv/bin/python" "$scripts/check_installed_wheel.py" "${@:3}"
clean_env=(env -u HSA_OVERRIDE_GFX_VERSION -u IREE_HAL_AMDGPU_LIBHSA_PATH \
  -u IREE_HAL_AMDGPU_LIBAQLPROFILE_PATH -u LD_LIBRARY_PATH)
"${clean_env[@]}" "$venv/bin/loom-serve" --help
"${clean_env[@]}" "$venv/bin/loom-serve-setup" --help
"${clean_env[@]}" "$venv/bin/loom-serve-start" --help
"${clean_env[@]}" "$venv/bin/loom-hem" --help
