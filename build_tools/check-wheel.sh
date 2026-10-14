#!/usr/bin/env bash
# Copyright 2026 The HRX Authors
# Licensed under the MIT License.
# SPDX-License-Identifier: MIT
# Install a wheel in a new venv and check it on the host, without a GPU.
# Usage: bash build_tools/check-wheel.sh WHEEL NEW_VENV_DIRECTORY
set -euo pipefail
if [[ $# != 2 || ! -f $1 || $1 != *.whl || -e $2 ]]; then
  echo "Usage: $0 WHEEL NEW_VENV_DIRECTORY (venv must not exist)" >&2
  exit 2
fi
wheel=$(realpath -- "$1")
venv=$(realpath -m -- "$2")
scripts=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
python3 -m venv "$venv"
"$venv/bin/python" -m pip install --disable-pip-version-check --no-index --no-deps "$wheel"
"$venv/bin/python" "$scripts/check_installed_wheel.py"
clean_env=(env -u HSA_OVERRIDE_GFX_VERSION -u IREE_HAL_AMDGPU_LIBHSA_PATH \
  -u IREE_HAL_AMDGPU_LIBAQLPROFILE_PATH -u LD_LIBRARY_PATH)
"${clean_env[@]}" "$venv/bin/loom-serve" --help
"${clean_env[@]}" "$venv/bin/loom-serve-setup" --help
