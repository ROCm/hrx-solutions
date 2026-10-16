#!/usr/bin/env bash
# Copyright 2026 The HRX Authors
# Licensed under the MIT License.
# SPDX-License-Identifier: MIT
# Usage: bash build_tools/model-smoke.sh WHEEL NEW_RUN_DIRECTORY [MODEL08_DIRECTORY]
# Without local fixtures, acquires the exact pinned 0.8B recipe. Requires gfx1151.
# HOME, HF credentials/cache, and unrelated services are preserved.
set -euo pipefail
umask 077
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
python3 -m venv "$run/venv"
"$run/venv/bin/python" -m pip install --disable-pip-version-check "$wheel" > "$run/install.log" 2>&1
cd -- "$run"
# The same helper drives local and CI qualification. Its fresh namespace owns
# server cleanup even if launch acceptance precedes a failed/timed-out CLI.
# 10s lifecycle budgets are unchanged; acquisition has a separate download cap.
pi_args=()
if [[ -n ${LOOM_PI_RUNTIME:-} ]]; then
  pi_args=(--pi-runtime "$(realpath -- "$LOOM_PI_RUNTIME")")
fi
"$run/venv/bin/python" "$scripts/hem_model_smoke.py" \
  --run "$run/lifecycle" --wheel "$wheel" --wheel-sha256 "$(sha256sum "$wheel" | cut -d ' ' -f 1)" \
  --source "$source_dir" --profile ci08 "${model_args[@]}" "${pi_args[@]}"
echo "0.8B empty-server, registration, query, retirement and shutdown passed: $run"
