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
exec python3 "$scripts/validation/model_smoke.py" "$wheel" "$run" "$scripts" "$source_dir" "$profile" "${model_args[@]}" </dev/null
