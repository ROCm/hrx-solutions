#!/usr/bin/env bash
# Copyright 2026 The HRX Authors
# Licensed under the MIT License.
# SPDX-License-Identifier: MIT
# Usage: bash build_tools/model-smoke.sh WHEEL NEW_RUN_DIRECTORY
# Requires Linux/gfx1151 GPU access, Python 3 with venv, curl, jq, and CA certificates.
# The caller owns the run directory. HOME, XDG settings and HF caching are unchanged.
set -euo pipefail

if [[ $# != 2 || ! -f $1 || $1 != *.whl ]]; then
  echo "Usage: $0 WHEEL NEW_RUN_DIRECTORY" >&2
  exit 2
fi
wheel=$(realpath -- "$1")
if [[ -e $2 ]]; then
  if [[ ! -d $2 || -n $(ls -A -- "$2") ]]; then
    echo "Run directory must be new or empty: $2" >&2
    exit 2
  fi
fi
mkdir -p -- "$2"
run=$(realpath -- "$2")
scripts=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export LOOM_SERVE_DIR="$run/loom"
server_id=

cleanup_after_failure() {
  if [[ -n $server_id ]]; then
    if ! loom-serve-setup servers stop "$server_id" --json > "$run/stop.json" 2> "$run/stop.log"; then
      echo "Cleanup could not stop the server; see $run/stop.log" >&2
    fi
  fi
}
trap cleanup_after_failure EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

python3 -m venv "$run/venv"
source "$run/venv/bin/activate"
cd -- "$run"
python -m pip install --disable-pip-version-check "$wheel" 'huggingface_hub[cli]==0.34.4' \
  2>&1 | tee "$run/install.log"

# The wheel's recipe owns pinned sources, cache aliases and model validation.
loom-serve-setup models download qwen35-08b-q5-k-xl --id ci-qwen08 \
  2>&1 | tee "$run/setup.log"
echo "Starting ci-qwen08 (readiness timeout: 180s)..."
if loom-serve-start ci-qwen08 --ready-timeout 180 --json \
    > "$run/start.json" 2> "$run/start.log"; then
  echo "Server ready. Sending test query..."
else
  start_status=$?
  cat "$run/start.log" >&2
  for log in "$run/loom/logs/"*.log; do
    if [[ -f $log ]]; then
      tail -n 100 "$log" >&2
    fi
  done
  exit "$start_status"
fi
server_id=$(jq -er '.server_id' "$run/start.json")
endpoint=$(jq -er '.endpoint' "$run/start.json")

curl --noproxy '*' --fail --silent --show-error --max-time 120 \
  -H 'Content-Type: application/json' \
  --data-binary '{"model":"ci-qwen08","messages":[{"role":"user","content":"Reply with exactly Hello! and nothing else."}],"temperature":0,"max_tokens":32,"stream":true}' \
  "$endpoint/chat/completions" > "$run/response.sse"
python "$scripts/smoke/assert_response.py" "$run/response.sse"

# Shutdown is a normal test step. Retire the fallback so it is never retried.
trap - EXIT
loom-serve-setup servers stop "$server_id" --json \
  > "$run/stop.json" 2> "$run/stop.log"
echo "0.8B setup, query and shutdown passed. Results: $run"
