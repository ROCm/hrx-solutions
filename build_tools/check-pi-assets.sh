#!/usr/bin/env bash
# Focused tests of our copied extension, discovery and budget policy.
set -euo pipefail
[[ $# == 2 ]] || { echo "Usage: $0 STAGED_RUNTIME ASSET_DIRECTORY" >&2; exit 2; }
runtime=$(realpath -- "$1")
export LOOM_PI_ASSET_DIR=$(realpath -- "$2")
export LOOM_PI_TEST_SCRATCH="$runtime/scratch"
mkdir -p -- "$LOOM_PI_TEST_SCRATCH"
scripts=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export PATH="$runtime/node-v22.22.1-linux-x64/bin:$PATH"
for test in pi_extension_test.mjs pi_discovery_test.mjs pi_budgets_test.mjs; do
  cp "$scripts/$test" "$runtime/$test"
  timeout --kill-after=5s 60s node "$runtime/$test"
done
