#!/usr/bin/env bash
# Same bounded adapter/public-host contracts for source and installed assets.
set -euo pipefail
[[ $# == 2 ]] || { echo "Usage: $0 STAGED_RUNTIME ASSET_DIRECTORY" >&2; exit 2; }
runtime=$(realpath -- "$1")
assets=$(realpath -- "$2")
scripts=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export LOOM_OMP_TEST_PACKAGE="$runtime/client/node_modules/@oh-my-pi/pi-coding-agent"
bun="$runtime/runtime/node_modules/.bin/bun"
timeout --kill-after=5s 30s "$bun" "$scripts/../../../tests/omp_adapter_test.mjs" "$assets"
timeout --kill-after=5s 30s "$bun" "$scripts/../../../tests/omp_loader_test.mjs" "$assets"
