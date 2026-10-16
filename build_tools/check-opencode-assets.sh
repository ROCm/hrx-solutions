#!/usr/bin/env bash
# Same bounded adapter contracts against source and repaired installed assets.
set -euo pipefail
[[ $# == 2 ]] || { echo "Usage: $0 STAGED_RUNTIME ASSET_DIRECTORY" >&2; exit 2; }
runtime=$(realpath -- "$1")
assets=$(realpath -- "$2")
scripts=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
"$runtime/node-v24.10.0-linux-x64/bin/node" --experimental-strip-types --no-warnings \
  "$scripts/../tests/opencode_adapter_test.mjs" "$assets"
