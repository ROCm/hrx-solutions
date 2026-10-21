#!/usr/bin/env bash
# CI/test-only qualified Bun route; configure never installs/upgrades OMP.
set -euo pipefail
umask 077
[[ $# == 1 && ! -e $1 ]] || { echo "Usage: $0 NEW_EXTERNAL_RUNTIME_DIRECTORY" >&2; exit 2; }
scripts=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
mkdir -p -- "$1"
runtime=$(realpath -- "$1")
cp -a "$scripts/runtime/client" "$scripts/runtime/runtime" "$runtime/"
cd -- "$runtime/runtime"
timeout --kill-after=10s 180s npm ci --ignore-scripts --no-audit --no-fund
# npm's Bun postinstall normally moves this exact native package into place.
# Select it explicitly instead, without executing dependency install scripts.
cp node_modules/@oven/bun-linux-x64/bin/bun node_modules/bun/bin/bun.exe
printf '%s  %s\n' 9fd36f87e4b90b07632b987a2e4ec81ca15a62c81bf983190cea6d715be2ad74 node_modules/bun/bin/bun.exe | sha256sum --check --status
bun="$runtime/runtime/node_modules/.bin/bun"
test "$("$bun" --version)" = '1.3.14'
cd -- "$runtime/client"
timeout --kill-after=10s 180s "$bun" install --frozen-lockfile --ignore-scripts
test "$("$bun" node_modules/@oh-my-pi/pi-coding-agent/src/cli.ts --version)" = 'omp/18.11.2'
