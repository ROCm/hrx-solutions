#!/usr/bin/env bash
# CI/test-only dependencies; configure never installs or upgrades a client.
# Node SHA: https://nodejs.org/dist/v24.10.0/SHASUMS256.txt
set -euo pipefail
umask 077
[[ $# == 1 && ! -e $1 ]] || { echo "Usage: $0 NEW_EXTERNAL_RUNTIME_DIRECTORY" >&2; exit 2; }
scripts=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
mkdir -p -- "$1"
runtime=$(realpath -- "$1")
archive=node-v24.10.0-linux-x64.tar.gz
curl --fail --location --silent --show-error --max-time 120 \
  "https://nodejs.org/dist/v24.10.0/$archive" --output "$runtime/$archive"
printf '%s  %s\n' 2b03c5417ce0b1076780df00e01da373bead3b4b80d1c78c1ad10ee7b918d90c "$runtime/$archive" | sha256sum --check --status
tar -xzf "$runtime/$archive" -C "$runtime"
export PATH="$runtime/node-v24.10.0-linux-x64/bin:$PATH"
cp "$scripts/runtime/package.json" "$scripts/runtime/package-lock.json" "$runtime/"
cd -- "$runtime"
timeout --kill-after=10s 180s npm ci --ignore-scripts --no-audit --no-fund
# Select the exact qualified Linux x64 native payload without running postinstall
# scripts or allowing postinstall to acquire a fallback binary outside the lock.
mkdir bin
ln -s ../node_modules/@opencode/cli-linux-x64/bin/opencode bin/opencode
printf '%s  %s\n' 2df936a20cf74af5706020342164628f9a31978c50c2b0ad306b42b614fe245d bin/opencode | sha256sum --check --status
node -e 'if (process.version !== "v24.10.0") process.exit(1); console.log(process.version)'
test "$(bin/opencode --version)" = 'opencode v2.0.31'
