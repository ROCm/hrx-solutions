#!/usr/bin/env bash
# CI/test dependency acquisition only; user configure never runs npm or downloads.
# Node archive SHA256: https://nodejs.org/dist/v22.22.1/SHASUMS256.txt
set -euo pipefail
if [[ $# != 1 || -e $1 ]]; then
  echo "Usage: $0 NEW_EXTERNAL_RUNTIME_DIRECTORY" >&2; exit 2
fi
scripts=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
mkdir -p -- "$1"
runtime=$(realpath -- "$1")
archive=node-v22.22.1-linux-x64.tar.gz
curl --fail --location --silent --show-error --max-time 120 \
  "https://nodejs.org/dist/v22.22.1/$archive" --output "$runtime/$archive"
printf '%s  %s\n' 07c8aafa60644fb81adefa1ee7da860eb1920851ffdc9a37020ab0be47fbc10e "$runtime/$archive" | sha256sum --check --status
tar -xzf "$runtime/$archive" -C "$runtime"
export PATH="$runtime/node-v22.22.1-linux-x64/bin:$PATH"
cp "$scripts/pi_runtime/package.json" "$scripts/pi_runtime/package-lock.json" "$runtime/"
cd -- "$runtime"
# Lock includes integrity for every transitive package; no install scripts run.
timeout --kill-after=10s 180s npm ci --ignore-scripts --no-audit --no-fund
node -e 'if (process.version !== "v22.22.1") process.exit(1); console.log(process.version)'
node -e 'const p=require("./node_modules/@earendil-works/pi-coding-agent/package.json"); if(p.version!=="1.3.0")process.exit(1); console.log(p.name,p.version)'
