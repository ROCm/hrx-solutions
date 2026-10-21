#!/usr/bin/env bash
# Copyright 2026 The HRX Authors
# Licensed under the MIT License.
# SPDX-License-Identifier: MIT
# Check the UI integrations shipped by one isolated wheel installation.
set -euo pipefail
umask 077

if [[ $# != 4 || ! -x $1/bin/python || ! -d $2 || ! -d $3 || ! -d $4 ]]; then
  echo "Usage: $0 INSTALLED_VENV PI_RUNTIME OPENCODE_RUNTIME OMP_RUNTIME" >&2
  exit 2
fi

venv=$(realpath -- "$1")
pi_runtime=$(realpath -- "$2")
opencode_runtime=$(realpath -- "$3")
omp_runtime=$(realpath -- "$4")
scripts=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
package=$(
  env -u PYTHONPATH -u PYTHONHOME "$venv/bin/python" -c '
import hrx_solutions, sys
from pathlib import Path
package = Path(hrx_solutions.__file__).resolve().parent
package.relative_to(Path(sys.prefix).resolve())
print(package)
'
)

echo '[ui-integrations] Pi'
bash "$scripts/pi/check-pi-assets.sh" "$pi_runtime" "$package/pi_extension"
echo '[ui-integrations] OpenCode'
bash "$scripts/opencode/check-opencode-assets.sh" "$opencode_runtime" "$package/opencode_plugin"
echo '[ui-integrations] OMP'
bash "$scripts/omp/check-omp-assets.sh" "$omp_runtime" "$package/omp_extension"
echo '[ui-integrations] passed'
