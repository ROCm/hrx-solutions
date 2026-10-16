#!/usr/bin/env bash
# Copyright 2026 The HRX Authors
# Licensed under the MIT License.
# SPDX-License-Identifier: MIT
set -euo pipefail
scripts=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1
exec python3 "$scripts/local_ci.py" "$@"
