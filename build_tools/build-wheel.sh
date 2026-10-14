#!/usr/bin/env bash
# Copyright 2026 The HRX Authors
# Licensed under the MIT License.
# SPDX-License-Identifier: MIT
# Build and repair the wheel in the pinned manylinux container.
# Run from a clean checkout with Docker available. Outputs: build/wheel/wheelhouse/.
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)

docker run --rm --user "$(id -u):$(id -g)" \
  --env USER="$(id -un)" \
  --volume "$repo:/workspace" --workdir /workspace \
  --env HRX_RELEASE_TYPE="${HRX_RELEASE_TYPE:-nightly}" \
  --env HRX_RUN_ID="${HRX_RUN_ID:-}" \
  --env HRX_ARTIFACT_SET="${HRX_ARTIFACT_SET:-core}" \
  ghcr.io/rocm/therock_build_manylinux_x86_64@sha256:702a5133851e6d1daf1207d2c9fbb01c2667914a5b6dc5a01faeb3ce66ea6421 \
  bash .github/scripts/build_python_wheel.sh
