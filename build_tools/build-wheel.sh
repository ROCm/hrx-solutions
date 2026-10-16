#!/usr/bin/env bash
# Copyright 2026 The HRX Authors
# Licensed under the MIT License.
# SPDX-License-Identifier: MIT
# Build and repair the wheel in the pinned manylinux container.
# Run from a clean checkout with Docker available. Outputs: build/wheel/wheelhouse/.
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)

cache_args=(--env HRX_BAZEL_CACHE_ENABLED=0)
if [[ -n "${HRX_BAZEL_CACHE_DIR:-}" && ${HRX_BAZEL_CACHE_ENABLED:-1} != 0 ]]; then
  if result=$(python3 "$repo/build_tools/wheel_cache.py" prepare --directory "$HRX_BAZEL_CACHE_DIR") &&
      [[ $result == *'"enabled": true'* ]]; then
    cache_args=(--volume "$HRX_BAZEL_CACHE_DIR:/ci-cache" --env HRX_BAZEL_CACHE_ENABLED=1)
  else
    echo "Optional Bazel cache unavailable; building without it: ${result:-probe failed}" >&2
  fi
fi
container=ghcr.io/rocm/therock_build_manylinux_x86_64@sha256:702a5133851e6d1daf1207d2c9fbb01c2667914a5b6dc5a01faeb3ce66ea6421

docker run --rm --user "$(id -u):$(id -g)" \
  "${cache_args[@]}" --env HRX_WHEEL_CONTAINER="$container" \
  --env GITHUB_RUN_ID="${GITHUB_RUN_ID:-local}" --env GITHUB_RUN_ATTEMPT="${GITHUB_RUN_ATTEMPT:-1}" \
  --env USER="$(id -un)" \
  --volume "$repo:/workspace" --workdir /workspace \
  --env HRX_RELEASE_TYPE="${HRX_RELEASE_TYPE:-nightly}" \
  --env HRX_RUN_ID="${HRX_RUN_ID:-}" \
  --env HRX_ARTIFACT_SET="${HRX_ARTIFACT_SET:-core}" \
  "$container" \
  bash .github/scripts/build_python_wheel.sh
