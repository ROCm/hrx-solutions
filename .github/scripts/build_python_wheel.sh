#!/usr/bin/env bash
# Copyright 2026 The HRX Authors
# Licensed under the MIT License.
# SPDX-License-Identifier: MIT
# Build inside the manylinux image; invoked by build_tools/build-wheel.sh.
set -euo pipefail
export HRX_OUTPUT_DIR="$PWD/build/wheel"
export HRX_BUILD_PYTHON="$HRX_OUTPUT_DIR/build-venv/bin/python"
export HRX_PYTHON="$HRX_OUTPUT_DIR/python/bin/python"
export HRX_ROCM_ROOT="$HRX_OUTPUT_DIR/rocm-root"
export HRX_DOWNLOAD_CACHE_DIR="$HRX_OUTPUT_DIR/downloads"
export HRX_ROCM_ARTIFACT_VARIANT=release
export PIP_CACHE_DIR="$HRX_OUTPUT_DIR/caches/pip"
export BAZELISK_HOME="$HRX_OUTPUT_DIR/caches/bazelisk"
# Bazel requires its repository cache outside the source checkout.
export TEST_TMPDIR=/tmp/hrx-wheel-bazel
export BAZEL="$HRX_OUTPUT_DIR/tools/bazel"
mkdir -p "$(dirname -- "$BAZEL")" "$TEST_TMPDIR"
/opt/python/cp312-cp312/bin/python3 -m venv "$HRX_OUTPUT_DIR/build-venv"
"$HRX_BUILD_PYTHON" -m pip install build 'setuptools>=77' 'wheel>=0.45'
curl -fL https://github.com/bazelbuild/bazelisk/releases/download/v1.29.0/bazelisk-linux-amd64 -o "$BAZEL"
echo "5a408715e932c0250d28bd84555f12edbf70117de42f9181691c736eacc4a992  $BAZEL" | sha256sum --check
chmod +x "$BAZEL"

HRX_SOURCE_DIR=$("$HRX_BUILD_PYTHON" -c 'from build_tools.python_packaging import discover_bazel, discover_hrx_source; print(discover_hrx_source(discover_bazel()))')
export HRX_SOURCE_DIR
source .github/scripts/fetch_rocm_toolchain.sh
"$CC" --version
/usr/local/bin/auditwheel --version
: "${AUDITWHEEL_PLAT:?}"
test -d "$HRX_ROCM_ROOT/include"
test -f "$HRX_ROCM_ROOT/lib/libhsa-runtime64.so.1"
"$HRX_BUILD_PYTHON" build_tools/setup_python.py --rocm="$HRX_ROCM_ROOT" --bazel="$BAZEL"
cache_enabled=()
cache_recorded=0
if [[ ${HRX_BAZEL_CACHE_ENABLED:-0} == 1 ]]; then
  "$HRX_BUILD_PYTHON" build_tools/wheel_cache.py invalidate
  if compiler_json=$("$HRX_BUILD_PYTHON" build_tools/bazel_compiler_inputs.py \
      --rocm "$HRX_ROCM_ROOT" --container-identity "$HRX_WHEEL_CONTAINER" --bazelrc .bazelrc.configured) &&
      HRX_COMPILER_FINGERPRINT=$("$HRX_BUILD_PYTHON" -c 'import json,sys; print(json.load(sys.stdin)["fingerprint"])' <<<"$compiler_json"); then
    export HRX_COMPILER_FINGERPRINT
    cache_enabled=(--enabled)
  else
    echo "Compiler tracking unavailable; building without optional cache" >&2
    HRX_BAZEL_CACHE_ENABLED=0
  fi
fi
# Remove our previous block even when caching is disabled on this invocation.
if configuration=$("$HRX_BUILD_PYTHON" build_tools/wheel_cache.py configure "${cache_enabled[@]}") &&
    [[ $configuration == *'"configured": true'* ]]; then
  :
else
  echo "Optional cache configuration unavailable; overriding disk cache off" >&2
  HRX_BAZEL_CACHE_ENABLED=0
fi
export HRX_BAZEL_CACHE_ENABLED
export HRX_BAZEL_METRICS_DIR="$HRX_OUTPUT_DIR/metrics"
mkdir -p "$HRX_BAZEL_METRICS_DIR"
if [[ ${HRX_BAZEL_CACHE_ENABLED:-0} == 1 ]]; then
  if metadata=$("$HRX_BUILD_PYTHON" build_tools/wheel_cache.py begin) && [[ $metadata == *'"recorded": true'* ]]; then
    cache_recorded=1
  fi
fi
"$HRX_BUILD_PYTHON" -m build --wheel --outdir "$HRX_OUTPUT_DIR/dist"
"$HRX_BUILD_PYTHON" build_tools/validate_wheel.py "$HRX_OUTPUT_DIR/dist/"*.whl
env LD_LIBRARY_PATH="$HRX_AUDITWHEEL_LD_LIBRARY_PATH" \
  /usr/local/bin/auditwheel show "$HRX_OUTPUT_DIR/dist/"*.whl
env LD_LIBRARY_PATH="$HRX_AUDITWHEEL_LD_LIBRARY_PATH" \
  /usr/local/bin/auditwheel repair --plat "$AUDITWHEEL_PLAT" \
  --wheel-dir "$HRX_OUTPUT_DIR/wheelhouse" "$HRX_OUTPUT_DIR/dist/"*.whl
shopt -s nullglob
wheels=("$HRX_OUTPUT_DIR/wheelhouse/"*.whl)
test "${#wheels[@]}" -eq 1
[[ ${wheels[0]} == *manylinux* ]]
"$HRX_BUILD_PYTHON" build_tools/validate_wheel.py "${wheels[0]}"
/usr/local/bin/auditwheel show "${wheels[0]}"

# Synchronous disk-cache writes have finished when the native build returns.
# The host admission check runs after this container has exited.
if [[ $cache_recorded == 1 ]]; then
  "$HRX_BUILD_PYTHON" build_tools/wheel_cache.py finish
fi
