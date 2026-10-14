#!/usr/bin/env bash
# Copyright 2026 The HRX Authors
#
# Licensed under the MIT License.
# SPDX-License-Identifier: MIT

set -euo pipefail

: "${HRX_OUTPUT_DIR:?}"
: "${HRX_BUILD_PYTHON:?}"
: "${HRX_PYTHON:?}"
: "${HRX_SOURCE_DIR:?}"
: "${HRX_ROCM_ROOT:?}"

"${HRX_BUILD_PYTHON}" -m venv "${HRX_OUTPUT_DIR}/python"
"${HRX_PYTHON}" -m pip install --upgrade pip boto3 zstandard
"${HRX_PYTHON}" "${HRX_SOURCE_DIR}/build_tools/ci/ci_core_linux.py" fetch-rocm

append_path() {
  local path="$1"
  if [[ -n "${GITHUB_PATH:-}" ]]; then
    echo "${path}" >>"${GITHUB_PATH}"
  else
    export PATH="${path}:${PATH}"
  fi
}

export_env() {
  local name="$1"
  local value="$2"
  if [[ -n "${GITHUB_ENV:-}" ]]; then
    echo "${name}=${value}" >>"${GITHUB_ENV}"
  else
    export "${name}=${value}"
    echo "${name}=${value}"
  fi
}

append_path "${HRX_ROCM_ROOT}/lib/llvm/bin"
append_path "${HRX_ROCM_ROOT}/bin"
export_env "HRX_AUDITWHEEL_LD_LIBRARY_PATH" "${LD_LIBRARY_PATH:-}"
export_env "CC" "${HRX_ROCM_ROOT}/lib/llvm/bin/clang"
export_env "CXX" "${HRX_ROCM_ROOT}/lib/llvm/bin/clang++"
export_env "CMAKE_PREFIX_PATH" \
  "${HRX_ROCM_ROOT}${CMAKE_PREFIX_PATH:+:${CMAKE_PREFIX_PATH}}"
export_env "IREE_ROCM_PATH" "${HRX_ROCM_ROOT}"
export_env "IREE_HAL_AMDGPU_LIBHSA_PATH" \
  "${HRX_ROCM_ROOT}/lib/libhsa-runtime64.so.1"
export_env "LD_LIBRARY_PATH" \
  "${HRX_ROCM_ROOT}/lib:${HRX_ROCM_ROOT}/lib/rocm_sysdeps/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
