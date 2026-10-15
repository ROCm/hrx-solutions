#!/usr/bin/env bash
# Copyright 2026 The HRX Authors
# Licensed under the MIT License.
# SPDX-License-Identifier: MIT
# Install only the diagnostic tool and its runtime in the Ubuntu 24.04 CI container.
set -euo pipefail

# Ubuntu's default ROCm runtime can silently enumerate only the CPU on gfx1151.
mkdir -p /etc/apt/keyrings
curl -fsSL https://repo.radeon.com/rocm/rocm.gpg.key -o /etc/apt/keyrings/rocm.asc
echo 'deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.asc] https://repo.radeon.com/rocm/apt/7.2.4 noble main' \
  > /etc/apt/sources.list.d/rocm.list
apt-get update
apt-get install -y --no-install-recommends 'rocminfo=1.0.0.70204-93~24.04'
