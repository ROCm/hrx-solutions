#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT/deps.lock"
HRX_DIR="${HRX_SOURCE_DIR:-$ROOT/.deps/hrx}"

if [[ ! -d "$HRX_DIR/.git" ]]; then
  mkdir -p "$(dirname "$HRX_DIR")"
  git init "$HRX_DIR"
  git -C "$HRX_DIR" remote add origin "$HRX_REPOSITORY"
fi

actual_repository="$(git -C "$HRX_DIR" remote get-url origin)"
[[ "$actual_repository" == "$HRX_REPOSITORY" ]] || {
  echo "Unexpected HRX source remote: $actual_repository" >&2
  exit 1
}
if [[ -n "$(git -C "$HRX_DIR" status --porcelain)" ]]; then
  echo "HRX source checkout has local changes: $HRX_DIR" >&2
  exit 1
fi

actual_commit="$(git -C "$HRX_DIR" rev-parse HEAD 2>/dev/null || true)"
if [[ "$actual_commit" != "$HRX_COMMIT" ]]; then
  git -C "$HRX_DIR" fetch --depth=1 origin "$HRX_COMMIT"
  git -C "$HRX_DIR" checkout --detach "$HRX_COMMIT"
fi
[[ "$(git -C "$HRX_DIR" rev-parse HEAD)" == "$HRX_COMMIT" ]] || {
  echo "HRX source does not match pinned commit $HRX_COMMIT" >&2
  exit 1
}
