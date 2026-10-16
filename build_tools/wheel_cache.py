#!/usr/bin/env python3
"""Optional wheel action-cache preparation, provenance, and publication admission."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time

SCHEMA = 1
CONTRACT = "hrx-wheel-compiler-v1"
MAX_BYTES = 3 * 1024**3
BEGIN = "# BEGIN HRX WHEEL CACHE\n"
END = "# END HRX WHEEL CACHE\n"


def snapshot_root(directory: str, checkout: Path | None = None) -> Path:
    root = Path(directory)
    if not root.is_absolute() or root.is_symlink() or root.resolve() != root:
        raise ValueError("cache must be an absolute path without symlink components")
    checkout = (checkout or Path(__file__).resolve().parent.parent).resolve()
    if root == Path("/") or root == checkout or root in checkout.parents or checkout in root.parents:
        raise ValueError("cache must be dedicated storage outside the checkout")
    return root


def layout(root: Path) -> None:
    if any(p.name not in {"bazel-actions", "manifest.json"} for p in root.iterdir()):
        raise ValueError("unexpected files in dedicated snapshot directory")
    if any(p.is_symlink() for p in root.iterdir()):
        raise ValueError("snapshot entries must not be symlinks")


def manifest(root: Path) -> dict:
    if (root / "manifest.json").stat().st_size > 65536:
        raise ValueError("snapshot metadata exceeds size limit")
    data = json.loads((root / "manifest.json").read_text())
    required = {"source_commit", "hrx_pin", "bazel_version", "container_digest", "artifact_set", "sdk_identity", "compiler_fingerprint", "native_configuration", "build_run", "build_attempt"}
    if not isinstance(data, dict) or data.get("schema") != SCHEMA or data.get("compiler_contract") != CONTRACT:
        raise ValueError("unsupported snapshot metadata")
    if not required <= data.keys() or any(not isinstance(data[k], str) or not data[k] for k in required):
        raise ValueError("incomplete snapshot provenance")
    if data.get("complete") is not True:
        raise ValueError("snapshot build did not complete")
    return data


def prepare(directory: str, restore_state: str = "fresh") -> dict:
    root = snapshot_root(directory)
    root.mkdir(parents=True, exist_ok=True)
    layout(root)
    if restore_state == "failed":
        # Never mount partially restored state. Leave it for runner teardown.
        return {"enabled": False, "reason": "restore failed; snapshot ignored"}
    action_cache = root / "bazel-actions"
    empty_seed = not (root / "manifest.json").exists() and (
        not action_cache.exists() or (action_cache.is_dir() and not any(action_cache.iterdir()))
    )
    if not empty_seed:
        manifest(root)
        if not action_cache.is_dir():
            raise ValueError("missing action cache")
    elif restore_state == "restored":
        raise ValueError("restored snapshot has no manifest")
    (root / "bazel-actions").mkdir(exist_ok=True)
    # Validate restored descendants before Docker or Bazel can follow them.
    sizes(root)
    with tempfile.TemporaryFile(dir=root):
        pass
    with tempfile.TemporaryFile(dir=root / "bazel-actions"):
        pass
    return {"enabled": True, "reason": "usable snapshot"}


def write_manifest(root: Path, data: dict) -> None:
    # Atomic replacement; no temporary entry can be uploaded after this call.
    with tempfile.NamedTemporaryFile(mode="w", dir=root, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(data, stream, sort_keys=True, indent=2)
        stream.write("\n")
    try:
        temporary.replace(root / "manifest.json")
    finally:
        temporary.unlink(missing_ok=True)


def configure(rc: Path, enabled: bool) -> None:
    contents = rc.read_text()
    contents = re.sub(re.escape(BEGIN) + r".*?" + re.escape(END), "", contents, flags=re.S)
    if not enabled:
        contents = "".join(line for line in contents.splitlines(keepends=True) if not any(
            line.startswith(f"common --{option}=HRX_WHEEL_COMPILER_INPUTS=")
            for option in ("action_env", "host_action_env")
        ))
    if enabled:
        contents += BEGIN + "build --disk_cache=/ci-cache/bazel-actions\nbuild --remote_cache_async=false\n" + END
    rc.write_text(contents)


def sizes(root: Path) -> tuple[int, int]:
    stat = root.stat()
    apparent, allocated = stat.st_size, stat.st_blocks * 512
    for parent, dirs, files in os.walk(root, followlinks=False):
        for name in dirs + files:
            path = Path(parent) / name
            if path.is_symlink():
                raise ValueError("snapshot contains a symlink")
            stat = path.stat()
            apparent += stat.st_size
            allocated += stat.st_blocks * 512
    return apparent, allocated


def admit(directory: str, max_bytes: int = MAX_BYTES) -> dict:
    root = snapshot_root(directory)
    layout(root)
    data = manifest(root)
    for environment, field in (("GITHUB_RUN_ID", "build_run"), ("GITHUB_RUN_ATTEMPT", "build_attempt")):
        if expected := os.environ.get(environment):
            if data[field] != expected:
                return {"ready": False, "reason": "snapshot was not completed by this build attempt"}
    first = sizes(root)
    time.sleep(0.1)
    second = sizes(root)
    if first != second:
        return {"ready": False, "reason": "snapshot is still changing"}
    return {"ready": max(second) <= max_bytes, "reason": "within budget" if max(second) <= max_bytes else "snapshot exceeds budget", "apparent_bytes": second[0], "allocated_bytes": second[1]}


def provenance(root: Path, repo: Path) -> None:
    (root / "manifest.json").unlink(missing_ok=True)
    pin = re.search(r'commit = "([a-f0-9]+)"', (repo / "MODULE.bazel").read_text()).group(1)
    sdk_manifest = Path(os.environ["HRX_ROCM_ROOT"]) / ".hrx-rocm-artifacts.json"
    sdk = json.loads(sdk_manifest.read_text())["artifact_identity"]
    fingerprint = os.environ.get("HRX_COMPILER_FINGERPRINT", "")
    if not fingerprint:
        raise ValueError("compiler tracking identity unavailable; publication disabled")
    data = dict(schema=SCHEMA, compiler_contract=CONTRACT, complete=False,
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
        hrx_pin=pin, bazel_version=(repo / ".bazelversion").read_text().strip(),
        container_digest=os.environ["HRX_WHEEL_CONTAINER"], artifact_set=os.environ.get("HRX_ARTIFACT_SET", "core"),
        sdk_identity=sdk, compiler_fingerprint=fingerprint,
        native_configuration="--config=opt; " + (repo / ".bazelrc").read_text(), build_run=os.environ.get("GITHUB_RUN_ID", "local"), build_attempt=os.environ.get("GITHUB_RUN_ATTEMPT", "1"))
    write_manifest(root, data)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "configure", "invalidate", "begin", "finish", "admit"])
    parser.add_argument("--directory", default="/ci-cache")
    parser.add_argument("--restore-state", choices=["fresh", "restored", "failed"], default="fresh")
    parser.add_argument("--max-bytes", type=int, default=MAX_BYTES)
    parser.add_argument("--bazelrc", type=Path, default=Path(".bazelrc.configured"))
    parser.add_argument("--enabled", action="store_true")
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            result = prepare(args.directory, args.restore_state)
        elif args.command == "configure":
            configure(args.bazelrc, args.enabled)
            result = {"configured": True, "enabled": args.enabled}
        elif args.command == "invalidate":
            root = Path(args.directory)
            (root / "manifest.json").unlink(missing_ok=True)
            result = {"recorded": True}
        elif args.command == "begin":
            provenance(Path(args.directory), Path.cwd())
            result = {"recorded": True}
        elif args.command == "finish":
            root = Path(args.directory)
            data = json.loads((root / "manifest.json").read_text())
            data["complete"] = True
            write_manifest(root, data)
            result = {"recorded": True}
        else:
            result = admit(args.directory, args.max_bytes)
    except (OSError, ValueError, TypeError, KeyError, subprocess.SubprocessError) as error:
        result = {"enabled": False, "ready": False, "reason": str(error)}
    print(json.dumps(result))
    if output := os.environ.get("GITHUB_OUTPUT"):
        try:
            with open(output, "a") as stream:
                for key in ("enabled", "ready"):
                    if key in result:
                        stream.write(f"{key}={str(result[key]).lower()}\n")
        except OSError:
            pass


if __name__ == "__main__":
    main()
