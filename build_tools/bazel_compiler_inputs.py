#!/usr/bin/env python3
"""Fingerprint unmanaged native toolchain inputs for the optional wheel cache.

Bazel's local_config_cc records absolute tool/include paths, not their contents.
The pinned container supplies system headers/tools; SDK-provided tools, their
loaded libraries, and implicit include trees need an explicit content identity.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

CONTRACT = "hrx-wheel-compiler-v1"
ENVIRONMENT_KEY = "HRX_WHEEL_COMPILER_INPUTS"


def _run(*command: str) -> str:
    return subprocess.run(command, check=True, text=True, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE).stdout


def _loaded_libraries(tool: Path) -> list[Path]:
    output = _run("ldd", str(tool))
    if "not found" in output:
        raise RuntimeError(f"Unresolved loaded library for {tool}")
    libraries = []
    for line in output.splitlines():
        # Both 'lib.so => /path/lib.so (address)' and '/lib/ld.so (address)'.
        match = re.search(r"(?:=>\s*)?(/\S+)\s+\(", line)
        if match:
            libraries.append(Path(match.group(1)))
    if not libraries:
        raise RuntimeError(f"No loaded-library inventory for {tool}")
    return libraries


def _files(path: Path, ancestors: frozenset[Path] = frozenset()) -> list[Path]:
    resolved = path.resolve(strict=True)
    if not path.is_dir():
        raise RuntimeError(f"Required compiler include tree missing: {path}")
    if resolved in ancestors:
        raise RuntimeError(f"Compiler resource directory symlink cycle: {path}")
    files = []
    for item in sorted(path.iterdir()):
        item.resolve(strict=True)  # Broken symlinks cannot silently omit inputs.
        if item.is_dir():
            files.extend(_files(item, ancestors | {resolved}))
        elif item.is_file():
            files.append(item)
        else:
            raise RuntimeError(f"Unsupported compiler resource: {item}")
    return files


def fingerprint(rocm: Path, container_identity: str) -> dict:
    rocm = rocm.resolve(strict=True)
    binary_dir = rocm / "lib/llvm/bin"
    tools = [binary_dir / name for name in ("clang", "clang++", "ld.lld")]
    inputs = set()
    for tool in tools:
        inputs.add(tool)
        inputs.update(_loaded_libraries(tool))
    # Clang can implicitly read driver configuration beside the executable.
    inputs.update(binary_dir.glob("*.cfg"))
    inputs.update((rocm / "bin").glob("*.cfg"))
    resource = Path(_run(str(tools[0]), "--print-resource-dir").strip())
    inputs.update(_files(resource))
    # rules_cc explicitly permits this implicit SDK C++ include directory.
    libcxx = rocm / "lib/llvm/include/c++/v1"
    if libcxx.exists():
        inputs.update(_files(libcxx))

    digest = hashlib.sha256()
    digest.update((CONTRACT + "\0" + container_identity + "\0").encode())
    records = []
    for path in sorted(inputs):
        resolved = path.resolve(strict=True)
        try:
            label = str(path.relative_to(rocm))
        except ValueError:
            label = str(path)
        content = hashlib.sha256()
        with resolved.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                content.update(chunk)
        record = {"path": label, "sha256": content.hexdigest()}
        digest.update((label + "\0" + record["sha256"] + "\0").encode())
        records.append(record)
    return {"contract": CONTRACT, "fingerprint": digest.hexdigest(),
            "container_identity": container_identity, "inputs": records}


def configure(path: Path, identity: str) -> None:
    lines = path.read_text().splitlines()
    prefix = f"{ENVIRONMENT_KEY}="
    lines = [line for line in lines if not any(
        line.startswith(f"common --{option}={prefix}")
        for option in ("action_env", "host_action_env"))]
    lines.extend(f"common --{option}={prefix}{identity}"
                 for option in ("action_env", "host_action_env"))
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rocm", required=True, type=Path)
    parser.add_argument("--container-identity", required=True,
                        help="Pinned image digest; also protects its system toolchain.")
    parser.add_argument("--bazelrc", type=Path)
    args = parser.parse_args()
    if not re.fullmatch(r"(?:\S+@)?sha256:[0-9a-f]{64}", args.container_identity):
        parser.error("--container-identity must be an immutable image digest")
    result = fingerprint(args.rocm, args.container_identity)
    if args.bazelrc:
        configure(args.bazelrc, result["fingerprint"])
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
