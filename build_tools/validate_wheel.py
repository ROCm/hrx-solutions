#!/usr/bin/env python3
"""Validates package metadata and native libraries in the release wheel."""

from __future__ import annotations

import argparse
import email.parser
import hashlib
import zipfile
from collections import defaultdict
from pathlib import Path


try:
    from .versioning import read_package_version
except ImportError:
    from versioning import read_package_version


REPO_ROOT = Path(__file__).resolve().parents[1]
README_PATH = REPO_ROOT / "PYPI_README.md"
NATIVE_LIBRARY_PREFIX = "hrx_solutions/_native/lib/"
PACKAGE_PREFIX = "hrx_solutions/"
PIPELINE_PREFIX = PACKAGE_PREFIX + "loom-pipeline/llm/"
REQUIRED_RUNTIME_LIBRARIES = {
    NATIVE_LIBRARY_PREFIX + "libhsa-runtime64.so.1",
    NATIVE_LIBRARY_PREFIX + "libhsa-amd-aqlprofile64.so.1",
}
STABLE_RUNTIME_NAMES = {
    "libhsa-runtime64": "libhsa-runtime64.so.1",
    "libhsa-amd-aqlprofile64": "libhsa-amd-aqlprofile64.so.1",
}
ROCM_LIBRARY_PREFIXES = ("libamd", "libhsa", "libroc")


def validate_wheel(path: Path) -> tuple[int, int]:
    duplicates: dict[str, list[str]] = defaultdict(list)
    native_libraries: list[str] = []
    with zipfile.ZipFile(path) as wheel:
        names = set(wheel.namelist())
        metadata_paths = sorted(
            name for name in names if name.endswith(".dist-info/METADATA")
        )
        if len(metadata_paths) != 1:
            raise RuntimeError(
                f"wheel must contain one METADATA file, found {metadata_paths!r}"
            )
        metadata = email.parser.BytesParser().parsebytes(
            wheel.read(metadata_paths[0])
        )
        if metadata["Name"] != "hrx-solutions":
            raise RuntimeError("wheel package name must be hrx-solutions")
        if metadata["Version"] != read_package_version():
            raise RuntimeError("wheel version does not match version.json")
        if metadata["Description-Content-Type"] != "text/markdown":
            raise RuntimeError(
                "wheel long description must have content type text/markdown"
            )
        expected_description = README_PATH.read_text(encoding="utf-8")
        if metadata.get_payload() != expected_description:
            raise RuntimeError("wheel long description does not match PYPI_README.md")

        missing = sorted(REQUIRED_RUNTIME_LIBRARIES - names)
        if missing:
            raise RuntimeError(
                "wheel is missing runtime libraries: " + ", ".join(missing)
            )

        required_files = {
            PACKAGE_PREFIX + "_native/bin/loom-serve",
            PIPELINE_PREFIX + "LICENSE-APACHE-2.0",
            PIPELINE_PREFIX + "qwen/qwen_27b/sources.txt",
            PIPELINE_PREFIX + "qwen/qwen_27b/prepare.loom",
            PIPELINE_PREFIX + "qwen/qwen_27b/control.loom",
            PIPELINE_PREFIX + "qwen/qwen_27b/weights.loom",
            PIPELINE_PREFIX + "qwen/qwen_27b/compatibility.json",
            PACKAGE_PREFIX + "gguf.py",
            PACKAGE_PREFIX + "qwen_validation.py",
            PACKAGE_PREFIX + "_native/share/loom-serve/LICENSE",
            PACKAGE_PREFIX + "pipeline_validation.py",
            PACKAGE_PREFIX + "setup_cli.py",
            PACKAGE_PREFIX + "model_store.py",
        }
        missing = sorted(required_files - names)
        if missing:
            raise RuntimeError("wheel is missing setup assets: " + ", ".join(missing))

        obsolete_checkers = sorted(
            name for name in names
            if Path(name).name in {"loom-qwen-epoch-check", "qwen-epoch-check", "epoch_check"}
        )
        if obsolete_checkers:
            raise RuntimeError("wheel contains obsolete model execution checkers: " + ", ".join(obsolete_checkers))

        entry_points_paths = [
            name for name in names if name.endswith(".dist-info/entry_points.txt")
        ]
        if len(entry_points_paths) != 1:
            raise RuntimeError("wheel must contain one entry_points.txt")
        entry_points = wheel.read(entry_points_paths[0]).decode("utf-8")
        if "loom-serve-setup = hrx_solutions.setup_cli:main" not in entry_points:
            raise RuntimeError("wheel does not expose the loom-serve-setup command")

        manifest_path = REPO_ROOT / "loom-pipeline" / "llm" / "qwen" / "qwen_27b" / "sources.txt"
        expected_sources = []
        for raw_line in manifest_path.read_text(encoding="utf-8").splitlines():
            relative = raw_line.strip()
            if not relative or relative.startswith("#"):
                continue
            expected_sources.append((manifest_path.parent / relative).resolve())
        catalog_root = (REPO_ROOT / "loom-pipeline" / "llm").resolve()
        expected_members = {
            PIPELINE_PREFIX + path.relative_to(catalog_root).as_posix()
            for path in expected_sources
        }
        missing = sorted(expected_members - names)
        if missing:
            raise RuntimeError("wheel is missing Qwen manifest sources: " + ", ".join(missing))

        weight_suffixes = {".gguf", ".safetensors", ".pt", ".pth", ".ckpt"}
        bundled_weights = sorted(
            name for name in names if Path(name).suffix.lower() in weight_suffixes
        )
        if bundled_weights:
            raise RuntimeError("wheel must not contain model weights: " + ", ".join(bundled_weights))

        for info in wheel.infolist():
            if info.is_dir():
                continue
            name = info.filename
            basename = Path(name).name
            if name.startswith(NATIVE_LIBRARY_PREFIX) and ".so" in basename:
                native_libraries.append(name)
                digest = hashlib.sha256(wheel.read(info)).hexdigest()
                duplicates[digest].append(name)
            if ".libs/" in name and basename.startswith(ROCM_LIBRARY_PREFIXES):
                raise RuntimeError(
                    f"auditwheel grafted a duplicate ROCm library: {name}"
                )

    for prefix, allowed_name in STABLE_RUNTIME_NAMES.items():
        aliases = sorted(
            name
            for name in native_libraries
            if Path(name).name.startswith(prefix + ".so")
            and Path(name).name != allowed_name
        )
        if aliases:
            raise RuntimeError(
                f"wheel contains unstable {prefix} aliases: " + ", ".join(aliases)
            )

    duplicate_groups = [names for names in duplicates.values() if len(names) > 1]
    if duplicate_groups:
        formatted = "; ".join(", ".join(names) for names in duplicate_groups)
        raise RuntimeError(f"wheel contains duplicate native libraries: {formatted}")

    return len(native_libraries), path.stat().st_size


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    args = parser.parse_args()
    count, size = validate_wheel(args.wheel)
    print(f"Validated {args.wheel}: {count} native libraries, {size} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
