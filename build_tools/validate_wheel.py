#!/usr/bin/env python3
"""Validates package metadata and native libraries in the release wheel."""

from __future__ import annotations

import argparse
import configparser
import email.parser
import hashlib
import json
import zipfile
from collections import defaultdict
from pathlib import Path


try:
    from .versioning import read_package_version
    from . import loom_catalog
except ImportError:
    from versioning import read_package_version
    import loom_catalog


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
HEM_MODULES = (
    "hem_arguments", "hem_bootstrap", "hem_cli", "hem_common", "hem_config",
    "hem_display", "hem_help", "hem_http", "hem_instance_cli", "hem_instances", "hem_render",
    "hem_model_cli", "hem_models", "hem_pipeline_cli", "hem_process",
    "hem_server_cli", "hem_shutdown", "hem_state",
)
OBSOLETE_MODULES = (
    "setup_cli", "serve_start", "connection_manager", "server_manager",
    "remote_registry", "state_store",
)
OBSOLETE_ENTRY_POINTS = {"loom-hem", "loom-serve-setup", "loom-serve-start"}
REQUIRED_ENTRY_POINTS = {
    "loom-control": "hrx_solutions.hem_cli:main",
    "loom-serve": "hrx_solutions.loom_serve:main",
}


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

        obsolete_modules = sorted(
            PACKAGE_PREFIX + name + ".py" for name in OBSOLETE_MODULES
            if PACKAGE_PREFIX + name + ".py" in names
        )
        if obsolete_modules:
            raise RuntimeError("wheel contains retired manager modules: " + ", ".join(obsolete_modules))

        required_files = {
            PACKAGE_PREFIX + "_native/bin/loom-serve",
            PIPELINE_PREFIX + "LICENSE-APACHE-2.0",
            PACKAGE_PREFIX + "gguf.py",
            PACKAGE_PREFIX + "qwen_validation.py",
            PACKAGE_PREFIX + "_native/share/loom-serve/LICENSE",
            PACKAGE_PREFIX + "pipeline_validation.py",
            PACKAGE_PREFIX + "model_store.py",
            PACKAGE_PREFIX + "runtime_paths.py",
            PACKAGE_PREFIX + "model_download.py",
        }
        required_files.update(PACKAGE_PREFIX + name + ".py" for name in HEM_MODULES)
        required_files.update(target.split(":")[0].replace(".", "/") + ".py"
                              for target in REQUIRED_ENTRY_POINTS.values())
        for architecture in ("qwen_27b", "qwen_08b"):
            required_files.update(
                PIPELINE_PREFIX + f"qwen/{architecture}/{filename}"
                for filename in ("pipeline.loombc", "pipeline.json", "compatibility.json")
            )
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
        parser = configparser.ConfigParser()
        parser.read_string(entry_points)
        retired_commands = sorted(OBSOLETE_ENTRY_POINTS & set(parser.options("console_scripts"))) if parser.has_section("console_scripts") else []
        if retired_commands:
            raise RuntimeError("wheel exposes retired commands: " + ", ".join(retired_commands))
        for command, target in REQUIRED_ENTRY_POINTS.items():
            if parser.get("console_scripts", command, fallback=None) != target:
                raise RuntimeError(f"wheel does not expose the {command} command")

        if any(name.endswith(".loom") for name in names):
            raise RuntimeError("wheel must not contain executable .loom sources")
        inventory = REPO_ROOT / "loom-pipeline/catalog_sources.bzl"
        loom_catalog.validate_catalog(inventory)
        variants = loom_catalog.variants(inventory)
        build_name = PIPELINE_PREFIX + "catalog-build.json"
        if build_name not in names:
            raise RuntimeError("wheel is missing catalog build hashes")
        build = json.loads(wheel.read(build_name))
        if build.get("format") != 2 or set(build.get("variants", {})) != set(variants):
            raise RuntimeError("wheel catalog build inventory does not match authored variants")
        allowed = {build_name, PIPELINE_PREFIX + "LICENSE-APACHE-2.0"}
        root = inventory.parent / "llm"
        if wheel.read(PIPELINE_PREFIX + "LICENSE-APACHE-2.0") != (root / "LICENSE-APACHE-2.0").read_bytes():
            raise RuntimeError("wheel catalog license does not match authored license")
        for architecture, variant in variants.items():
            relative = f"qwen/{architecture}/"
            name = PIPELINE_PREFIX + variant["output"]
            content = wheel.read(name)
            digest = hashlib.sha256(content).hexdigest()
            expected = {
                "identifier": variant["identifier"],
                "sources": {source: hashlib.sha256((root / source).read_bytes()).hexdigest()
                            for source in variant["sources"]},
                "bytecode_sha256": digest,
            }
            if not content.startswith(b"LOOM") or build["variants"][architecture] != expected:
                raise RuntimeError(f"wheel catalog source/bytecode hash mismatch: {name}")
            descriptor = PIPELINE_PREFIX + relative + "pipeline.json"
            if json.loads(wheel.read(descriptor)) != {
                    "format": 1, "variant": variant["identifier"],
                    "module": "pipeline.loombc", "sha256": digest}:
                raise RuntimeError(f"wheel catalog descriptor mismatch: {descriptor}")
            profile = PIPELINE_PREFIX + relative + "compatibility.json"
            if wheel.read(profile) != (root / relative / "compatibility.json").read_bytes():
                raise RuntimeError(f"wheel compatibility profile mismatch: {profile}")
            allowed.update({name, descriptor, profile})
        actual = {name for name in names if name.startswith(PIPELINE_PREFIX) and not name.endswith("/")}
        if actual != allowed:
            raise RuntimeError(f"wheel catalog file inventory mismatch: extra={sorted(actual-allowed)}, missing={sorted(allowed-actual)}")

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
