"""Executes the Loom Serve binary shipped in the wheel."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, NoReturn, Optional


_LIBHSA_ENV = "IREE_HAL_AMDGPU_LIBHSA_PATH"
_LIBAQLPROFILE_ENV = "IREE_HAL_AMDGPU_LIBAQLPROFILE_PATH"


def _required_library(directory: Path, name: str) -> Path:
    candidate = directory / name
    if candidate.is_file():
        return candidate
    raise RuntimeError(f"HRX wheel is missing a required runtime library: {name}")


def native_environment(
    environ: Optional[Dict[str, str]] = None,
    *, native_root: Optional[Path] = None,
) -> Dict[str, str]:
    """Return a bundled environment, optionally pinned to a retained closure.

    Explicit retained roots replace library overrides/search paths: their payload
    must remain usable after the installed package is replaced. Ordinary direct
    launcher calls retain the existing override behavior.
    """
    result = dict(os.environ if environ is None else environ)
    retained = native_root is not None
    native_root = Path(native_root) if retained else Path(__file__).resolve().parent / "_native"
    if retained:
        for key in (_LIBHSA_ENV, _LIBAQLPROFILE_ENV, "LD_LIBRARY_PATH", "LD_PRELOAD"):
            result.pop(key, None)
    library_dir = native_root / "lib"

    if _LIBHSA_ENV not in result:
        result[_LIBHSA_ENV] = str(
            _required_library(library_dir, "libhsa-runtime64.so.1")
        )
        aqlprofile = library_dir / "libhsa-amd-aqlprofile64.so.1"
        if aqlprofile.is_file():
            result.setdefault(_LIBAQLPROFILE_ENV, str(aqlprofile))

        search_paths = [
            str(library_dir),
            str(library_dir / "rocm_sysdeps" / "lib"),
        ]
        existing_search_path = result.get("LD_LIBRARY_PATH")
        if existing_search_path:
            search_paths.append(existing_search_path)
        result["LD_LIBRARY_PATH"] = os.pathsep.join(search_paths)

    return result


def exec_loom_serve() -> NoReturn:
    """Runs the bundled server, forwarding all native command-line arguments."""
    package_root = Path(__file__).resolve().parent
    native_root = package_root / "_native"
    executable = native_root / "bin" / "loom-serve"
    arguments = list(sys.argv[1:])
    from . import pipeline_validation
    # Native deployment documents pass through unchanged. Retain the catalog
    # invocation accepted by earlier wheels by translating its model flags.
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--model")
    parser.add_argument("--weights")
    parser.add_argument("--tokenizer")
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument("--context_capacity", type=int, default=16384)
    parser.add_argument("--pool_capacity", type=int, default=65536)
    parser.add_argument("--prefill_capacity", type=int, default=512)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--mtp", nargs="?", const="true", default="false")
    parser.add_argument("--mtp_depth", type=int)
    legacy, remaining = parser.parse_known_args(arguments)
    catalog = Path(legacy.model) if legacy.model else None
    catalog_mode = (catalog is not None and catalog.is_dir()) or bool(legacy.weights or legacy.tokenizer)
    if catalog_mode:
        catalog = catalog or pipeline_validation.resolve_catalog("qwen")
        requested = legacy.mtp.lower() not in {"false", "0"} or legacy.mtp_depth is not None
        pipeline_validation.mtp_enabled(catalog, requested)
        if not legacy.weights or not legacy.tokenizer:
            raise ValueError("Catalog serving requires --weights and --tokenizer; native --model accepts a deployment JSON")
        deployment = pipeline_validation.deployment_for_model(
            catalog, pipeline_validation.ModelFiles(Path(legacy.weights), Path(legacy.tokenizer)),
            catalog.name, rows=legacy.rows, context_capacity=legacy.context_capacity,
            pool_capacity=legacy.pool_capacity, prefill_capacity=legacy.prefill_capacity,
            draft_depth=(legacy.mtp_depth if legacy.mtp_depth is not None else 3) if requested else 0,
        )
        arguments = [f"--model={deployment}", *remaining]
        if not any(arg == "--device" or arg.startswith("--device=") for arg in remaining):
            arguments.append("--device=amdgpu")
        if not any(arg == "--listen" or arg.startswith("--listen=") for arg in remaining):
            arguments.append(f"--listen=127.0.0.1:{legacy.port}")
    if not executable.is_file():
        raise RuntimeError("HRX wheel is missing native executable: loom-serve")
    os.execve(
        str(executable),
        [str(executable), *arguments],
        native_environment(),
    )
