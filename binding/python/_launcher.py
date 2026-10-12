"""Executes the Loom Serve binary shipped in the wheel."""

from __future__ import annotations

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
) -> Dict[str, str]:
    """Returns the environment used for a bundled native executable."""
    result = dict(os.environ if environ is None else environ)
    native_root = Path(__file__).resolve().parent / "_native"
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
    """Runs the bundled text server with its packaged source catalog."""
    package_root = Path(__file__).resolve().parent
    native_root = package_root / "_native"
    executable = native_root / "bin" / "loom-text-server"
    if not executable.is_file():
        raise RuntimeError("HRX wheel is missing native executable: loom-text-server")
    arguments = list(sys.argv[1:])
    if not any(argument == "--model" or argument.startswith("--model=") for argument in arguments):
        arguments.insert(
            0,
            "--model="
            + str(
                native_root
                / "share"
                / "loom-serve"
                / "experimental"
                / "loom_serve"
                / "models"
                / "qwen"
            ),
        )
    os.execve(
        str(executable),
        [str(executable), *arguments],
        native_environment(),
    )
