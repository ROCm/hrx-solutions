"""Build and staging helpers for the HRX Solutions Python wheel."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
import platform
import re
import shlex
import shutil
import subprocess
from pathlib import Path
from types import MappingProxyType
from typing import Callable, Mapping, Optional

try:
    from ..versioning import read_package_version
    from ..model_tools import package_families
except ImportError:
    from build_tools.versioning import read_package_version
    from build_tools.model_tools import package_families

from setuptools import Distribution
from setuptools.command.bdist_wheel import bdist_wheel
from setuptools.command.build_py import build_py


REPO_ROOT = Path(__file__).resolve().parents[2]
BAZEL_TARGETS = {"loom-serve": "//:loom-serve"}
BAZEL_BUILD_OPTIONS = ("--config=opt",)
ROCM_RUNTIME_ROOTS = (
    "lib/libhsa-runtime64.so.1",
    "lib/libhsa-amd-aqlprofile64.so.1",
)
ROCM_LIBRARY_DIRS = (
    "lib",
    "lib/rocm_sysdeps/lib",
    "lib/llvm/lib",
)
ROCM_LIBRARY_PREFIXES = (
    "libamd",
    "libhsa",
    "libroc",
)
ROCM_DOCUMENTS = (
    "share/doc/rocr/LICENSE.md",
    "share/doc/hsa-amd-aqlprofile/LICENSE.md",
    "share/doc/rocprofiler-register/LICENSE.md",
)
NeededLibraries = Callable[[Path], tuple[str, ...]]


def package_version() -> str:
    """Returns the package version selected for this build."""
    return read_package_version()


def discover_bazel(explicit: Optional[str] = None) -> Path:
    """Finds Bazel from an override, PATH, or a containing workspace."""
    requested = explicit or os.environ.get("BAZEL")
    if requested:
        candidate = shutil.which(requested) or requested
        path = Path(candidate).expanduser().resolve()
        if path.is_file() and os.access(path, os.X_OK):
            return path
        raise RuntimeError(f"Bazel is not executable: {requested}")

    path_candidate = shutil.which("bazel")
    if path_candidate:
        return Path(path_candidate).resolve()

    for parent in (REPO_ROOT, *REPO_ROOT.parents):
        candidate = parent / "programs" / "bazel" / "bin" / "bazel"
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate.resolve()

    raise RuntimeError(
        "Bazel was not found. Run `python -m build_tools.wheel_tools.setup_python --help` "
        "for the pinned Bazelisk installation instructions."
    )


def read_bazel_repo_env(path: Path) -> dict[str, str]:
    """Reads --repo_env assignments from a Bazel rc file."""
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        for token in shlex.split(line):
            prefix = "--repo_env="
            if not token.startswith(prefix):
                continue
            assignment = token[len(prefix) :]
            if "=" in assignment:
                name, value = assignment.split("=", 1)
                values[name] = value
    return values


def discover_rocm_root() -> Path:
    """Finds the ROCm root used by the Bazel repository configuration."""
    configured = os.environ.get("IREE_ROCM_PATH")
    if not configured:
        values = read_bazel_repo_env(REPO_ROOT / ".bazelrc.configured")
        values.update(read_bazel_repo_env(REPO_ROOT / ".bazelrc.local"))
        configured = values.get("IREE_ROCM_PATH")
    if not configured:
        raise RuntimeError(
            "IREE_ROCM_PATH is not configured; run "
            "`python -m build_tools.wheel_tools.setup_python --rocm /path/to/rocm`."
        )
    root = Path(configured).expanduser().resolve()
    if not (root / "include").is_dir():
        raise RuntimeError(f"ROCm root has no include directory: {root}")
    if not (root / "lib" / "libhsa-runtime64.so.1").is_file():
        raise RuntimeError(f"ROCm root has no HSA runtime: {root}")
    return root


def bazel_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment["USE_BAZEL_VERSION"] = (REPO_ROOT / ".bazelversion").read_text().strip()
    return environment


def discover_hrx_source(context: BazelBuildContext) -> Path:
    """Returns Bazel's selected HRX checkout, including local module overrides."""
    subprocess.run(
        [str(context.executable), "fetch", "--repo=@hrx_system"],
        cwd=context.repository,
        check=True,
        env=dict(context.environment),
    )

    def query(*arguments: str) -> str:
        return subprocess.run(
            [str(context.executable), *arguments],
            cwd=context.repository,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            env=dict(context.environment),
        ).stdout.strip()

    mapping = json.loads(query("mod", "dump_repo_mapping", ""))
    output_base = Path(query("info", "output_base"))
    source_root = output_base / "external" / mapping["hrx_system"]
    if not (source_root / "MODULE.bazel").is_file():
        raise RuntimeError(f"Bazel HRX source directory is missing: {source_root}")
    return source_root.resolve()


@dataclass(frozen=True)
class BazelBuildContext:
    """One build's executable, repository, options and snapshotted environment.

    Queries resolve locators against the captured repository, preserving symlink
    names until family-specific artifact identity selection. No global cache or
    artifact interpretation belongs here.
    """
    executable: Path
    repository: Path
    options: tuple[str, ...]
    environment: Mapping[str, str]

    def __post_init__(self):
        object.__setattr__(self, 'executable', Path(self.executable).absolute())
        object.__setattr__(self, 'repository', Path(self.repository).absolute())
        object.__setattr__(self, 'options', tuple(self.options))
        object.__setattr__(self, 'environment', MappingProxyType(dict(self.environment)))

    def query(self, target: str) -> tuple[Path, ...]:
        result = subprocess.run(
            [str(self.executable), "cquery", *self.options, "--output=files", target],
            cwd=self.repository, check=True, text=True, stdout=subprocess.PIPE,
            env=dict(self.environment),
        )
        # Cardinality, file presence/content and model identities remain caller
        # decisions. In particular, do not resolve symlinks before Qwen parsing.
        paths = (Path(line.strip()) for line in result.stdout.splitlines() if line.strip())
        return tuple(path if path.is_absolute() else self.repository / path for path in paths)


def _native_output(context: BazelBuildContext, target: str) -> Path:
    outputs = context.query(target)
    if len(outputs) != 1:
        raise RuntimeError(f"Expected one Bazel output for {target}, got {list(map(str, outputs))!r}")
    output = outputs[0]
    if not output.is_file():
        raise RuntimeError(f"Bazel output for {target} does not exist: {output}")
    return output.resolve()


def wheel_build_options(rocm_root: Path) -> tuple[str, ...]:
    """Use identical configuration/cache identity for build and all output queries."""
    options = (
        *BAZEL_BUILD_OPTIONS,
        f"--repo_env=IREE_ROCM_PATH={rocm_root}",
    )
    # An explicit container disable also overrides stale generated rc entries.
    if os.environ.get("HRX_BAZEL_CACHE_ENABLED") == "0":
        options += (
            "--disk_cache=",
            "--action_env=HRX_WHEEL_COMPILER_INPUTS=",
            "--host_action_env=HRX_WHEEL_COMPILER_INPUTS=",
        )
    return options


def build_native_executables(context: BazelBuildContext) -> dict[str, Path]:
    """Builds the root Bazel targets included in the wheel."""
    metrics_options = []
    if metrics_dir := context.environment.get("HRX_BAZEL_METRICS_DIR"):
        metrics_path = Path(metrics_dir)
        metrics_path.mkdir(parents=True, exist_ok=True)
        metrics_options = [
            f"--profile={metrics_path / 'native-build.profile.gz'}",
            f"--build_event_json_file={metrics_path / 'native-build.events.json'}",
        ]
    subprocess.run(
        [str(context.executable), "build", *context.options, *metrics_options, *BAZEL_TARGETS.values(), *package_families.build_targets()],
        cwd=context.repository,
        check=True,
        env=dict(context.environment),
    )
    return {
        name: _native_output(context, target)
        for name, target in BAZEL_TARGETS.items()
    }


def _copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source.resolve(), destination)


def _find_llvm_readelf(rocm_root: Path) -> Path:
    for relative_name in (
        "lib/llvm/bin/llvm-readelf",
        "llvm/bin/llvm-readelf",
        "bin/llvm-readelf",
    ):
        candidate = rocm_root / relative_name
        if candidate.is_file() and os.access(candidate, os.X_OK):
            # llvm-readelf may be a multicall symlink to llvm-readobj. Preserve
            # argv[0], which selects the readelf-compatible output format.
            return candidate
    raise RuntimeError(f"ROCm llvm-readelf was not found under {rocm_root}")


def _read_needed_libraries(readelf: Path, library: Path) -> tuple[str, ...]:
    result = subprocess.run(
        [str(readelf), "--dynamic-table", str(library)],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return tuple(
        match.group(1)
        for match in re.finditer(
            r"\(NEEDED\).*Shared library: \[([^]]+)\]", result.stdout
        )
    )


def _find_rocm_library(rocm_root: Path, library_name: str) -> Optional[Path]:
    for relative_directory in ROCM_LIBRARY_DIRS:
        candidate = rocm_root / relative_directory / library_name
        if candidate.is_file():
            return candidate
    return None


def resolve_rocm_runtime_payload(
    rocm_root: Path,
    *,
    needed_libraries: Optional[NeededLibraries] = None,
) -> dict[Path, Path]:
    """Resolves the bundled ROCm ELF closure by loader-visible names."""
    dependency_reader = needed_libraries
    if dependency_reader is None:
        readelf = _find_llvm_readelf(rocm_root)

        def dependency_reader(path: Path) -> tuple[str, ...]:
            return _read_needed_libraries(readelf, path)

    payload: dict[Path, Path] = {}
    resolved_sources: dict[Path, Path] = {}
    pending: list[tuple[Path, Path]] = []
    for relative_name in ROCM_RUNTIME_ROOTS:
        source = rocm_root / relative_name
        if not source.is_file():
            raise RuntimeError(f"Required ROCm runtime library is missing: {source}")
        pending.append((Path(relative_name), source))

    while pending:
        relative_name, source = pending.pop()
        real_source = source.resolve()
        existing_name = resolved_sources.get(real_source)
        if existing_name is not None and existing_name != relative_name:
            raise RuntimeError(
                f"ROCm runtime library {real_source} is required as both "
                f"{existing_name} and {relative_name}; wheel aliases are unsupported"
            )
        if relative_name in payload:
            continue
        payload[relative_name] = real_source
        resolved_sources[real_source] = relative_name

        for dependency_name in dependency_reader(real_source):
            dependency = _find_rocm_library(rocm_root, dependency_name)
            if dependency is None:
                if dependency_name.startswith(ROCM_LIBRARY_PREFIXES):
                    raise RuntimeError(
                        f"ROCm dependency {dependency_name} required by {source} "
                        f"was not found under {rocm_root}"
                    )
                continue
            dependency_relative = dependency.relative_to(rocm_root)
            pending.append((dependency_relative, dependency))

    return dict(sorted(payload.items(), key=lambda item: item[0].as_posix()))


def stage_native_payload(
    native_root: Path,
    executables: dict[str, Path],
    rocm_root: Path,
    hrx_source_root: Path,
    *,
    needed_libraries: Optional[NeededLibraries] = None,
) -> list[Path]:
    """Stages Bazel tools and the redistributable ROCm runtime payload."""
    if native_root.exists():
        shutil.rmtree(native_root)
    native_root.mkdir(parents=True)

    for installed_name, source in executables.items():
        destination = native_root / "bin" / installed_name
        _copy_file(source, destination)
        destination.chmod(destination.stat().st_mode | 0o111)

    _copy_file(
        hrx_source_root / "LICENSE",
        native_root / "share" / "loom-serve" / "LICENSE",
    )

    runtime_payload = resolve_rocm_runtime_payload(
        rocm_root, needed_libraries=needed_libraries
    )
    for relative_path, source in runtime_payload.items():
        _copy_file(source, native_root / relative_path)

    for relative_name in ROCM_DOCUMENTS:
        source = rocm_root / relative_name
        if not source.is_file():
            continue
        component = Path(relative_name).parent.name
        destination = native_root / "licenses" / f"{component}-LICENSE.md"
        _copy_file(source, destination)

    return [path for path in native_root.rglob("*") if path.is_file()]


def validate_wheel_platform() -> None:
    machine = platform.machine().lower()
    if os.name != "posix" or platform.system() != "Linux":
        raise RuntimeError("HRX Solutions wheels currently require Linux")
    if machine not in ("x86_64", "amd64"):
        raise RuntimeError(f"HRX Solutions wheels currently require x86-64, got {machine}")


class BinaryDistribution(Distribution):
    """Marks the distribution as platform-specific without a Python extension."""

    def has_ext_modules(self) -> bool:
        return True


class Py3PlatformWheel(bdist_wheel):
    """Tags native executable wheels as Python-minor independent."""

    def finalize_options(self) -> None:
        super().finalize_options()
        self.root_is_pure = False

    def get_tag(self) -> tuple[str, str, str]:
        _, _, platform_tag = super().get_tag()
        return "py3", "none", platform_tag


class BazelBuildPy(build_py):
    """Builds and stages the native payload into setuptools' build tree."""

    def run(self) -> None:
        validate_wheel_platform()
        bazel = discover_bazel()
        rocm_root = discover_rocm_root()
        context = BazelBuildContext(bazel, REPO_ROOT, wheel_build_options(rocm_root), bazel_environment())
        executables = build_native_executables(context)
        products = package_families.prepare(context.query, context.repository)
        package_root = Path(self.build_lib) / "hrx_solutions"
        if package_root.exists():
            shutil.rmtree(package_root)
        super().run()
        self._catalog_outputs = package_families.stage(package_root, products, context.repository)
        native_root = package_root / "_native"
        hrx_source = discover_hrx_source(context)
        self._native_outputs = stage_native_payload(native_root, executables, rocm_root, hrx_source)
        self._native_outputs.extend(package_families.stage_helpers(native_root, products, hrx_source))

    def get_outputs(self, include_bytecode: int = 1) -> list[str]:
        outputs = super().get_outputs(include_bytecode)
        outputs.extend(str(path) for path in getattr(self, "_native_outputs", ()))
        outputs.extend(str(path) for path in getattr(self, "_catalog_outputs", ()))
        return outputs
