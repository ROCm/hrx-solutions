"""Validate model assets against registered Loom pipeline implementations.

This module deliberately knows about pipeline identifiers, not model families.
Pipeline-specific compatibility checks live in adapters registered here.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Optional, Protocol, Union

from ._launcher import native_environment


@dataclass(frozen=True)
class ModelFiles:
    """The local files a pipeline needs to check a model."""

    weights: Path
    tokenizer: Path


@dataclass(frozen=True)
class ValidationResult:
    """Outcome and diagnostic details from a pipeline compatibility check."""

    compatible: bool
    pipeline: str
    message: str
    returncode: Optional[int] = None
    stdout: str = ""
    stderr: str = ""
    command: tuple[str, ...] = ()


class Pipeline(Protocol):
    """A selected pipeline's model compatibility check."""

    identifier: str

    def validate(
        self,
        model: ModelFiles,
        *,
        checker: Optional[Union[str, os.PathLike[str]]] = None,
        timeout: float = 1800,
    ) -> ValidationResult: ...


_PIPELINES: dict[str, Pipeline] = {}


def register_pipeline(pipeline: Pipeline) -> None:
    """Register a pipeline implementation by its stable CLI identifier."""
    if not pipeline.identifier or pipeline.identifier in _PIPELINES:
        raise ValueError(f"Pipeline identifier is empty or already registered: {pipeline.identifier!r}")
    _PIPELINES[pipeline.identifier] = pipeline


def available_pipelines() -> tuple[str, ...]:
    """Return the registered pipeline identifiers in stable order."""
    return tuple(sorted(_PIPELINES))


def catalog_for_model(pipeline: str, catalog: Path, model_id: str) -> Path:
    """Return a serving catalog configured for this installed model ID.

    Pipeline adapters may customize their source catalog when the server's
    endpoint identity is encoded in its source files. Adapters without this
    capability keep using their shared catalog unchanged.
    """
    implementation = _PIPELINES.get(pipeline)
    configure = getattr(implementation, "catalog_for_model", None)
    if configure is None:
        return catalog
    return configure(catalog, model_id)


def validate_model(
    model: ModelFiles,
    pipeline: str,
    *,
    checker: Optional[Union[str, os.PathLike[str]]] = None,
    timeout: float = 1800,
) -> ValidationResult:
    """Validate local model files with the selected registered pipeline.

    ``checker`` is an optional pipeline-specific executable override. The
    generic function only resolves the pipeline and checks that model files
    exist; each adapter owns the interpretation and checker invocation.
    """
    implementation = _PIPELINES.get(pipeline)
    if implementation is None:
        choices = ", ".join(available_pipelines()) or "none registered"
        return ValidationResult(
            False, pipeline, f"Unknown pipeline {pipeline!r}. Available pipelines: {choices}."
        )

    missing = [str(path) for path in (model.weights, model.tokenizer) if not Path(path).is_file()]
    if missing:
        return ValidationResult(
            False,
            pipeline,
            "Model validation requires existing weights and tokenizer files; missing: "
            + ", ".join(missing),
        )
    return implementation.validate(model, checker=checker, timeout=timeout)


class _QwenPipeline:
    """First adapter, backed by HRX's Qwen epoch_check correctness witness."""

    identifier = "qwen"

    def catalog_for_model(self, catalog: Path, model_id: str) -> Path:
        """Make the Qwen HTTP model name match the installed model ID."""
        catalog = Path(catalog).resolve()
        source_root = catalog.parent
        control = catalog / "control.loom"
        if not control.is_file():
            raise FileNotFoundError(f"Qwen pipeline control source is missing: {control}")

        digest = hashlib.sha256()
        for source in sorted(path for path in source_root.rglob("*") if path.is_file()):
            digest.update(source.relative_to(source_root).as_posix().encode("utf-8"))
            digest.update(source.read_bytes())
        fingerprint = digest.hexdigest()[:16]

        cache_root = Path.home() / ".cache" / "hrx-solutions" / "catalogs" / self.identifier
        output = cache_root / f"{model_id}-{fingerprint}"
        configured_control = output / "qwen" / "control.loom"
        if configured_control.is_file():
            return output / "qwen"

        cache_root.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{model_id}-", dir=cache_root))
        try:
            staging_catalog = staging / "llm"
            shutil.copytree(source_root, staging_catalog)
            configured_control = staging_catalog / "qwen" / "control.loom"
            source_text = configured_control.read_text(encoding="utf-8")
            marker = 'global.rodata.def @chat_model_name = bytes('
            lines = source_text.splitlines()
            matching = [index for index, line in enumerate(lines) if line.startswith(marker)]
            if len(matching) != 1:
                raise ValueError(
                    "Qwen catalog must define exactly one @chat_model_name byte string"
                )
            lines[matching[0]] = (
                f'global.rodata.def @chat_model_name = bytes("{model_id.encode("utf-8").hex()}")'
            )
            configured_control.write_text("\n".join(lines) + "\n", encoding="utf-8")
            output.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staging_catalog, output)
        finally:
            shutil.rmtree(staging, ignore_errors=True)
        return output / "qwen"

    def validate(
        self,
        model: ModelFiles,
        *,
        checker: Optional[Union[str, os.PathLike[str]]] = None,
        timeout: float = 1800,
    ) -> ValidationResult:
        executable = _find_qwen_checker(checker)
        if executable is None:
            return ValidationResult(
                False,
                self.identifier,
                "The Qwen compatibility checker is unavailable. The current wheel "
                "does not bundle the retired HRX epoch_check executable. Supply a "
                "compatible external checker with --checker or HRX_QWEN_EPOCH_CHECK; "
                "the Qwen validation adapter still needs migration to current HRX.",
            )

        catalog = _qwen_catalog()
        if not catalog.is_dir():
            return ValidationResult(
                False,
                self.identifier,
                f"Qwen pipeline source catalog is unavailable at {catalog}. "
                "Install the Qwen pipeline assets with the hrx-solutions package.",
            )

        command = (
            str(executable),
            f"--model={catalog}",
            # Keep the visible filenames when these point through Hugging
            # Face snapshot symlinks. The checker chooses the input format
            # from the filename extension, while the resolved blob name has
            # no extension.
            f"--weights={Path(model.weights).absolute()}",
            f"--tokenizer={Path(model.tokenizer).absolute()}",
            "--epoch=32:8",
        )
        try:
            result = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=native_environment(),
            )
        except subprocess.TimeoutExpired as exc:
            return ValidationResult(
                False,
                self.identifier,
                f"Qwen compatibility check exceeded {timeout:g} seconds. "
                "Confirm the GPU is available and retry with a longer timeout.",
                stdout=_as_text(exc.stdout),
                stderr=_as_text(exc.stderr),
                command=command,
            )
        except OSError as exc:
            return ValidationResult(
                False,
                self.identifier,
                f"Could not run Qwen compatibility checker {executable}: {exc}",
                command=command,
            )

        combined = "\n".join(part for part in (result.stdout.strip(), result.stderr.strip()) if part)
        if result.returncode == 0:
            return ValidationResult(
                True,
                self.identifier,
                "Model passed the Qwen pipeline compatibility check.",
                returncode=0,
                stdout=result.stdout,
                stderr=result.stderr,
                command=command,
            )
        details = combined or "The checker exited without diagnostic output."
        return ValidationResult(
            False,
            self.identifier,
            f"Model failed the Qwen pipeline compatibility check (exit {result.returncode}). "
            f"Checker output: {details}",
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
            command=command,
        )


def _find_qwen_checker(
    override: Optional[Union[str, os.PathLike[str]]],
) -> Optional[Path]:
    candidates: list[str] = []
    if override is not None:
        candidates.append(os.fspath(override))
    configured = os.environ.get("HRX_QWEN_EPOCH_CHECK")
    if configured:
        candidates.append(configured)
    for name in ("loom-qwen-epoch-check", "qwen-epoch-check"):
        found = shutil.which(name)
        if found:
            candidates.append(found)

    for candidate in candidates:
        path = Path(candidate).expanduser()
        if path.is_file() and os.access(path, os.X_OK):
            return path.resolve()
    return None


def _qwen_catalog() -> Path:
    # Source checkout layout. Wheel packaging can move this behind the same
    # adapter by placing assets adjacent to the Python package.
    repo_catalog = Path(__file__).resolve().parents[2] / "loom-pipeline" / "llm" / "qwen"
    if repo_catalog.is_dir():
        return repo_catalog
    return Path(__file__).resolve().parent / "loom-pipeline" / "llm" / "qwen"


def _as_text(value: Optional[Union[str, bytes]]) -> str:
    if value is None:
        return ""
    return value.decode(errors="replace") if isinstance(value, bytes) else value


register_pipeline(_QwenPipeline())
