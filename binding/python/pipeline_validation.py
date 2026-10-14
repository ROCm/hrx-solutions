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
import tempfile
from typing import Protocol

from .gguf import inspect_gguf
from .qwen_validation import compare_gguf, compare_tokenizer, load_profile


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


class Pipeline(Protocol):
    """A selected pipeline's model compatibility check."""

    identifier: str

    def validate(self, model: ModelFiles) -> ValidationResult: ...


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
) -> ValidationResult:
    """Inspect local model files with the selected registered pipeline."""
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
    return implementation.validate(model)


class _QwenPipeline:
    """Header-only compatibility with the packaged Qwen architecture contract."""

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

    def validate(self, model: ModelFiles) -> ValidationResult:
        try:
            profile = load_profile(_qwen_catalog())
            weights = inspect_gguf(model.weights)
            problems = compare_gguf(weights, profile) + compare_tokenizer(model.tokenizer, profile)
        except (OSError, ValueError) as exc:
            return ValidationResult(False, self.identifier, f"Could not inspect Qwen model assets: {exc}")
        if problems:
            details = "\n".join(f"  - {problem}" for problem in problems[:32])
            if len(problems) > 32:
                details += f"\n  - ... {len(problems) - 32} additional mismatches"
            return ValidationResult(
                False, self.identifier,
                f"Model does not match {profile['identifier']} ({len(problems)} mismatches):\n{details}",
            )
        return ValidationResult(
            True, self.identifier,
            f"Model matches {profile['identifier']}: 64 base layers and MTP, "
            f"{len(weights.tensors)} tensors checked from GGUF descriptors; matching tokenizer configuration.",
        )


def _qwen_catalog() -> Path:
    # Source checkout layout. Wheel packaging can move this behind the same
    # adapter by placing assets adjacent to the Python package.
    repo_catalog = Path(__file__).resolve().parents[2] / "loom-pipeline" / "llm" / "qwen"
    if repo_catalog.is_dir():
        return repo_catalog
    return Path(__file__).resolve().parent / "loom-pipeline" / "llm" / "qwen"



register_pipeline(_QwenPipeline())
