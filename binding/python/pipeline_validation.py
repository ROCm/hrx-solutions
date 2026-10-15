"""Validate model assets against registered Loom pipeline implementations.

This module deliberately knows about pipeline identifiers, not model families.
Pipeline-specific compatibility checks live in adapters registered here.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Protocol, Optional
import sys
import threading

from .runtime_paths import runtime_cache_dir
from .gguf import GGUF, inspect_gguf_set
from .model_store import ModelFiles
from .qwen_validation import compare_gguf, compare_tokenizer, load_profile


@dataclass(frozen=True)
class ValidationResult:
    """Outcome and diagnostic details from a pipeline compatibility check."""

    compatible: bool
    pipeline: str
    message: str
    inspection: Optional[GGUF] = None


class Pipeline(Protocol):
    """A selected pipeline's model compatibility check."""

    identifier: str

    def validate(self, model: ModelFiles, *, tokenizer_data: Optional[dict] = None) -> ValidationResult: ...


_PIPELINES: dict[str, Pipeline] = {}
_QWEN_CATALOGS = {"qwen": "qwen_27b", "qwen-08b": "qwen_08b"}


def resolve_catalog(pipeline: str) -> Path:
    """Resolve an explicitly selected catalog in source or installed layouts."""
    package = Path(__file__).resolve().parent
    relative = Path("qwen") / _QWEN_CATALOGS[pipeline] if pipeline in _QWEN_CATALOGS else Path(pipeline)
    candidates = (
        package / "loom-pipeline" / "llm" / relative,
        package.parents[1] / "loom-pipeline" / "llm" / relative,
    )
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()
    raise FileNotFoundError(
        f"No source catalog is installed for pipeline {pipeline!r}; searched: "
        + ", ".join(str(path) for path in candidates)
    )


def mtp_enabled(catalog: Path, requested: bool | None = None) -> bool:
    """Apply the selected catalog's MTP capability before native execution."""
    if not (Path(catalog) / "compatibility.json").is_file():
        return requested is not False
    profile = load_profile(catalog)
    supported = profile.get("capabilities", {}).get("mtp", profile["mtp_layer"] is not None)
    if requested is True and not supported:
        raise ValueError(
            f"Model profile {profile['identifier']!r} does not support MTP: "
            "this checkpoint has no MTP weights."
        )
    return supported if requested is None else requested


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
    *, tokenizer_data: Optional[dict] = None,
) -> ValidationResult:
    """Inspect local model files with the selected registered pipeline."""
    implementation = _PIPELINES.get(pipeline)
    if implementation is None:
        choices = ", ".join(available_pipelines()) or "none registered"
        return ValidationResult(
            False, pipeline, f"Unknown pipeline {pipeline!r}. Available pipelines: {choices}."
        )

    missing = [str(path) for path in (*model.gguf_paths, model.tokenizer) if not Path(path).is_file()]
    if missing:
        return ValidationResult(
            False,
            pipeline,
            "Model validation requires existing weights and tokenizer files; missing: "
            + ", ".join(missing),
        )
    print("Validating GGUF descriptors and tokenizer compatibility on CPU…", file=sys.stderr, flush=True)
    done = threading.Event()
    def heartbeat():
        while not done.wait(5):
            print("Still validating model descriptors and tokenizer…", file=sys.stderr, flush=True)
    worker = threading.Thread(target=heartbeat, daemon=True)
    worker.start()
    try:
        return implementation.validate(model, tokenizer_data=tokenizer_data)
    finally:
        done.set()
        worker.join()


class _QwenPipeline:
    """Header-only compatibility with the packaged Qwen architecture contract."""

    def __init__(self, identifier: str = "qwen"):
        self.identifier = identifier

    def catalog_for_model(self, catalog: Path, model_id: str) -> Path:
        """Make the Qwen HTTP model name match the installed model ID."""
        catalog = Path(catalog).resolve()
        source_root = catalog.parents[1]
        catalog_relative = catalog.relative_to(source_root)
        control = catalog / "control.loom"
        if not control.is_file():
            raise FileNotFoundError(f"Qwen pipeline control source is missing: {control}")

        digest = hashlib.sha256()
        for source in sorted(path for path in source_root.rglob("*") if path.is_file()):
            digest.update(source.relative_to(source_root).as_posix().encode("utf-8"))
            digest.update(source.read_bytes())
        fingerprint = digest.hexdigest()[:16]

        cache_root = runtime_cache_dir() / "catalogs" / self.identifier
        output = cache_root / f"{model_id}-{fingerprint}"
        configured_control = output / catalog_relative / "control.loom"
        if configured_control.is_file():
            return output / catalog_relative

        cache_root.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{model_id}-", dir=cache_root))
        try:
            staging_catalog = staging / "llm"
            shutil.copytree(source_root, staging_catalog)
            configured_control = staging_catalog / catalog_relative / "control.loom"
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
        return output / catalog_relative

    def validate(self, model: ModelFiles, *, tokenizer_data: Optional[dict] = None) -> ValidationResult:
        try:
            profile = load_profile(resolve_catalog(self.identifier))
            weights = inspect_gguf_set(model.gguf_paths)
            problems = compare_gguf(weights, profile) + compare_tokenizer(model.tokenizer, profile, data=tokenizer_data)
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
            f"Model matches {profile['identifier']}: {profile['base_layer_count']} base layers "
            f"{'and MTP' if profile['mtp_layer'] is not None else 'without MTP'}, "
            f"{len(weights.tensors)} tensors checked from GGUF descriptors; matching tokenizer configuration.",
            inspection=weights,
        )


for _identifier in _QWEN_CATALOGS:
    register_pipeline(_QwenPipeline(_identifier))


def deployment_for_model(
    catalog: Path, model: ModelFiles, model_id: str, *, rows: int = 4,
    context_capacity: int = 16384, pool_capacity: int = 65536,
    prefill_capacity: int = 512, draft_depth: int = 0,
) -> Path:
    """Write a native deployment document outside the packaged source catalog."""
    if draft_depth < 0 or draft_depth > 3:
        raise ValueError("Qwen catalogs support --mtp-depth from 1 through 3 (0 disables MTP)")
    mtp_enabled(catalog, draft_depth > 0)
    if pool_capacity <= 0:
        raise ValueError("The pinned Loom Serve runtime requires positive --pool-capacity")
    sources = [line.strip() for line in (catalog / "sources.txt").read_text().splitlines()
               if line.strip() and not line.strip().startswith("#")]
    definition = {
        "kind": "text", "prepare": "prepare.loom", "control": "control.loom",
        "sources": [entry for entry in sources
                    if entry not in {"prepare.loom", "control.loom", "weights.loom"}],
        # Preserve parser-selecting extensions on symlinked model assets.
        "weights": str(model.weights.absolute()),
        "tokenizer": str(model.tokenizer.absolute()),
        "rows": rows, "context_capacity": context_capacity,
        "pool_capacity": pool_capacity, "prefill_capacity": prefill_capacity,
        "draft_depth": draft_depth, "options": {},
    }
    document = {"name": model_id, "device": 0,
                "base_directory": str(catalog.absolute()), "definition": definition}
    content = json.dumps(document, indent=2) + "\n"
    digest = hashlib.sha256(content.encode()).hexdigest()[:20]
    directory = runtime_cache_dir() / "deployments"
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / f"{digest}.json"
    if not output.exists():
        with tempfile.NamedTemporaryFile(mode="w", dir=directory, delete=False) as temporary:
            temporary.write(content)
            staging = Path(temporary.name)
        staging.replace(output)
    return output
