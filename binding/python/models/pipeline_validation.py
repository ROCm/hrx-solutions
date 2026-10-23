"""Validate model assets against fixed packaged Loom pipeline contracts."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Optional
import sys
import threading

from . import curated_catalog
from ..runtime_paths import runtime_cache_dir
from .gguf import GGUF
from .model_store import ModelFiles
from .qwen_validation import compare_profile, inspect_model, load_profile


@dataclass(frozen=True)
class ValidationResult:
    """Outcome and diagnostic details from a pipeline compatibility check."""

    compatible: bool
    pipeline: str
    message: str
    inspection: Optional[GGUF] = None


_QWEN_CATALOGS = {"qwen": "qwen_27b", "qwen-08b": "qwen_08b"}


def catalog_roots(pipeline: str) -> tuple[Path, ...]:
    """Authoritative installed/source roots for an existing pipeline family."""
    if pipeline in curated_catalog.CATALOGS:
        return curated_catalog.catalog_roots()
    package = Path(__file__).resolve().parents[1]
    return package / "loom-pipeline", package.parents[1] / "loom-pipeline"


def resolve_catalog(pipeline: str) -> Path:
    """Resolve an explicitly selected catalog in source or installed layouts."""
    if pipeline in curated_catalog.CATALOGS:
        return curated_catalog.catalog_path(pipeline)
    relative = Path("qwen") / _QWEN_CATALOGS[pipeline] if pipeline in _QWEN_CATALOGS else Path(pipeline)
    candidates = tuple(root / "llm" / relative for root in catalog_roots(pipeline))
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()
    raise FileNotFoundError(
        f"No pipeline catalog is installed for pipeline {pipeline!r}; searched: "
        + ", ".join(str(path) for path in candidates)
    )


@dataclass(frozen=True)
class CatalogModules:
    """Validated role/provider selection shared by native deployment builders."""

    prepare: str
    control: str
    weights: str
    providers: tuple[str, ...]


def catalog_modules(catalog: Path) -> CatalogModules:
    """Resolve a checked installed singleton or an authored source catalog."""
    catalog = Path(catalog).resolve()
    descriptor = catalog / "pipeline.json"
    if descriptor.exists():
        metadata = json.loads(descriptor.read_text(encoding="utf-8"))
        profile = load_profile(catalog)
        if (not isinstance(metadata, dict)
                or set(metadata) != {"format", "variant", "module", "sha256"}
                or metadata["format"] != 1
                or metadata["variant"] != profile["identifier"]
                or metadata["module"] != "pipeline.loombc"):
            raise ValueError("Invalid singleton pipeline descriptor or compatibility variant")
        module = catalog / metadata["module"]
        module.resolve(strict=True).relative_to(catalog)
        content = module.read_bytes()
        if (not content.startswith(b"LOOM")
                or hashlib.sha256(content).hexdigest() != metadata["sha256"]):
            raise ValueError(f"Invalid Loom bytecode module or hash: {module}")
        executable = {p.relative_to(catalog).as_posix() for p in catalog.rglob("*")
                      if p.suffix in (".loom", ".loombc")}
        if executable != {"pipeline.loombc"}:
            raise ValueError("Singleton catalog contains unexpected executable modules")
        return CatalogModules("pipeline.loombc", "pipeline.loombc", "pipeline.loombc", ("pipeline.loombc",))
    root = catalog.parents[1]
    entries = [line.strip() for line in (catalog / "sources.txt").read_text(encoding="utf-8").splitlines()
               if line.strip() and not line.strip().startswith("#")]
    roles = {name: [] for name in ("prepare", "control", "weights")}
    identities = set()
    providers = []
    for entry in entries:
        path = Path(entry)
        if path.is_absolute() or path.suffix != ".loom":
            raise ValueError(f"Unsupported pipeline module: {entry}")
        module = catalog / path
        identity = os.path.normpath(str(module))
        if identity in identities:
            raise ValueError(f"Duplicate pipeline module: {entry}")
        identities.add(identity)
        module.resolve(strict=True).relative_to(root)
        if not module.is_file():
            raise ValueError(f"Pipeline module is not a file: {module}")
        if path.stem in roles:
            roles[path.stem].append(entry)
        else:
            providers.append(entry)
    for name, matches in roles.items():
        if len(matches) != 1:
            raise ValueError(f"Pipeline catalog must contain exactly one {name} module")
    return CatalogModules(*(roles[name][0] for name in roles), tuple(providers))


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


def available_pipelines() -> tuple[str, ...]:
    """Return the fixed packaged pipeline identifiers in stable order."""
    return tuple(sorted(set(_QWEN_CATALOGS) | set(curated_catalog.CATALOGS)))


def validate_model(
    model: ModelFiles,
    pipeline: str,
    *, tokenizer_data: Optional[dict] = None,
) -> ValidationResult:
    """Inspect local model files with the selected Qwen contract."""
    if pipeline not in _QWEN_CATALOGS:
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
        return _validate_qwen(model, pipeline, tokenizer_data=tokenizer_data)
    finally:
        done.set()
        worker.join()


def _validate_qwen(
    model: ModelFiles, pipeline: str, *, tokenizer_data: Optional[dict] = None,
) -> ValidationResult:
    """Present shared inspection/comparison through the public validation result."""
    try:
        profile = load_profile(resolve_catalog(pipeline))
        inspection = inspect_model(model, tokenizer_data=tokenizer_data)
        problems = compare_profile(inspection, profile)
    except (OSError, UnicodeError, ValueError) as exc:
        return ValidationResult(False, pipeline, f"Could not inspect Qwen model assets: {exc}")
    if problems:
        details = "\n".join(f"  - {problem}" for problem in problems[:32])
        if len(problems) > 32:
            details += f"\n  - ... {len(problems) - 32} additional mismatches"
        return ValidationResult(
            False, pipeline,
            f"Model does not match {profile['identifier']} ({len(problems)} mismatches):\n{details}",
        )
    return ValidationResult(
        True, pipeline,
        f"Model matches {profile['identifier']}: {profile['base_layer_count']} base layers "
        f"{'and MTP' if profile['mtp_layer'] is not None else 'without MTP'}, "
        f"{len(inspection.weights.tensors)} tensors checked from GGUF descriptors; matching tokenizer configuration.",
        inspection=inspection.weights,
    )


def text_definition(
    model: ModelFiles, modules: CatalogModules, *, rows: int,
    context_capacity: int, pool_capacity: int, prefill_capacity: int, draft_depth: int,
) -> dict:
    """Assemble explicit text fields; callers own validation/defaults/envelopes."""
    return {
        "kind": "text", "prepare": modules.prepare, "control": modules.control,
        "sources": list(modules.providers),
        # Preserve parser-selecting extensions on symlinked model assets.
        "weights": str(model.weights.absolute()),
        "tokenizer": str(model.tokenizer.absolute()),
        "rows": rows, "context_capacity": context_capacity,
        "pool_capacity": pool_capacity, "prefill_capacity": prefill_capacity,
        "draft_depth": draft_depth, "options": {},
    }


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
    modules = catalog_modules(catalog)
    definition = text_definition(model, modules, rows=rows, context_capacity=context_capacity,
                                 pool_capacity=pool_capacity, prefill_capacity=prefill_capacity,
                                 draft_depth=draft_depth)
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
