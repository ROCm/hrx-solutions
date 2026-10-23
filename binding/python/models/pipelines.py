"""Fixed selected-pipeline access; implementations are imported only on selection."""
from __future__ import annotations

from importlib import import_module
from pathlib import Path

from ..hem_common import HemError, InvalidInput
from .recipe_types import PipelineBinding, PipelineDescriptor


# Implementation bindings, not another recipe/model-support registry.
_BINDINGS = {
    "qwen": "pipelines_qwen",
    "qwen-08b": "pipelines_qwen",
    "moonshine-streaming-tiny": "pipelines_moonshine",
}


def catalog_roots() -> tuple[Path, ...]:
    """Existing installed/source catalog locations; never initialize any state."""
    package = Path(__file__).resolve().parents[1]
    return package / "loom-pipeline", package.parents[1] / "loom-pipeline"


def get(pipeline_id: str) -> PipelineBinding:
    if type(pipeline_id) is not str or pipeline_id not in _BINDINGS:
        raise InvalidInput(f"Unknown pipeline {pipeline_id!r}")
    module_name = f"{__package__}.{_BINDINGS[pipeline_id]}"
    try:
        module = import_module(module_name)
    except ImportError as error:
        raise HemError(f"Cannot load pipeline {pipeline_id!r}: its implementation is unavailable") from error
    binding = module.get(pipeline_id)
    descriptor = binding.describe()
    if not isinstance(descriptor, PipelineDescriptor) or descriptor.id != pipeline_id:
        raise HemError(f"Pipeline {pipeline_id!r} returned an inconsistent descriptor")
    return binding
