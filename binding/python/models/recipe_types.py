"""Shared immutable recipe, pipeline and prepared-artifact values; no I/O."""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Optional, Protocol, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from ..hem_common import Deadline


def freeze_json(value):
    """Copy finite JSON to recursively immutable values without scalar coercion."""
    if isinstance(value, Mapping):
        if any(type(key) is not str for key in value):
            raise ValueError("JSON object keys must be strings")
        return MappingProxyType({key: freeze_json(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(freeze_json(item) for item in value)
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    raise ValueError("expected finite JSON value")


def json_document(value):
    """Return mutable JSON containers for serialization at an existing boundary."""
    if isinstance(value, Mapping):
        return {key: json_document(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [json_document(item) for item in value]
    return value


@dataclass(frozen=True)
class HFSource:
    repo_id: str
    revision: str
    filename: str
    kind: str = field(default="huggingface", init=False)


@dataclass(frozen=True)
class GeneratedSource:
    kind: str = field(default="generated", init=False)


@dataclass(frozen=True)
class Artifact:
    id: str
    role: str
    format: str
    path: str
    source: Union[HFSource, GeneratedSource]


@dataclass(frozen=True)
class Recipe:
    format: int
    id: str
    name: str
    modality: str
    pipeline: str
    artifacts: tuple[Artifact, ...]
    configuration: Mapping[str, Any]
    # SHA256 of admitted authored bytes, never another authored revision field.
    identity: str

    def __post_init__(self):
        object.__setattr__(self, "artifacts", tuple(self.artifacts))
        object.__setattr__(self, "configuration", freeze_json(self.configuration))


@dataclass(frozen=True)
class Constraints:
    minimum: Optional[int] = None
    maximum: Optional[int] = None
    allowed_keys: Optional[tuple[str, ...]] = None

    def __post_init__(self):
        if self.allowed_keys is not None:
            object.__setattr__(self, "allowed_keys", tuple(self.allowed_keys))


@dataclass(frozen=True)
class Configurable:
    type: str
    default: Any
    constraints: Constraints = field(default_factory=Constraints)
    active_when: Optional[Mapping[str, Any]] = None
    state: str = field(default="configurable", init=False)

    def __post_init__(self):
        object.__setattr__(self, "default", freeze_json(self.default))
        if self.active_when is not None:
            object.__setattr__(self, "active_when", freeze_json(self.active_when))


@dataclass(frozen=True)
class Fixed:
    type: str
    value: Any
    state: str = field(default="fixed", init=False)

    def __post_init__(self):
        object.__setattr__(self, "value", freeze_json(self.value))


@dataclass(frozen=True)
class Unsupported:
    type: str
    reason: str
    state: str = field(default="unsupported", init=False)


ParameterDescriptor = Union[Configurable, Fixed, Unsupported]


@dataclass(frozen=True)
class ProtocolDescriptor:
    operation: str
    native_kind: str
    endpoint: str
    request_encoding: str
    input_modalities: tuple[str, ...]
    output_modalities: tuple[str, ...]

    def __post_init__(self):
        object.__setattr__(self, "input_modalities", tuple(self.input_modalities))
        object.__setattr__(self, "output_modalities", tuple(self.output_modalities))


@dataclass(frozen=True)
class ArtifactRequirement:
    role: str
    formats: tuple[str, ...]
    minimum_count: int
    maximum_count: Optional[int]
    paths: Optional[tuple[str, ...]] = None

    def __post_init__(self):
        object.__setattr__(self, "formats", tuple(self.formats))
        if self.paths is not None:
            object.__setattr__(self, "paths", tuple(self.paths))


@dataclass(frozen=True)
class HFInputConstraint:
    role: str
    format: str
    source: HFSource


@dataclass(frozen=True)
class GeneratedOutput:
    role: str
    format: str
    path: str


@dataclass(frozen=True)
class PreparationRequirement:
    required_inputs: tuple[HFInputConstraint, ...]
    outputs: tuple[GeneratedOutput, ...]

    def __post_init__(self):
        object.__setattr__(self, "required_inputs", tuple(self.required_inputs))
        object.__setattr__(self, "outputs", tuple(self.outputs))


@dataclass(frozen=True)
class BundleFile:
    path: str
    purpose: str


@dataclass(frozen=True)
class BundleManifest:
    catalog_relative_path: str
    files: tuple[BundleFile, ...]

    def __post_init__(self):
        object.__setattr__(self, "files", tuple(self.files))


@dataclass(frozen=True)
class MetricDescriptor:
    name: str
    event: str
    field: str
    unit: str


@dataclass(frozen=True)
class TelemetryDescriptor:
    raw_events: str = "unverified"
    metrics: tuple[MetricDescriptor, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "metrics", tuple(self.metrics))


@dataclass(frozen=True)
class TextPrograms:
    prepare: str
    control: str
    providers: tuple[str, ...]

    def __post_init__(self):
        object.__setattr__(self, "providers", tuple(self.providers))


@dataclass(frozen=True)
class RecognitionPrograms:
    prepare: str
    control: str
    sources: tuple[str, ...]

    def __post_init__(self):
        object.__setattr__(self, "sources", tuple(self.sources))


@dataclass(frozen=True)
class PipelineDescriptor:
    id: str
    modality: str
    parameters: Mapping[str, ParameterDescriptor]
    protocol: ProtocolDescriptor
    artifacts: tuple[ArtifactRequirement, ...]
    preparation: Optional[PreparationRequirement]
    programs: Union[TextPrograms, RecognitionPrograms]
    bundle: BundleManifest
    telemetry: TelemetryDescriptor = field(default_factory=TelemetryDescriptor)

    def __post_init__(self):
        object.__setattr__(self, "parameters", MappingProxyType(dict(self.parameters)))
        object.__setattr__(self, "artifacts", tuple(self.artifacts))


@dataclass(frozen=True)
class AcquisitionPlan:
    recipe_id: str
    pipeline: str
    inputs: tuple[Artifact, ...]
    generated: tuple[Artifact, ...]
    preparation: Optional[PreparationRequirement]

    def __post_init__(self):
        object.__setattr__(self, "inputs", tuple(self.inputs))
        object.__setattr__(self, "generated", tuple(self.generated))


@dataclass(frozen=True)
class GeneratedArtifact:
    role: str
    format: str
    path: str
    file: Path


@dataclass(frozen=True)
class PreparedArtifact:
    id: str
    role: str
    format: str
    logical_path: str
    path: Path
    ownership: str


@dataclass(frozen=True)
class PreparedModel:
    local_id: str
    recipe_id: str
    recipe_identity: str
    pipeline: str
    directory: Path
    files: tuple[PreparedArtifact, ...]
    preparation_evidence: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "files", tuple(self.files))
        object.__setattr__(self, "preparation_evidence", freeze_json(self.preparation_evidence))


@dataclass(frozen=True)
class ArtifactLocation:
    id: str
    logical_path: str
    source: Union[HFSource, GeneratedSource]
    cache_path: Optional[Path]
    cached: bool
    ownership: str


@dataclass(frozen=True)
class AcquisitionResult:
    model_id: str
    recipe_id: str
    pipeline: str
    state: str
    artifacts: tuple[ArtifactLocation, ...]

    def __post_init__(self):
        object.__setattr__(self, "artifacts", tuple(self.artifacts))


@dataclass(frozen=True)
class LogMetric:
    name: str
    value: Union[int, float]
    unit: str


@dataclass(frozen=True)
class LogView:
    text: str
    metrics: tuple[LogMetric, ...] = ()
    recognized: bool = False

    def __post_init__(self):
        object.__setattr__(self, "metrics", tuple(self.metrics))


class PipelineBinding(Protocol):
    def describe(self) -> PipelineDescriptor: ...

    def validate_artifacts(self, artifacts: tuple[Artifact, ...]) -> None: ...

    def preflight(self, plan: AcquisitionPlan, deadline: Deadline) -> None:
        """Check preparation tools before acquisition only when outputs are needed.

        Discovery, preview and complete prepared imports never invoke this hook.
        Bindings without generated preparation implement a no-op.
        """
        ...

    def prepare(self, plan: AcquisitionPlan, inputs: Mapping[str, Path],
                attempt: Path, deadline: Deadline) -> tuple[GeneratedArtifact, ...]: ...
