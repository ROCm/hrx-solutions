"""Shared model asset types and checks; Hem owns registration in SQLite."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

_MODEL_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class ModelImportError(Exception):
    """Raised when a model cannot be safely registered."""


@dataclass(frozen=True)
class ModelFiles:
    """Runtime weights and the GGUF members used for compatibility checks."""
    weights: Path
    tokenizer: Path
    fragments: tuple[Path, ...] = ()

    @property
    def gguf_paths(self) -> tuple[Path, ...]:
        return self.fragments or (self.weights,)


@dataclass(frozen=True)
class NamedAsset:
    """One catalog-declared file; roles need not be unique."""
    role: str
    logical_path: str
    path: Path


@dataclass(frozen=True)
class CatalogAssets:
    """Complete caller-prepared files, without GGUF or tokenizer assumptions."""
    pipeline: str
    contract: str
    directory: Path
    files: tuple[NamedAsset, ...]


def readable_file(path: Path) -> bool:
    try:
        if not path.is_file():
            return False
        with path.open("rb"):
            return True
    except OSError:
        return False


def validate_model_id(model_id: str) -> None:
    if not isinstance(model_id, str) or model_id in (".", "..") or not _MODEL_ID_RE.fullmatch(model_id):
        raise ModelImportError("Model ID must start with a letter or digit and contain only letters, digits, '.', '_' or '-'")


@dataclass(frozen=True)
class AcquisitionInput:
    """One immutable caller source or pinned HF locator, independent of its use."""
    input_id: str
    path: Path
    kind: str = 'caller'
    repo_id: str | None = None
    revision: str | None = None
    filename: str | None = None


@dataclass(frozen=True)
class RuntimeMapping:
    """A runtime destination selects exactly one input or generated output role."""
    role: str
    logical_path: str
    ownership: str
    input_id: str | None = None
    generated_role: str | None = None


def mapped_assets(inputs, mappings, *, generated_directory=None, named=True):
    """Translate explicit selections to existing durable asset records.

    Unselected inputs are preparation-only and never become runtime references.
    Generated source locators name the final owned file, preserving recipe/retry
    identity; completed attempt output is separately copied into reserved staging.
    """
    selected = {item.input_id: item for item in inputs}
    if len(selected) != len(inputs) or any(not item.input_id or not item.path.is_absolute() for item in inputs):
        raise ModelImportError('Acquisition inputs require unique IDs and absolute paths')
    records, destinations = [], []
    for mapping in mappings:
        path = Path(mapping.logical_path)
        if (not mapping.logical_path or path.is_absolute() or '\\' in mapping.logical_path
                or any(part in ('', '.', '..') for part in mapping.logical_path.split('/'))
                or mapping.ownership not in ('owned', 'reference')
                or (mapping.input_id is None) == (mapping.generated_role is None)
                or any(path == other or path in other.parents or other in path.parents for other in destinations)):
            raise ModelImportError('Invalid or overlapping runtime mapping')
        if mapping.input_id is not None:
            if mapping.input_id not in selected:
                raise ModelImportError('Runtime mapping selects an unknown input')
            source = selected[mapping.input_id].path
        else:
            if generated_directory is None or mapping.ownership != 'owned':
                raise ModelImportError('Generated runtime outputs require an owned destination')
            source = Path(generated_directory) / path
            if not source.is_absolute():
                raise ModelImportError('Generated runtime destination must be absolute')
        destinations.append(path)
        record = dict(logical_path=mapping.logical_path, source_path=str(source), ownership=mapping.ownership)
        if named:
            record['role'] = mapping.role
        records.append(record)
    return records


@dataclass(frozen=True)
class PreparationAttempt:
    """Caller-owned private directory; the helper requires a nonexistent output."""
    directory: Path

    @property
    def output(self):
        return self.directory / 'output'

    @property
    def stdout(self):
        return self.directory / 'stdout.log'

    @property
    def stderr(self):
        return self.directory / 'stderr.log'

    @property
    def receipt(self):
        return self.directory / 'receipt.json'


@dataclass(frozen=True)
class PreparationResult:
    """One complete attempt's declared outputs and finite immutable facts."""
    attempt_directory: Path
    outputs: tuple[NamedAsset, ...]
    facts: tuple = ()


def prepared_outputs(attempt, result, expected):
    """Refuse incomplete, aliased or cross-attempt outputs before any staging."""
    if (not attempt.directory.is_absolute() or attempt.directory.is_symlink()
            or not isinstance(result, PreparationResult) or result.attempt_directory != attempt.directory
            or type(result.outputs) is not tuple or type(result.facts) is not tuple or result.facts
            or any(not isinstance(item, NamedAsset) or not isinstance(item.path, Path) for item in result.outputs)):
        raise ModelImportError('Preparation result does not match its complete declared attempt')
    actual = [(item.role, item.logical_path) for item in result.outputs]
    if (len({role for role, _ in actual}) != len(actual)
            or len({path for _, path in actual}) != len(actual) or set(actual) != set(expected)):
        raise ModelImportError('Preparation result has missing, duplicate or unknown generated outputs')
    paths = {}
    for item in result.outputs:
        path = item.path
        try:
            relative = path.relative_to(attempt.output)
            if (not path.is_absolute() or '..' in relative.parts or path.is_symlink()
                    or any(parent.is_symlink() for parent in path.parents
                           if parent == attempt.output or attempt.output in parent.parents)
                    or not readable_file(path)):
                raise ValueError()
            path.resolve(strict=True).relative_to(attempt.output.resolve(strict=True))
        except (OSError, ValueError, RuntimeError) as error:
            raise ModelImportError('Preparation output is missing, aliased or outside its attempt') from error
        paths[item.role] = path
    return paths
