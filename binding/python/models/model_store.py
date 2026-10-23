"""Shared filesystem checks for declared acquisition artifacts.

Public recipe/prepared values live in recipe_types; Registry owns persistence.
"""
from __future__ import annotations

from pathlib import Path
import re

from .recipe_types import GeneratedArtifact

_MODEL_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class ModelImportError(Exception):
    """Raised when model acquisition cannot safely publish its artifacts."""


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


def generated_outputs(attempt, outputs, expected):
    """Admit one complete typed result without collapsing repeated roles."""
    if (not isinstance(attempt, Path) or not attempt.is_absolute() or attempt.is_symlink()
            or type(outputs) is not tuple
            or any(not isinstance(item, GeneratedArtifact) or not isinstance(item.file, Path)
                   for item in outputs)):
        raise ModelImportError('Preparation result does not match its complete declared attempt')
    wanted = {(item.role, item.format, item.path) for item in expected}
    actual = [(item.role, item.format, item.path) for item in outputs]
    if len(actual) != len(wanted) or set(actual) != wanted:
        raise ModelImportError('Preparation result has missing, duplicate or unknown generated outputs')
    paths, identities = {}, set()
    for item in outputs:
        path = item.file
        try:
            relative = path.relative_to(attempt)
            if (not path.is_absolute() or '..' in relative.parts or path == attempt
                    or any(part.is_symlink() for part in (path, *path.parents)
                           if part == attempt or attempt in part.parents)
                    or not readable_file(path)):
                raise ValueError()
            path.resolve(strict=True).relative_to(attempt.resolve(strict=True))
            stat = path.stat()
            identity = stat.st_dev, stat.st_ino
            if identity in identities:
                raise ValueError()
            identities.add(identity)
        except (OSError, ValueError, RuntimeError) as error:
            raise ModelImportError('Preparation output is missing, aliased or outside its attempt') from error
        paths[item.path] = path
    return paths
