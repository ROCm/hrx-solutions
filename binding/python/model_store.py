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
