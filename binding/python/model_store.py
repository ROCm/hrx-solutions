"""Import GGUF models and tokenizers into the user's Loom model store.

Hugging Face files are resolved through the ``hf`` CLI into its standard
cache. Local files are copied or aliased into the same canonical layout:

    ~/.loom/models/<model-id>/model.gguf
    ~/.loom/models/<model-id>/tokenizer.json
    ~/.loom/models/<model-id>/manifest.json

An import is staged beside its destination and published with one directory
rename, so incomplete downloads never appear as installed models.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Union


DEFAULT_MODEL_ROOT = Path.home() / ".loom" / "models"
_MODEL_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_CHUNK_SIZE = 1024 * 1024


class ModelImportError(Exception):
    """Raised when a model cannot be safely imported."""


@dataclass(frozen=True)
class HuggingFaceFile:
    """A file in a Hugging Face model repository."""

    repo_id: str
    filename: str
    revision: str = "main"


@dataclass(frozen=True)
class ImportedModel:
    """Paths and metadata for a successfully imported model."""

    model_id: str
    directory: Path
    gguf_path: Path
    tokenizer_path: Path
    manifest_path: Path


Source = Union[Path, str, HuggingFaceFile]


def import_model(
    model_id: str,
    *,
    gguf: Source,
    tokenizer: Source,
    model_root: Optional[Path] = None,
    token: Optional[str] = None,
    alias: bool = False,
) -> ImportedModel:
    """Install a local or Hugging Face GGUF and tokenizer atomically.

    Pass a local path for either file, or a :class:`HuggingFaceFile` for a
    Hugging Face repository file. Files are obtained through ``hf download``.
    Existing imports are left intact;
    remove one explicitly before importing a replacement.

    With ``alias=True``, local files and Hugging Face cache paths are symlinked
    in place, so large weights are not duplicated.

    ``token`` overrides ``HF_TOKEN`` and ``HUGGING_FACE_HUB_TOKEN`` for
    private or gated Hugging Face repositories. ``alias`` avoids a second
    copy of model files in the model store.
    """

    _validate_model_id(model_id)
    root = Path(model_root).expanduser() if model_root is not None else DEFAULT_MODEL_ROOT
    root.mkdir(parents=True, exist_ok=True)
    root = root.resolve()
    destination = root / model_id
    if destination.exists() or destination.is_symlink():
        raise ModelImportError(f"Model destination already exists: {destination}")

    stage = Path(tempfile.mkdtemp(prefix=f".{model_id}.import-", dir=root))
    try:
        gguf_source = _resolve_source(gguf, token)
        tokenizer_source = _resolve_source(tokenizer, token)
        gguf_path = stage / "model.gguf"
        tokenizer_path = stage / "tokenizer.json"
        link_targets = {}
        if alias:
            link_targets["gguf"] = gguf_source["path"]
            link_targets["tokenizer"] = tokenizer_source["path"]
            os.symlink(link_targets["gguf"], gguf_path)
            os.symlink(link_targets["tokenizer"], tokenizer_path)
            gguf_bytes = gguf_path.stat().st_size
            tokenizer_bytes = tokenizer_path.stat().st_size
        else:
            gguf_bytes = _copy_source(gguf_source, gguf_path)
            tokenizer_bytes = _copy_source(tokenizer_source, tokenizer_path)
        _validate_gguf(gguf_path)
        _validate_tokenizer(tokenizer_path)

        gguf_entry = {
            "path": "model.gguf",
            "size_bytes": gguf_bytes,
            "sha256": _sha256(gguf_path),
            "source": _manifest_source(gguf_source),
        }
        tokenizer_entry = {
            "path": "tokenizer.json",
            "size_bytes": tokenizer_bytes,
            "sha256": _sha256(tokenizer_path),
            "source": _manifest_source(tokenizer_source),
        }
        if alias:
            gguf_entry["link_target"] = str(link_targets["gguf"])
            tokenizer_entry["link_target"] = str(link_targets["tokenizer"])
        manifest = {
            "format_version": 1,
            "model_id": model_id,
            "imported_at": datetime.now(timezone.utc).isoformat(),
            "install_mode": "alias" if alias else "copy",
            "files": {
                "gguf": gguf_entry,
                "tokenizer": tokenizer_entry,
            },
        }
        (stage / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

        # The target is required to be absent, so this same-filesystem rename
        # makes the completed model visible in one operation.
        os.replace(stage, destination)
    except ModelImportError:
        raise
    except (OSError, ValueError) as exc:
        raise ModelImportError(f"Could not import model {model_id!r}: {exc}") from exc
    finally:
        if stage.exists():
            shutil.rmtree(stage, ignore_errors=True)

    return ImportedModel(
        model_id=model_id,
        directory=destination,
        gguf_path=destination / "model.gguf",
        tokenizer_path=destination / "tokenizer.json",
        manifest_path=destination / "manifest.json",
    )


def read_manifest(model_id: str, model_root: Optional[Path] = None) -> dict:
    """Read metadata for an installed model, rejecting invalid IDs."""

    _validate_model_id(model_id)
    root = Path(model_root).expanduser() if model_root is not None else DEFAULT_MODEL_ROOT
    path = root / model_id / "manifest.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ModelImportError(f"No installed model manifest at {path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelImportError(f"Could not read model manifest {path}: {exc}") from exc


def record_pipeline(model_id: str, pipeline: str, model_root: Optional[Path] = None) -> dict:
    """Record a pipeline after the caller has successfully validated the model."""
    _validate_model_id(model_id)
    if not pipeline or not isinstance(pipeline, str):
        raise ModelImportError("Pipeline identifier must be a non-empty string")
    root = Path(model_root).expanduser() if model_root is not None else DEFAULT_MODEL_ROOT
    path = root / model_id / "manifest.json"
    manifest = read_manifest(model_id, root)
    manifest["pipeline"] = pipeline
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        mode = path.stat().st_mode & 0o777
        temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.chmod(mode)
        os.replace(temporary, path)
    except OSError as exc:
        raise ModelImportError(f"Could not update model manifest {path}: {exc}") from exc
    finally:
        if temporary.exists():
            temporary.unlink()
    return manifest


def _validate_model_id(model_id: str) -> None:
    if not isinstance(model_id, str) or not _MODEL_ID_RE.fullmatch(model_id):
        raise ModelImportError(
            "Model ID must start with a letter or digit and contain only letters, digits, '.', '_' or '-'"
        )


def _resolve_source(source: Source, token: Optional[str]) -> dict:
    if isinstance(source, HuggingFaceFile):
        repo_id = source.repo_id.strip("/")
        if not repo_id or any(part in (".", "..") for part in repo_id.split("/")):
            raise ModelImportError(f"Invalid Hugging Face repository ID: {source.repo_id!r}")
        filename = source.filename.lstrip("/")
        if not filename or any(part in (".", "..") for part in filename.split("/")):
            raise ModelImportError(f"Invalid Hugging Face filename: {source.filename!r}")
        revision = source.revision or "main"
        auth_token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        command = ["hf", "download", repo_id, filename, "--revision", revision]
        if auth_token:
            command.extend(("--token", auth_token))
        try:
            result = subprocess.run(command, check=True, capture_output=True, text=True)
        except FileNotFoundError as exc:
            raise ModelImportError(
                "The Hugging Face CLI is required for Hugging Face imports; install it with `python -m pip install -U huggingface_hub[cli]`"
            ) from exc
        except subprocess.CalledProcessError as exc:
            details = (exc.stderr or exc.stdout or str(exc)).strip()
            raise ModelImportError(f"Hugging Face CLI download failed: {details}") from exc
        cache_path = Path(result.stdout.strip()).expanduser()
        if not cache_path.is_file():
            raise ModelImportError(
                f"Hugging Face CLI did not return an existing file path for {repo_id}/{filename}: {result.stdout.strip()!r}"
            )
        return {
            "kind": "huggingface",
            "repo_id": repo_id,
            "filename": filename,
            "revision": revision,
            # Keep the snapshot filename (for example, ``model.gguf``) in
            # the path. Hugging Face snapshots symlink to extensionless blob
            # files, and resolving here breaks consumers that select parsers
            # by the path extension.
            "path": cache_path.absolute(),
        }
    path = Path(source).expanduser()
    if not path.is_file():
        raise ModelImportError(f"Source file does not exist or is not a regular file: {path}")
    # Preserve a symlink's filename for alias mode. This is important for
    # Hugging Face snapshot paths, which link named files to extensionless
    # content-addressed blobs.
    return {"kind": "local", "path": path.absolute()}


def _copy_source(source: dict, destination: Path) -> int:
    digest_size = 0
    input_stream = source["path"].open("rb")
    try:
        with input_stream, destination.open("wb") as output:
            while True:
                chunk = input_stream.read(_CHUNK_SIZE)
                if not chunk:
                    break
                output.write(chunk)
                digest_size += len(chunk)
    finally:
        if not input_stream.closed:
            input_stream.close()
    if digest_size == 0:
        raise ModelImportError(f"Source file is empty: {source['path']}")
    return digest_size


def _validate_gguf(path: Path) -> None:
    with path.open("rb") as stream:
        if stream.read(4) != b"GGUF":
            raise ModelImportError(f"File does not have the GGUF magic header: {path}")


def _validate_tokenizer(path: Path) -> None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ModelImportError(f"Tokenizer is not valid UTF-8 JSON: {path}") from exc
    if not isinstance(data, dict) or not isinstance(data.get("model"), dict):
        raise ModelImportError(f"Tokenizer JSON does not contain a model object: {path}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(_CHUNK_SIZE)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_source(source: dict) -> dict:
    if source["kind"] == "local":
        return {"kind": "local", "path": str(source["path"])}
    result = {
        "kind": "huggingface",
        "repo_id": source["repo_id"],
        "filename": source["filename"],
        "revision": source["revision"],
    }
    if "requested_revision" in source:
        result["requested_revision"] = source["requested_revision"]
    return result
