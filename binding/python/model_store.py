"""Atomic local model registration; downloaded files remain owned by HF's cache."""
from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import time
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

DEFAULT_MODEL_ROOT = Path.home() / ".loom" / "models"
_MODEL_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_CHUNK_SIZE = 1024 * 1024

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
class ImportedModel:
    model_id: str
    directory: Path
    files: ModelFiles
    manifest_path: Path


@dataclass(frozen=True)
class RegisteredFile:
    name: str
    path: Path
    source: dict
    link_target: Optional[Path]


@dataclass(frozen=True)
class RegisteredModel:
    model_id: str
    manifest: dict
    files: Optional[ModelFiles]
    entries: tuple[RegisteredFile, ...]
    availability: str
    error: Optional[str] = None

    def require_files(self) -> ModelFiles:
        if self.availability != "ready" or self.files is None:
            raise ModelImportError(self.error or f"Model {self.model_id!r} is unavailable")
        return self.files


def _progress(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def import_model(model_id: str, *, gguf, tokenizer, model_root: Optional[Path] = None,
                 alias: bool = False) -> ImportedModel:
    """Import user-owned local files. Use alias to avoid copying large weights."""
    paths = (gguf,) if isinstance(gguf, (str, Path)) else tuple(gguf)
    return register_model(model_id, ggufs=[_local_source(path) for path in paths],
                          tokenizer=_local_source(tokenizer), model_root=model_root, alias=alias)


def register_model(model_id: str, *, ggufs: Sequence[dict], tokenizer: dict,
                   model_root: Optional[Path] = None, alias: bool = False,
                   recipe: Optional[dict] = None,
                   pipeline: Optional[str] = None) -> ImportedModel:
    """Stage files, validate them, then publish an absent model ID atomically."""
    validate_model_id(model_id)
    if not ggufs:
        raise ModelImportError("At least one GGUF file is required")
    root = (Path(model_root).expanduser() if model_root is not None else DEFAULT_MODEL_ROOT).absolute()
    root.mkdir(parents=True, exist_ok=True)
    destination = root / model_id
    lock = root / f".{model_id}.lock"
    try:
        lock_fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise ModelImportError(f"Another registration of {model_id!r} is active (lock: {lock})") from exc
    stage = None
    try:
        os.close(lock_fd)
        if destination.exists() or destination.is_symlink():
            raise ModelImportError(f"Model destination already exists: {destination}; choose a different --id")
        stage = Path(tempfile.mkdtemp(prefix=f".{model_id}.import-", dir=root))
        entries = {}
        for index, source in enumerate(ggufs):
            name = "model.gguf" if len(ggufs) == 1 else f"model-{index+1:05d}-of-{len(ggufs):05d}.gguf"
            key = "gguf" if index == 0 else f"gguf_{index+1}"
            entries[key] = _stage_file(source, stage / name, alias)
            _validate_gguf(stage / name)
        entries["tokenizer"] = _stage_file(tokenizer, stage / "tokenizer.json", alias)
        tokenizer_data = _validate_tokenizer(stage / "tokenizer.json")
        paths = tuple(stage / entry["path"] for key, entry in entries.items() if key.startswith("gguf"))
        manifest = {"format_version": 2, "model_id": model_id,
                    "imported_at": datetime.now(timezone.utc).isoformat(),
                    "install_mode": "alias" if alias else "copy", "files": entries,
                    "gguf_files": [path.name for path in paths], "weights_path": paths[0].name}
        if recipe is not None:
            manifest["recipe"] = recipe
        from .gguf import inspect_gguf_set
        # Local single-file imports retain their structural magic check. Recipes
        # and fragmented imports need the complete descriptor inspection.
        inspection = None
        if pipeline is not None:
            from .pipeline_validation import validate_model
            result = validate_model(ModelFiles(paths[0], stage / "tokenizer.json", paths),
                                    pipeline, tokenizer_data=tokenizer_data)
            if not result.compatible:
                raise ModelImportError(result.message)
            inspection = result.inspection
            manifest["pipeline"] = result.pipeline
            _progress(result.message)
        if len(paths) > 1:
            if inspection is None:
                _progress("Inspecting GGUF fragment descriptors…")
                inspection = inspect_gguf_set(paths)
            # The pinned native reader requires this suffix and dispatches
            # members by extension; aliases must retain their .gguf extension.
            native_index = stage / "model.safetensors.index.json"
            weight_map = {name: path.name for name, path in inspection.tensor_files.items()}
            native_index.write_text(json.dumps({"weight_map": weight_map}) + "\n", encoding="utf-8")
            manifest["weights_path"] = native_index.name
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if destination.exists() or destination.is_symlink():
            raise ModelImportError(f"Model destination already exists: {destination}")
        os.rename(stage, destination)
        _progress(f"Registered {model_id}; {'cache/source aliases' if alias else 'copied files'} are ready.")
    except ModelImportError:
        raise
    except (OSError, ValueError) as exc:
        raise ModelImportError(f"Could not register model {model_id!r}: {exc}") from exc
    finally:
        if stage is not None and stage.exists():
            shutil.rmtree(stage, ignore_errors=True)
        lock.unlink(missing_ok=True)
    return ImportedModel(model_id, destination,
                         ModelFiles(destination / manifest["weights_path"], destination / "tokenizer.json",
                                    tuple(destination / name for name in manifest["gguf_files"])),
                         destination / "manifest.json")


def _stage_file(source: dict, destination: Path, alias: bool) -> dict:
    path = Path(source["path"]).absolute()
    _progress(f"{'Linking' if alias else 'Copying'} {path.name}…")
    if alias:
        destination.symlink_to(path)
    else:
        _copy_source(path, destination)
    size = destination.stat().st_size
    if not size:
        raise ModelImportError(f"Source file is empty: {path}")
    entry = {"path": destination.name, "size_bytes": size,
             "source": {key: str(value) if isinstance(value, Path) else value
                        for key, value in source.items() if key != "path"}}
    if source["kind"] == "local":
        entry["source"]["path"] = str(path)
    if alias:
        entry["link_target"] = str(path)
    return entry


def _local_source(source) -> dict:
    path = Path(source).expanduser().absolute()
    if not readable_file(path):
        raise ModelImportError(f"Source file is unavailable or unreadable: {path}")
    return {"kind": "local", "path": path}


def readable_file(path: Path) -> bool:
    try:
        if not path.is_file():
            return False
        with path.open("rb"):
            return True
    except OSError:
        return False


def _manifest_file_entries(manifest: dict) -> dict:
    entries = manifest.get("files", {})
    if not isinstance(entries, dict) or any(not isinstance(entry, dict) for entry in entries.values()):
        raise ModelImportError("Malformed manifest file entries")
    return entries


def registered_model(model_id: str, model_root: Optional[Path] = None) -> RegisteredModel:
    """Read one manifest and normalize assets, provenance and availability.

    Malformed manifests are invalid; missing files retain their registration
    and validation information. Legacy manifests use conventional filenames.
    """
    root = Path(model_root).expanduser() if model_root is not None else DEFAULT_MODEL_ROOT
    directory = root / model_id
    manifest = {}
    try:
        manifest = read_manifest(model_id, root)
        if manifest.get("pipeline") is not None and not isinstance(manifest["pipeline"], str):
            raise ModelImportError("Malformed pipeline identifier")
        raw = _manifest_file_entries(manifest)
        entries = []
        for name, entry in raw.items():
            path = _manifest_path(directory, entry.get("path", ""))
            source = entry.get("source", {})
            if not isinstance(source, dict):
                raise ModelImportError(f"Malformed source for {name}")
            if any(key in source and not isinstance(source[key], str)
                   for key in ("repo_id", "revision", "filename")):
                raise ModelImportError(f"Malformed source coordinates for {name}")
            target = None
            if path.is_symlink():
                target = Path(os.readlink(path))
                target = target if target.is_absolute() else (path.parent / target).absolute()
            entries.append(RegisteredFile(name, path, source, target))
        names = manifest.get("gguf_files", [raw.get("gguf", {}).get("path", "model.gguf")])
        if not isinstance(names, list) or not names:
            raise ModelImportError("Malformed GGUF file list")
        paths = tuple(_manifest_path(directory, name) for name in names)
        tokenizer = _manifest_path(directory, raw.get("tokenizer", {}).get("path", "tokenizer.json"))
        weights = _manifest_path(directory, manifest.get("weights_path", paths[0].name))
        files = ModelFiles(weights, tokenizer, paths)
    except (ModelImportError, OSError) as exc:
        return RegisteredModel(model_id, {}, None, (), "invalid", str(exc))
    alias = manifest.get("install_mode") == "alias"
    for entry in entries:
        if not readable_file(entry.path) or (alias and entry.link_target is None):
            error = (f"Model {'alias' if alias else 'file'} {entry.name} is unavailable or unreadable: "
                     f"{entry.path} -> {entry.link_target or entry.path}; restore the source/cache file, "
                     "or download under a new --id. Existing registration was preserved.")
            return RegisteredModel(model_id, manifest, files, tuple(entries),
                                   "broken alias" if alias else "incomplete", error)
    for path in (*files.gguf_paths, files.tokenizer, files.weights):
        if not readable_file(path):
            return RegisteredModel(model_id, manifest, files, tuple(entries),
                                   "broken alias" if alias else "incomplete",
                                   f"Model file is unavailable or unreadable: {path}; restore the source/cache file")
    return RegisteredModel(model_id, manifest, files, tuple(entries), "ready")


def model_inventory(model_root: Optional[Path] = None) -> tuple[RegisteredModel, ...]:
    root = Path(model_root).expanduser() if model_root is not None else DEFAULT_MODEL_ROOT
    if not root.is_dir():
        return ()
    return tuple(registered_model(directory.name, root) for directory in sorted(root.iterdir())
                 if directory.is_dir() and not directory.name.startswith("."))


def model_files(model_id: str, model_root: Optional[Path] = None) -> ModelFiles:
    """Resolve registered assets and reject unreadable files or broken aliases."""
    return registered_model(model_id, model_root).require_files()


def _manifest_path(directory: Path, name: str) -> Path:
    if not isinstance(name, str) or not name or Path(name).is_absolute() or ".." in Path(name).parts:
        raise ModelImportError(f"Invalid manifest file path: {name!r}")
    return directory / name


def read_manifest(model_id: str, model_root: Optional[Path] = None) -> dict:
    validate_model_id(model_id)
    root = Path(model_root).expanduser() if model_root is not None else DEFAULT_MODEL_ROOT
    path = root / model_id / "manifest.json"
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise ValueError("expected a JSON object")
        return manifest
    except (OSError, ValueError) as exc:
        raise ModelImportError(f"Could not read model manifest {path}: {exc}") from exc


def record_pipeline(model_id: str, pipeline: str, model_root: Optional[Path] = None) -> dict:
    validate_model_id(model_id)
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
        temporary.unlink(missing_ok=True)
    return manifest


def validate_model_id(model_id: str) -> None:
    if not isinstance(model_id, str) or model_id in (".", "..") or not _MODEL_ID_RE.fullmatch(model_id):
        raise ModelImportError("Model ID must start with a letter or digit and contain only letters, digits, '.', '_' or '-'")


def _copy_source(path: Path, destination: Path) -> int:
    size = 0
    last_update = time.monotonic()
    total = path.stat().st_size
    with path.open("rb") as input_stream, destination.open("wb") as output:
        while True:
            chunk = input_stream.read(_CHUNK_SIZE)
            if not chunk:
                break
            output.write(chunk)
            size += len(chunk)
            if time.monotonic() - last_update >= 2:
                _progress(f"Copying {destination.name}: {size:,}/{total:,} bytes")
                last_update = time.monotonic()
    return size


def _validate_gguf(path: Path) -> None:
    with path.open("rb") as stream:
        if stream.read(4) != b"GGUF":
            raise ModelImportError(f"File does not have the GGUF magic header: {path}")


def _validate_tokenizer(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ModelImportError(f"Tokenizer is not valid UTF-8 JSON: {path}") from exc
    if not isinstance(data, dict) or not isinstance(data.get("model"), dict):
        raise ModelImportError(f"Tokenizer JSON does not contain a model object: {path}")
    return data
