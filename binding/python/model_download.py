"""Bundled recipes and one streamed Hugging Face CLI acquisition path.

Cache inspection uses the documented Hub snapshots layout, not a Hub Python
backend. No catalog/status operation performs a network request.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import threading
import time
from typing import Optional

from . import model_store, pipeline_validation
from .qwen_validation import load_profile

_COMMIT = re.compile(r"^[0-9a-f]{40}$")


def recipes() -> list[dict]:
    catalog = []
    for pipeline in pipeline_validation.available_pipelines():
        profile = load_profile(pipeline_validation.resolve_catalog(pipeline))
        if "download_recipe" not in profile:
            continue
        descriptor = dict(profile["download_recipe"])
        reference = profile["reference"]
        filenames = reference.get("gguf_filenames", [reference.get("gguf_filename")])
        descriptor.update({"pipeline": pipeline, "profile": profile["identifier"],
                           "gguf": {"repo_id": reference["gguf_repo"],
                                    "revision": reference["gguf_revision"], "filenames": filenames},
                           "tokenizer": {"repo_id": reference["tokenizer_repo"],
                                         "revision": reference["tokenizer_revision"],
                                         "filename": reference.get("tokenizer_filename", "tokenizer.json")}})
        _sources(descriptor)  # Reject malformed bundled coordinates before using them.
        catalog.append(descriptor)
    return catalog


def recipe_by_id(identifier: str) -> dict:
    for recipe in recipes():
        if recipe["id"] == identifier:
            return recipe
    raise model_store.ModelImportError(f"Unknown download recipe {identifier!r}; run 'loom-serve-setup models available'")


def cache_root() -> Path:
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"]).expanduser().absolute()
    if os.environ.get("HUGGINGFACE_HUB_CACHE"):
        return Path(os.environ["HUGGINGFACE_HUB_CACHE"]).expanduser().absolute()
    home = os.environ.get("HF_HOME")
    if home:
        return Path(home).expanduser().absolute() / "hub"
    return Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))).expanduser().absolute() / "huggingface" / "hub"


def _sources(recipe: dict) -> list[dict]:
    gguf = recipe["gguf"]
    tokenizer = recipe["tokenizer"]
    sources = [{"role": "gguf", "kind": "huggingface", "repo_id": gguf["repo_id"],
                "revision": gguf["revision"], "filename": filename} for filename in gguf["filenames"]]
    sources.append({"role": "tokenizer", "kind": "huggingface", "repo_id": tokenizer["repo_id"],
                    "revision": tokenizer["revision"], "filename": tokenizer["filename"]})
    if len(sources) < 2:
        raise model_store.ModelImportError("Recipe must specify GGUF and tokenizer files")
    for source in sources:
        if not _COMMIT.fullmatch(source["revision"]):
            raise model_store.ModelImportError("Bundled recipes must pin each repository to an immutable commit")
        for key in ("repo_id", "filename"):
            value = source[key]
            if not isinstance(value, str) or not value or value.startswith(("/", "-")) or "\\" in value or any(part in ("", ".", "..") for part in value.split("/")):
                raise model_store.ModelImportError(f"Invalid recipe {key}: {value!r}")
    if len({(s['repo_id'], s['revision'], s['filename']) for s in sources}) != len(sources):
        raise model_store.ModelImportError("Recipe contains duplicate files")
    return sources


def _cache_path(source: dict) -> Path:
    return cache_root() / ("models--" + source["repo_id"].replace("/", "--")) / "snapshots" / source["revision"] / source["filename"]


def source_status(source: dict) -> dict:
    path = _cache_path(source)
    cached = model_store.readable_file(path)
    return {**source, "cached": cached, "cache_path": str(path),
            "size_bytes": path.stat().st_size if cached else None}


def _cache_status(files: list[dict]) -> str:
    count = sum(file["cached"] for file in files)
    return "cached" if count == len(files) else "partially cached" if count else "not downloaded"


def _registration_status(recipe: dict, inventory: tuple[model_store.RegisteredModel, ...]) -> list[dict]:
    registrations = []
    expected = {(s["repo_id"], s["revision"], s["filename"]) for s in _sources(recipe)}
    for registered in inventory:
        manifest = registered.manifest
        actual = {(entry.source.get("repo_id"), entry.source.get("revision"), entry.source.get("filename"))
                  for entry in registered.entries}
        matches = manifest.get("recipe") == recipe or actual == expected
        if not matches and registered.model_id != recipe["id"]:
            continue
        status = registered.availability
        if status != "invalid":
            if not matches:
                status = "conflicting registration"
            elif status == "ready" and manifest.get("pipeline") != recipe["pipeline"]:
                status = "unvalidated"
        item = {"id": registered.model_id, "status": status}
        if registered.error:
            item["error"] = registered.error
        registrations.append(item)
    return registrations


def available(model_root: Optional[Path] = None) -> list[dict]:
    catalog = []
    inventory = model_store.model_inventory(model_root)
    for recipe in recipes():
        files = [source_status(source) for source in _sources(recipe)]
        registrations = _registration_status(recipe, inventory)
        catalog.append({**recipe, "cache_status": _cache_status(files), "files": files,
                        "registration_status": registrations[0]["status"] if len(registrations) == 1 else "multiple registrations" if registrations else "not registered",
                        "registrations": registrations})
    return catalog


def _redact(value: str) -> str:
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        token = os.environ.get(key)
        if token:
            value = value.replace(token, "[REDACTED]")
    return re.sub(r"hf_[A-Za-z0-9]+", "[REDACTED]", value)


def _run_hf(command: list[str]) -> str:
    """Stream both CLI channels to stderr and retain bounded stdout metadata."""
    _progress("Running: " + _redact(shlex.join(command)))
    environment = os.environ.copy()
    environment["HF_HUB_DISABLE_TELEMETRY"] = "1"
    environment["HF_HUB_DISABLE_PROGRESS_BARS"] = "0"
    # HF otherwise hides byte progress when its stderr is piped to our streamer.
    environment["TQDM_POSITION"] = "-1"
    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   text=True, env=environment, errors="replace")
    except FileNotFoundError as exc:
        raise model_store.ModelImportError("Hugging Face CLI 'hf' is required for downloads; install huggingface_hub with its CLI, then run 'hf auth login' for gated repositories. See the setup documentation for Python 3.9.") from exc
    output = []
    output_size = 0
    last_update = time.monotonic()
    finished = threading.Event()
    def heartbeat():
        while not finished.wait(5):
            if time.monotonic() - last_update >= 5:
                _progress("Waiting for Hugging Face CLI…")
    def pump(stream, capture):
        nonlocal output_size, last_update
        pending = []
        while True:
            character = stream.read(1)
            if not character:
                break
            last_update = time.monotonic()
            pending.append(character)
            if character in "\n\r" or len(pending) >= 16384:
                line = _redact("".join(pending))
                sys.stderr.write(line)
                sys.stderr.flush()
                if capture and output_size < 1024 * 1024:
                    output.append(line)
                    output_size += len(line)
                pending = []
        if pending:
            line = _redact("".join(pending))
            sys.stderr.write(line)
            sys.stderr.flush()
            if capture and output_size < 1024 * 1024:
                output.append(line)
        stream.close()
    workers = [threading.Thread(target=pump, args=(process.stdout, True), daemon=True),
               threading.Thread(target=pump, args=(process.stderr, False), daemon=True)]
    monitor = threading.Thread(target=heartbeat, daemon=True)
    monitor.start()
    for worker in workers:
        worker.start()
    try:
        returncode = process.wait()
    except BaseException:
        process.terminate()
        process.wait()
        raise
    finally:
        finished.set()
        monitor.join()
        for worker in workers:
            worker.join()
    if returncode:
        raise model_store.ModelImportError(f"hf download failed (exit {returncode}); see the streamed error above. Check repository access with 'hf auth login', pinned filenames, cache permissions, and connectivity.")
    return "".join(output)


def _download_source(source: dict, offline: bool) -> dict:
    status = source_status(source)
    if status["cached"]:
        _progress(f"Using cached {source['repo_id']}/{source['filename']}")
    elif offline:
        raise model_store.ModelImportError(f"Offline cache is missing {source['repo_id']}@{source['revision']}/{source['filename']}: {status['cache_path']}")
    else:
        _run_hf(["hf", "download", source["repo_id"], source["filename"], "--revision", source["revision"], "--cache-dir", str(cache_root())])
        status = source_status(source)
        if not status["cached"]:
            raise model_store.ModelImportError(f"hf download did not populate the pinned snapshot: {status['cache_path']}")
    return {key: value for key, value in source.items() if key != "role"} | {"path": Path(status["cache_path"])}


def _preview_missing(files: list[dict]) -> dict:
    """Normalize the two supported hf contracts: old local plan, new JSON sizes.

    HF's remote size strings are rounded estimates. Exact sizes come only
    from readable cache files; unfamiliar preview formats remain unknown.
    """
    help_text = _run_hf(["hf", "download", "--help"])
    if "--dry-run" not in help_text or "--json" not in help_text:
        return {"size_note": "This hf CLI lacks JSON --dry-run; uncached sizes are unknown. No files were downloaded."}
    unknown = False
    previews = [dict(file) for file in files]
    for file in previews:
        if file["cached"]:
            continue
        preview = _run_hf(["hf", "download", file["repo_id"], file["filename"],
                           "--revision", file["revision"], "--cache-dir", str(cache_root()),
                           "--dry-run", "--json"])
        try:
            records = json.loads(preview)
            size = records[0]["size"] if isinstance(records, list) and len(records) == 1 else None
            match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)\s*([KMGT]?)B?", size, re.IGNORECASE) if isinstance(size, str) else None
        except (ValueError, KeyError, TypeError):
            match = None
        if match:
            scale = 1000 ** " KMGT".index(match[2].upper() or " ")
            file["expected_size"] = size
            file["estimated_size_bytes"] = int(float(match[1]) * scale)
        else:
            unknown = True
    if unknown:
        return {"files": previews, "size_note": "Unrecognized hf preview size; some uncached sizes are unknown. No files were downloaded."}
    return {"files": previews}


def _progress(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def download(identifier: str, *, model_id: Optional[str] = None, model_root: Optional[Path] = None,
             offline: bool = False, dry_run: bool = False) -> dict:
    recipe = recipe_by_id(identifier)
    model_id = model_id or identifier
    model_store.validate_model_id(model_id)
    offline = offline or os.environ.get("HF_HUB_OFFLINE", "").upper() in ("1", "ON", "YES", "TRUE")
    sources = _sources(recipe)
    _progress("Checking pinned Hugging Face cache files…")
    files = [source_status(source) for source in sources]
    root = Path(model_root).expanduser() if model_root is not None else model_store.DEFAULT_MODEL_ROOT
    destination = root / model_id
    plan = {"recipe": recipe, "id": model_id, "directory": str(destination),
            "storage": "alias", "offline": offline, "dry_run": dry_run,
            "cache_status": _cache_status(files), "files": files,
            "additional_model_store_payload_bytes": 0,
            "storage_note": "Files remain in the Hugging Face cache; model-store space is limited to aliases, manifest and optional fragment index.",
            "download_bytes": sum(file["size_bytes"] for file in files if not file["cached"]) if all(file["size_bytes"] is not None for file in files) else None}
    if dry_run:
        plan["destination_exists"] = destination.exists() or destination.is_symlink()
        if not offline and any(not file["cached"] for file in files):
            plan.update(_preview_missing(files))
            files = plan["files"]
        missing = [file for file in files if not file["cached"]]
        if all(file.get("size_bytes") is not None for file in missing):
            plan["download_bytes"] = sum(file["size_bytes"] for file in missing)
        elif all(file.get("estimated_size_bytes") is not None or file.get("size_bytes") is not None for file in missing):
            plan["estimated_download_bytes"] = sum(file.get("size_bytes") if file.get("size_bytes") is not None else file["estimated_size_bytes"] for file in missing)
        return plan
    if destination.exists() or destination.is_symlink():
        registered = model_store.registered_model(model_id, root)
        manifest = registered.manifest
        expected = {(source["repo_id"], source["revision"], source["filename"]): _cache_path(source).absolute() for source in sources}
        actual = {(entry.source.get("repo_id"), entry.source.get("revision"), entry.source.get("filename")): entry.link_target
                  for entry in registered.entries if entry.link_target is not None}
        if manifest.get("recipe") == recipe and manifest.get("pipeline") == recipe["pipeline"] and manifest.get("install_mode") == "alias" and actual == expected:
            existing = registered.require_files()
            result = pipeline_validation.validate_model(existing, recipe["pipeline"])
            if not result.compatible:
                raise model_store.ModelImportError(result.message)
            _progress(f"{model_id} is already downloaded, registered, and compatible.")
            return {**plan, "status": "already registered"}
        raise model_store.ModelImportError(f"Model destination already exists: {destination}; existing installation preserved. Choose a different --id.")
    if offline:
        missing = [f"{file['repo_id']}@{file['revision']}/{file['filename']}" for file in files if not file["cached"]]
        if missing:
            raise model_store.ModelImportError("Offline cache is missing pinned files:\n  " + "\n  ".join(missing))
    acquired = [_download_source(source, offline) for source in sources]
    model_store.register_model(model_id, ggufs=acquired[:-1], tokenizer=acquired[-1], model_root=root,
                               alias=True, recipe=recipe,
                               pipeline=recipe["pipeline"])
    return {**plan, "files": [source_status(source) for source in sources], "cache_status": "cached", "status": "registered", "pipeline": recipe["pipeline"]}
