"""Pinned recipes, local Hub cache inspection and a streamed HF CLI primitive.

Hem owns model publication and inventory; these helpers never import old state.
"""
from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time

from . import curated_catalog, hem_display, model_store, pipeline_validation
from .qwen_validation import load_profile

_COMMIT = re.compile(r"^[0-9a-f]{40}$")


def recipes() -> list[dict]:
    catalog = []
    pipelines = pipeline_validation.available_pipelines()
    # Keep existing Qwen recipe ordering when adding curated acquisition entries.
    for pipeline in (*[item for item in pipelines if item not in curated_catalog.CATALOGS],
                     *[item for item in pipelines if item in curated_catalog.CATALOGS]):
        if pipeline in curated_catalog.CATALOGS:
            entry = curated_catalog.load_catalog(pipeline)
            if entry.recipe is not None:
                descriptor = dict(entry.recipe, pipeline=pipeline,
                                  asset_contract=entry.contract, check_kind='acquisition-presence')
                _sources(descriptor)
                catalog.append(descriptor)
            continue
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
    raise model_store.ModelImportError(f"Unknown download recipe {identifier!r}; run 'loom-control model available'")


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
    if 'sources' in recipe:
        sources = recipe['sources']
        if not isinstance(sources, list) or not sources:
            raise model_store.ModelImportError('Named recipe must specify pinned HF sources')
        for source in sources:
            if (not isinstance(source, dict) or
                    set(source) != {'role', 'kind', 'repo_id', 'revision', 'filename'} or
                    source['kind'] != 'huggingface' or not isinstance(source['role'], str) or
                    not source['role']):
                raise model_store.ModelImportError('Invalid named recipe source')
        _check_sources(sources)
        return sources
    gguf = recipe["gguf"]
    tokenizer = recipe["tokenizer"]
    sources = [{"role": "gguf", "kind": "huggingface", "repo_id": gguf["repo_id"],
                "revision": gguf["revision"], "filename": filename} for filename in gguf["filenames"]]
    sources.append({"role": "tokenizer", "kind": "huggingface", "repo_id": tokenizer["repo_id"],
                    "revision": tokenizer["revision"], "filename": tokenizer["filename"]})
    if len(sources) < 2:
        raise model_store.ModelImportError("Recipe must specify GGUF and tokenizer files")
    _check_sources(sources)
    return sources


def acquisition_inputs(sources):
    """Snapshot source identity separately from runtime filenames or preparation."""
    return tuple(model_store.AcquisitionInput(
        f'source-{index}', _cache_path(source), source['kind'], source['repo_id'],
        source['revision'], source['filename']) for index, source in enumerate(sources))


def runtime_mappings(recipe, sources, *, mappings=None):
    """Admit explicit input-to-path associations, never expand repeated roles.

    Existing serialized recipes use their exact source filename as the runtime
    locator. Adapt that legacy association once; nonidentity mappings must be
    explicit rather than guessed from a role or basename. An unselected input in
    an explicit mapping set remains preparation-only.
    """
    catalog = curated_catalog.load_catalog(recipe['pipeline'])
    declared = {asset['path']: asset['role'] for asset in catalog.assets}
    generated = {item['path'] for item in recipe.get('generated', ())}
    by_id = {f'source-{index}': source for index, source in enumerate(sources)}
    if mappings is None:
        mappings = tuple(model_store.RuntimeMapping(source['role'], source['filename'], 'reference',
                                                   input_id=input_id) for input_id, source in by_id.items())
    for mapping in mappings:
        source = by_id.get(mapping.input_id)
        if (source is None or mapping.generated_role is not None or mapping.ownership != 'reference'
                or mapping.logical_path in generated or declared.get(mapping.logical_path) != mapping.role
                or source['role'] != mapping.role):
            raise model_store.ModelImportError('Named source mapping must select its declared runtime path and role, not a generated output')
    return tuple(mappings)


def _check_sources(sources):
    for source in sources:
        if not isinstance(source['revision'], str) or not _COMMIT.fullmatch(source["revision"]):
            raise model_store.ModelImportError("Bundled recipes must pin each repository to an immutable commit")
        for key in ("repo_id", "filename"):
            value = source[key]
            if not isinstance(value, str) or not value or value.startswith(("/", "-")) or "\\" in value or any(part in ("", ".", "..") for part in value.split("/")):
                raise model_store.ModelImportError(f"Invalid recipe {key}: {value!r}")
    if len({(s['repo_id'], s['revision'], s['filename']) for s in sources}) != len(sources):
        raise model_store.ModelImportError("Recipe contains duplicate files")


def _cache_path(source: dict) -> Path:
    return cache_root() / ("models--" + source["repo_id"].replace("/", "--")) / "snapshots" / source["revision"] / source["filename"]


def source_status(source: dict) -> dict:
    path = _cache_path(source)
    cached = model_store.readable_file(path)
    return {**source, "cached": cached, "cache_path": str(path),
            "size_bytes": path.stat().st_size if cached else None}


def _redact(value: str) -> str:
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        token = os.environ.get(key)
        if token:
            value = value.replace(token, "[REDACTED]")
    return re.sub(r"hf_[A-Za-z0-9]+", "[REDACTED]", value)


def _run_hf(command: list[str]) -> None:
    """Stream and drain both CLI channels to stderr without retaining output."""
    hem_display.emit(hem_display.command(command, secrets=[
        argument for argument in command if _redact(argument) != argument]))
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
    last_update = time.monotonic()
    finished = threading.Event()
    def heartbeat():
        while not finished.wait(5):
            if time.monotonic() - last_update >= 5:
                _progress("Waiting for Hugging Face CLI…")
    def pump(stream):
        nonlocal last_update
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
                pending = []
        if pending:
            line = _redact("".join(pending))
            sys.stderr.write(line)
            sys.stderr.flush()
        stream.close()
    workers = [threading.Thread(target=pump, args=(process.stdout,), daemon=True),
               threading.Thread(target=pump, args=(process.stderr,), daemon=True)]
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


def _progress(message: str) -> None:
    print(message, file=sys.stderr, flush=True)
