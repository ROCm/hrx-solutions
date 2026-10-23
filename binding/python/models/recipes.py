"""One authored recipe grammar and packaged lookup, without acquisition effects."""
from __future__ import annotations

import builtins
import hashlib
import json
import math
from pathlib import PurePosixPath, PureWindowsPath
import re

from ..hem_common import HemError
from . import pipelines
from .recipe_types import Artifact, GeneratedSource, HFSource, Recipe

MAX_BYTES = 1024 * 1024
MAX_DEPTH = 64
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
_REVISION = re.compile(r"[0-9a-f]{40}\Z")
_FORMATS = ("gguf", "safetensors", "json")
_MODALITIES = ("chat", "asr")


def _fail(path, message):
    raise HemError(f"Invalid recipe {path}: {message}")


def _object(value, keys, path):
    if type(value) is not dict or set(value) != set(keys):
        _fail(path, "expected exactly " + ", ".join(keys))


def _name(value, path):
    if type(value) is not str or not _NAME.fullmatch(value) or value in (".", ".."):
        _fail(path, "expected a safe identifier")
    return value


def _relative(value, path, *, locator=False):
    if (type(value) is not str or not value or "\\" in value or "\0" in value
            or PurePosixPath(value).is_absolute()
            or PureWindowsPath(value).drive
            or any(part in ("", ".", "..") for part in value.split("/"))
            or locator and any(part.startswith("-") for part in value.split("/"))):
        _fail(path, "expected a normalized relative path/locator")
    return value


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            _fail(key, "duplicate object key")
        result[key] = value
    return result


def _finite(value, path, depth=0):
    if depth > MAX_DEPTH:
        _fail(path, "JSON nesting exceeds limit")
    if type(value) is dict:
        for key, child in value.items():
            _finite(child, f"{path}.{key}", depth + 1)
    elif type(value) is builtins.list:
        for index, child in enumerate(value):
            _finite(child, f"{path}[{index}]", depth + 1)
    elif type(value) is float and not math.isfinite(value):
        _fail(path, "expected finite JSON")


def parse(data: bytes) -> Recipe:
    """Validate only the shared envelope/source/path grammar, never model policy."""
    if type(data) is not bytes or len(data) > MAX_BYTES:
        _fail("document", "expected at most 1 MiB of UTF-8 JSON bytes")
    try:
        value = json.loads(data.decode("utf-8"), object_pairs_hook=_unique,
                           parse_constant=lambda token: _fail("document", f"nonfinite value {token}"))
        _finite(value, "document")
    except (ValueError, UnicodeError, RecursionError) as error:
        raise HemError("Invalid recipe document: malformed UTF-8 JSON or excessive nesting") from error
    _object(value, ("format", "id", "name", "modality", "pipeline", "artifacts", "configuration"), "document")
    if type(value["format"]) is not int or value["format"] != 1:
        _fail("format", "expected integer 1")
    identifier = _name(value["id"], "id")
    pipeline = _name(value["pipeline"], "pipeline")
    if type(value["name"]) is not str or not value["name"].strip():
        _fail("name", "expected nonempty string")
    if type(value["modality"]) is not str or value["modality"] not in _MODALITIES:
        _fail("modality", "expected chat or asr")
    if type(value["configuration"]) is not dict:
        _fail("configuration", "expected an object")
    if type(value["artifacts"]) is not builtins.list or not value["artifacts"]:
        _fail("artifacts", "expected a nonempty ordered array")
    artifacts, ids, paths, coordinates = [], set(), [], set()
    for index, item in enumerate(value["artifacts"]):
        where = f"artifacts[{index}]"
        _object(item, ("id", "role", "format", "path", "source"), where)
        aid = _name(item["id"], where + ".id")
        role = _name(item["role"], where + ".role")
        if aid in ids:
            _fail(where + ".id", "duplicate artifact ID")
        ids.add(aid)
        if type(item["format"]) is not str or item["format"] not in _FORMATS:
            _fail(where + ".format", "unsupported artifact format")
        path = _relative(item["path"], where + ".path")
        parts = PurePosixPath(path)
        if any(parts == other or parts in other.parents or other in parts.parents for other in paths):
            _fail(where + ".path", "duplicate or overlapping artifact destination")
        paths.append(parts)
        source = item["source"]
        if type(source) is not dict:
            _fail(where + ".source", "expected an object")
        if source.get("kind") == "huggingface":
            _object(source, ("kind", "repo_id", "revision", "filename"), where + ".source")
            repo = _relative(source["repo_id"], where + ".source.repo_id", locator=True)
            filename = _relative(source["filename"], where + ".source.filename", locator=True)
            revision = source["revision"]
            if type(revision) is not str or not _REVISION.fullmatch(revision):
                _fail(where + ".source.revision", "expected immutable lowercase 40-hex revision")
            coordinate = repo, revision, filename
            if coordinate in coordinates:
                _fail(where + ".source", "duplicate HF coordinates")
            coordinates.add(coordinate)
            selected_source = HFSource(repo, revision, filename)
        elif source.get("kind") == "generated":
            _object(source, ("kind",), where + ".source")
            selected_source = GeneratedSource()
        else:
            _fail(where + ".source.kind", "expected huggingface or generated")
        artifacts.append(Artifact(aid, role, item["format"], path, selected_source))
    return Recipe(1, identifier, value["name"], value["modality"], pipeline,
                  tuple(artifacts), value["configuration"], hashlib.sha256(data).hexdigest())


def _paths(identifier=None):
    for root in pipelines.catalog_roots():
        if root.is_dir():
            pattern = "**/recipes/*.json" if identifier is None else f"**/recipes/{identifier}.json"
            paths = tuple(sorted(root.glob(pattern)))
            for path in paths:
                try:
                    path.resolve(strict=True).relative_to(root.resolve(strict=True))
                except (ValueError, OSError, RuntimeError) as error:
                    raise HemError("Packaged recipe escapes its catalog root") from error
            return paths
    raise HemError("No packaged recipe catalog is available")


def _read(path):
    try:
        with path.open("rb") as stream:
            return parse(stream.read(MAX_BYTES + 1))
    except OSError as error:
        raise HemError(f"Cannot read packaged recipe {path.name!r}") from error


def _binding(recipe):
    binding = pipelines.get(recipe.pipeline)
    if binding.describe().modality != recipe.modality:
        _fail(recipe.id, "recipe and selected pipeline modalities differ")
    binding.validate_artifacts(recipe.artifacts)


def load(recipe_id: str) -> Recipe:
    """Resolve only the selected declaration/binding, never unrelated recipes."""
    _name(recipe_id, "id")
    paths = _paths(recipe_id)
    if not paths:
        raise HemError(f"Unknown recipe {recipe_id!r}")
    if len(paths) != 1:
        raise HemError(f"Duplicate packaged recipe ID {recipe_id!r}")
    recipe = _read(paths[0])
    if recipe.id != recipe_id:
        _fail(recipe_id, "filename and declared ID differ")
    _binding(recipe)
    return recipe


def list(modality=None) -> tuple[Recipe, ...]:
    """Discover authored recipes in ID order, rejecting broken support entries."""
    if modality is not None and (type(modality) is not str or modality not in _MODALITIES):
        _fail("modality", "expected chat, asr or omitted")
    result, ids = [], set()
    for path in _paths():
        recipe = _read(path)
        if recipe.id in ids:
            raise HemError(f"Duplicate packaged recipe ID {recipe.id!r}")
        ids.add(recipe.id)
        if path.name != recipe.id + ".json":
            _fail(recipe.id, "filename and declared ID differ")
        _binding(recipe)
        if modality is None or recipe.modality == modality:
            result.append(recipe)
    return tuple(sorted(result, key=lambda recipe: recipe.id))
