"""One authored recipe grammar and packaged lookup, without acquisition effects."""
from __future__ import annotations

import builtins
import hashlib
import json
import math
from pathlib import PurePosixPath, PureWindowsPath
import re

from ..hem_common import HemError

MAX_BYTES = 1024 * 1024
MAX_DEPTH = 64
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
_REVISION = re.compile(r"[0-9a-f]{40}\Z")


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


def _decode(data: bytes):
    """Bounded strict JSON shared by recipe and executable declarations."""
    if type(data) is not bytes or len(data) > MAX_BYTES:
        _fail("document", "expected at most 1 MiB of UTF-8 JSON bytes")
    try:
        value = json.loads(data.decode("utf-8"), object_pairs_hook=_unique,
                           parse_constant=lambda token: _fail("document", f"nonfinite value {token}"))
        _finite(value, "document")
    except (ValueError, UnicodeError, RecursionError) as error:
        raise HemError("Invalid recipe document: malformed UTF-8 JSON or excessive nesting") from error
    return value, hashlib.sha256(data).hexdigest()


def parse(data: bytes) -> tuple[dict, str]:
    """Read authored format3 without imports, acquisition or implementation policy."""
    value, identity = _decode(data)
    _object(value, ("format", "id", "name", "executable", "files", "configuration"), "document")
    if type(value["format"]) is not int or value["format"] != 3:
        _fail("format", "expected integer 3")
    _name(value["id"], "id")
    _relative(value["executable"], "executable", locator=True)
    if not value["executable"].endswith("/executable.json"):
        _fail("executable", "expected a direct catalog-relative executable.json reference")
    if type(value["name"]) is not str or not value["name"].strip():
        _fail("name", "expected nonempty string")
    if type(value["configuration"]) is not dict:
        _fail("configuration", "expected an object")
    if type(value["files"]) is not dict or not value["files"]:
        _fail("files", "expected a nonempty purpose-keyed object")
    paths, coordinates = [], set()
    for purpose, item in value["files"].items():
        _name(purpose, "files purpose")
        where = f"files.{purpose}"
        _object(item, ("path", "source"), where)
        path = _relative(item["path"], where + ".path")
        parts = PurePosixPath(path)
        if any(parts == other or parts in other.parents or other in parts.parents for other in paths):
            _fail(where + ".path", "duplicate or overlapping file destination")
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
        elif source.get("kind") == "generated":
            _object(source, ("kind",), where + ".source")
        else:
            _fail(where + ".source.kind", "expected huggingface or generated")
    return value, identity


def _paths(identifier=None, *, executable=None):
    from . import pipelines
    if executable is None:
        root = pipelines._root()
        pattern = "**/recipes/*.json" if identifier is None else f"**/recipes/{identifier}.json"
    else:
        manifest, _ = pipelines._manifest(executable)
        root = manifest.parent
        pattern = "recipes/*.json"
    paths = tuple(sorted(root.glob(pattern)))
    for path in paths:
        pipelines._inside(path, root)
    return paths


def _read_bytes(path):
    try:
        if not path.is_file():
            raise OSError("declaration must be a regular file")
        with path.open("rb") as stream:
            return stream.read(MAX_BYTES + 1)
    except OSError as error:
        raise HemError(f"Cannot read packaged declaration {path.name!r}") from error


def _read(path):
    return parse(_read_bytes(path))


def _admit(document, path):
    from . import pipelines
    expected = (path.parent.parent.relative_to(pipelines._root()) / 'executable.json').as_posix()
    if document['executable'] != expected:
        _fail(document['id'], 'recipe location differs from its executable catalog')
    executable, _, _ = pipelines.load(document["executable"])
    pipelines.validate_files(executable, document["files"])


def load(recipe_id: str) -> tuple[dict, str]:
    """Read/admit selected recipe and direct executable, never unrelated metadata."""
    _name(recipe_id, "id")
    paths = _paths(recipe_id)
    if not paths:
        raise HemError(f"Unknown recipe {recipe_id!r}")
    if len(paths) != 1:
        raise HemError(f"Duplicate packaged recipe ID {recipe_id!r}")
    document, identity = _read(paths[0])
    if document["id"] != recipe_id:
        _fail(recipe_id, "filename and declared ID differ")
    _admit(document, paths[0])
    return document, identity


def list(executable=None) -> tuple[tuple[dict, str], ...]:
    """Selected catalog recipes or strict global inventory, sorted by recipe ID."""
    if executable is not None:
        from . import pipelines
        pipelines.load(executable)
    result, ids = [], set()
    for path in _paths(executable=executable):
        document, identity = _read(path)
        if document["id"] in ids:
            raise HemError(f"Duplicate packaged recipe ID {document['id']!r}")
        ids.add(document["id"])
        if path.name != document["id"] + ".json":
            _fail(document["id"], "filename and declared ID differ")
        if executable is not None and document["executable"] != executable:
            _fail(document["id"], "declaration differs from selected catalog executable")
        _admit(document, path)
        result.append((document, identity))
    return tuple(sorted(result, key=lambda item: item[0]["id"]))
