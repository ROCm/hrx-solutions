"""Required bytecode catalog generation and staging (independent of native ELF)."""
from __future__ import annotations

import argparse
import hashlib
import json
import ast
from pathlib import Path
import posixpath
import re
import shutil
import subprocess
import tempfile

ARCHITECTURES = ("qwen_27b", "qwen_08b")
CATALOG_TARGET = "//loom-pipeline:bytecode_catalogs"
DECODER_TARGET = "@hrx_system//loom/src/loom/tools/loom-format"
ROLES = ("prepare.loom", "control.loom", "weights.loom")


def variants(inventory: Path) -> dict:
    """Read the single explicit variant literal, without executing Starlark."""
    tree = ast.parse(inventory.read_text(encoding="utf-8"))
    if len(tree.body) != 1:
        raise ValueError("Catalog inventory must define only CATALOG_VARIANTS")
    statement = tree.body[0]
    if (not isinstance(statement, ast.Assign) or len(statement.targets) != 1
            or not isinstance(statement.targets[0], ast.Name)
            or statement.targets[0].id != "CATALOG_VARIANTS"):
        raise ValueError("Catalog inventory must contain only the literal CATALOG_VARIANTS assignment")
    # literal_eval silently overwrites duplicate dictionary keys. Reject them
    # before evaluation so the checked contract has no hidden declarations.
    try:
        for node in ast.walk(statement.value):
            if isinstance(node, ast.Dict):
                keys = [ast.literal_eval(key) for key in node.keys]
                if len(keys) != len(set(keys)):
                    raise ValueError("Duplicate catalog inventory key")
        value = ast.literal_eval(statement.value)
    except TypeError as error:
        raise ValueError("Malformed catalog inventory literal") from error
    if not isinstance(value, dict) or set(value) != set(ARCHITECTURES):
        raise ValueError("Catalog must declare exactly the supported variants")
    paths = []
    for architecture, variant in value.items():
        if (not isinstance(variant, dict)
                or set(variant) != {"identifier", "sources", "output"}
                or not isinstance(variant["identifier"], str) or not variant["identifier"]
                or variant["output"] != f"qwen/{architecture}/pipeline.loombc"):
            raise ValueError(f"Invalid catalog variant contract: {architecture}")
        sources = variant["sources"]
        if not isinstance(sources, list) or not all(isinstance(p, str) for p in sources):
            raise ValueError("Catalog inventory must be a list of source paths")
        if len(sources) != len(set(sources)):
            raise ValueError("Duplicate source identity in catalog inventory")
        paths.extend(sources)
    for path in paths:
        if (posixpath.normpath(path) != path or path.startswith(("/", "../"))
                or "\\" in path or not path.endswith(".loom")):
            raise ValueError(f"Invalid catalog source identity: {path}")
    return value


def catalog_sources(inventory: Path) -> tuple[str, ...]:
    """Derive aggregate inputs in variant declaration and source-list order."""
    return tuple(dict.fromkeys(source for variant in variants(inventory).values()
                               for source in variant["sources"]))


def manifest_entries(manifest: Path, source_root: Path) -> tuple[str, ...]:
    """Validate authored closure, preserving manifest order and symlink identity."""
    entries = tuple(line.strip() for line in manifest.read_text(encoding="utf-8").splitlines()
                    if line.strip() and not line.strip().startswith("#"))
    if len(entries) != len(set(entries)):
        raise ValueError(f"Duplicate manifest entry: {manifest}")
    normalized = []
    for entry in entries:
        if Path(entry).is_absolute() or "\\" in entry or not entry.endswith(".loom"):
            raise ValueError(f"Invalid source manifest entry: {entry}")
        identity = posixpath.normpath((manifest.parent.relative_to(source_root) / entry).as_posix())
        if identity.startswith("../"):
            raise ValueError(f"Source manifest escapes catalog: {entry}")
        source = source_root / identity
        source.resolve(strict=True).relative_to(source_root.resolve())
        if not source.is_file():
            raise ValueError(f"Source manifest entry is not a file: {entry}")
        normalized.append(identity)
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"Duplicate normalized source identity: {manifest}")
    for role in (*ROLES, "config.loom"):
        if [entry for entry in entries if Path(entry).name == role] != [role]:
            raise ValueError(f"Expected exactly one local {role} role: {manifest}")
    return entries


def validate_catalog(inventory: Path) -> tuple[str, ...]:
    source_root = inventory.resolve().parent / "llm"
    expected = catalog_sources(inventory)
    actual = set()
    for architecture, variant in variants(inventory).items():
        relative = Path("qwen") / architecture
        manifest = source_root / relative / "sources.txt"
        entries = manifest_entries(manifest, source_root)
        closure = [posixpath.normpath((relative / entry).as_posix()) for entry in entries]
        if (set(variant) != {"identifier", "sources", "output"}
                or variant["sources"] != closure
                or variant["output"] != (relative / "pipeline.loombc").as_posix()):
            raise ValueError(f"Variant inventory/manifest drift: {architecture}")
        actual.update(closure)
    if actual != set(expected):
        raise ValueError(f"Catalog inventory/manifest drift: missing={sorted(actual-set(expected))}, "
                         f"extra={sorted(set(expected)-actual)}")
    # Include internal symlink targets, but reject executable additions omitted
    # from both manifests. Bazel's input glob makes additions invalidate the gate.
    allowed = actual | {(source_root / p).resolve().relative_to(source_root.resolve()).as_posix()
                        for p in actual}
    # Setuptools materializes source symlinks in sdists. Keep the explicitly
    # authored shared control backing file allowed even after those flatten.
    allowed.add("qwen/control.loom")
    authored = {p.relative_to(source_root).as_posix() for p in source_root.rglob("*.loom")}
    if authored != allowed:
        raise ValueError(f"Undeclared authored catalog sources: {sorted(authored ^ allowed)}")
    return expected


def preparation_bytecode_source(source: str, policy: str = "pipeline.loombc") -> str:
    """Change precisely the known policy locator, rejecting symbol/value drift."""
    old = 'global.rodata.def @policy = bytes("' + b"weights.loom".hex() + '")'
    new = 'global.rodata.def @policy = bytes("' + policy.encode().hex() + '")'
    declarations = re.findall(r"^global\.rodata\.def\s+@policy\s*=.*$", source, re.MULTILINE)
    if declarations != [old]:
        raise ValueError("Preparation must define exactly one expected @policy weights.loom locator")
    return source.replace(old, new, 1)


def _is_bytecode(path: Path) -> bool:
    with path.open("rb") as stream:
        return stream.read(4) == b"LOOM"


def discover_bytecode_outputs(paths: list[Path], inventory: Path, repository: Path) -> dict[str, Path]:
    """Map cquery-discovered artifacts to declared catalog identities."""
    validate_catalog(inventory)
    declared = variants(inventory)
    suffixes = {"loom-pipeline/bytecode/" + v["output"]: arch for arch, v in declared.items()}
    outputs = {}
    for path in paths:
        matches = [arch for suffix, arch in suffixes.items()
                   if path.as_posix().endswith("/" + suffix) or path.as_posix() == suffix]
        if len(matches) != 1 or matches[0] in outputs:
            raise ValueError(f"Unexpected or duplicate Bazel bytecode output: {path}")
        arch = matches[0]
        output = path if path.is_absolute() else repository / path
        if not output.is_file() or not _is_bytecode(output):
            raise ValueError(f"Missing or invalid required bytecode output: {output}")
        outputs[arch] = output.resolve()
    if set(outputs) != set(declared):
        raise ValueError(f"Missing required bytecode outputs: {sorted(set(declared)-set(outputs))}")
    return outputs


def query_bytecode_outputs(query, repository: Path) -> dict[str, Path]:
    """Interpret the build-bound query using Qwen's declared catalog identities."""
    return discover_bytecode_outputs(query(CATALOG_TARGET),
                                     repository / "loom-pipeline/catalog_sources.bzl", repository)


def query_decoder(query) -> Path:
    paths = query(DECODER_TARGET)
    if len(paths) != 1:
        raise ValueError("Expected exactly one pinned Loom decoder executable")
    return paths[0].resolve(strict=True)


def validate_bytecode_artifact(artifact: Path, variant: dict, root: Path, decoder: Path) -> None:
    """Decode and prove full authored symbol/config closure before blessing bytes."""
    with tempfile.TemporaryDirectory() as temporary:
        decoded = Path(temporary) / "module.loom"
        subprocess.run([str(decoder), str(artifact), "--from=bc", "--to=text", "--output=" + str(decoded)],
                       check=True)
        text = decoded.read_text(encoding="utf-8")
    inputs = [(root / source).read_text(encoding="utf-8") for source in variant["sources"]]
    symbols = lambda value: set(re.findall(r"^func\.def public @([^\s(]+)", value, re.MULTILINE))
    expected = set().union(*(symbols(source) for source in inputs))
    configs = lambda value: set(re.findall(r"^config\.(?:def|decl) @([^\s=:(]+)", value, re.MULTILINE))
    configuration = set().union(*(configs(source) for source in inputs))
    policy = 'global.rodata.def @policy = bytes("' + b"pipeline.loombc".hex() + '")'
    if (symbols(text) != expected or not configuration <= configs(text)
            or sum(len(re.findall(r"^func\.def ", source, re.MULTILINE)) for source in inputs)
                != len(re.findall(r"^func\.def ", text, re.MULTILINE))
            or text.count(policy) != 1):
        raise ValueError(f"Bytecode does not preserve complete variant roots/configuration/policy: {artifact}")


def stage_bytecode_catalog(destination: Path, outputs: dict[str, Path], inventory: Path, *, decoder: Path) -> list[Path]:
    """Stage an explicit runtime allowlist, never copy/prune authored source."""
    validate_catalog(inventory)
    declared = variants(inventory)
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    if set(outputs) != set(declared):
        raise ValueError("Bytecode output inventory does not match declared variants")
    build = {}
    root = inventory.resolve().parent / "llm"
    for arch, variant in declared.items():
        artifact = outputs[arch]
        if not artifact.is_file() or not _is_bytecode(artifact):
            raise ValueError(f"Missing or invalid required bytecode output: {artifact}")
        validate_bytecode_artifact(artifact, variant, root, decoder)
        target = destination / variant["output"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(artifact, target)
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        descriptor = {"format": 1, "variant": variant["identifier"],
                      "module": "pipeline.loombc", "sha256": digest}
        (target.parent / "pipeline.json").write_text(json.dumps(descriptor, sort_keys=True, indent=2)+"\n")
        recipes = sorted((root / "qwen" / arch / "recipes").glob("*.json"))
        if not recipes:
            raise ValueError(f"Missing authored recipes: {arch}")
        (target.parent / "recipes").mkdir()
        for recipe in recipes:
            recipe.resolve(strict=True).relative_to(root.resolve())
            shutil.copyfile(recipe, target.parent / "recipes" / recipe.name)
        build[arch] = {"identifier": variant["identifier"], "bytecode_sha256": digest,
                       "sources": {p: hashlib.sha256((root / p).read_bytes()).hexdigest()
                                   for p in variant["sources"]}}
    (destination / "catalog-build.json").write_text(
        json.dumps({"format": 2, "variants": build}, sort_keys=True, indent=2)+"\n")
    shutil.copyfile(root / "LICENSE-APACHE-2.0", destination / "LICENSE-APACHE-2.0")
    return sorted(path for path in destination.rglob("*") if path.is_file())


def stage_package(package_root, outputs, repository, *, decoder):
    """Own Qwen's installed catalog destination and authored inventory binding."""
    return stage_bytecode_catalog(package_root / "loom-pipeline/llm", outputs,
                                  repository / "loom-pipeline/catalog_sources.bzl", decoder=decoder)


def validate_archive(archive, repository, *, prefix='hrx_solutions/loom-pipeline/llm/'):
    """Prove Qwen archive layout, authored closure and exact payload metadata."""
    names = set(archive.namelist())
    for name in names:
        if name.startswith(prefix) and archive.namelist().count(name) != 1:
            raise RuntimeError(f"wheel duplicates catalog member: {name}")
    inventory = repository / "loom-pipeline/catalog_sources.bzl"
    validate_catalog(inventory)
    declared = variants(inventory)
    build_name = prefix + "catalog-build.json"
    if build_name not in names:
        raise RuntimeError("wheel is missing catalog build hashes")
    build = json.loads(archive.read(build_name))
    if build.get("format") != 2 or set(build.get("variants", {})) != set(declared):
        raise RuntimeError("wheel catalog build inventory does not match authored variants")
    allowed = {build_name, prefix + "LICENSE-APACHE-2.0"}
    root = inventory.parent / "llm"
    if archive.read(prefix + "LICENSE-APACHE-2.0") != (root / "LICENSE-APACHE-2.0").read_bytes():
        raise RuntimeError("wheel catalog license does not match authored license")
    for architecture, variant in declared.items():
        relative = f"qwen/{architecture}/"
        name = prefix + variant["output"]
        content = archive.read(name)
        digest = hashlib.sha256(content).hexdigest()
        expected = {
            "identifier": variant["identifier"],
            "sources": {source: hashlib.sha256((root / source).read_bytes()).hexdigest()
                        for source in variant["sources"]},
            "bytecode_sha256": digest,
        }
        if not content.startswith(b"LOOM") or build["variants"][architecture] != expected:
            raise RuntimeError(f"wheel catalog source/bytecode hash mismatch: {name}")
        descriptor = prefix + relative + "pipeline.json"
        if json.loads(archive.read(descriptor)) != {
                "format": 1, "variant": variant["identifier"],
                "module": "pipeline.loombc", "sha256": digest}:
            raise RuntimeError(f"wheel catalog descriptor mismatch: {descriptor}")
        allowed.update({name, descriptor})
        for recipe in sorted((root / relative / "recipes").glob("*.json")):
            member = prefix + relative + "recipes/" + recipe.name
            if archive.read(member) != recipe.read_bytes():
                raise RuntimeError(f"wheel authored recipe mismatch: {member}")
            allowed.add(member)
    actual = {name for name in names if name.startswith(prefix) and not name.endswith("/")}
    if actual != allowed:
        raise RuntimeError(f"wheel catalog file inventory mismatch: extra={sorted(actual-allowed)}, missing={sorted(allowed-actual)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate")
    validate.add_argument("--inventory", type=Path, required=True)
    validate.add_argument("--stamp", type=Path, required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--source", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--policy", choices=("pipeline.loombc", "weights.loombc"), default="pipeline.loombc")
    args = parser.parse_args()
    if args.command == "validate":
        validate_catalog(args.inventory)
        args.stamp.write_text("validated\n", encoding="utf-8")
    else:
        args.output.write_text(preparation_bytecode_source(args.source.read_text(encoding="utf-8"), args.policy), encoding="utf-8")


if __name__ == "__main__":
    main()
