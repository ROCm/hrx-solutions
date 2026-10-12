"""Command line setup workflow for local Loom Serve models."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import NoReturn, Optional, Sequence

from . import model_store, pipeline_validation
from ._launcher import native_environment
from .pipeline_validation import ModelFiles


def _model_path(model_id: str, model_root: Optional[Path]) -> Path:
    root = model_root.expanduser() if model_root else model_store.DEFAULT_MODEL_ROOT
    manifest = model_store.read_manifest(model_id, root)
    path = root / model_id
    if not path.is_dir():
        raise model_store.ModelImportError(f"Model is not installed: {model_id} (looked in {path})")
    if manifest.get("install_mode") == "alias":
        for name, entry in manifest.get("files", {}).items():
            linked_path = path / entry.get("path", "")
            if not linked_path.is_symlink() or not linked_path.is_file():
                target = entry.get("link_target", "unknown target")
                raise model_store.ModelImportError(
                    f"Model alias {name} is unavailable: {linked_path} -> {target}; "
                    "restore the source file or reimport the model"
                )
    return path


def _model_files(model_id: str, model_root: Optional[Path]) -> ModelFiles:
    directory = _model_path(model_id, model_root)
    return ModelFiles(directory / "model.gguf", directory / "tokenizer.json")


def _catalog(pipeline: str) -> Path:
    """Resolve the selected pipeline catalog in source or installed layouts."""
    package = Path(__file__).resolve().parent
    native = package / "_native" / "share" / "loom-serve" / "experimental" / "loom_serve" / "models" / pipeline
    candidates = (
        package / "loom-pipeline" / "llm" / pipeline,
        native,
        package.parents[1] / "loom-pipeline" / "llm" / pipeline,
    )
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()
    raise FileNotFoundError(
        f"No source catalog is installed for pipeline {pipeline!r}; searched: "
        + ", ".join(str(path) for path in candidates)
    )


def _cmd_list(args: argparse.Namespace) -> int:
    root = args.model_root.expanduser() if args.model_root else model_store.DEFAULT_MODEL_ROOT
    if not root.is_dir():
        print(f"No models installed in {root}")
        return 0
    found = False
    for directory in sorted(root.iterdir()):
        if not directory.is_dir() or directory.name.startswith("."):
            continue
        found = True
        try:
            manifest = model_store.read_manifest(directory.name, root)
            files = manifest.get("files", {})
            gguf_ok = (directory / "model.gguf").is_file()
            tokenizer_ok = (directory / "tokenizer.json").is_file()
            if manifest.get("install_mode") == "alias":
                invalid_links = [
                    name
                    for name, entry in files.items()
                    if not (directory / entry.get("path", "")).is_symlink()
                    or not (directory / entry.get("path", "")).is_file()
                ]
                state = (
                    "broken alias (" + ", ".join(invalid_links) + ")"
                    if invalid_links
                    else "ready (alias)"
                )
            else:
                state = "ready" if gguf_ok and tokenizer_ok else "incomplete"
            source = files.get("gguf", {}).get("source", {})
            if source.get("kind") == "huggingface":
                source_label = f"hf:{source.get('repo_id', '?')}:{source.get('filename', '?')}"
            elif source.get("kind") == "local":
                source_label = str(source.get("path", "unknown"))
            else:
                source_label = "unknown"
            print(
                f"{directory.name}\tpipeline={manifest.get('pipeline', 'unvalidated')}"
                f"\tsource={source_label}\tmode={manifest.get('install_mode', 'unknown')}"
                f"\tavailability={state}"
            )
        except model_store.ModelImportError as exc:
            print(f"{directory.name}\tinvalid\t{exc}")
    if not found:
        print(f"No models installed in {root}")
    return 0


def _cmd_import(args: argparse.Namespace) -> int:
    if args.hf_repo:
        if not args.gguf_file or not args.tokenizer_file:
            raise model_store.ModelImportError(
                "Hugging Face imports require both --gguf-file and --tokenizer-file"
            )
        if args.tokenizer:
            raise model_store.ModelImportError(
                "Use --tokenizer-file for Hugging Face imports or --tokenizer for local imports"
            )
        revision = args.revision or "main"
        gguf = model_store.HuggingFaceFile(args.hf_repo, args.gguf_file, revision)
        tokenizer = model_store.HuggingFaceFile(
            args.tokenizer_hf_repo or args.hf_repo,
            args.tokenizer_file,
            args.tokenizer_revision or revision,
        )
    else:
        if args.gguf_file or args.tokenizer_file or args.tokenizer_hf_repo or args.tokenizer_revision:
            raise model_store.ModelImportError(
                "Hugging Face file and revision options are only valid with --hf-repo"
            )
        if not args.tokenizer:
            raise model_store.ModelImportError("Local imports require --tokenizer")
        gguf = Path(args.gguf).expanduser()
        tokenizer = Path(args.tokenizer).expanduser()
    imported = model_store.import_model(
        args.model_id,
        gguf=gguf,
        tokenizer=tokenizer,
        model_root=args.model_root,
        token=args.token,
        alias=args.alias,
    )
    print(f"Imported {imported.model_id} into {imported.directory}")
    print(f"  weights:   {imported.gguf_path}")
    print(f"  tokenizer: {imported.tokenizer_path}")
    return 0


def _cmd_pipelines(_args: argparse.Namespace) -> int:
    pipelines = pipeline_validation.available_pipelines()
    if not pipelines:
        print("No pipeline compatibility adapters are installed")
        return 0
    print("\n".join(pipelines))
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    try:
        files = _model_files(args.model_id, args.model_root)
    except model_store.ModelImportError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    result = pipeline_validation.validate_model(
        files, args.pipeline, checker=args.checker, timeout=args.timeout
    )
    print(result.message)
    if result.stdout:
        print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
    if result.stderr:
        print(result.stderr, file=sys.stderr, end="" if result.stderr.endswith("\n") else "\n")
    if result.compatible:
        model_store.record_pipeline(args.model_id, args.pipeline, args.model_root)
    return 0 if result.compatible else 1


def _cmd_serve(args: argparse.Namespace) -> NoReturn:
    manifest = model_store.read_manifest(args.model_id, args.model_root)
    pipeline = args.pipeline or manifest.get("pipeline")
    if not pipeline:
        raise model_store.ModelImportError(
            f"Model {args.model_id!r} has no validated pipeline recorded. Run "
            f"'loom-serve-setup models validate {args.model_id} --pipeline <pipeline>' "
            "or pass --pipeline explicitly."
        )
    files = _model_files(args.model_id, args.model_root)
    catalog = pipeline_validation.catalog_for_model(pipeline, _catalog(pipeline), args.model_id)
    package = Path(__file__).resolve().parent
    executable = package / "_native" / "bin" / "loom-text-server"
    if not executable.is_file():
        raise FileNotFoundError(f"Loom Serve binary is missing: {executable}")
    argv = [
        str(executable),
        f"--model={catalog}",
        f"--weights={files.weights.resolve()}",
        f"--tokenizer={files.tokenizer.resolve()}",
        f"--rows={args.rows}",
        f"--context_capacity={args.context_capacity}",
        f"--pool_capacity={args.pool_capacity}",
        f"--prefill_capacity={args.prefill_capacity}",
        f"--port={args.port}",
        "--heartbeat_ms=1000",
    ]
    if not args.no_mtp:
        argv.extend(("--mtp", f"--mtp_depth={args.mtp_depth}"))
    os.execve(str(executable), argv, native_environment())


def _cmd_pi_config(args: argparse.Namespace) -> int:
    # Refuse to point Pi at a model that has not been imported into Loom's
    # model store. This also catches broken aliases before changing user config.
    files = _model_files(args.model_id, args.model_root)
    for label, path in (("GGUF", files.weights), ("tokenizer", files.tokenizer)):
        if not path.is_file():
            raise model_store.ModelImportError(
                f"Installed model {args.model_id!r} is missing its {label}: {path}"
            )

    output = args.output.expanduser()
    base_url = args.base_url.rstrip("/")
    if not base_url.endswith("/v1"):
        base_url += "/v1"
    expected_model = {
        "id": args.model_id,
        "name": args.display_name or args.model_id,
        "contextWindow": args.context_window,
        "maxTokens": args.max_tokens,
        "input": ["text"],
    }
    if output.exists():
        try:
            config = json.loads(output.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Cannot read Pi configuration {output}: {exc}") from exc
        if not isinstance(config, dict):
            raise ValueError(f"Malformed Pi configuration {output}: expected a JSON object")
    else:
        config = {}
    configured_providers = config.get("providers", {})
    if not isinstance(configured_providers, dict):
        raise ValueError(f"Malformed Pi configuration {output}: 'providers' must be an object")
    # Attach an initially empty provider map so a previously absent "providers"
    # key is persisted when configuring the first Loom model.
    providers = config.setdefault("providers", {})
    provider = providers.get(args.provider)
    if provider is not None and not isinstance(provider, dict):
        raise ValueError(
            f"Malformed Pi configuration {output}: provider {args.provider!r} must be an object"
        )
    if isinstance(provider, dict) and "models" in provider and not isinstance(provider["models"], list):
        raise ValueError(
            f"Malformed Pi configuration {output}: provider {args.provider!r} 'models' must be an array"
        )

    if args.check:
        configured_model = None
        if isinstance(provider, dict) and isinstance(provider.get("models"), list):
            matches = [
                model for model in provider["models"]
                if isinstance(model, dict) and model.get("id") == args.model_id
            ]
            if len(matches) == 1:
                configured_model = matches[0]
        if (
            isinstance(provider, dict)
            and provider.get("baseUrl", "").rstrip("/") == base_url
            and provider.get("api") == "openai-completions"
            and configured_model is not None
        ):
            print(f"Pi is configured for {args.model_id!r} at {base_url} ({output})")
            return 0
        print(
            f"Pi is not configured for model {args.model_id!r} at {base_url} "
            f"under provider {args.provider!r} ({output})",
            file=sys.stderr,
        )
        return 1

    provider = providers.setdefault(args.provider, {})
    models = provider.setdefault("models", [])
    if not isinstance(models, list):
        raise ValueError(
            f"Malformed Pi configuration {output}: provider {args.provider!r} 'models' must be an array"
        )
    matches = [
        model for model in models
        if isinstance(model, dict) and model.get("id") == args.model_id
    ]
    if len(matches) > 1:
        raise ValueError(
            f"Malformed Pi configuration {output}: provider {args.provider!r} has "
            f"multiple entries for model {args.model_id!r}"
        )
    if matches:
        # Retain Pi-specific extension fields the user may have added.
        matches[0].update(expected_model)
    else:
        models.append(expected_model)
    provider.update({
        "name": provider.get("name") or "HRX Loom",
        "baseUrl": base_url,
        "api": "openai-completions",
        "apiKey": provider.get("apiKey") or "local",
    })

    output.parent.mkdir(parents=True, exist_ok=True)
    original_mode = output.stat().st_mode & 0o777 if output.exists() else None
    temporary_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=output.parent,
            prefix=f".{output.name}.", suffix=".tmp", delete=False
        ) as stream:
            temporary_path = Path(stream.name)
            stream.write(json.dumps(config, indent=2) + "\n")
        if original_mode is not None:
            temporary_path.chmod(original_mode)
        os.replace(temporary_path, output)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    print(f"Configured Pi for Loom model {args.model_id!r}: {output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="loom-serve-setup")
    subparsers = parser.add_subparsers(dest="command", required=True)

    models = subparsers.add_parser("models", help="manage locally installed models")
    model_commands = models.add_subparsers(dest="models_command", required=True)
    listing = model_commands.add_parser("list", help="list installed models")
    listing.add_argument("--model-root", type=Path)
    listing.set_defaults(handler=_cmd_list)

    model_commands.add_parser("pipelines", help="list registered compatibility pipelines").set_defaults(
        handler=_cmd_pipelines
    )

    importing = model_commands.add_parser("import", help="import local files or files downloaded by the Hugging Face CLI")
    importing.add_argument("model_id")
    source = importing.add_mutually_exclusive_group(required=True)
    source.add_argument("--hf-repo", help="Hugging Face repository containing both files (requires the `hf` CLI)")
    source.add_argument("--gguf", help="local GGUF file; use with --tokenizer")
    importing.add_argument("--gguf-file", help="GGUF path within the Hugging Face repository")
    importing.add_argument("--tokenizer-file", help="tokenizer path within the Hugging Face repository")
    importing.add_argument("--tokenizer-hf-repo", help="Hugging Face repository for the tokenizer (defaults to --hf-repo)")
    importing.add_argument("--revision", default="main")
    importing.add_argument("--tokenizer-revision", help="tokenizer revision (defaults to --revision)")
    importing.add_argument("--token", help="Hugging Face access token passed to `hf download` (or use HF_TOKEN)")
    importing.add_argument("--tokenizer", help="local tokenizer.json file")
    importing.add_argument(
        "--alias",
        action="store_true",
        help="symlink local files or the Hugging Face CLI cache paths instead of copying",
    )
    importing.add_argument("--model-root", type=Path)
    importing.set_defaults(handler=_cmd_import)

    validating = model_commands.add_parser("validate", help="check model compatibility with a pipeline")
    validating.add_argument("model_id")
    validating.add_argument("--pipeline", required=True)
    validating.add_argument("--checker", help="optional pipeline checker executable")
    validating.add_argument("--timeout", type=float, default=1800)
    validating.add_argument("--model-root", type=Path)
    validating.set_defaults(handler=_cmd_validate)

    serve = subparsers.add_parser("serve", help="launch Loom Serve for an installed model")
    serve.add_argument("model_id")
    serve.add_argument("--pipeline", help="override the model's recorded pipeline/catalog")
    serve.add_argument("--model-root", type=Path)
    serve.add_argument("--port", type=int, default=8080)
    serve.add_argument("--rows", type=int, default=4)
    serve.add_argument("--context-capacity", type=int, default=8192)
    serve.add_argument("--pool-capacity", type=int, default=32768)
    serve.add_argument("--prefill-capacity", type=int, default=128)
    serve.add_argument("--no-mtp", action="store_true", help="disable multi-token prediction")
    serve.add_argument("--mtp-depth", type=int, default=3)
    serve.set_defaults(handler=_cmd_serve)

    pi = subparsers.add_parser("pi-config", help="configure or verify Pi for an installed Loom model")
    pi.add_argument("model_id", help="ID of a model already installed in the Loom model store")
    pi.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    pi.add_argument("--provider", default="hrx-loom")
    pi.add_argument("--display-name")
    pi.add_argument("--context-window", type=int, default=8192)
    pi.add_argument("--max-tokens", type=int, default=4096)
    pi.add_argument("--output", type=Path, default=Path.home() / ".pi" / "agent" / "models.json")
    pi.add_argument("--model-root", type=Path, help="override the Loom model store directory")
    pi.add_argument("--check", action="store_true", help="verify the config without changing it")
    pi.set_defaults(handler=_cmd_pi_config)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.handler(args)
    except (model_store.ModelImportError, FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        print(f"loom-serve-setup: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
