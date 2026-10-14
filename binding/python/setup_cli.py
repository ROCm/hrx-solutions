"""Command line setup workflow for local Loom Serve models."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Optional, Sequence

from . import connection_manager, model_store, pipeline_validation, remote_registry, server_manager
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
    candidates = (
        package / "loom-pipeline" / "llm" / pipeline,
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
    if args.json:
        print(remote_registry.inventory_json(args.model_root))
        return 0
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


def _cmd_remote_register(args: argparse.Namespace) -> int:
    remote = remote_registry.register_remote(
        args.name,
        args.ssh_target,
        server_port=args.server_port,
        local_port=args.local_port,
        replace=args.replace,
    )
    print(
        f"Registered remote {remote.name}: ssh={remote.ssh_target} "
        f"server_port={remote.server_port} local_port={remote.local_port}"
    )
    return 0


def _cmd_remote_list(_args: argparse.Namespace) -> int:
    remotes = remote_registry.list_remotes()
    if not remotes:
        print("No remote hosts registered")
        return 0
    for remote in remotes:
        print(
            f"{remote.name}\tssh={remote.ssh_target} "
            f"server_port={remote.server_port}\tlocal_port={remote.local_port}"
        )
    return 0


def _cmd_remote_unregister(args: argparse.Namespace) -> int:
    remote_registry.unregister_remote(args.name)
    print(f"Unregistered remote {args.name}")
    return 0


def _cmd_remote_models(args: argparse.Namespace) -> int:
    inventory = remote_registry.query_remote_models(args.name, timeout=args.timeout)
    print(json.dumps(inventory, indent=2, sort_keys=True))
    return 0


def _cmd_remote_start(args: argparse.Namespace) -> int:
    record = remote_registry.start_remote_server(args.name, args.model_id, timeout=args.timeout)
    if args.json:
        _print_server_json(record)
    else:
        print(
            f"Started remote Loom Serve server {record['server_id']} for {record['model_id']}"
            f" on {args.name}\n  endpoint: {record.get('endpoint')}\n"
            f"  pid:      {record.get('pid')}\n  log:      {record.get('log_path')}"
        )
    return 0


def _cmd_remote_servers_list(args: argparse.Namespace) -> int:
    records = remote_registry.list_remote_servers(args.name, timeout=args.timeout)
    if args.json:
        _print_server_json(records)
    elif not records:
        print(f"No managed Loom Serve processes on {args.name}")
    else:
        for record in records:
            print(_server_summary(record))
    return 0


def _cmd_remote_servers_action(args: argparse.Namespace) -> int:
    record, returncode = remote_registry.remote_server_action(
        args.name, args.server_id, args.servers_action, timeout=args.timeout
    )
    if args.json:
        _print_server_json(record)
    else:
        print(_server_summary(record))
    if args.servers_action == "stop":
        return 0 if returncode == 0 and record.get("state") == "stopped" else 1
    return 0


def _connection_summary(record: dict) -> str:
    return (
        f"{record.get('connection_id')}\t{record.get('state')}"
        f"\tremote={record.get('remote_name')}\turl={record.get('local_url')}"
        f"\tpid={record.get('pid')}\tlog={record.get('log_path')}"
        + (f"\tendpoint={record['endpoint_health']}" if record.get("endpoint_health") else "")
    )


def _cmd_remote_connect(args: argparse.Namespace) -> int:
    record = connection_manager.connect(
        args.name, args.model, ready_timeout=args.ready_timeout
    )
    if args.json:
        _print_server_json(record)
    else:
        print(
            f"Connected to remote Loom Serve on {record['remote_name']}\n"
            f"  connection: {record['connection_id']}\n"
            f"  endpoint:   {record['local_url']}\n"
            f"  pid:        {record['pid']}\n"
            f"  log:        {record['log_path']}"
        )
    return 0


def _cmd_remote_connections_list(args: argparse.Namespace) -> int:
    records = connection_manager.list_connections()
    if args.json:
        _print_server_json(records)
    elif not records:
        print("No managed SSH connections")
    else:
        for record in records:
            print(_connection_summary(record))
    return 0


def _cmd_remote_connections_status(args: argparse.Namespace) -> int:
    record = connection_manager.get_connection(args.connection_id)
    if args.json:
        _print_server_json(record)
    else:
        print(_connection_summary(record))
    return 0


def _cmd_remote_connections_disconnect(args: argparse.Namespace) -> int:
    record = connection_manager.disconnect(args.connection_id)
    if args.json:
        _print_server_json(record)
    else:
        print(_connection_summary(record))
    return 0 if record.get("state") == "disconnected" else 1


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


def _server_summary(record: dict) -> str:
    return (
        f"{record.get('server_id')}\t{record.get('state')}\tmodel={record.get('model_id')}"
        f"\tpid={record.get('pid')}\tendpoint={record.get('endpoint')}"
        f"\tlog={record.get('log_path')}"
        + (f"\thealth={record['endpoint_health']}" if record.get("endpoint_health") else "")
    )


def _print_server_json(value: object) -> None:
    """Write one JSON value to stdout for machine callers."""
    print(json.dumps(value, indent=2, sort_keys=True))


def _cmd_servers_list(args: argparse.Namespace) -> int:
    records = server_manager.list_servers()
    if args.json:
        _print_server_json(records)
        return 0
    if not records:
        print("No managed Loom Serve processes")
        return 0
    for record in records:
        print(_server_summary(record))
    return 0


def _cmd_servers_status(args: argparse.Namespace) -> int:
    record = server_manager.get_server(args.server_id)
    if args.json:
        _print_server_json(record)
    else:
        print(_server_summary(record))
    return 0


def _cmd_servers_stop(args: argparse.Namespace) -> int:
    record = server_manager.stop_server(args.server_id)
    if args.json:
        _print_server_json(record)
    else:
        print(_server_summary(record))
    return 0 if record.get("state") == "stopped" else 1


def _cmd_serve(args: argparse.Namespace) -> int:
    if args.foreground and args.json:
        raise ValueError("--json is only available for managed background servers")
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
    executable = package / "_native" / "bin" / "loom-serve"
    if not executable.is_file():
        raise FileNotFoundError(f"Loom Serve binary is missing: {executable}")
    argv = [
        str(executable),
        f"--model={catalog}",
        # Preserve the visible file extensions on Hugging Face snapshot paths.
        # Resolving the symlinks follows them to extensionless blob names,
        # which Loom cannot use to select its GGUF parser.
        f"--weights={files.weights.absolute()}",
        f"--tokenizer={files.tokenizer.absolute()}",
        f"--rows={args.rows}",
        f"--context_capacity={args.context_capacity}",
        f"--pool_capacity={args.pool_capacity}",
        f"--prefill_capacity={args.prefill_capacity}",
        f"--port={args.port}",
        "--heartbeat_ms=1000",
    ]
    if not args.no_mtp:
        argv.extend(("--mtp", f"--mtp_depth={args.mtp_depth}"))
    if args.foreground:
        os.execve(str(executable), argv, native_environment())
    record = server_manager.launch(
        model_id=args.model_id,
        pipeline=pipeline,
        executable=executable,
        argv=argv,
        port=args.port,
        model_paths=(files.weights.absolute(), files.tokenizer.absolute(), catalog.resolve()),
        launch_options={
            "rows": args.rows, "context_capacity": args.context_capacity,
            "pool_capacity": args.pool_capacity, "prefill_capacity": args.prefill_capacity,
            "mtp": not args.no_mtp, "mtp_depth": args.mtp_depth,
        },
        ready_timeout=args.ready_timeout,
        environment=native_environment(),
    )
    if args.json:
        _print_server_json(record)
    else:
        print(
            f"Started Loom Serve server {record['server_id']} for {args.model_id}\n"
            f"  endpoint: {record['endpoint']}\n"
            f"  pid:      {record['pid']}\n"
            f"  log:      {record['log_path']}"
        )
    return 0


def _update_pi_config(
    model_id: str,
    base_url: str,
    output: Path,
    provider_name: str,
    display_name: Optional[str],
    context_window: int,
    max_tokens: int,
    *,
    check: bool = False,
) -> int:
    output = output.expanduser()
    base_url = base_url.rstrip("/")
    if not base_url.endswith("/v1"):
        base_url += "/v1"
    expected_model = {
        "id": model_id,
        "name": display_name or model_id,
        "contextWindow": context_window,
        "maxTokens": max_tokens,
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
    provider = providers.get(provider_name)
    if provider is not None and not isinstance(provider, dict):
        raise ValueError(
            f"Malformed Pi configuration {output}: provider {provider_name!r} must be an object"
        )
    if isinstance(provider, dict) and "models" in provider and not isinstance(provider["models"], list):
        raise ValueError(
            f"Malformed Pi configuration {output}: provider {provider_name!r} 'models' must be an array"
        )

    if check:
        configured_model = None
        if isinstance(provider, dict) and isinstance(provider.get("models"), list):
            matches = [
                model for model in provider["models"]
                if isinstance(model, dict) and model.get("id") == model_id
            ]
            if len(matches) == 1:
                configured_model = matches[0]
        if (
            isinstance(provider, dict)
            and provider.get("baseUrl", "").rstrip("/") == base_url
            and provider.get("api") == "openai-completions"
            and configured_model is not None
        ):
            print(f"Pi is configured for {model_id!r} at {base_url} ({output})")
            return 0
        print(
            f"Pi is not configured for model {model_id!r} at {base_url} "
            f"under provider {provider_name!r} ({output})",
            file=sys.stderr,
        )
        return 1

    provider = providers.setdefault(provider_name, {})
    models = provider.setdefault("models", [])
    if not isinstance(models, list):
        raise ValueError(
            f"Malformed Pi configuration {output}: provider {provider_name!r} 'models' must be an array"
        )
    matches = [
        model for model in models
        if isinstance(model, dict) and model.get("id") == model_id
    ]
    if len(matches) > 1:
        raise ValueError(
            f"Malformed Pi configuration {output}: provider {provider_name!r} has "
            f"multiple entries for model {model_id!r}"
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
    print(f"Configured Pi for Loom model {model_id!r}: {output}")
    return 0


def _cmd_pi_config(args: argparse.Namespace) -> int:
    # Refuse to point Pi at a model that has not been imported into Loom's
    # model store. This also catches broken aliases before changing user config.
    files = _model_files(args.model_id, args.model_root)
    for label, path in (("GGUF", files.weights), ("tokenizer", files.tokenizer)):
        if not path.is_file():
            raise model_store.ModelImportError(
                f"Installed model {args.model_id!r} is missing its {label}: {path}"
            )
    return _update_pi_config(
        args.model_id, args.base_url, args.output, args.provider, args.display_name,
        args.context_window, args.max_tokens, check=args.check,
    )


def _cmd_remote_pi_config(args: argparse.Namespace) -> int:
    inventory = remote_registry.query_remote_models(args.name, timeout=args.timeout)
    model = next(
        (item for item in inventory["models"] if item.get("id") == args.model_id),
        None,
    )
    if model is None:
        raise remote_registry.RemoteRegistryError(
            f"Model {args.model_id!r} is not installed on remote {args.name!r}"
        )
    if not model.get("validated") or model.get("availability") != "ready":
        raise remote_registry.RemoteRegistryError(
            f"Model {args.model_id!r} on remote {args.name!r} is not validated and ready "
            f"(validated={model.get('validated')!r}, availability={model.get('availability')!r})"
        )
    remote = remote_registry.get_remote(args.name)
    base_url = f"http://127.0.0.1:{remote.local_port}/v1"
    if args.check_connection:
        connection_manager.verify_connection(args.name, args.model_id)
    result = _update_pi_config(
        args.model_id, base_url, Path.home() / ".pi" / "agent" / "models.json",
        "hrx-loom", None, 8192, 4096,
    )
    if result == 0:
        result = _update_pi_config(
            args.model_id, base_url, Path.home() / ".pi" / "agent" / "models.json",
            "hrx-loom", None, 8192, 4096, check=True,
        )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="loom-serve-setup",
        description=(
            "Import and validate Loom models, manage local servers, and connect to "
            "prepared remote hosts over SSH."
        ),
        epilog=(
            "Typical remote flow: remote register → remote models → remote start → "
            "remote connect → remote pi-config. Stop the server with remote servers "
            "NAME stop ID and close the tunnel with remote connections disconnect ID."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    models = subparsers.add_parser("models", help="manage locally installed models")
    model_commands = models.add_subparsers(dest="models_command", required=True)
    listing = model_commands.add_parser("list", help="list installed models")
    listing.add_argument("--model-root", type=Path)
    listing.add_argument("--json", action="store_true", help="print stable machine-readable model inventory")
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

    serve = subparsers.add_parser(
        "serve",
        help="start Loom Serve for an installed model in the background",
        description=(
            "By default, start a managed background server and wait for its API to become ready. "
            "Use --foreground to run it in this terminal; --json prints the background server record."
        ),
    )
    serve.add_argument("model_id", help="ID used when the model was imported")
    serve.add_argument("--pipeline", help="override the model's recorded pipeline/catalog")
    serve.add_argument("--model-root", type=Path)
    serve.add_argument("--port", type=int, default=8080)
    serve.add_argument("--rows", type=int, default=4)
    serve.add_argument("--context-capacity", type=int, default=8192)
    serve.add_argument("--pool-capacity", type=int, default=32768)
    serve.add_argument("--prefill-capacity", type=int, default=128)
    serve.add_argument("--no-mtp", action="store_true", help="disable multi-token prediction")
    serve.add_argument("--mtp-depth", type=int, default=3)
    serve.add_argument("--foreground", action="store_true", help="run Loom Serve in the foreground")
    serve.add_argument("--json", action="store_true", help="print the managed server record as JSON")
    serve.add_argument("--ready-timeout", type=float, default=180.0, help="seconds to wait for API readiness")
    serve.set_defaults(handler=_cmd_serve)

    servers = subparsers.add_parser(
        "servers",
        help="inspect and stop managed Loom Serve processes",
        description="List, inspect, or stop servers started by loom-serve-setup.",
    )
    server_commands = servers.add_subparsers(dest="servers_command", required=True)
    server_list = server_commands.add_parser("list", help="list managed server processes")
    server_list.add_argument("--json", action="store_true", help="print server records as a JSON array")
    server_list.set_defaults(handler=_cmd_servers_list)
    status = server_commands.add_parser("status", help="show a managed server")
    status.add_argument("server_id")
    status.add_argument("--json", action="store_true", help="print the server record as JSON")
    status.set_defaults(handler=_cmd_servers_status)
    stop = server_commands.add_parser("stop", help="stop a verified managed server")
    stop.add_argument("server_id")
    stop.add_argument("--json", action="store_true", help="print the server record as JSON")
    stop.set_defaults(handler=_cmd_servers_stop)

    remote = subparsers.add_parser(
        "remote",
        help="manage SSH remotes, servers, and tunnels",
        description=(
            "Use a registered SSH host with loom-serve-setup and the model installed. "
            "Remote server lifecycle and client tunnel lifecycle are separate."
        ),
        epilog=(
            "Flow: register NAME --ssh-target USER@HOST, inspect with models NAME, "
            "start with start NAME MODEL_ID, then open a client tunnel with connect NAME. "
            "Pi configuration is local to this client."
        ),
    )
    remote_commands = remote.add_subparsers(dest="remote_command", required=True)
    remote_register = remote_commands.add_parser(
        "register", help="save a prepared SSH host and its server/tunnel ports"
    )
    remote_register.add_argument("name", help="short name used by other remote commands")
    remote_register.add_argument("--ssh-target", required=True, help="SSH destination, such as user@host")
    remote_register.add_argument("--server-port", type=int, default=8080, help="Loom Serve port on the remote host (default: 8080)")
    remote_register.add_argument("--local-port", type=int, default=18080, help="loopback port opened by connect (default: 18080)")
    remote_register.add_argument("--replace", action="store_true", help="replace an unused registration")
    remote_register.set_defaults(handler=_cmd_remote_register)
    remote_commands.add_parser("list", help="list registered SSH hosts").set_defaults(handler=_cmd_remote_list)
    remote_unregister = remote_commands.add_parser("unregister", help="remove an SSH host definition")
    remote_unregister.add_argument("name")
    remote_unregister.set_defaults(handler=_cmd_remote_unregister)
    remote_models = remote_commands.add_parser("models", help="list installed models on a remote host")
    remote_models.add_argument("name", help="registered remote host name")
    remote_models.add_argument("--timeout", type=float, default=30.0, help="SSH command timeout in seconds")
    remote_models.set_defaults(handler=_cmd_remote_models)

    remote_pi = remote_commands.add_parser(
        "pi-config",
        help="configure this client's Pi to use a validated remote model",
        description=(
            "Writes the registered remote's loopback tunnel URL into this client's Pi config. "
            "It does not start the server or open the tunnel."
        ),
    )
    remote_pi.add_argument("name", help="registered remote host name")
    remote_pi.add_argument("model_id", help="exact model ID installed on the remote host")
    remote_pi.add_argument(
        "--check-connection", action="store_true",
        help="require an active tunnel and verify it advertises this model before configuring Pi",
    )
    remote_pi.add_argument("--timeout", type=float, default=30.0, help="SSH command timeout in seconds")
    remote_pi.set_defaults(handler=_cmd_remote_pi_config)

    remote_start = remote_commands.add_parser(
        "start", help="start a validated model as a managed background server on a remote host"
    )
    remote_start.add_argument("name", help="registered remote host name")
    remote_start.add_argument("model_id", help="exact ID of an installed, validated model")
    remote_start.add_argument("--timeout", type=float, default=240.0, help="remote startup timeout in seconds")
    remote_start.add_argument("--json", action="store_true", help="print the remote server record as JSON")
    remote_start.set_defaults(handler=_cmd_remote_start)

    remote_servers = remote_commands.add_parser(
        "servers", help="inspect or stop managed servers on a remote host"
    )
    remote_servers.add_argument("name", help="registered remote host name")
    remote_server_commands = remote_servers.add_subparsers(dest="servers_action", required=True)
    remote_server_list = remote_server_commands.add_parser("list", help="list remote managed servers")
    remote_server_list.add_argument("--timeout", type=float, default=30.0)
    remote_server_list.add_argument("--json", action="store_true", help="print remote server records as JSON")
    remote_server_list.set_defaults(handler=_cmd_remote_servers_list)
    for action in ("status", "stop"):
        remote_server_action = remote_server_commands.add_parser(
            action,
            help=("show a remote managed server" if action == "status" else "stop a verified remote managed server"),
        )
        remote_server_action.add_argument("server_id")
        remote_server_action.add_argument("--timeout", type=float, default=30.0)
        remote_server_action.add_argument("--json", action="store_true", help="print the remote server record as JSON")
        remote_server_action.set_defaults(handler=_cmd_remote_servers_action)

    remote_connect = remote_commands.add_parser(
        "connect",
        help="open a managed SSH tunnel to an already running remote server",
        description=(
            "Forward the registered remote server port to the configured client loopback port. "
            "This manages only the SSH tunnel; start and stop the remote server with remote start/servers."
        ),
    )
    remote_connect.add_argument("name", help="registered remote host name")
    remote_connect.add_argument("--model", help="wait until this exact model is advertised by the server")
    remote_connect.add_argument("--ready-timeout", type=float, default=30.0, help="tunnel/API readiness timeout in seconds")
    remote_connect.add_argument("--json", action="store_true", help="print the connection record as JSON")
    remote_connect.set_defaults(handler=_cmd_remote_connect)

    remote_connections = remote_commands.add_parser(
        "connections",
        help="inspect or close client-side managed SSH tunnels",
        description="Tunnel records and lifecycle are managed on this client, separately from remote servers.",
    )
    connection_commands = remote_connections.add_subparsers(
        dest="connection_action", required=True
    )
    connection_list = connection_commands.add_parser("list", help="list managed SSH tunnels")
    connection_list.add_argument("--json", action="store_true", help="print connection records as JSON")
    connection_list.set_defaults(handler=_cmd_remote_connections_list)
    connection_status = connection_commands.add_parser("status", help="show a managed SSH tunnel")
    connection_status.add_argument("connection_id")
    connection_status.add_argument("--json", action="store_true", help="print the connection record as JSON")
    connection_status.set_defaults(handler=_cmd_remote_connections_status)
    connection_disconnect = connection_commands.add_parser(
        "disconnect", help="stop a verified managed SSH tunnel"
    )
    connection_disconnect.add_argument("connection_id")
    connection_disconnect.add_argument("--json", action="store_true", help="print the connection record as JSON")
    connection_disconnect.set_defaults(handler=_cmd_remote_connections_disconnect)

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
