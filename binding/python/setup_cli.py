"""Command line setup workflow for local Loom Serve models."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Optional, Sequence

from . import connection_manager, model_download, model_store, pipeline_validation, remote_registry, server_manager
from ._launcher import native_environment


def _cmd_list(args: argparse.Namespace) -> int:
    inventory = remote_registry.local_model_inventory(args.model_root)
    if args.json:
        _print_json(inventory)
    elif not inventory["models"]:
        print(f"No models installed in {args.model_root or model_store.default_model_root()}")
    else:
        for item in inventory["models"]:
            print(f"{item['id']}\tpipeline={item.get('pipeline') or 'unvalidated'}"
                  f"\tmode={item.get('install_mode')}\tavailability={item['availability']}")
    return 0


def _cmd_available(args: argparse.Namespace) -> int:
    catalog = model_download.available(args.model_root)
    if args.json:
        _print_json(catalog)
    else:
        for recipe in catalog:
            print(f"{recipe['id']} — {recipe['description']}")
            print(f"  variant: {recipe['variant']}; quantization: {recipe['quantization']}; pipeline: {recipe['pipeline']}")
            print(f"  download: {recipe['cache_status']}; registration: {recipe['registration_status']}")
            for file in recipe['files']:
                print(f"  {file['role']}: {file['repo_id']}@{file['revision']}/{file['filename']}"
                      f" ({'cached' if file['cached'] else 'not downloaded'})")
            for registration in recipe['registrations']:
                print(f"  local ID: {registration['id']} ({registration['status']})")
    return 0


def _cmd_download(args: argparse.Namespace) -> int:
    result = model_download.download(args.recipe, model_id=args.id, model_root=args.model_root,
                                     offline=args.offline, dry_run=args.dry_run)
    if args.json:
        _print_json(result)
    elif args.dry_run:
        print(f"Download plan for {result['id']}: {result['cache_status']}; cache aliases in {result['directory']}")
        for file in result['files']:
            size = file.get('size_bytes')
            print(f"  {file['repo_id']}@{file['revision']}/{file['filename']}: "
                  f"{'cached' if file['cached'] else 'download required'}, size={size if size is not None else file.get('expected_size', 'unknown')}")
        if result.get('destination_exists'):
            print("  Destination already exists; a conflicting installation will be preserved.")
        if result.get('size_note'):
            print("  " + result['size_note'])
    else:
        print(f"{result['id']}: {result['status']} in {result['directory']}")
        print("  Files are aliases to the Hugging Face cache; retain those cache files.")
        print(f"  Start: loom-serve-setup serve {result['id']}" +
              (f" --model-root {args.model_root}" if args.model_root else ""))
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
        _print_json(record)
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
        _print_json(records)
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
        _print_json(record)
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
        _print_json(record)
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
        _print_json(records)
    elif not records:
        print("No managed SSH connections")
    else:
        for record in records:
            print(_connection_summary(record))
    return 0


def _cmd_remote_connections_status(args: argparse.Namespace) -> int:
    record = connection_manager.get_connection(args.connection_id)
    if args.json:
        _print_json(record)
    else:
        print(_connection_summary(record))
    return 0


def _cmd_remote_connections_disconnect(args: argparse.Namespace) -> int:
    record = connection_manager.disconnect(args.connection_id)
    if args.json:
        _print_json(record)
    else:
        print(_connection_summary(record))
    return 0 if record.get("state") == "disconnected" else 1


def _cmd_import(args: argparse.Namespace) -> int:
    imported = model_store.import_model(args.model_id, gguf=args.gguf, tokenizer=args.tokenizer,
                                        model_root=args.model_root, alias=args.alias)
    print(f"Imported {imported.model_id} into {imported.directory}")
    print(f"  weights:   {imported.files.weights}")
    print(f"  tokenizer: {imported.files.tokenizer}")
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
        files = model_store.model_files(args.model_id, args.model_root)
    except model_store.ModelImportError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    result = pipeline_validation.validate_model(
        files, args.pipeline
    )
    print(result.message)
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


def _print_json(value: object) -> None:
    """Write one JSON value to stdout for machine callers."""
    print(json.dumps(value, indent=2, sort_keys=True))


def _cmd_servers_list(args: argparse.Namespace) -> int:
    records = server_manager.list_servers()
    if args.json:
        _print_json(records)
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
        _print_json(record)
    else:
        print(_server_summary(record))
    return 0


def _cmd_servers_stop(args: argparse.Namespace) -> int:
    record = server_manager.stop_server(args.server_id)
    if args.json:
        _print_json(record)
    else:
        print(_server_summary(record))
    return 0 if record.get("state") == "stopped" else 1


def _cmd_serve(args: argparse.Namespace) -> int:
    if args.foreground and args.json:
        raise ValueError("--json is only available for managed background servers")
    registered = model_store.registered_model(args.model_id, args.model_root)
    files = registered.require_files()
    pipeline = args.pipeline or registered.manifest.get("pipeline")
    if not pipeline:
        raise model_store.ModelImportError(
            f"Model {args.model_id!r} has no validated pipeline recorded. Run "
            f"'loom-serve-setup models validate {args.model_id} --pipeline <pipeline>' "
            "or pass --pipeline explicitly."
        )
    selected_catalog = pipeline_validation.resolve_catalog(pipeline)
    requested_mtp = args.mtp
    if args.mtp_depth is not None:
        if requested_mtp is False:
            raise ValueError("--mtp-depth cannot be combined with --no-mtp")
        requested_mtp = True
    enabled_mtp = pipeline_validation.mtp_enabled(selected_catalog, requested_mtp)
    mtp_depth = args.mtp_depth if args.mtp_depth is not None else 3
    catalog = pipeline_validation.catalog_for_model(pipeline, selected_catalog, args.model_id)
    package = Path(__file__).resolve().parent
    executable = package / "_native" / "bin" / "loom-serve"
    if not executable.is_file():
        raise FileNotFoundError(f"Loom Serve binary is missing: {executable}")
    deployment = pipeline_validation.deployment_for_model(
        catalog, files, args.model_id, rows=args.rows,
        context_capacity=args.context_capacity, pool_capacity=args.pool_capacity,
        prefill_capacity=args.prefill_capacity,
        draft_depth=mtp_depth if enabled_mtp else 0,
    )
    argv = [str(executable), "--device=amdgpu", f"--model={deployment}",
            f"--listen=127.0.0.1:{args.port}", "--heartbeat_ms=1000"]
    if args.foreground:
        os.execve(str(executable), argv, native_environment())
    record = server_manager.launch(
        model_id=args.model_id,
        pipeline=pipeline,
        executable=executable,
        argv=argv,
        port=args.port,
        model_paths=tuple(path.absolute() for path in (*files.gguf_paths, files.weights, files.tokenizer, catalog, deployment)),
        launch_options={
            "rows": args.rows, "context_capacity": args.context_capacity,
            "pool_capacity": args.pool_capacity, "prefill_capacity": args.prefill_capacity,
            "mtp": enabled_mtp, "mtp_depth": mtp_depth,
        },
        ready_timeout=args.ready_timeout,
        environment=native_environment(),
    )
    if args.json:
        _print_json(record)
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
    files = model_store.model_files(args.model_id, args.model_root)
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

    available = model_commands.add_parser("available", help="show bundled download recipes, variants and local cache/registration status")
    available.add_argument("--model-root", type=Path)
    available.add_argument("--json", action="store_true")
    available.set_defaults(handler=_cmd_available)

    downloading = model_commands.add_parser("download", help="download and validate a known recipe; alias Hugging Face cache files")
    downloading.add_argument("recipe", help="recipe ID shown by models available")
    downloading.add_argument("--id", help="local model/HTTP identity (defaults to recipe ID)")
    downloading.add_argument("--model-root", type=Path)
    downloading.add_argument("--offline", action="store_true", help="use pinned cached files only; make no network requests")
    downloading.add_argument("--dry-run", action="store_true", help="preview sources, cache state and sizes without downloading or registering")
    downloading.add_argument("--json", action="store_true", help="JSON result on stdout; progress streams to stderr")
    downloading.set_defaults(handler=_cmd_download)

    importing = model_commands.add_parser("import", help="import local GGUF/tokenizer files", epilog="Hugging Face options were removed: use models available and models download RECIPE instead.")
    importing.add_argument("model_id")
    importing.add_argument("--gguf", required=True, type=Path, nargs="+", help="one or more local GGUF fragments")
    importing.add_argument("--tokenizer", required=True, type=Path, help="local tokenizer.json")
    importing.add_argument("--alias", action="store_true", help="alias user-owned files instead of copying large checkpoints")
    importing.add_argument("--model-root", type=Path)
    importing.set_defaults(handler=_cmd_import)

    validating = model_commands.add_parser("validate", help="inspect GGUF architecture and tensor encodings for a pipeline")
    validating.add_argument("model_id")
    validating.add_argument("--pipeline", required=True)
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
    mtp_options = serve.add_mutually_exclusive_group()
    mtp_options.add_argument("--mtp", dest="mtp", action="store_true", help="require multi-token prediction")
    mtp_options.add_argument("--no-mtp", dest="mtp", action="store_false", help="disable multi-token prediction")
    serve.add_argument("--mtp-depth", type=int, help="require MTP with this depth (default: 3 for supported models)")
    serve.add_argument("--foreground", action="store_true", help="run Loom Serve in the foreground")
    serve.add_argument("--json", action="store_true", help="print the managed server record as JSON")
    serve.add_argument("--ready-timeout", type=float, default=180.0, help="seconds to wait for API readiness")
    serve.set_defaults(handler=_cmd_serve, mtp=None)

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
    tokens = list(argv) if argv is not None else sys.argv[1:]
    removed = {"--hf-repo", "--gguf-file", "--tokenizer-file", "--tokenizer-hf-repo", "--revision", "--tokenizer-revision", "--token"}
    if tokens[:2] == ["models", "import"] and any(token.split("=", 1)[0] in removed for token in tokens[2:]):
        parser.error("models import accepts local files only; Hugging Face options were removed. Run models available, then models download RECIPE.")
    args = parser.parse_args(tokens)
    try:
        return args.handler(args)
    except (model_store.ModelImportError, FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        print(f"loom-serve-setup: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
