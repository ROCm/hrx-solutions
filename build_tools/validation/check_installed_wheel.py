#!/usr/bin/env python3
# Copyright 2026 The HRX Authors
# Licensed under the MIT License.
# SPDX-License-Identifier: MIT
"""Validate installed assets without a GPU; optionally qualify native lifecycle."""
import argparse
import hashlib
import importlib
import importlib.metadata
import os
import site
import sys
import json
import socket
import time
from pathlib import Path
import hrx_solutions

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--wheel", type=Path, required=True, help="exact wheel installed in this venv")
parser.add_argument("--native-lifecycle", action="store_true",
                    help="also run an empty AMDGPU server (requires an available AMD GPU)")
args = parser.parse_args()
assert sys.prefix != sys.base_prefix, "wheel checks require a standalone venv"
assert not site.ENABLE_USER_SITE
assert not os.environ.get("PYTHONPATH") and not os.environ.get("PYTHONHOME")
assert "include-system-site-packages = false" in (Path(sys.prefix) / "pyvenv.cfg").read_text()
state_root = Path(os.environ["LOOM_SERVE_DIR"])
assert state_root.is_absolute() and not state_root.exists(), "use fresh isolated LOOM_SERVE_DIR"
assert not Path.cwd().is_relative_to(Path(__file__).resolve().parents[2]), "run outside the source checkout"
print(f"hrx-solutions {importlib.metadata.version('hrx-solutions')}")
print(f"package: {hrx_solutions.__file__}")
package_root = Path(hrx_solutions.__file__).resolve().parent
package_root.relative_to(Path(sys.prefix).resolve())
for name in (
    "hem_integrations", "hem_client_documents", "hem_openwebui_wire", "hem_openwebui_receipts",
    "hem_bootstrap", "hem_cli", "hem_common",
    "hem_display", "hem_http", "hem_instances", "hem_process",
    "hem_shutdown", "hem_state",
):
    module = importlib.import_module("hrx_solutions." + name)
    assert Path(module.__file__).resolve().parent == package_root
for name in (
    'curated_catalog',
    'gguf',
    'hem_config',
    'hem_models',
    'hem_model_operations',
    'model_download',
    'model_store',
    'moonshine_preparation',
    'moonshine_preparer_identity',
    'pipeline_validation',
    'qwen_validation',
):
    module = importlib.import_module("hrx_solutions.models." + name)
    assert Path(module.__file__).resolve().parent == package_root / "models"
for name in (
    'cli',
    'cli.hem_arguments',
    'cli.hem_help',
    'cli.hem_render',
    'cli.hem_instance_cli',
    'cli.hem_model_cli',
    'cli.hem_server_cli',
    'cli.hem_pipeline_cli',
    'cli.hem_pi_cli',
    'cli.hem_omp_cli',
    'cli.hem_opencode_cli',
    'cli.hem_openwebui_cli',
):
    module = importlib.import_module("hrx_solutions." + name)
    expected = package_root / (name.replace(".", "/") + ".py" if name != "cli" else "cli/__init__.py")
    assert Path(module.__file__).resolve() == expected.resolve()
entrypoints = {entry.name: entry.value for entry in
               importlib.metadata.distribution("hrx-solutions").entry_points
               if entry.group == "console_scripts"}
required_entrypoints = {
    "loom-control": "hrx_solutions.hem_cli:main",
    "loom-serve": "hrx_solutions.loom_serve:main",
}
assert entrypoints == required_entrypoints, entrypoints
for command, target in required_entrypoints.items():
    assert entrypoints[command] == target
    assert (Path(sys.prefix) / "bin" / command).is_file()
for name in ("setup_cli", "serve_start", "connection_manager", "server_manager", "remote_registry", "state_store"):
    assert importlib.util.find_spec("hrx_solutions." + name) is None, name
for name in ("loom-hem", "loom-serve-setup", "loom-serve-start"):
    assert not (Path(sys.prefix) / "bin" / name).exists(), name
from build_tools.validation.check_installed_cpu import verify_wheel_payload
provenance = verify_wheel_payload(args.wheel.resolve(strict=True), package_root)
assert (package_root / "_native/bin/loom-serve").is_file()
assert (package_root / "_native/share/loom-serve/LICENSE").is_file()
assert not (package_root / "_native/bin/loom-qwen-epoch-check").exists()
from hrx_solutions.models.gguf import inspect_gguf
from hrx_solutions.models.qwen_validation import load_profile
from hrx_solutions.models.pipeline_validation import resolve_catalog, catalog_modules
catalog_root = package_root / "loom-pipeline/llm"
for pipeline, architecture in (("qwen", "qwen_27b"), ("qwen-08b", "qwen_08b")):
    catalog = catalog_root / "qwen" / architecture
    assert resolve_catalog(pipeline) == catalog
    modules = catalog_modules(catalog)
    assert all(name.endswith(".loombc") for name in (modules.prepare, modules.control, modules.weights, *modules.providers))
    contract = load_profile(catalog)
    assert contract["metadata"]["general.architecture"] == "qwen35"
    assert (modules.prepare, modules.control, modules.weights, modules.providers) == (
        "pipeline.loombc", "pipeline.loombc", "pipeline.loombc", ("pipeline.loombc",))
    artifact = catalog / "pipeline.loombc"
    descriptor = json.loads((catalog / "pipeline.json").read_text())
    assert descriptor == {"format": 1, "variant": contract["identifier"],
                          "module": "pipeline.loombc", "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest()}
    assert {p.name for p in catalog.iterdir()} == {"pipeline.loombc", "pipeline.json", "compatibility.json"}
assert (package_root / "loom-pipeline/llm/LICENSE-APACHE-2.0").is_file()
assert not list(package_root.rglob("*.loom"))
import struct
import tempfile
from hrx_solutions.models.pipeline_validation import ModelFiles, validate_model
with tempfile.TemporaryDirectory() as directory:
    weights = Path(directory) / "model.gguf"
    tokenizer = Path(directory) / "tokenizer.json"
    weights.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 0, 0) + bytes(8))
    tokenizer.write_text('{"model": {}}')
    assert inspect_gguf(weights).version == 3
    for pipeline in ("qwen", "qwen-08b"):
        result = validate_model(ModelFiles(weights, tokenizer), pipeline)
        assert not result.compatible
        assert "Missing required metadata" in result.message
# --help does not load HSA. Check the installed runtime API without
# requiring a GPU or allowing host runtime overrides.
import os
import subprocess
import sys
from hrx_solutions._launcher import native_environment
clean_environment = dict(os.environ)
for name in ("IREE_HAL_AMDGPU_LIBHSA_PATH", "IREE_HAL_AMDGPU_LIBAQLPROFILE_PATH", "LD_LIBRARY_PATH"):
    clean_environment.pop(name, None)
subprocess.run(
    [sys.executable, "-c", "import ctypes, os; hsa = ctypes.CDLL(os.environ['IREE_HAL_AMDGPU_LIBHSA_PATH']); hsa.hsa_init; hsa.hsa_amd_queue_create"],
    env=native_environment(clean_environment),
    check=True,
)

# Exercise the public installed CLI without downloads or GPU initialization.
cli = str(Path(sys.prefix) / "bin/loom-control")
root_help = subprocess.run(
    [cli, "--help"], capture_output=True, text=True, timeout=15, check=False)
assert root_help.returncode == 0, root_help.stdout + root_help.stderr
assert ("tui" in root_help.stdout
        and "Interactive, read-only terminal dashboard" in root_help.stdout)
tui_help = subprocess.run(
    [cli, "tui", "--help"], capture_output=True, text=True, timeout=15, check=False)
assert tui_help.returncode == 0, tui_help.stdout + tui_help.stderr
assert "usage: loom-control tui" in tui_help.stdout
def invoke(*arguments, expected=0, timeout=15):
    process = subprocess.run([cli, *arguments, "--json"], capture_output=True,
                             text=True, timeout=timeout, check=False)
    print("loom-control", *arguments, "exit", process.returncode)
    if process.stderr:
        print(process.stderr, file=sys.stderr, end="")
    expected_codes = expected if isinstance(expected, tuple) else (expected,)
    assert process.returncode in expected_codes, process.stdout + process.stderr
    answer = json.loads(process.stdout)
    assert "state" in answer and "completed" in answer
    return answer, process.stderr

pipelines, _ = invoke("pipeline", "list")
pipeline_entries = {item["pipeline"]:item for item in pipelines["items"]}
assert set(pipeline_entries) == {"qwen", "qwen-08b", "moonshine-streaming-tiny"}
for identifier in ("qwen", "qwen-08b"):
    assert "compatibility" in pipeline_entries[identifier] and "mtp" in pipeline_entries[identifier]["capabilities"]
recognition = pipeline_entries["moonshine-streaming-tiny"]
assert recognition["kind"] == "recognition" and recognition["check_kind"] == "acquisition-presence"
assert recognition["defaults"] == {"pending_requests":4,"options":{}}
assert {asset["path"] for asset in recognition["assets"]} == {"model.safetensors","tokenizer.json","config.json","rotary.safetensors"}
assert "compatibility" not in recognition
recipes, _ = invoke("model", "available")
for recipe in recipes["items"]:
    preview, diagnostic = invoke("model", "download", recipe["id"], "--id", "preview", "--dry-run")
    assert preview["state"] == "preview" and "hf download" in diagnostic
invoke("server", "start", "--name", "invalid", "--timeout", "0", expected=2)
assert not state_root.exists(), "help, preview and invalid input must not create state"

def check_native_lifecycle():
    """Real pinned driver is AMDGPU-only; opt in on a GPU host, never CPU CI."""
    from hrx_solutions.hem_common import Deadline
    from hrx_solutions.hem_process import package_identity, identity, _confirmed_exit

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    started_at = time.monotonic()
    record = None
    moved = []
    try:
        started, diagnostic = invoke("server", "start", "--name", "wheel-check",
                                    "--device", "amdgpu", "--listen", f"127.0.0.1:{port}")
        assert time.monotonic() - started_at < 10
        assert "loom-serve" in diagnostic
        record = started["record"]
        assert started["state"] == "running" and started["completed"]
        status, diagnostic = invoke("server", "status", "wheel-check")
        assert status["native"]["data"] == [] and "curl" in diagnostic
        retained = Path(record["build"]["directory"])
        assert package_identity(retained, Deadline.after()) == package_identity(package_root, Deadline.after())
        assert identity(record["process"]["pid"])["executable"] == record["build"]["executable"]
        # Simulate an installer replacing the native/source package trees. Retained
        # artifacts and the live process must remain usable; Python stays installed
        # so the public status/shutdown commands still exercise the real entrypoint.
        for name in ("_native", "loom-pipeline"):
            source, backup = package_root / name, package_root / (name + ".wheel-check-backup")
            assert not backup.exists()
            source.rename(backup)
            moved.append((source, backup))
        status, _ = invoke("server", "status", "wheel-check")
        assert status["native"]["data"] == []
        assert (retained / "loom-pipeline/llm/qwen/qwen_08b/pipeline.loombc").is_file()
        stopped, _ = invoke("server", "shutdown", "wheel-check")
        assert stopped["completed"]
        assert _confirmed_exit(record["process"])
        if record["process"].get("logger"):
            assert _confirmed_exit(record["process"]["logger"])
        assert not retained.exists()
        record = None
    finally:
        original_error = sys.exc_info()[1]
        try:
            for source, backup in reversed(moved):
                backup.rename(source)
            # A timed-out accepted start may still own a pending bootstrap/native.
            # Reconcile before aborting, using one cleanup budget and only this
            # check's named server; never sweep other rows in the same namespace.
            from hrx_solutions.hem_state import Registry
            if state_root.exists():
                budget = Deadline.after()
                owned = next((row for row in Registry(state_root, deadline=budget).list_servers()
                              if row['id'] == 'wheel-check'), None)
                if owned is not None:
                    token = owned['launch_token']
                    if record is not None and token not in (None, record['launch_token']):
                        raise RuntimeError('checker launch changed; cleanup ownership is unresolved')
                    process = owned['process'] or (record or {}).get('process')
                    while (owned.get('pending') or {}).get('action') == 'start':
                        observed, _ = invoke('server', 'reconcile', 'wheel-check',
                                             expected=(0, 1, 4), timeout=budget.remaining())
                        owned = observed['record']
                        if owned['launch_token'] not in (None, token):
                            raise RuntimeError('checker launch changed during cleanup reconciliation')
                        process = process or owned['process']
                        if owned.get('pending') and owned['pending']['action'] == 'start':
                            time.sleep(min(.01, budget.remaining()))
                    if owned['process']:
                        invoke('server', 'shutdown', 'wheel-check', '--abort',
                               '--timeout', str(budget.remaining()), timeout=budget.remaining())
                    if process:
                        assert _confirmed_exit(process), 'checker native process remains unresolved'
                        logger = process.get('logger')
                        while logger and not _confirmed_exit(logger):
                            time.sleep(min(.01, budget.remaining()))
        except Exception as cleanup_error:
            print(f'Checker cleanup unresolved for wheel-check in {state_root}: {cleanup_error}',
                  file=sys.stderr)
            if original_error is None:
                raise
            # Keep the original gate failure as the exception being reported;
            # stderr separately records any unresolved owned process cleanup.
if args.native_lifecycle:
    check_native_lifecycle()
    print("Installed native lifecycle, payload identity and replacement retention passed")
print("Installed Loom Control inventory, entrypoint and CPU descriptor checks passed")

provenance.update(python=sys.executable, package=str(package_root), entrypoints=entrypoints,
                  native_lifecycle=args.native_lifecycle, success=True)
Path("installed-wheel-evidence.json").write_text(json.dumps(provenance, indent=2) + "\n")
