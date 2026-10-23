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
    "hem_bootstrap", "hem_cli", "hem_common",
    "hem_display",
):
    module = importlib.import_module("hrx_solutions." + name)
    assert Path(module.__file__).resolve().parent == package_root
for name in (
    'gguf',
    'hem_config',
    'hem_models',
    'model_download',
    'model_store',
    'moonshine_preparation',
    'moonshine_preparer_identity',
    'recipe_types', 'recipes', 'pipelines', 'pipelines_qwen', 'pipelines_moonshine',
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
for name in ('__init__', 'hem_tui', 'hem_tui_details', 'hem_tui_logs', 'hem_tui_provider'):
    module = importlib.import_module('hrx_solutions.tui' + ('' if name == '__init__' else '.' + name))
    assert Path(module.__file__).resolve() == package_root / 'tui' / (name + '.py')


for relative in (
    'integrations/__init__.py',
    'integrations/omp/__init__.py',
    'integrations/omp/hem_omp.py',
    'integrations/omp/hem_omp_yaml.py',
):
    parts = Path(relative).with_suffix('').parts
    if parts[-1] == '__init__':
        parts = parts[:-1]
    module = importlib.import_module('.'.join(('hrx_solutions', *parts)))
    installed = Path(module.__file__).resolve()
    assert installed.is_relative_to(package_root)
    assert installed == (package_root / relative).resolve()
for relative in (
    'integrations/__init__.py',
    'integrations/opencode/__init__.py',
    'integrations/opencode/hem_opencode.py',
):
    parts = Path(relative).with_suffix('').parts
    if parts[-1] == '__init__':
        parts = parts[:-1]
    module = importlib.import_module('.'.join(('hrx_solutions', *parts)))
    installed = Path(module.__file__).resolve()
    assert installed.is_relative_to(package_root)
    assert installed == (package_root / relative).resolve()
for relative in (
    'integrations/__init__.py',
    'integrations/openwebui/__init__.py',
    'integrations/openwebui/hem_openwebui.py',
    'integrations/openwebui/hem_openwebui_http.py',
    'integrations/openwebui/hem_openwebui_state.py',
    'integrations/openwebui/hem_openwebui_stt.py',
    'integrations/openwebui/hem_openwebui_receipts.py',
    'integrations/openwebui/hem_openwebui_wire.py',
):
    parts = Path(relative).with_suffix('').parts
    if parts[-1] == '__init__':
        parts = parts[:-1]
    module = importlib.import_module('.'.join(('hrx_solutions', *parts)))
    installed = Path(module.__file__).resolve()
    assert installed.is_relative_to(package_root)
    assert installed == (package_root / relative).resolve()
for name in (
    'integrations',
    'integrations.pi',
    'integrations.pi.hem_pi',
    'integrations.pi.hem_pi_assets',
    'integrations.pi.hem_pi_contracts',
    'integrations.pi.hem_pi_endpoints',
    'integrations.pi.hem_pi_files',
    'integrations.pi.hem_pi_state',
):
    module = importlib.import_module("hrx_solutions." + name)
    relative = name.replace(".", "/")
    relative += "/__init__.py" if name in ("integrations", "integrations.pi") else ".py"
    assert Path(module.__file__).resolve() == package_root / relative
for relative in (
    'integrations/__init__.py',
    'integrations/hem_integrations.py',
    'integrations/hem_client_documents.py',
    'integrations/hem_clients_endpoints.py',
):
    parts = Path(relative).with_suffix('').parts
    if parts[-1] == '__init__':
        parts = parts[:-1]
    module = importlib.import_module('.'.join(('hrx_solutions', *parts)))
    installed = Path(module.__file__).resolve()
    assert installed.is_relative_to(package_root)
    assert installed == (package_root / relative).resolve()
for name in ('__init__', 'hem_http', 'hem_instances', 'hem_operation_targets', 'hem_process', 'hem_shutdown', 'hem_state'):
    module = importlib.import_module('hrx_solutions.control' + ('' if name == '__init__' else '.' + name))
    assert Path(module.__file__).resolve() == package_root / 'control' / (name + '.py')
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
from hrx_solutions.models import recipes as authored_recipes, pipelines as pipeline_bindings, modalities
from hrx_solutions.models.hem_config import effective_settings
from hrx_solutions.models.recipe_types import PreparedModel, PreparedArtifact
from hrx_solutions.models.gguf import inspect_gguf
from build_tools.validation.recipe_inventory import validate_installed, EXPECTED_RECIPES
import tempfile
import struct

validate_installed(package_root, authored_recipes.parse)
expected_ids = {value[0] for value in EXPECTED_RECIPES.values()}
assert {recipe.id for recipe in authored_recipes.list()} == expected_ids
assert {recipe.id for recipe in authored_recipes.list('chat')} == {'qwen35-08b-q5-k-xl', 'qwen38-27b-q5-k-xl'}
assert {recipe.id for recipe in authored_recipes.list('asr')} == {'moonshine-streaming-tiny'}
# Exercise the real parser -> binding -> effective settings -> modality definition
# chain with typed prepared file references. This is pure setup, not acquisition
# or a claim that these synthetic paths are usable for native inference.
with tempfile.TemporaryDirectory() as temporary:
    directory = Path(temporary)
    for identifier in sorted(expected_ids):
        recipe = authored_recipes.load(identifier)
        binding = pipeline_bindings.get(recipe.pipeline)
        descriptor = binding.describe()
        assert descriptor.id == recipe.pipeline and descriptor.modality == recipe.modality
        for entry in descriptor.bundle.files:
            path = package_root / entry.path
            path.resolve(strict=True).relative_to(package_root)
            assert path.is_file() and not path.is_symlink(), entry.path
        assert all(name.endswith('.loombc') for name in
                   (descriptor.programs.prepare, descriptor.programs.control,
                    *(descriptor.programs.providers if recipe.modality == 'chat' else descriptor.programs.sources)))
        prepared = PreparedModel('wheel-setup', recipe.id, recipe.identity, recipe.pipeline, directory,
            tuple(PreparedArtifact(a.id, a.role, a.format, a.path, directory / a.path,
                                   'owned' if a.source.kind == 'generated' else 'hf-cache')
                  for a in recipe.artifacts))
        settings, sources = effective_settings(recipe, descriptor)
        policy = modalities.get(recipe.modality)
        policy.validate(settings, descriptor)
        definition = policy.service_definition(settings, prepared, descriptor.programs)
        assert definition['kind'] == descriptor.protocol.native_kind
        assert definition['prepare'] == descriptor.programs.prepare
        assert definition['control'] == descriptor.programs.control
        if recipe.modality == 'chat':
            assert definition['rows'] == 4 and definition['context_capacity'] == 8192
            assert definition['pool_capacity'] == 32768 and definition['prefill_capacity'] == 128
            assert definition['draft_depth'] == (3 if recipe.pipeline == 'qwen' else 0)
            assert definition['weights'] == str(directory / 'model.gguf')
            assert definition['tokenizer'] == str(directory / 'tokenizer.json')
        else:
            assert definition['pending_requests'] == 4 and definition['options'] == {}
            assert definition['assets'] == str(directory)
    weights = directory / 'minimal.gguf'
    weights.write_bytes(b'GGUF' + struct.pack('<IQQ', 3, 0, 0) + bytes(8))
    assert inspect_gguf(weights).version == 3  # generic format admission remains
assert (package_root / 'loom-pipeline/llm/LICENSE-APACHE-2.0').is_file()
assert not list(package_root.rglob('*.loom'))
assert not list((package_root / 'loom-pipeline').rglob('compatibility.json'))
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
for identifier in ('qwen', 'qwen-08b'):
    entry = pipeline_entries[identifier]
    assert entry['modality'] == 'chat' and entry['protocol']['operation'] == 'chat'
    assert set(entry['parameters']) == {'rows', 'context_capacity', 'pool_capacity', 'prefill_capacity', 'mtp', 'mtp_depth'}
recognition = pipeline_entries['moonshine-streaming-tiny']
assert recognition['modality'] == 'asr' and recognition['protocol']['operation'] == 'transcription'
assert recognition['parameters']['pending_requests']['value'] == 4
assert recognition['parameters']['options']['value'] == {}
assert set(recognition['parameters']) == {'pending_requests', 'options'}
recipes, _ = invoke("model", "available")
for recipe in recipes["items"]:
    preview, diagnostic = invoke("model", "download", recipe["id"], "--id", "preview", "--dry-run")
    assert preview["state"] == "preview" and "hf download" in diagnostic
invoke("server", "start", "--name", "invalid", "--timeout", "0", expected=2)
assert not state_root.exists(), "help, preview and invalid input must not create state"

def check_native_lifecycle():
    """Real pinned driver is AMDGPU-only; opt in on a GPU host, never CPU CI."""
    from hrx_solutions.hem_common import Deadline
    from hrx_solutions.control.hem_process import package_identity, identity, _confirmed_exit

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
            from hrx_solutions.control.hem_state import Registry
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
