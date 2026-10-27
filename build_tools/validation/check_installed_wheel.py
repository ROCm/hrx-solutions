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
    'recipes', 'pipelines',
):
    module = importlib.import_module("hrx_solutions.models." + name)
    assert Path(module.__file__).resolve().parent == package_root / "models"
for name in ('preparation', 'preparation.moonshine', 'preparation.moonshine_contract'):
    module = importlib.import_module('hrx_solutions.models.' + name)
    relative = name.replace('.', '/') + ('.py' if '.' in name else '/__init__.py')
    assert Path(module.__file__).resolve() == (package_root / 'models' / relative).resolve()
for name in (
    'cli',
    'cli.hem_arguments',
    'cli.hem_help',
    'cli.hem_render',
    'cli.hem_instance_cli',
    'cli.hem_model_cli',
    'cli.hem_server_cli',
    'cli.hem_modality_cli',
    'cli.hem_executable_cli',
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


def check_recipe_consumers(package_root, *, release=None):
    """Exercise installed generic readers and finite CHAT/ASR policy against release DATA."""
    from hrx_solutions.models import recipes, pipelines, modalities, hem_config
    from hrx_solutions.hem_common import Deadline
    from hrx_solutions.models.gguf import inspect_gguf
    from build_tools.validation import recipe_inventory as inventory
    import tempfile
    import struct

    release = inventory.load_release() if release is None else release
    inventory.validate_installed_catalogs(package_root, recipes.parse, pipelines.parse, release=release)
    expected_ids = {document['id'] for document in release['recipes'].values()}
    assert {document['id'] for document, _ in recipes.list()} == expected_ids
    for reference in release['executables']:
        selected_ids = {document['id'] for document in release['recipes'].values()
                        if document['executable'] == reference}
        assert {document['id'] for document, _ in recipes.list(executable=reference)} == selected_ids
    # Pure policy checks use synthetic paths; real import/checked resolution,
    # preparation and inference belong to the composed installed gates.
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        for wanted in release['recipes'].values():
            document, identity = recipes.load(wanted['id'])
            assert inventory._same(document, wanted)
            reference = document['executable']
            executable, catalog, _ = pipelines.load(reference)
            catalog.resolve(strict=True).relative_to(package_root)
            programs = pipelines.programs(executable, catalog, Deadline.after())
            reviewed = release['executables'][reference]
            assert programs == {key: reviewed['compiled'][key] for key in ('prepare', 'control', 'sources')}
            policy = modalities.get(executable['modality'])
            settings, _ = hem_config.effective_settings(document, executable['serving'], policy)
            expected = dict(reviewed['source']['serving']['defaults'], **wanted['configuration'])
            if executable['modality'] == 'chat' and expected['mtp'] is False:
                expected.pop('mtp_depth', None)
            assert inventory._same(hem_config.json_document(settings), expected)
            paths = {purpose: directory / item['path'] for purpose, item in document['files'].items()}
            definition = policy.definition(settings, paths, directory, programs)
            if executable['modality'] == 'chat':
                native = dict(kind='text', **programs, weights=str(paths['weights']),
                    tokenizer=str(paths['tokenizer']),
                    **{key: expected[key] for key in
                       ('rows', 'context_capacity', 'pool_capacity', 'prefill_capacity')},
                    draft_depth=expected['mtp_depth'] if expected['mtp'] else 0, options={})
            else:
                native = dict(kind='recognition', **programs, assets=str(directory),
                              pending_requests=expected['pending_requests'], options=expected['options'])
            assert inventory._same(definition, native)
        weights = directory / 'minimal.gguf'
        weights.write_bytes(b'GGUF' + struct.pack('<IQQ', 3, 0, 0) + bytes(8))
        assert inspect_gguf(weights).version == 3
    assert not list(package_root.rglob('*.loom'))
    assert not list((package_root / 'loom-pipeline').rglob('compatibility.json'))
    return tuple(sorted(expected_ids))


check_recipe_consumers(package_root)
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

def check_public_recipe_queries(invoke, *, release=None):
    """Public inventories and capability projections have independent DATA authority."""
    from build_tools.validation import recipe_inventory as inventory
    release = inventory.load_release() if release is None else release
    modalities, _ = invoke('modality', 'list')
    modality_rows = {row['modality']: row for row in modalities['items']}
    assert len(modalities['items']) == len(modality_rows) == 2
    assert set(modality_rows) == {'chat', 'asr'}
    executables, _ = invoke('executable', 'list')
    rows = {row['executable']: row for row in executables['items']}
    assert len(rows) == len(executables['items'])
    assert set(rows) == set(release['executables'])
    for modality, row in modality_rows.items():
        assert 'parameters' not in row and 'pipeline' not in row and 'defaults' not in row
        assert row['native_contract'] == ('text-v1' if modality == 'chat' else 'recognition-v1')
        assert row['protocol']['operation'] == ('chat' if modality == 'chat' else 'transcription')
        assert {item['executable'] for item in row['executables']} == {
            ref for ref, expected in release['executables'].items()
            if expected['source']['modality'] == modality}
    for reference, expected in release['executables'].items():
        entry = rows[reference]
        source = expected['source']
        assert entry['executable_id'] == source['id']
        assert entry['modality'] == source['modality']
        assert entry['native_contract'] == source['native_contract']
        assert 'bundle' not in entry
        shown, _ = invoke('executable', 'show', reference)
        assert all(inventory._same(shown[key], value) for key, value in entry.items())
        serving = source['serving']
        parameters = entry['parameters']
        intervals = (serving['bounds'] if source['modality'] == 'chat'
                     else {'pending_requests': serving['pending_requests']})
        for key, (minimum, maximum) in intervals.items():
            value = parameters[key]
            assert value['type'] == 'integer'
            assert value['state'] == ('fixed' if minimum == maximum else 'configurable')
            if minimum == maximum:
                assert type(value['value']) is int and value['value'] == minimum
            else:
                assert type(value['default']) is int and value['default'] == serving['defaults'][key]
                assert value['constraints']['minimum'] == minimum and value['constraints']['maximum'] == maximum
        if source['modality'] == 'chat':
            assert set(parameters) == {*intervals, 'mtp', 'mtp_depth'}
            maximum = serving['mtp_depth_max']
            assert parameters['mtp']['state'] == ('configurable' if maximum else 'fixed')
            assert parameters['mtp'].get('default', parameters['mtp'].get('value')) is serving['defaults']['mtp']
            assert parameters['mtp_depth']['state'] == ('configurable' if maximum else 'unsupported')
            if maximum:
                assert parameters['mtp_depth']['constraints']['maximum'] == maximum
                assert parameters['mtp_depth']['default'] == serving['defaults'].get('mtp_depth')
        else:
            assert set(parameters) == {'pending_requests', 'options'}
            assert parameters['options'] == {'type': 'object', 'state': 'fixed', 'value': {}}
    available, _ = invoke('model', 'available')
    expected_recipes = {item['id']: item for item in release['recipes'].values()}
    actual_recipes = {item['id']: item for item in available['items']}
    assert len(actual_recipes) == len(available['items']) and set(actual_recipes) == set(expected_recipes)
    for identifier, wanted in expected_recipes.items():
        row = actual_recipes[identifier]
        executable = release['executables'][wanted['executable']]['source']
        assert row['recipe_id'] == identifier and row['executable'] == wanted['executable']
        assert row['executable_id'] == executable['id'] and row['modality'] == executable['modality']
        preview, diagnostic = invoke('model', 'download', identifier, '--id', 'preview', '--dry-run')
        assert preview['state'] == 'preview' and 'hf download' in diagnostic
        assert preview['recipe_id'] == identifier and preview['executable'] == wanted['executable']
    return tuple(sorted(actual_recipes))


check_public_recipe_queries(invoke)
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
        from build_tools.validation.recipe_inventory import catalog_paths
        assert all((retained / 'loom-pipeline' / name).is_file() for name in catalog_paths())
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
