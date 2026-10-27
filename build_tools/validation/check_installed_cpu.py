#!/usr/bin/env python3
# Copyright 2026 The HRX Authors
# Licensed under the MIT License.
# SPDX-License-Identifier: MIT
"""Run existing CPU contract tests against installed code, outside the checkout.

Only test imports/embedded child imports are rebased from binding.python to
hrx_solutions. Product files are never rewritten. The existing tests select their
own temporary ELF, sparse GGUFs, loopback ports and Loom namespace; these checks
do not qualify native execution, GPU readiness, or model inference.
"""
import argparse
import ast
import hashlib
import importlib
import json
import os
from pathlib import Path
import site
import stat
import sys
import types
import unittest
import zipfile

try:
    from . import recipe_inventory
except ImportError:
    import recipe_inventory


CLI_FILES = (
    'cli/__init__.py',
    'cli/hem_arguments.py',
    'cli/hem_help.py',
    'cli/hem_render.py',
    'cli/hem_instance_cli.py',
    'cli/hem_model_cli.py',
    'cli/hem_server_cli.py',
    'cli/hem_modality_cli.py',
    'cli/hem_executable_cli.py',
    'cli/hem_pi_cli.py',
    'cli/hem_omp_cli.py',
    'cli/hem_opencode_cli.py',
    'cli/hem_openwebui_cli.py',
)

RETIRED_MODULES = (
    'models/moonshine_preparation.py', 'models/moonshine_preparer_identity.py',
    'models/pipelines_qwen.py', 'models/pipelines_moonshine.py',
                   'models/recipe_types.py', 'cli/hem_pipeline_cli.py')



SHARED_INTEGRATION_PYTHON_PATHS = (
    'integrations/__init__.py',
    'integrations/hem_integrations.py',
    'integrations/hem_client_documents.py',
    'integrations/hem_clients_endpoints.py',
)


OPENCODE_PYTHON_PATHS = (
    'integrations/__init__.py',
    'integrations/opencode/__init__.py',
    'integrations/opencode/hem_opencode.py',
)

OMP_PYTHON_PATHS = (
    'integrations/__init__.py',
    'integrations/omp/__init__.py',
    'integrations/omp/hem_omp.py',
    'integrations/omp/hem_omp_yaml.py',
)

OPENWEBUI_PYTHON_PATHS = (
    'integrations/__init__.py',
    'integrations/openwebui/__init__.py',
    'integrations/openwebui/hem_openwebui.py',
    'integrations/openwebui/hem_openwebui_http.py',
    'integrations/openwebui/hem_openwebui_state.py',
    'integrations/openwebui/hem_openwebui_stt.py',
    'integrations/openwebui/hem_openwebui_receipts.py',
    'integrations/openwebui/hem_openwebui_wire.py',
)

PI_PYTHON_PATHS = (
    'integrations/__init__.py',
    'integrations/pi/__init__.py',
    'integrations/pi/hem_pi.py',
    'integrations/pi/hem_pi_assets.py',
    'integrations/pi/hem_pi_contracts.py',
    'integrations/pi/hem_pi_endpoints.py',
    'integrations/pi/hem_pi_files.py',
    'integrations/pi/hem_pi_state.py',
)


MODEL_PYTHON_PATHS = (
    'models/__init__.py',
    'models/gguf.py',
    'models/hem_config.py',
    'models/hem_models.py',
    'models/model_download.py',
    'models/model_store.py',
    'models/preparation/__init__.py',
    'models/preparation/moonshine.py',
    'models/preparation/moonshine_contract.py',
    'models/recipes.py',
    'models/pipelines.py',
    'models/modalities/__init__.py',
    'models/modalities/chat.py',
    'models/modalities/asr.py',
)


HELPERS = ('gguf_fixtures', 'hem_lifecycle_fixture', 'recipe_config_fixture')
# Installer concurrency fixtures import source-only test helpers in children.
# Keep those in source CPU discovery; installed CLI cases still exercise the
# real installed installer through its public entry point.
# generic_modality_onboarding_test is the separately required source/producer
# gate: its child packages intentionally copy source and cannot qualify a wheel.
CONTRACTS = ('recipes_test', 'generic_discovery_cli_test',
             'trusted_preparation_test', 'preparation_boundary_test', 'moonshine_helper_test',
             'launcher_test', 'hem_operation_targets_test', 'hem_process_test', 'hem_model_cli_test', 'hem_server_cli_test',
             'hem_instance_cli_test', 'hem_shutdown_test', 'hem_help_test',
             'hem_render_test', 'hem_contracts_test', 'hem_config_test', 'pipeline_catalog_test', 'hem_instances_test',
             'hem_pi_contracts_test',
             'hem_pi_state_test', 'hem_pi_endpoints_test', 'hem_pi_cli_test', 'hem_clients_endpoints_test',
             'hem_client_documents_test', 'hem_omp_cli_test', 'hem_omp_installer_test', 'curated_assets_test', 'hem_models_test', 'model_download_test', 'moonshine_acquisition_test')


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest() if hasattr(hashlib, 'file_digest') else hashlib.sha256(stream.read()).hexdigest()


def snapshot(package):
    """Ignore Python bytecode, but include every shipped source/catalog/native file."""
    return {str(p.relative_to(package)): digest(p) for p in sorted(package.rglob('*'))
            if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc'}


def verify_wheel_payload(wheel, package):
    """Prove installed source, catalog and native bytes came from this exact wheel."""
    checked = {}
    with zipfile.ZipFile(wheel) as archive:
        for relative in RETIRED_MODULES:
            assert 'hrx_solutions/' + relative not in archive.namelist(), 'retired module in wheel: ' + relative
            assert not (package / relative).exists(), 'retired installed module: ' + relative
        for relative in PI_PYTHON_PATHS:
            name = 'hrx_solutions/' + relative
            members = [info for info in archive.infolist() if info.filename == name]
            assert (len(members) == 1 and not members[0].is_dir()
                    and stat.S_IFMT(members[0].external_attr >> 16) in (0, stat.S_IFREG)), f'missing or malformed Pi payload: {name}'
            assert not (package / relative).is_symlink(), f'installed Pi payload is not regular: {name}'
        release = recipe_inventory.load_release()
        for relative in (*MODEL_PYTHON_PATHS,
                         *('loom-pipeline/' + path for path in recipe_inventory.catalog_paths(release)),
                         *recipe_inventory.operation_paths(release)):
            name = 'hrx_solutions/' + relative
            members = [info for info in archive.infolist() if info.filename == name]
            assert (len(members) == 1 and not members[0].is_dir()
                    and stat.S_IFMT(members[0].external_attr >> 16) in (0, stat.S_IFREG)), f'missing or malformed model payload: {name}'
            assert not (package / relative).is_symlink(), f'installed model payload is not regular: {name}'
        for relative in OMP_PYTHON_PATHS:
            name = 'hrx_solutions/' + relative
            members = [info for info in archive.infolist() if info.filename == name]
            assert (len(members) == 1 and not members[0].is_dir()
                    and stat.S_IFMT(members[0].external_attr >> 16) in (0, stat.S_IFREG)), f'missing or malformed OMP payload: {name}'
            assert not (package / relative).is_symlink(), f'installed OMP payload is not regular: {name}'
        for relative in OPENCODE_PYTHON_PATHS:
            name = 'hrx_solutions/' + relative
            members = [info for info in archive.infolist() if info.filename == name]
            assert (len(members) == 1 and not members[0].is_dir()
                    and stat.S_IFMT(members[0].external_attr >> 16) in (0, stat.S_IFREG)), f'missing or malformed OpenCode payload: {name}'
            assert not (package / relative).is_symlink(), f'installed OpenCode payload is not regular: {name}'
        for relative in OPENWEBUI_PYTHON_PATHS:
            name = 'hrx_solutions/' + relative
            members = [info for info in archive.infolist() if info.filename == name]
            assert (len(members) == 1 and not members[0].is_dir()
                    and stat.S_IFMT(members[0].external_attr >> 16) in (0, stat.S_IFREG)), f'missing or malformed OpenWebUI payload: {name}'
            assert not (package / relative).is_symlink(), f'installed OpenWebUI payload is not regular: {name}'
        for relative in SHARED_INTEGRATION_PYTHON_PATHS:
            name = 'hrx_solutions/' + relative
            members = [info for info in archive.infolist() if info.filename == name]
            assert (len(members) == 1 and not members[0].is_dir()
                    and stat.S_IFMT(members[0].external_attr >> 16) in (0, stat.S_IFREG)), f'missing or malformed shared integration payload: {name}'
            assert not (package / relative).is_symlink(), f'installed shared integration payload is not regular: {name}'
        for name in archive.namelist():
            if not name.startswith('hrx_solutions/') or name.endswith('/'):
                continue
            relative = name[len('hrx_solutions/'):]
            target = package / relative
            target.resolve().relative_to(package.resolve())
            expected = hashlib.sha256(archive.read(name)).hexdigest()
            assert target.is_file() and digest(target) == expected, f'installed wheel mismatch: {name}'
            checked[relative] = expected
    assert set(CLI_FILES) <= checked.keys(), 'wheel is missing required CLI modules'
    for name in ('tui/__init__.py', 'tui/hem_tui.py', 'tui/hem_tui_details.py',
                     'tui/hem_tui_logs.py', 'tui/hem_tui_provider.py'):
        assert name in checked, f'wheel is missing TUI module: {name}'
    for name in ('control/__init__.py', 'control/hem_http.py', 'control/hem_instances.py', 'control/hem_operation_targets.py', 'control/hem_process.py', 'control/hem_shutdown.py', 'control/hem_state.py'):
        assert name in checked, f'wheel is missing Control module: {name}'
    assert '__init__.py' in checked and '_native/bin/loom-serve' in checked
    assert any(name.startswith('loom-pipeline/') for name in checked)
    return {'wheel': str(wheel), 'wheel_sha256': digest(wheel), 'installed_files': checked}


def source_parity(source_root, package):
    """Reject a stale installed wheel before running any contracts."""
    checked = {}
    python_root = source_root / 'binding/python'
    for source in sorted(python_root.rglob('*.py')):
        relative = source.relative_to(python_root)
        parts = relative.with_suffix('').parts
        if parts[-1] == '__init__':
            parts = parts[:-1]
        module = importlib.import_module('.'.join(('hrx_solutions', *parts)))
        installed = Path(module.__file__).resolve()
        assert installed.is_relative_to(package.resolve()), f'import escaped installed package: {module}'
        assert installed == (package / relative).resolve(), f'import escaped installed package: {module}'
        assert source.read_bytes() == installed.read_bytes(), f'stale installed module: {source}'
        checked[str(source.relative_to(source_root))] = digest(source)
    catalog_root = source_root / 'loom-pipeline'
    if catalog_root.is_dir():
        recipe_module = importlib.import_module('hrx_solutions.models.recipes')
        executable_module = importlib.import_module('hrx_solutions.models.pipelines')
        release = recipe_inventory.load_release()
        recipe_inventory.validate_installed_catalogs(package, recipe_module.parse,
            executable_module.parse, source_root, release=release)
        references = [path.relative_to(catalog_root).as_posix()
                      for path in catalog_root.glob('**/executable.json')]
        documents = recipe_inventory.validate_executables(
            lambda relative: (catalog_root / relative).read_bytes(), references,
            executable_module.parse, release=release, encoding='source')
        for reference, document in documents.items():
            checked['loom-pipeline/' + reference] = digest(catalog_root / reference)
            for relative, checksum in document['programs']['files'].items():
                checked['loom-pipeline/' + document['programs']['root'] + '/' + relative] = checksum
        for relative in release['recipes']:
            checked['loom-pipeline/' + relative] = digest(catalog_root / relative)
        notices = {relative for record in release['executables'].values() for relative in record['notices']}
        for relative in notices:
            assert (catalog_root / relative).read_bytes() == (package / 'loom-pipeline' / relative).read_bytes(), 'installed notice differs'
        assert not list(package.rglob('*.loom')), 'installed wheel contains executable sources'
    assert checked, 'no source files checked'
    return checked


def admit_fixtures(source_root, names):
    """Check the complete fixture inventory before executing any fixture code.

    Walk function bodies too: some contracts import another test's fixtures only
    when the test runs. Selection must never trim these transitive dependencies.
    """
    sources = {}
    for name in names:
        path = source_root / 'tests' / (name + '.py')
        assert path.is_file(), f'missing installed-contract fixture: tests.{name}'
        sources[name] = path.read_text()
    for name, code in sources.items():
        dependencies = set()
        for node in ast.walk(ast.parse(code, filename=str(source_root / 'tests' / (name + '.py')))):
            if isinstance(node, ast.ImportFrom) and node.module == 'tests':
                dependencies.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module and node.module.startswith('tests.'):
                dependencies.add(node.module.split('.')[1])
            elif isinstance(node, ast.Import):
                dependencies.update(alias.name.split('.')[1] for alias in node.names
                                    if alias.name.startswith('tests.'))
        missing = dependencies - sources.keys()
        assert not missing, f'incomplete installed-contract fixtures for tests.{name}: {sorted(missing)}'
    return sources


def load_contracts(source_root, package):
    """Load only test fixtures from source; every product import uses the wheel.

    Rebasing also covers mock targets and Python snippets executed by children.
    Complete fixture admission runs before selection; product imports always
    resolve to installed modules.
    """
    sources = admit_fixtures(source_root, dict.fromkeys((*HELPERS, *CONTRACTS)))
    test_package = types.ModuleType('tests')
    test_package.__path__ = []
    sys.modules['tests'] = test_package
    suite = unittest.TestSuite()
    for name, text in sources.items():
        source = source_root / 'tests' / (name + '.py')
        module = types.ModuleType('tests.' + name)
        module.__file__ = str(source)
        module.__package__ = 'tests'
        sys.modules[module.__name__] = module
        setattr(test_package, name, module)
        code = text.replace('binding.python', 'hrx_solutions')
        exec(compile(code, str(source), 'exec'), module.__dict__)
        if name in CONTRACTS:
            suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(module))
    assert suite.countTestCases() > 0, 'installed contract suite is empty'
    return suite


def contract_cases(suite):
    for test in suite:
        if isinstance(test, unittest.TestSuite):
            yield from contract_cases(test)
        else:
            yield test


def select_contracts(suite, test_ids):
    """Select exact discovered IDs after loading the complete fixture inventory."""
    if not test_ids:
        return suite
    if len(set(test_ids)) != len(test_ids):
        raise ValueError('duplicate installed-contract test IDs')
    cases = {test.id(): test for test in contract_cases(suite)}
    missing = set(test_ids) - cases.keys()
    if missing:
        raise ValueError(f'unknown installed-contract test IDs: {sorted(missing)}')
    return unittest.TestSuite(cases[test_id] for test_id in test_ids)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--test', action='append', default=[], metavar='EXACT_ID',
                        help='Run this exact discovered test ID; repeat to select several. '
                             'All fixture modules are still loaded; omission runs the full suite.')
    args = parser.parse_args(argv)
    source_root = args.source_root.resolve()
    assert sys.prefix != sys.base_prefix and not site.ENABLE_USER_SITE
    assert 'include-system-site-packages = false' in (Path(sys.prefix) / 'pyvenv.cfg').read_text()
    assert not os.environ.get('PYTHONPATH') and not os.environ.get('PYTHONHOME')
    assert not Path.cwd().resolve().is_relative_to(source_root), 'run outside checkout'
    state = Path(os.environ['LOOM_SERVE_DIR'])
    assert state.is_absolute() and not state.exists(), 'use a new private Loom namespace'
    os.environ['HEM_TEST_PACKAGE'] = 'hrx_solutions'
    import hrx_solutions
    package = Path(hrx_solutions.__file__).resolve().parent
    package.relative_to(Path(sys.prefix).resolve())
    before = snapshot(package)
    evidence = {'success': False, 'python': sys.executable, 'package': str(package),
                'source_root': str(source_root), 'state': str(state),
                'qualification': 'Installed Python CPU contracts with test-owned ELF and sparse GGUF; no GPU/native inference qualification'}
    try:
        evidence['source_files'] = source_parity(source_root, package)
        suite = select_contracts(load_contracts(source_root, package), args.test)
        evidence['fixture_admission'] = True
        evidence['test_ids'] = [test.id() for test in contract_cases(suite)]
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        evidence['tests_run'] = result.testsRun
        evidence['failures'] = len(result.failures)
        evidence['errors'] = len(result.errors)
        evidence['skipped'] = len(result.skipped)
        evidence['failed_test_ids'] = [test.id() for test, _ in result.failures]
        evidence['error_test_ids'] = [test.id() for test, _ in result.errors]
        assert result.wasSuccessful(), 'installed CPU contracts failed'
        assert snapshot(package) == before, 'installed wheel payload changed during tests'
        evidence['payload_unchanged'] = True
        evidence['success'] = True
    finally:
        args.evidence.parent.mkdir(parents=True, exist_ok=True)
        args.evidence.write_text(json.dumps(evidence, indent=2) + '\n')
    print(f"{evidence['tests_run']} installed CPU contracts passed; payload unchanged")


if __name__ == '__main__':
    main()
