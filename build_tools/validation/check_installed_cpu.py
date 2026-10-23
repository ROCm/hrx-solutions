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


CLI_FILES = (
    'cli/__init__.py',
    'cli/hem_arguments.py',
    'cli/hem_help.py',
    'cli/hem_render.py',
    'cli/hem_instance_cli.py',
    'cli/hem_model_cli.py',
    'cli/hem_server_cli.py',
    'cli/hem_pipeline_cli.py',
    'cli/hem_pi_cli.py',
    'cli/hem_omp_cli.py',
    'cli/hem_opencode_cli.py',
    'cli/hem_openwebui_cli.py',
)


MODEL_PYTHON_PATHS = (
    'models/__init__.py',
    'models/curated_catalog.py',
    'models/gguf.py',
    'models/hem_config.py',
    'models/hem_models.py',
    'models/hem_model_operations.py',
    'models/model_download.py',
    'models/model_store.py',
    'models/moonshine_preparation.py',
    'models/moonshine_preparer_identity.py',
    'models/pipeline_validation.py',
    'models/qwen_validation.py',
)


HELPERS = ('gguf_fixtures', 'hem_lifecycle_fixture')
# Installer concurrency fixtures import source-only test helpers in children.
# Keep those in source CPU discovery; installed CLI cases still exercise the
# real installed installer through its public entry point.
CONTRACTS = ('hem_operation_targets_test', 'hem_process_test', 'hem_model_cli_test', 'hem_server_cli_test',
             'hem_instance_cli_test', 'hem_shutdown_test', 'hem_help_test',
             'hem_render_test', 'hem_contracts_test', 'hem_config_test', 'pipeline_catalog_test', 'hem_instances_test',
             'hem_pi_contracts_test',
             'hem_pi_state_test', 'hem_pi_endpoints_test', 'hem_pi_cli_test', 'hem_clients_endpoints_test',
             'hem_client_documents_test', 'hem_omp_cli_test', 'hem_omp_installer_test', 'curated_assets_test', 'moonshine_acquisition_test')


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
        for relative in (*MODEL_PYTHON_PATHS, 'pipeline_validation.py'):
            name = 'hrx_solutions/' + relative
            members = [info for info in archive.infolist() if info.filename == name]
            assert (len(members) == 1 and not members[0].is_dir()
                    and stat.S_IFMT(members[0].external_attr >> 16) in (0, stat.S_IFREG)), f'missing or malformed model payload: {name}'
            assert not (package / relative).is_symlink(), f'installed model payload is not regular: {name}'
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
        assert installed == (package / relative).resolve(), f'import escaped installed package: {module}'
        assert source.read_bytes() == installed.read_bytes(), f'stale installed module: {source}'
        checked[str(source.relative_to(source_root))] = digest(source)
    authored_root = source_root / 'loom-pipeline/llm'
    if authored_root.is_dir():
        from hrx_solutions.models.pipeline_validation import catalog_modules
        installed_root = package / 'loom-pipeline/llm'
        build = json.loads((installed_root / 'catalog-build.json').read_text())
        assert build.get('format') == 2, 'unsupported catalog build metadata'
        architectures = {'qwen_27b', 'qwen_08b'}
        assert set(build.get('variants', {})) == architectures, 'catalog build inventory mismatch'
        allowed = {'catalog-build.json', 'LICENSE-APACHE-2.0'}
        for architecture in sorted(architectures):
            relative = Path('qwen') / architecture
            authored_manifest = authored_root / relative / 'sources.txt'
            entries = [line.strip() for line in authored_manifest.read_text().splitlines()
                       if line.strip() and not line.strip().startswith('#')]
            expected = {os.path.normpath((relative / entry).as_posix()) for entry in entries}
            assert all(not p.startswith(('../', '/')) for p in expected), 'authored catalog path escapes root'
            variant = build['variants'][architecture]
            assert set(variant) == {'identifier', 'sources', 'bytecode_sha256'}, 'catalog variant metadata mismatch'
            assert set(variant.get('sources', {})) == expected, 'catalog source inventory mismatch'
            for identity in sorted(expected):
                authored = authored_root / identity
                authored.resolve(strict=True).relative_to(authored_root.resolve())
                assert digest(authored) == variant['sources'][identity], f'stale installed catalog input: {identity}'
                checked['loom-pipeline/llm/' + identity] = variant['sources'][identity]
            profile = relative / 'compatibility.json'
            assert (authored_root / profile).read_bytes() == (installed_root / profile).read_bytes(), 'installed compatibility profile mismatch'
            identifier = json.loads((authored_root / profile).read_text())['identifier']
            assert variant['identifier'] == identifier, 'installed variant mismatch'
            artifact = installed_root / relative / 'pipeline.loombc'
            assert artifact.read_bytes().startswith(b'LOOM') and digest(artifact) == variant['bytecode_sha256'], f'installed bytecode mismatch: {architecture}'
            descriptor = relative / 'pipeline.json'
            assert json.loads((installed_root / descriptor).read_text()) == {
                'format': 1, 'variant': identifier, 'module': 'pipeline.loombc',
                'sha256': variant['bytecode_sha256']}, 'installed pipeline descriptor mismatch'
            modules = catalog_modules(installed_root / relative)
            assert (modules.prepare, modules.control, modules.weights, modules.providers) == (
                'pipeline.loombc', 'pipeline.loombc', 'pipeline.loombc', ('pipeline.loombc',)), 'installed pipeline roles mismatch'
            allowed.update({(relative / 'pipeline.loombc').as_posix(), profile.as_posix(), descriptor.as_posix()})
        actual = {p.relative_to(installed_root).as_posix() for p in installed_root.rglob('*') if p.is_file()}
        assert actual == allowed, 'installed catalog file inventory mismatch'
        assert not list(package.rglob('*.loom')), 'installed wheel contains executable sources'
        assert (authored_root / 'LICENSE-APACHE-2.0').read_bytes() == (installed_root / 'LICENSE-APACHE-2.0').read_bytes()
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
    Fixture discovery is the sole path adaptation, selecting the installed
    compatibility contract instead of a checkout-relative catalog.
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
        if name == 'gguf_fixtures':
            module.PROFILE_PATH = package / 'loom-pipeline/llm/qwen/qwen_27b/compatibility.json'
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
