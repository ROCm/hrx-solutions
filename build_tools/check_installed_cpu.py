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
import hashlib
import importlib
import json
import os
from pathlib import Path
import site
import sys
import types
import unittest
import zipfile


HELPERS = ('gguf_fixtures', 'hem_http_test', 'hem_process_test', 'hem_instances_test')
# Installer concurrency fixtures import source-only test helpers in children.
# Keep those in source CPU discovery; installed CLI cases still exercise the
# real installed installer through its public entry point.
CONTRACTS = ('hem_process_test', 'hem_model_cli_test', 'hem_server_cli_test',
             'hem_instance_cli_test', 'hem_shutdown_test', 'hem_pi_contracts_test',
             'hem_pi_state_test', 'hem_pi_endpoints_test', 'hem_pi_cli_test')


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
        for name in archive.namelist():
            if not name.startswith('hrx_solutions/') or name.endswith('/'):
                continue
            relative = name[len('hrx_solutions/'):]
            target = package / relative
            target.resolve().relative_to(package.resolve())
            expected = hashlib.sha256(archive.read(name)).hexdigest()
            assert target.is_file() and digest(target) == expected, f'installed wheel mismatch: {name}'
            checked[relative] = expected
    assert '__init__.py' in checked and '_native/bin/loom-serve' in checked
    assert any(name.startswith('loom-pipeline/') for name in checked)
    return {'wheel': str(wheel), 'wheel_sha256': digest(wheel), 'installed_files': checked}


def source_parity(source_root, package):
    """Reject a stale installed wheel before running any contracts."""
    checked = {}
    for source in sorted((source_root / 'binding/python').glob('*.py')):
        module = importlib.import_module('hrx_solutions.' + source.stem)
        installed = Path(module.__file__).resolve()
        assert installed == (package / source.name).resolve(), f'import escaped installed package: {module}'
        assert source.read_bytes() == installed.read_bytes(), f'stale installed module: {source}'
        checked[str(source.relative_to(source_root))] = digest(source)
    for source in sorted((source_root / 'loom-pipeline').rglob('*')):
        if source.is_file():
            installed = package / source.relative_to(source_root)
            assert source.read_bytes() == installed.read_bytes(), f'stale installed catalog: {source}'
            checked[str(source.relative_to(source_root))] = digest(source)
    assert checked, 'no source files checked'
    return checked


def load_contracts(source_root, package):
    """Load only test fixtures from source; every product import uses the wheel.

    Rebasing also covers mock targets and Python snippets executed by children.
    Fixture discovery is the sole path adaptation, selecting the installed
    compatibility contract instead of a checkout-relative catalog.
    """
    test_package = types.ModuleType('tests')
    test_package.__path__ = []
    sys.modules['tests'] = test_package
    suite = unittest.TestSuite()
    for name in dict.fromkeys((*HELPERS, *CONTRACTS)):
        source = source_root / 'tests' / (name + '.py')
        module = types.ModuleType('tests.' + name)
        module.__file__ = str(source)
        module.__package__ = 'tests'
        sys.modules[module.__name__] = module
        setattr(test_package, name, module)
        code = source.read_text().replace('binding.python', 'hrx_solutions')
        exec(compile(code, str(source), 'exec'), module.__dict__)
        if name == 'gguf_fixtures':
            module.PROFILE_PATH = package / 'loom-pipeline/llm/qwen/qwen_27b/compatibility.json'
        if name in CONTRACTS:
            suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(module))
    assert suite.countTestCases() > 0, 'installed contract suite is empty'
    return suite


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--evidence', type=Path, required=True)
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
        suite = load_contracts(source_root, package)
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        evidence['tests_run'] = result.testsRun
        evidence['failures'] = len(result.failures)
        evidence['errors'] = len(result.errors)
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
