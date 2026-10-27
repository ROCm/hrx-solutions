"""Failure cleanup of the actual checker with owned, independently exec'd ELF."""
import ast
from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import unittest

from tests import hem_process_test as process_fixtures
from binding.python.hem_common import Deadline
from binding.python.control.hem_shutdown import ShutdownManager
from binding.python.control import hem_process


class PublicCatalogOracleTest(unittest.TestCase):
    """Real source CLI composition; this never substitutes for installed checks."""
    def setUp(self):
        from contextlib import redirect_stdout, redirect_stderr
        from binding.python import hem_cli
        from build_tools.validation import recipe_inventory
        self.release = recipe_inventory.load_release()
        self.cli = hem_cli
        root = Path(__file__).resolve().parents[2]
        checker = root / 'build_tools/validation/check_installed_wheel.py'
        tree = ast.parse(checker.read_text())
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                     and node.name == 'check_public_recipe_queries']
        namespace = {}
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(checker), 'exec'), namespace)
        self.check = namespace['check_public_recipe_queries']

    def invoke(self, *arguments):
        output, diagnostic = io.StringIO(), io.StringIO()
        with redirect_stdout(output), redirect_stderr(diagnostic):
            status = self.cli.main([*arguments, '--json'])
        self.assertEqual(status, 0, diagnostic.getvalue() + output.getvalue())
        return json.loads(output.getvalue()), diagnostic.getvalue()

    def test_real_source_public_queries_match_reviewed_inventory_and_capabilities(self):
        from unittest.mock import patch
        import tempfile
        with tempfile.TemporaryDirectory() as temporary:
            state = Path(temporary) / 'must-not-exist'
            with patch.dict(os.environ, {'LOOM_SERVE_DIR': str(state)}):
                actual = self.check(self.invoke)
            self.assertEqual(set(actual), {doc['id'] for doc in self.release['recipes'].values()})
            self.assertFalse(state.exists())

    def test_unseen_recipe_less_executable_is_expected_and_projection_drift_fails(self):
        from build_tools.validation.recipe_inventory_test import fixture
        from binding.python.models import pipelines
        from unittest.mock import patch
        import tempfile
        release, source, _ = fixture()
        # Data-only third executable with no recipe; public CLI and checker must
        # include it without extending any Python inventory.
        from copy import deepcopy
        original = next(iter(release['executables']))
        reference = 'independent/extra/executable.json'
        entry = deepcopy(release['executables'][original])
        entry['source']['id'] = 'additional-program'
        entry['source']['programs']['root'] = 'independent/extra'
        release['executables'][reference] = entry
        source[reference] = json.dumps(entry['source']).encode()
        source['independent/extra/main.loom'] = source['unseen/fresh-chat/main.loom']
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / 'catalogs'
            for relative, content in source.items():
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(content)
            with patch.object(pipelines, 'catalog_roots', return_value=(root,)), \
                 patch.dict(os.environ, {'LOOM_SERVE_DIR': str(Path(temporary) / 'state')}):
                self.check(self.invoke, release=release)
                for kind in ('missing', 'identity', 'limit', 'modality-default', 'recipe'):
                    def corrupt(*arguments):
                        result, diagnostic = self.invoke(*arguments)
                        if arguments == ('executable', 'list'):
                            if kind == 'missing': result['items'].pop()
                            elif kind == 'identity': result['items'][0]['executable_id'] = 'wrong'
                            elif kind == 'limit': result['items'][0]['parameters']['rows']['constraints']['maximum'] += 1
                        if kind == 'modality-default' and arguments == ('modality', 'list'):
                            result['items'][0]['defaults'] = {'rows': 1}
                        if kind == 'recipe' and arguments == ('model', 'available'):
                            result['items'].pop()
                        return result, diagnostic
                    with self.subTest(kind=kind), self.assertRaises(AssertionError):
                        self.check(corrupt, release=release)


class CheckerCleanupTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        process_fixtures.ProcessTests.setUpClass()

    @classmethod
    def tearDownClass(cls):
        process_fixtures.ProcessTests.tearDownClass()

    def setUp(self):
        self.case = process_fixtures.ProcessTests('test_two_servers_are_independent_reuse_and_shutdown')
        self.case.setUp()
        self.addCleanup(self.case.tearDown)
        self.calls = []
        self.start_error = None
        self.owned = None
        root = Path(__file__).resolve().parents[2]
        self.cli = self.case.root / 'checker-cli'
        self.cli.write_text(
            '#!' + sys.executable + '\n'
            'import functools,sys\n'
            f'sys.path.insert(0,{str(root)!r})\n'
            'from binding.python import hem_cli; from binding.python.cli import hem_server_cli\n'
            f'hem_server_cli.ProcessManager=functools.partial(hem_server_cli.ProcessManager,package_root={str(self.case.package)!r})\n'
            'args=sys.argv[1:]\n'
            'if args[:2]==["server","start"]:\n'
            ' args[args.index("--device")+1]="slow"\n'
            ' args += ["--timeout","0.15"]\n'
            'raise SystemExit(hem_cli.main(args))\n')
        self.cli.chmod(0o700)
        checker = root / 'build_tools/validation/check_installed_wheel.py'
        tree = ast.parse(checker.read_text())
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                     and node.name in ('invoke', 'check_native_lifecycle')]
        # Only source-import namespace changes: execute the actual checker bodies.
        for node in ast.walk(ast.Module(body=functions, type_ignores=[])):
            if isinstance(node, ast.ImportFrom) and node.module.startswith('hrx_solutions'):
                node.module = node.module.replace('hrx_solutions', 'binding.python', 1)
        namespace = dict(socket=socket, time=time, Path=Path, json=json, sys=sys,
                         subprocess=subprocess, cli=str(self.cli),
                         package_root=self.case.package, state_root=self.case.registry.root)
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(checker), 'exec'), namespace)
        self.original_invoke = namespace['invoke']
        self.namespace = namespace
        self.environment = dict(os.environ, LOOM_SERVE_DIR=str(self.case.registry.root),
                                PYTHONNOUSERSITE='1')
        self.environment.pop('PYTHONPATH', None)
        self.environment.pop('PYTHONHOME', None)
        from unittest.mock import patch
        self.environment_patch = patch.dict(os.environ, self.environment, clear=True)
        self.environment_patch.start()
        self.addCleanup(self.environment_patch.stop)

    def invoke(self, *arguments, **kwargs):
        self.calls.append(arguments)
        try:
            return self.original_invoke(*arguments, **kwargs)
        except AssertionError as error:
            if arguments[:2] == ('server', 'start'):
                self.start_error = error
                self.owned = self.case.registry.get_server('wheel-check')
                self.assertEqual(self.owned['pending']['action'], 'start')
                process = self.owned['process']
                if process is not None:
                    self.assertFalse(hem_process._confirmed_exit(process))
            raise

    def finish_own_launch(self):
        if self.owned:
            budget = Deadline.after(3)
            record = self.case.registry.get_server('wheel-check')
            while (record.get('pending') or {}).get('action') == 'start':
                result = self.case.manager.reconcile('wheel-check', budget)
                record = result.get('record') or self.case.registry.get_server('wheel-check')
                if (record.get('pending') or {}).get('action') == 'start':
                    time.sleep(min(.01, budget.remaining()))
            process = record['process']
            if process and not hem_process._confirmed_exit(process):
                ShutdownManager(self.case.registry, processes=self.case.manager).shutdown(
                    'wheel-check', budget, abort=True)
            if process:
                self.assertTrue(hem_process._confirmed_exit(process))
                logger = process.get('logger')
                if logger:
                    self.assertTrue(hem_process._confirmed_exit(logger))

    def test_timed_out_start_reconciles_and_aborts_only_its_owned_server(self):
        other = self.case.start('other')
        self.namespace['invoke'] = self.invoke
        try:
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                with self.assertRaises(AssertionError) as caught:
                    self.namespace['check_native_lifecycle']()
            self.assertIs(caught.exception, self.start_error)
            self.assertTrue(hem_process._confirmed_exit(self.owned['process']))
            self.assertTrue(hem_process._confirmed_exit(self.owned['process']['logger']))
            self.assertIn(('server', 'reconcile', 'wheel-check'), self.calls)
            self.assertTrue(any(call[:3] == ('server', 'shutdown', 'wheel-check')
                                and '--abort' in call for call in self.calls))
            self.assertFalse(any('other' in call for call in self.calls))
            with self.case.manager.pin(other) as pinned:
                self.assertEqual(self.case.manager.client(other, pinned, Deadline.after()).status()['data'], [])
        finally:
            self.finish_own_launch()
            ShutdownManager(self.case.registry, processes=self.case.manager).shutdown(
                'other', Deadline.after(3), abort=True)

    def test_cleanup_failure_reports_unresolved_process_without_hiding_start_error(self):
        def fail_cleanup(*arguments, **kwargs):
            if arguments[:2] == ('server', 'reconcile'):
                raise RuntimeError('injected reconciliation failure')
            return self.invoke(*arguments, **kwargs)
        self.namespace['invoke'] = fail_cleanup
        diagnostic = io.StringIO()
        try:
            with redirect_stdout(io.StringIO()), redirect_stderr(diagnostic):
                with self.assertRaises(AssertionError) as caught:
                    self.namespace['check_native_lifecycle']()
            self.assertIs(caught.exception, self.start_error)
            self.assertIn('cleanup', diagnostic.getvalue().lower())
            self.assertIn('injected reconciliation failure', diagnostic.getvalue())
            process = self.owned['process']
            self.assertTrue(process is None or not hem_process._confirmed_exit(process))
        finally:
            self.finish_own_launch()


if __name__ == '__main__':
    unittest.main()
