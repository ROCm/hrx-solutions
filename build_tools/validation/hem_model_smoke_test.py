"""Failure propagation in the installed real-model smoke harness."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import tempfile
import sqlite3
import socket
import signal
import runpy
import unittest
import zipfile
from contextlib import redirect_stderr, redirect_stdout
import io

from build_tools.validation import check_installed_wheel_test as checker_fixtures
from binding.python import hem_process, hem_display
from binding.python.hem_common import Deadline
from binding.python.hem_shutdown import ShutdownManager
from binding.python.hem_state import Registry
from unittest import mock

from build_tools.validation import hem_model_smoke
from build_tools.validation import check_installed_cpu
import types


class SmokeFailureTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.gate = hem_model_smoke.Gate(argparse.Namespace(run=self.root))
        self.gate.states['a'] = self.root / 'venv/bin/python'

    def test_source_commit_uses_github_identity_without_inspecting_checkout(self):
        with mock.patch.dict(os.environ, {'GITHUB_SHA': 'ci-commit'}, clear=True), \
             mock.patch.object(subprocess, 'check_output',
                               side_effect=AssertionError('Git must not inspect the CI checkout')):
            self.assertEqual(hem_model_smoke.source_commit(self.root), 'ci-commit')

    def test_source_commit_falls_back_to_checkout_identity_locally(self):
        with mock.patch.dict(os.environ, {}, clear=True), \
             mock.patch.object(subprocess, 'check_output', return_value='local-commit\n') as git:
            self.assertEqual(hem_model_smoke.source_commit(self.root), 'local-commit')
        git.assert_called_once_with(
            ['git', '-C', str(self.root), 'rev-parse', 'HEAD'], text=True)

    def test_internal_private_root_separates_state_from_report_and_venv_paths(self):
        from build_tools.ci_tools.local_ci_runtime import create
        import shutil
        owner = create(self.root)
        runtime = Path(owner['root'])
        self.addCleanup(shutil.rmtree, runtime)
        with mock.patch.dict(os.environ, {'HRX_LOCAL_CI_RUNTIME_ROOT': str(runtime / 'model')}):
            gate = hem_model_smoke.Gate(argparse.Namespace(run=self.root))
        gate.states['a'] = self.root / 'a/venv/bin/python'
        (runtime / 'model/a').mkdir(mode=0o700)
        registry = Registry(gate.state_path('a'))
        self.assertEqual(gate.verify_empty_state('a')['pi_target'], 0)
        gate.namespace_roots['c'] = 'a'
        self.assertEqual(gate.state_path('c'), gate.state_path('a'))
        child = subprocess.CompletedProcess([], 0, '{"completed":true}', '')
        with mock.patch.object(hem_model_smoke, 'run_child', return_value=child) as run:
            gate.command('a', 'server', 'list')
        self.assertEqual(run.call_args.kwargs['env']['LOOM_SERVE_DIR'], str(runtime / 'model/a/state'))
        self.assertEqual(run.call_args.args[0][0], str(self.root / 'a/venv/bin/loom-control'))
        self.assertTrue((self.root / 'results.json').is_file())
        self.assertFalse((runtime / 'model/results.json').exists())
        from binding.python.hem_pi_state import integration_lock
        from binding.python.hem_common import Deadline
        with integration_lock(registry.root, Deadline.after()):
            pass

    def test_full_profile_accepts_current_schema_and_requires_empty_pi_reservation(self):
        registry = Registry(self.root / 'a/state')
        counts = self.gate.verify_empty_state('a')
        self.assertEqual(set(counts), {'models', 'servers', 'instances', 'pi_target'})
        self.assertEqual(counts['pi_target'], 0)
        with sqlite3.connect(registry.database) as connection:
            connection.execute('INSERT INTO pi_target VALUES(1,?,?)', ('/unused/pi', 'test-integration'))
        with self.assertRaises(AssertionError):
            self.gate.verify_empty_state('a')

    def test_failed_child_is_recorded_and_fails_gate(self):
        failed = subprocess.CompletedProcess([], 1, '{"completed":false}', 'native failure')
        with mock.patch.object(hem_model_smoke, 'run_child', return_value=failed):
            with self.assertRaises(AssertionError):
                self.gate.command('a', 'instance', 'start', 'model')
        evidence = json.loads((self.root / 'results.json').read_text())
        self.assertEqual(evidence['commands'][0]['exit'], 1)
        self.assertFalse(evidence['success'])

    def test_staged_pi_environment_is_used_by_public_configure(self):
        child = subprocess.CompletedProcess([], 0, '{"completed":true}', '')
        overrides = {'PATH': '/isolated-node/bin:/isolated-pi/node_modules/.bin',
                     'PI_CODING_AGENT_DIR': '/isolated/pi', 'PI_OFFLINE': '1'}
        with mock.patch.object(hem_model_smoke, 'run_child', return_value=child) as run:
            self.gate.command('a', 'pi', 'configure', '--pi-dir', '/isolated/pi', env_overrides=overrides)
        environment = run.call_args.kwargs['env']
        self.assertEqual(environment['PI_CODING_AGENT_DIR'], '/isolated/pi')
        self.assertIn('/isolated-pi/node_modules/.bin', environment['PATH'])
        self.assertEqual(environment['LOOM_SERVE_DIR'], str(self.root / 'a/state'))
        self.assertNotIn('PYTHONPATH', environment)
        self.assertNotIn('PYTHONHOME', environment)

    def test_multiline_equivalents_preserve_native_order_and_document_check(self):
        document = self.root / 'registration.json'
        document.write_text('{"admission":"drained"}')
        output = io.StringIO()
        for action in ('register', 'activate', 'undrain'):
            argv = ['curl', '--fail-with-body', '--request', 'POST',
                    'http://localhost/v1/models/' + action]
            if action == 'register':
                argv += ['--data-binary', '@' + str(document)]
            hem_display.emit(hem_display.command(argv), stream=output)
        diagnostic = output.getvalue()
        self.assertIn('\\\n', diagnostic)
        child = subprocess.CompletedProcess([], 0, '{"completed":true}', diagnostic)
        with mock.patch.object(hem_model_smoke, 'run_child', return_value=child):
            self.gate.command('a', 'instance', 'start', 'model')
        self.assertEqual(self.gate.results['commands'][0]['native_order'],
                         ['register', 'activate', 'undrain'])

    def test_setup_uses_installed_catalog_authority_and_propagates_rejection(self):
        wheel = self.root / 'wheel.whl'
        wheel.write_bytes(b'fixture identity')
        source = self.root / 'source'
        package = Path(sys.prefix) / 'lib/fixture-hrx-solutions'
        self.gate.args = argparse.Namespace(wheel=wheel, wheel_sha256=hem_model_smoke.digest(wheel),
                                            source=source, run=self.root)
        fake_package = types.ModuleType('hrx_solutions')
        fake_package.__file__ = str(package / '__init__.py')
        with mock.patch.dict(sys.modules, hrx_solutions=fake_package), \
             mock.patch.object(check_installed_cpu, 'verify_wheel_payload') as verify, \
             mock.patch.object(check_installed_cpu, 'source_parity',
                               side_effect=AssertionError('installed bytecode mismatch')) as parity:
            with self.assertRaisesRegex(AssertionError, 'installed bytecode mismatch'):
                self.gate.setup()
        verify.assert_called_once_with(wheel, package)
        parity.assert_called_once_with(source, package)
        self.assertFalse(self.gate.attempts)

    def test_successful_exit_cannot_mask_invalid_native_order(self):
        diagnostic = '\n'.join('curl --request POST http://localhost/v1/models/' + action
                               for action in ('register', 'undrain', 'activate'))
        child = subprocess.CompletedProcess([], 0, '{"completed":true}', diagnostic)
        with mock.patch.object(hem_model_smoke, 'run_child', return_value=child):
            with self.assertRaises(AssertionError):
                self.gate.command('a', 'instance', 'start', 'model')
        self.assertFalse(self.gate.results['success'])


class SmokeNamespacePayloadTest(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory();self.addCleanup(self.temporary.cleanup)
        self.root=Path(self.temporary.name)
        self.files={'__init__.py':b'',
                    '_native/bin/loom-serve':b'ELF fixture',
                    'loom-pipeline/fixture/pipeline.loombc':b'LOOM fixture'}
        source = Path(__file__).resolve().parents[2] / 'binding/python'
        for name in (
            'integrations/__init__.py',
            'integrations/opencode/__init__.py',
            'integrations/opencode/hem_opencode.py',
            'integrations/omp/__init__.py',
            'integrations/omp/hem_omp.py',
            'integrations/omp/hem_omp_yaml.py',
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
            'pipeline_validation.py',
):
            self.files[name] = (source / name).read_bytes()
        for relative in ('tui/__init__.py', 'tui/hem_tui.py', 'tui/hem_tui_details.py',
                     'tui/hem_tui_logs.py', 'tui/hem_tui_provider.py'):
            self.files[relative] = (source / relative).read_bytes()
        from build_tools.validation.check_installed_cpu import CLI_FILES
        for name in CLI_FILES:
            self.files[name] = (source / name).read_bytes()
        self.wheel=self.root/'fixture.whl'
        with zipfile.ZipFile(self.wheel,'w') as archive:
            for name,data in self.files.items():
                archive.writestr('hrx_solutions/'+name,data)
        self.coordinator=self.root/'venv'
        self.coordinator.mkdir(mode=0o755)
        self.package=self.coordinator/'hrx_solutions'
        self.install(self.package)

    def install(self,package):
        for name,data in self.files.items():
            target=package/name;target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(data)

    def setup_namespaces(self,*,corrupt=False,reuse=False,invalid=None):
        lifecycle=self.root/'lifecycle';lifecycle.mkdir(mode=0o700)
        args=argparse.Namespace(run=lifecycle,wheel=self.wheel,
            wheel_sha256=hem_model_smoke.digest(self.wheel),source=self.root/'source',
            profile='full',acquire08=False,native_provenance='synthetic fixture')
        if reuse:
            (self.coordinator/'bin').mkdir(mode=0o700)
            (self.coordinator/'bin/python').write_text('fixture interpreter')
            (self.coordinator/'pyvenv.cfg').write_text('include-system-site-packages = false\n')
            owner=dict(pid=os.getppid(),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                start_ticks=int(Path(f'/proc/{os.getppid()}/stat').read_text().rsplit(')',1)[1].split()[19]),python=str(self.coordinator/'bin/python'),wheel_sha256=args.wheel_sha256)
            if invalid=='wheel':owner['wheel_sha256']='wrong wheel'
            if invalid=='parent':owner['pid']+=1
            if invalid=='generation':owner['start_ticks']+=1
            if invalid=='boot':owner['boot_id']='prior boot'
            if invalid=='python':owner['python']=str(self.root/'unowned/bin/python')
            args.coordinator_owner=self.root/'coordinator.json'
            args.coordinator_owner.write_text(json.dumps(owner,sort_keys=True)+'\n')
            args.coordinator_owner.chmod(0o600)
            if invalid=='private':self.coordinator.chmod(0o777)
            if invalid=='symlink':
                original=args.coordinator_owner.with_suffix('.original');args.coordinator_owner.rename(original)
                args.coordinator_owner.symlink_to(original)
            if invalid=='state':(lifecycle/'a/state').mkdir(parents=True)
            if invalid=='system':(self.coordinator/'pyvenv.cfg').write_text('include-system-site-packages = true\n')
        gate=hem_model_smoke.Gate(args)
        packages=[self.package]
        def child(argv,**kwargs):
            if argv[:3]==['python3','-m','venv']:
                Path(argv[3]).mkdir(parents=True)
            else:
                self.assertEqual(argv[1:4],['-m','pip','install'])
                venv=Path(argv[0]).parent.parent
                package=venv/'lib/hrx_solutions'
                self.install(package);packages.append(package)
                if corrupt and venv.parent.name=='b':
                    (package/'models/model_store.py').write_bytes(b'stale namespace module')
            return subprocess.CompletedProcess(argv,0)
        def identity(argv,**kwargs):
            venv=Path(argv[0]).parent.parent
            package=self.package if venv==self.coordinator else venv/'lib/hrx_solutions'
            return json.dumps(dict(package=str(package/'__init__.py'),
                                   prefix=str(venv),user_site=False))
        fake=types.ModuleType('hrx_solutions');fake.__file__=str(self.package/'__init__.py')
        with mock.patch.dict(sys.modules,hrx_solutions=fake), \
             mock.patch.object(sys,'prefix',str(self.coordinator)), \
             mock.patch.object(sys,'executable',str(self.coordinator/'bin/python') if reuse else sys.executable), \
             mock.patch.object(hem_model_smoke,'run_child',side_effect=child), \
             mock.patch.object(subprocess,'check_output',side_effect=identity), \
             mock.patch.object(check_installed_cpu,'source_parity',return_value={'source':'fixture'}) as source, \
             mock.patch.object(check_installed_cpu,'verify_wheel_payload',wraps=check_installed_cpu.verify_wheel_payload) as verify:
            if invalid:
                with self.assertRaises((AssertionError,OSError)):gate.setup()
                self.assertFalse(gate.states)
                self.assertEqual(len(packages),1)
                return gate
            if corrupt:
                with self.assertRaisesRegex(AssertionError,'installed wheel mismatch: hrx_solutions/models/model_store.py'):
                    gate.setup()
            else:
                gate.setup()
            self.assertEqual([call.args for call in verify.call_args_list],
                             [(self.wheel,package) for package in ([self.package,*packages] if reuse else packages)])
            self.assertEqual(len(packages),2 if reuse else 3)
            source.assert_called_once_with(args.source,self.package)
        return gate

    def test_setup_records_shared_exact_payload_for_both_namespaces(self):
        gate=self.setup_namespaces()
        for namespace in ('a','b'):
            parity=gate.results[namespace]['installed_parity']
            self.assertEqual(set(parity),set(self.files))
            self.assertEqual(parity['models/model_store.py'],hashlib.sha256(self.files['models/model_store.py']).hexdigest())
            self.assertEqual(gate.results[namespace]['native_sha256'],
                             hashlib.sha256(self.files['_native/bin/loom-serve']).hexdigest())

    def test_setup_refuses_second_namespace_drift_outside_old_filtered_members(self):
        self.setup_namespaces(corrupt=True)

    def test_explicit_owned_coordinator_reuses_only_A_and_keeps_B_state_independent(self):
        gate=self.setup_namespaces(reuse=True)
        self.assertEqual(gate.states['a'],self.coordinator/'bin/python')
        self.assertNotEqual(gate.states['a'],gate.states['b'])
        self.assertTrue(gate.results['a']['reused_coordinator'])
        self.assertFalse(gate.results['b']['reused_coordinator'])
        self.assertFalse((gate.root/'a/venv').exists())
        self.assertTrue((gate.root/'b/venv').is_dir())
        self.assertNotEqual(gate.state_path('a'),gate.state_path('b'))
        self.assertFalse(gate.state_path('a').exists() or gate.state_path('b').exists())
        self.assertEqual(gate.state_path('a'),gate.root/'a/state')

    def test_missing_handoff_uses_standalone_A_even_in_matching_interpreter_layout(self):
        gate=self.setup_namespaces()
        self.assertNotEqual(gate.states['a'],self.coordinator/'bin/python')
        self.assertFalse(gate.results['a']['reused_coordinator'])

    def test_wrong_wheel_or_unowned_handoff_is_refused_before_namespace_setup(self):
        for reason in ('wheel','parent','python','private','generation','boot','symlink','state','system'):
            with self.subTest(reason=reason):
                # Each case needs an independent fresh report/receipt namespace.
                with tempfile.TemporaryDirectory() as temporary:
                    previous=(self.root,self.coordinator,self.package,self.wheel)
                    self.root=Path(temporary)
                    self.coordinator=self.root/'venv';self.coordinator.mkdir(mode=0o755)
                    self.package=self.coordinator/'hrx_solutions';self.install(self.package)
                    self.wheel=self.root/'fixture.whl'
                    with zipfile.ZipFile(self.wheel,'w') as archive:
                        for name,data in self.files.items():archive.writestr('hrx_solutions/'+name,data)
                    try:self.setup_namespaces(reuse=True,invalid=reason)
                    finally:self.root,self.coordinator,self.package,self.wheel=previous

    def test_supervisor_emits_private_exact_wheel_handoff_to_real_helper_entry(self):
        script=Path(hem_model_smoke.__file__).with_name('model_smoke.py')
        run=self.root/'supervisor';run.mkdir(mode=0o700)
        commands=[]
        def child(argv,**kwargs):
            commands.append(argv)
            return mock.Mock(poll=mock.Mock(return_value=0),wait=mock.Mock(return_value=0))
        previous=Path.cwd()
        try:
            with mock.patch.object(sys,'argv',[str(script),str(self.wheel),str(run),
                    str(script.parent.parent),str(script.parents[2]),'ci08','--acquire08']), \
                 mock.patch.object(subprocess,'Popen',side_effect=child), \
                 mock.patch.object(signal,'signal'), redirect_stdout(io.StringIO()):
                with self.assertRaises(SystemExit) as exited:
                    runpy.run_path(str(script),run_name='__main__')
                self.assertEqual(exited.exception.code,0)
        finally:
            os.chdir(previous)
        marker=run/'coordinator.json'
        owner=json.loads(marker.read_text())
        self.assertEqual(owner,dict(pid=os.getpid(),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
            start_ticks=int(Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]),python=str(run/'venv/bin/python'),
            wheel_sha256=hem_model_smoke.digest(self.wheel)))
        self.assertEqual(marker.stat().st_mode & 0o777,0o600)
        self.assertIn('huggingface_hub[cli]==0.34.4',commands[1])
        helper=commands[-1]
        self.assertEqual(helper[helper.index('--coordinator-owner')+1],str(marker))
        self.assertEqual(helper[helper.index('--run')+1],str(run/'lifecycle'))


    def test_supervisor_refuses_wheel_changed_during_install_before_handoff(self):
        script=Path(hem_model_smoke.__file__).with_name('model_smoke.py')
        run=self.root/'changed-wheel';run.mkdir(mode=0o700)
        commands=[]
        def child(argv,**kwargs):
            commands.append(argv)
            if len(commands)==2:self.wheel.write_bytes(b'changed during install')
            return mock.Mock(poll=mock.Mock(return_value=0),wait=mock.Mock(return_value=0))
        previous=Path.cwd()
        try:
            with mock.patch.object(sys,'argv',[str(script),str(self.wheel),str(run),
                    str(script.parent.parent),str(script.parents[2]),'ci08']), \
                 mock.patch.object(subprocess,'Popen',side_effect=child), \
                 mock.patch.object(signal,'signal'):
                with self.assertRaisesRegex(AssertionError,'wheel changed during coordinator installation'):
                    runpy.run_path(str(script),run_name='__main__')
        finally:
            os.chdir(previous)
        self.assertEqual(len(commands),2)
        self.assertFalse((run/'coordinator.json').exists())


class ShutdownVerificationTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.gate = hem_model_smoke.Gate(argparse.Namespace(run=self.root))
        python = self.root / 'python'
        source = Path(__file__).resolve().parents[2]
        python.write_text('#!' + sys.executable + '\nimport sys\n'
                          f'sys.path.insert(0, {str(source)!r})\n'
                          'code=sys.argv[2].replace("hrx_solutions", "binding.python")\n'
                          'sys.argv=["-c",*sys.argv[3:]]\nexec(code)\n')
        python.chmod(0o700)
        self.gate.states['a'] = python
        self.listener = socket.socket()
        self.listener.bind(('127.0.0.1', 0))
        self.port = self.listener.getsockname()[1]
        self.addCleanup(self.listener.close)
        self.command = mock.patch.object(self.gate, 'command', return_value={
            'completed': True, 'state': 'stopped'})
        self.command.start()
        self.addCleanup(self.command.stop)

    def child(self):
        child = subprocess.Popen([sys.executable, '-c', 'import sys; sys.stdin.read()'],
                                 stdin=subprocess.PIPE)
        def cleanup():
            child.stdin.close()
            child.wait(timeout=3)
        self.addCleanup(cleanup)
        return child, hem_process.identity(child.pid)

    def zombie(self):
        child, process = self.child()
        child.stdin.close()
        os.waitid(os.P_PID, child.pid, os.WEXITED | os.WNOWAIT)
        self.assertTrue(Path(f'/proc/{child.pid}').exists())
        return process

    def configure(self, server, logger):
        process = dict(server, logger=logger)
        self.gate.servers['a'] = ('server-a', f'http://127.0.0.1:{self.port}', process, self.port)
        self.gate.attempts['a'] = {'process': process}

    def test_shutdown_accepts_real_unreaped_server_and_logger(self):
        self.configure(self.zombie(), self.zombie())
        self.listener.close()
        self.gate.shutdown('a')
        self.assertTrue(self.gate.results['cleanup'][0]['processes_exited'])
        self.assertNotIn('a', self.gate.attempts)

    def test_live_server_or_logger_times_out_and_preserves_recovery(self):
        exited = self.zombie()
        _, live = self.child()
        for server, logger in ((live, exited), (exited, live)):
            with self.subTest(server=server['pid'], logger=logger['pid']):
                self.configure(server, logger)
                # Exercise one real installed-inspection equivalent before the bound.
                self.assertFalse(self.gate.inspect_server('a', 'server-a',
                                 self.gate.servers['a'][2])['exited'])
                with mock.patch.object(hem_model_smoke.time, 'monotonic', side_effect=[0, 0, 11, 11]):
                    with self.assertRaises(TimeoutError):
                        self.gate.shutdown('a')
                self.assertIn('a', self.gate.attempts)
                self.assertFalse(self.gate.results['cleanup'])
                self.assertTrue((self.root / 'shutdown-a.json').exists())

    def test_open_port_fails_even_after_confirmed_exit(self):
        self.configure(self.zombie(), self.zombie())
        self.listener.listen()
        with self.assertRaisesRegex(AssertionError, 'owned port still open'):
            self.gate.shutdown('a')
        self.assertIn('a', self.gate.attempts)
        self.assertFalse(self.gate.results['cleanup'])

    def test_inspection_error_cannot_become_success(self):
        self.configure(self.zombie(), self.zombie())
        with mock.patch.object(self.gate, 'inspect_server', side_effect=RuntimeError('unreadable')):
            with self.assertRaisesRegex(RuntimeError, 'unreadable'):
                self.gate.shutdown('a')
        self.assertIn('a', self.gate.attempts)
        self.assertFalse(self.gate.results['cleanup'])
        evidence = json.loads((self.root / 'shutdown-a.json').read_text())
        self.assertEqual(evidence['server']['state'], 'Z')


class AcquisitionPathTest(unittest.TestCase):
    def test_download_subprocess_uses_selected_venv_hf_with_clean_or_hostile_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            gate = hem_model_smoke.Gate(argparse.Namespace(run=root))
            source = Path(__file__).resolve().parents[2]
            empty = root / 'empty-bin'
            global_bin = root / 'global-bin'
            empty.mkdir()
            global_bin.mkdir()
            global_hf = global_bin / 'hf'
            global_hf.write_text('#!' + sys.executable + '\nraise SystemExit(73)\n')
            global_hf.chmod(0o700)
            for namespace in ('a', 'b'):
                binary = root / namespace / 'venv/bin'
                binary.mkdir(parents=True)
                gate.states[namespace] = binary / 'python'
                hf = binary / 'hf'
                hf.write_text(
                    '#!' + sys.executable + '\nimport json,os,sys,site\nfrom pathlib import Path\n'
                    'evidence=dict(hf=sys.argv[0], home=os.environ["HOME"],'
                    'xdg=os.environ["XDG_CONFIG_HOME"], cache=os.environ["HF_HUB_CACHE"],'
                    'credentials_preserved=os.environ.get("HF_TOKEN")=="synthetic-test-token",'
                    'python_overrides_absent="PYTHONPATH" not in os.environ and "PYTHONHOME" not in os.environ,'
                    'user_site=site.ENABLE_USER_SITE)\n'
                    'Path(__file__).with_name("hf-evidence.json").write_text(json.dumps(evidence))\n'
                    'print(json.dumps(evidence))\n')
                hf.chmod(0o700)
                cli = binary / 'loom-control'
                cli.write_text(
                    '#!' + sys.executable + '\nimport json,sys\nfrom pathlib import Path\n'
                    f'sys.path.insert(0,{str(source)!r})\n'
                    'from binding.python.models.model_download import _run_hf\n'
                    'assert sys.argv[1:3]==["model","download"]\n'
                    '# Exercise the actual download subprocess primitive, without network or weights.\n'
                    '_run_hf(["hf","--help"])\n'
                    'print(json.dumps(dict(completed=True,hf=json.loads(Path(__file__).with_name("hf-evidence.json").read_text()))))\n')
                cli.chmod(0o700)
                for inherited_path in (empty, global_bin):
                    with self.subTest(namespace=namespace, inherited_path=inherited_path), mock.patch.dict(
                            os.environ, dict(PATH=str(inherited_path), HOME=str(root / 'home'),
                            XDG_CONFIG_HOME=str(root / 'xdg'), HF_HUB_CACHE=str(root / 'cache'),
                            HF_TOKEN='synthetic-test-token', PYTHONPATH='must-be-cleared',
                            PYTHONHOME='must-be-cleared')):
                        result = gate.command(namespace, 'model', 'download', 'qwen35-08b-q5-k-xl')
                        self.assertEqual(result['hf']['hf'], str(hf))
                        self.assertEqual(result['hf']['home'], str(root / 'home'))
                        self.assertEqual(result['hf']['xdg'], str(root / 'xdg'))
                        self.assertEqual(result['hf']['cache'], str(root / 'cache'))
                        self.assertTrue(result['hf']['credentials_preserved'])
                        self.assertTrue(result['hf']['python_overrides_absent'])
                        self.assertFalse(result['hf']['user_site'])


class CiProfileTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        database = self.root / 'a/state/control.sqlite3'
        database.parent.mkdir(parents=True)
        with sqlite3.connect(database) as connection:
            connection.execute('CREATE TABLE instances (id TEXT)')
        self.gate = hem_model_smoke.Gate(argparse.Namespace(
            run=self.root, acquire08=False, model08=self.root / 'immutable-model'))

    def test_required_profile_imports_only_qualified08_and_retires_before_shutdown(self):
        calls = []
        def record(name):
            return lambda *args, **kwargs: calls.append((name, args))
        def command(*args, **kwargs):
            calls.append(('command', args))
            return {'items': []}
        with mock.patch.object(self.gate, 'start_server', side_effect=record('server')), \
             mock.patch.object(self.gate, 'import_model', side_effect=record('import')), \
             mock.patch.object(self.gate, 'start_instance', side_effect=record('activate')), \
             mock.patch.object(self.gate, 'query', side_effect=record('query')), \
             mock.patch.object(self.gate, 'shutdown', side_effect=record('shutdown')), \
             mock.patch.object(self.gate, 'command', side_effect=command):
            self.gate.run_ci08()
        self.assertEqual([name for name, args in calls],
                         ['server', 'import', 'activate', 'query', 'command', 'command', 'shutdown'])
        self.assertEqual(calls[1][1][-2:], ('Qwen3.5-0.8B-UD-Q5_K_XL.gguf', 'qwen-08b'))
        self.assertEqual(calls[4][1][1:3], ('instance', 'stop'))
        self.assertTrue(self.gate.results['success'])

    def test_ci_acquisition_selects_pinned08_recipe(self):
        self.gate.args.acquire08 = True
        def command(*args, **kwargs):
            if args[1:3] == ('model', 'download'):
                return {'completed': True, 'validation': {'valid': True}}
            return {'items': []}
        with mock.patch.object(self.gate, 'start_server'), \
             mock.patch.object(self.gate, 'import_model') as imported, \
             mock.patch.object(self.gate, 'start_instance'), \
             mock.patch.object(self.gate, 'query'), \
             mock.patch.object(self.gate, 'shutdown'), \
             mock.patch.object(self.gate, 'command', side_effect=command) as invoked:
            self.gate.run_ci08()
        imported.assert_not_called()
        self.assertEqual(invoked.call_args_list[0].args,
                         ('a', 'model', 'download', 'qwen35-08b-q5-k-xl',
                          '--id', 'small-a', '--pipeline', 'qwen-08b'))
        self.assertTrue(self.gate.results['success'])

    def test_diagnostics_are_bounded_redacted_and_exclude_weights(self):
        (self.root / 'query.sse').write_text('x' * 80000 + ' hf_secret123 token-value')
        (self.root / 'model.gguf').write_bytes(b'weights')
        with mock.patch.dict('os.environ', {'HF_TOKEN': 'token-value'}):
            self.gate.artifacts()
        diagnostic = self.root / 'diagnostics'
        text = (diagnostic / 'query.sse').read_text()
        self.assertLessEqual((diagnostic / 'query.sse').stat().st_size, 65536)
        self.assertNotIn('hf_secret123', text)
        self.assertNotIn('token-value', text)
        self.assertFalse((diagnostic / 'model.gguf').exists())
        with sqlite3.connect(diagnostic / 'a-control.sqlite3') as connection:
            self.assertEqual(connection.execute('PRAGMA integrity_check').fetchone()[0], 'ok')


class SmokePendingLaunchTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        checker_fixtures.CheckerCleanupTest.setUpClass()

    @classmethod
    def tearDownClass(cls):
        checker_fixtures.CheckerCleanupTest.tearDownClass()

    def setUp(self):
        self.fixture = checker_fixtures.CheckerCleanupTest('test_timed_out_start_reconciles_and_aborts_only_its_owned_server')
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.case = self.fixture.case
        cli = self.case.root / 'loom-control'
        # Source-only fixture: the detached bootstrap must resolve binding.python
        # from this checkout too, without PYTHONPATH or an installed wheel.
        cli.write_text(self.fixture.cli.read_text().replace(
            'import functools,sys',
            'import functools,sys,os\nos.chdir(' + repr(str(Path(__file__).resolve().parents[2])) + ')'))
        cli.chmod(0o700)
        self.gate = hem_model_smoke.Gate(argparse.Namespace(run=self.case.root))
        python = self.case.root / 'python'
        python.write_text(
            '#!' + sys.executable + '\nimport sys\n'
            f'sys.path.insert(0,{str(Path(__file__).resolve().parents[2])!r})\n'
            'assert sys.argv[1] == "-c"\n'
            'code=sys.argv[2].replace("hrx_solutions", "binding.python")\n'
            'sys.argv=["-c",*sys.argv[3:]]\n'
            'exec(code)\n')
        python.chmod(0o700)
        self.gate.states['a'] = python
        self.gate.namespace_roots['a'] = '.'
        self.gate.setup = lambda: None
        self.addCleanup(self.recover_owned_fixture)

    def recover_owned_fixture(self):
        # Dispose only our fresh test name, before the fixture removes its DB.
        # This runs after success assertions, so recovery cannot qualify a gate.
        record = next((row for row in self.case.registry.list_servers()
                       if row['id'] == 'server-a'), None)
        if record and record['process'] and not hem_process._confirmed_exit(record['process']):
            self.gate.attempts['a'] = dict(name='server-a', token=record['launch_token'],
                process=record['process'], port=int(record['endpoint'].rsplit(':', 1)[1]))
            hem_model_smoke.Gate.cleanup_attempt(self.gate, 'a')

    def assert_failed_launch_cleaned(self, outer_timeout=False, cleanup_failure=False):
        other = self.case.start('unrelated')
        original = self.gate.command
        captured = {}
        if outer_timeout:
            cli = self.case.root / 'loom-control'
            cli.write_text(cli.read_text().replace('["--timeout","0.15"]', '["--timeout","5"]'))

        def command(namespace, *args, **kwargs):
            if cleanup_failure and args[:2] == ('server', 'reconcile'):
                raise RuntimeError('injected cleanup observation failure')
            try:
                if outer_timeout and args[:2] == ('server', 'start'):
                    kwargs['command_timeout'] = .3
                return original(namespace, *args, **kwargs)
            except (AssertionError, subprocess.TimeoutExpired) as error:
                if args[:2] == ('server', 'start'):
                    owned = self.case.registry.get_server('server-a')
                    captured['process'] = owned['process']
                    self.case.owned.append((owned['process'], owned['process']['executable']))
                    captured['error'] = error
                    raise error
                raise

        diagnostic = io.StringIO()
        with mock.patch.object(self.gate, 'command', side_effect=command), redirect_stderr(diagnostic):
            with self.assertRaises(subprocess.TimeoutExpired if outer_timeout else AssertionError) as caught:
                self.gate.run()
        self.assertIs(caught.exception, captured['error'])
        if cleanup_failure:
            self.assertIn('Cleanup unresolved', diagnostic.getvalue())
            self.assertIn('injected cleanup observation failure', diagnostic.getvalue())
            self.assertFalse(hem_process._confirmed_exit(captured['process']))
            self.assertIn('error', self.gate.results['cleanup'][-1])
            budget = Deadline.after(3)
            while self.case.manager.reconcile('server-a', budget)['phase'] != 'running':
                time.sleep(min(.01, budget.remaining()))
            ShutdownManager(self.case.registry, processes=self.case.manager).shutdown('server-a', budget, abort=True)
        self.assertTrue(hem_process._confirmed_exit(captured['process']))
        self.assertTrue(hem_process._confirmed_exit(captured['process']['logger']))
        self.assertTrue(self.gate.results['cleanup'])
        self.assertFalse(self.gate.results['success'])
        self.assertFalse(hem_process._confirmed_exit(other['process']))
        with self.case.manager.pin(other) as pinned:
            self.assertEqual(self.case.manager.client(other, pinned, Deadline.after()).status()['data'], [])
        ShutdownManager(self.case.registry, processes=self.case.manager).shutdown('unrelated', Deadline.after(3), abort=True)

    def test_accepted_slow_elf_start_failure_reconciles_and_aborts(self):
        self.assert_failed_launch_cleaned()

    def test_outer_timeout_after_submission_preserves_error_and_cleans_real_elf(self):
        self.assert_failed_launch_cleaned(outer_timeout=True)

    def test_unknown_cleanup_preserves_start_error_and_reports_live_process(self):
        self.assert_failed_launch_cleaned(cleanup_failure=True)

    def test_prelaunch_failure_does_not_abort_unrelated_server(self):
        other = self.case.start('unrelated')
        failure = FileNotFoundError('injected CLI launch failure')
        with mock.patch.object(self.gate, 'command', side_effect=failure):
            with self.assertRaises(FileNotFoundError) as caught:
                self.gate.run()
        self.assertIs(caught.exception, failure)
        self.assertFalse(self.gate.attempts)
        self.assertTrue(self.gate.results['cleanup'][-1]['absent'])
        self.assertFalse(hem_process._confirmed_exit(other['process']))
        ShutdownManager(self.case.registry, processes=self.case.manager).shutdown('unrelated', Deadline.after(3), abort=True)


    def assert_signal_cleanup(self, signum, registered=False, cleanup_failure=False):
        other = self.case.start('unrelated')
        cleanup_file = self.case.root / 'cleanup.json'
        self.gate.args.profile = 'ci08'
        cli = self.case.root / 'loom-control'
        cli.write_text(cli.read_text().replace('["--timeout","0.15"]', '["--timeout","5"]'))
        def exercise():
            self.gate.start_server('a', 'server-a')
            signal.pause()
        self.gate.run_ci08 = exercise
        if cleanup_failure:
            self.gate.cleanup_attempt = lambda namespace: (_ for _ in ()).throw(RuntimeError('injected cleanup failure'))
        pid = os.fork()
        if pid == 0:
            try:
                os.environ['HRX_LOCAL_CI_CLEANUP_FILE'] = str(cleanup_file)
                with (self.case.root / 'cancel.log').open('w') as log:
                    os.dup2(log.fileno(), 1)
                    os.dup2(log.fileno(), 2)
                    self.gate.run()
            except hem_model_smoke.Cancelled as error:
                os._exit(128 + error.signum)
            except BaseException:
                import traceback
                traceback.print_exc()
                os._exit(1)
            os._exit(0)
        reaped = False
        try:
            end = time.monotonic() + 8
            while True:
                record = next((row for row in self.case.registry.list_servers() if row['id'] == 'server-a'), None)
                if record and record['process'] and record['process'].get('logger') and (not registered or record['phase'] == 'running'):
                    break
                self.assertLess(time.monotonic(), end, 'owned launch not observed')
                time.sleep(.01)
            owned = record['process']
            self.case.owned.append((owned, owned['executable']))
            os.kill(pid, signum)
            # A second signal while the finally path waits must not interrupt it.
            time.sleep(.05)
            os.kill(pid, signum)
            while True:
                waited, status = os.waitpid(pid, os.WNOHANG)
                if waited:
                    reaped = True
                    break
                self.assertLess(time.monotonic(), end + 12, 'cooperative helper did not exit')
                time.sleep(.01)
            self.assertEqual(os.waitstatus_to_exitcode(status), 128 + signum)
            evidence = json.loads(cleanup_file.read_text())
            self.assertEqual(evidence['status'], 'unresolved' if cleanup_failure else 'complete')
            if not cleanup_failure:
                self.assertTrue(hem_process._confirmed_exit(owned))
                self.assertTrue(hem_process._confirmed_exit(owned['logger']))
                with socket.socket() as sock:
                    self.assertNotEqual(sock.connect_ex(('127.0.0.1', int(record['endpoint'].rsplit(':', 1)[1]))), 0)
            self.assertFalse(hem_process._confirmed_exit(other['process']))
            with self.case.manager.pin(other) as pinned:
                self.assertEqual(self.case.manager.client(other, pinned, Deadline.after()).status()['data'], [])
        finally:
            if not reaped:
                os.kill(pid, signal.SIGKILL)
                os.waitpid(pid, 0)
            # This recovery is fixture disposal, after assertions about helper
            # cleanup. It cannot turn unresolved evidence into a passing result.
            if cleanup_failure:
                deadline = Deadline.after(4)
                while self.case.manager.reconcile('server-a', deadline)['phase'] != 'running':
                    time.sleep(.01)
                ShutdownManager(self.case.registry, processes=self.case.manager).shutdown('server-a', deadline, abort=True)
            ShutdownManager(self.case.registry, processes=self.case.manager).shutdown('unrelated', Deadline.after(3), abort=True)

    def test_sigterm_during_synchronous_pending_start_reconciles_and_reaps(self):
        self.assert_signal_cleanup(signal.SIGTERM)

    def test_sigint_registered_server_and_repeated_signal_preserves_unrelated(self):
        self.assert_signal_cleanup(signal.SIGINT, registered=True)

    def test_sigterm_cleanup_failure_preserves_signal_and_unresolved_evidence(self):
        self.assert_signal_cleanup(signal.SIGTERM, cleanup_failure=True)


if __name__ == '__main__':
    unittest.main()
