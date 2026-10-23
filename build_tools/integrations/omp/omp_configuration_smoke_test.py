"""The delivery gate must reject stale/optimistic metadata and empty responses."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

from build_tools.integrations.omp.omp_configuration_smoke import ConfigurationGate, configuration_snapshot
from build_tools.integrations.common import provider


class AcceptanceProofTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        private = root / 'private-runtime'; private.mkdir()
        gate = Mock(root=root, runtime_root=private, state_path=Mock(return_value=private/'namespace-a/state'),
                    args=argparse.Namespace(omp_runtime=root,source=root), results={})
        self.check = ConfigurationGate(gate)
        self.row = dict(provider=provider('local'),id='a',baseUrl='http://127.0.0.1:18080/v1',
            contextWindow=32768,maxTokens=16384,supportsTools=False,reasoning=False,input=['text'])
        self.expected = {self.row['provider']+'/a':dict(base_url=self.row['baseUrl'],limit=dict(context=32768,output=16384))}

    def test_client_and_configure_share_private_gate_namespace(self):
        gate = self.check.gate
        self.assertNotEqual(gate.root, gate.runtime_root)
        self.assertEqual(self.check.env['LOOM_SERVE_DIR'], str(gate.state_path('a')))
        for path in (self.check.target, self.check.project):
            self.assertTrue(path.is_relative_to(gate.runtime_root))
        for key in ('XDG_DATA_HOME', 'XDG_CACHE_HOME', 'XDG_STATE_HOME'):
            self.assertTrue(Path(self.check.env[key]).is_relative_to(gate.runtime_root))
        self.check.command('--check')
        overrides = gate.command.call_args.kwargs['env_overrides']
        self.assertEqual(overrides['LOOM_SERVE_DIR'], str(gate.state_path('a')))
        self.assertEqual(overrides['PI_CODING_AGENT_DIR'], str(self.check.target))
        self.assertFalse((gate.root / 'a/state').exists())
        self.assertEqual(self.check.endpoints, gate.state_path('a') / 'clients/endpoints.json')
        self.check.endpoints.parent.mkdir(parents=True)
        self.check.endpoints.write_text('{}')
        self.assertTrue(self.check.endpoints.is_file())
        self.assertFalse((gate.root / 'a/state/clients/endpoints.json').exists())

    def metadata(self, expected=None):
        self.check.observe_models = Mock(return_value=[json.dumps(dict(models=[self.row])),json.dumps(dict(models=[self.row]))])
        return self.check.models(self.expected if expected is None else expected)

    def test_exact_routes_and_metadata_required(self):
        self.metadata()
        for key,value in [('id','other'),('baseUrl','http://127.0.0.1:18081/v1'),('contextWindow',200000),
                          ('maxTokens',32000),('supportsTools',True),('reasoning',True),('input',['text','image'])]:
            with self.subTest(key=key):
                before = self.row[key]; self.row[key] = value
                with self.assertRaises(AssertionError): self.metadata()
                self.row[key] = before
        with self.assertRaises(AssertionError): self.metadata({})

    def test_empty_or_error_text_not_a_successful_request(self):
        for text in ('', '  ', 'Error: rejected', 'error: unavailable'):
            self.check.run_client = Mock(return_value=text)
            with self.assertRaises(AssertionError): self.check.request('p/a')
            self.assertEqual(self.check.evidence['requests'],[])
        self.check.run_client = Mock(return_value='Hello!')
        self.check.request('p/a')
        self.assertEqual(self.check.evidence['requests'][0]['selected'],'p/a')

    def test_identity_failure_reaps_exact_owned_child(self):
        self.check.project.mkdir()
        child = Mock(pid=12345); child.poll.return_value=None; child.wait.return_value=143
        error=RuntimeError('identity unavailable')
        module=types.ModuleType('hrx_solutions.hem_process'); module.identity=Mock(side_effect=error)
        with patch.dict('sys.modules',{'hrx_solutions.hem_process':module}), \
             patch('build_tools.integrations.omp.omp_configuration_smoke.subprocess.Popen',return_value=child):
            with self.assertRaises(RuntimeError) as caught: self.check.run_client('--version')
        self.assertIs(caught.exception,error)
        child.terminate.assert_called_once(); child.wait.assert_called_once_with(timeout=10)
        self.assertIsNone(self.check.child)
        self.assertFalse(self.check.evidence['client_cleanup'][0]['identity_verified'])

    def observation_scripts(self, first, second):
        self.check.project.mkdir()
        self.check.bun = Path(sys.executable)
        self.check.client = self.check.gate.root / 'cli.py'
        self.check.client.write_text(first)
        probe = self.check.gate.args.source / 'tests/omp_configuration_probe.mjs'
        probe.parent.mkdir()
        probe.write_text(second)
        module = types.ModuleType('hrx_solutions.hem_process')
        module.identity = lambda pid: pid
        def exited(pid):
            try: os.kill(pid, 0)
            except ProcessLookupError: return True
            return False
        module._confirmed_exit = exited
        return patch.dict('sys.modules', {'hrx_solutions.hem_process': module})

    def assert_pair_retired(self):
        self.assertIsNone(self.check.child)
        cleanup = self.check.evidence['client_cleanup']
        self.assertEqual(len(cleanup), 2)
        self.assertEqual(len({row['pid'] for row in cleanup}), 2)
        self.assertTrue(all(row['identity_verified'] and row['processes_exited'] and not row['forced']
                            for row in cleanup))

    def test_pair_starts_both_before_join_and_preserves_distinct_output(self):
        # A serial implementation times out: each public host waits for the
        # other to start, then emits its independent output.
        def script(own, other):
            return ("from pathlib import Path\nimport time\n"
                    f"Path('{own}').touch()\n"
                    f"while not Path('{other}').exists(): time.sleep(.01)\n"
                    f"print('{own}')\n")
        with self.observation_scripts(script('cli', 'probe'), script('probe', 'cli')):
            self.assertEqual(self.check.observe_models(timeout=2), ['cli\n', 'probe\n'])
        self.assert_pair_retired()
        self.assertEqual([row['probe'] for row in self.check.evidence['commands']], [False, True])
        self.assertEqual(len(self.check.evidence['observation_pairs']), 1)

    def test_either_child_failure_retires_running_sibling(self):
        for failing in (0, 1):
            with self.subTest(failing=failing):
                scripts = ['import time; time.sleep(30)', 'import time; time.sleep(30)']
                scripts[failing] = 'raise SystemExit(7)'
                with self.observation_scripts(*scripts):
                    with self.assertRaisesRegex(AssertionError, 'OMP observation failed'):
                        self.check.observe_models(timeout=2)
                self.assert_pair_retired()
                self.check.project.rmdir()
                self.check.evidence['client_cleanup'].clear()
                (self.check.gate.root / 'tests/omp_configuration_probe.mjs').unlink()
                (self.check.gate.root / 'tests').rmdir()

    def test_pair_timeout_and_cancellation_retire_both(self):
        with self.observation_scripts('import time; time.sleep(30)', 'import time; time.sleep(30)'):
            with self.assertRaises(subprocess.TimeoutExpired):
                self.check.observe_models(timeout=.1)
            self.assert_pair_retired()

            self.check.evidence['client_cleanup'].clear()
            sleep = time.sleep
            interrupted = False
            def cancel_once(seconds):
                nonlocal interrupted
                if not interrupted:
                    interrupted = True
                    raise KeyboardInterrupt
                sleep(seconds)
            with patch('build_tools.integrations.omp.omp_configuration_smoke.time.sleep', side_effect=cancel_once):
                with self.assertRaises(KeyboardInterrupt): self.check.observe_models(timeout=2)
            self.assert_pair_retired()


    def test_second_identity_failure_retires_both_owned_children(self):
        with self.observation_scripts('import time; time.sleep(30)', 'import time; time.sleep(30)'):
            module = sys.modules['hrx_solutions.hem_process']
            module.identity = Mock(side_effect=[123, RuntimeError('second identity unavailable')])
            module._confirmed_exit = Mock(return_value=True)
            with self.assertRaisesRegex(RuntimeError, 'second identity unavailable'):
                self.check.observe_models(timeout=2)
        self.assertIsNone(self.check.child)
        cleanup = self.check.evidence['client_cleanup']
        self.assertEqual(len(cleanup), 2)
        self.assertEqual([row['identity_verified'] for row in cleanup], [True, False])
        self.assertTrue(all(row['processes_exited'] and not row['forced'] for row in cleanup))

    def test_launch_boundary_cancellation_publishes_and_retires_child(self):
        popen = subprocess.Popen
        original_mask = signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGINT, signal.SIGTERM})
        expected_mask = signal.pthread_sigmask(signal.SIG_BLOCK, set())
        try:
            for cancel_at in (1, 2):
                with self.subTest(cancel_at=cancel_at), self.observation_scripts(
                        'import time; time.sleep(30)', 'import time; time.sleep(30)'):
                    children = []
                    def launch(*args, **kwargs):
                        child = popen(*args, **kwargs)
                        children.append(child)
                        self.assertTrue({signal.SIGINT, signal.SIGTERM}.issubset(
                            signal.pthread_sigmask(signal.SIG_BLOCK, set())))
                        if len(children) == cancel_at:
                            # The subprocess exists but Popen has not returned
                            # to its caller: cancellation must remain pending.
                            os.kill(os.getpid(), signal.SIGINT)
                            self.assertIn(signal.SIGINT, signal.sigpending())
                        return child
                    with patch('build_tools.integrations.omp.omp_configuration_smoke.subprocess.Popen', side_effect=launch):
                        with self.assertRaises(KeyboardInterrupt): self.check.observe_models(timeout=2)
                    self.assertEqual(len(children), cancel_at)
                    self.assertTrue(all(child.poll() is not None for child in children))
                    self.assertEqual(len(self.check.evidence['client_cleanup']), cancel_at)
                    self.assertIsNone(self.check.child)
                    self.assertEqual(signal.pthread_sigmask(signal.SIG_BLOCK, set()), expected_mask)
                self.check.project.rmdir()
                self.check.evidence['client_cleanup'].clear()
                (self.check.gate.root / 'tests/omp_configuration_probe.mjs').unlink()
                (self.check.gate.root / 'tests').rmdir()
        finally:
            signal.pthread_sigmask(signal.SIG_SETMASK, original_mask)

    def test_children_restore_exact_original_signal_mask(self):
        original_mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGUSR1})
        try:
            expected = sorted(int(value) for value in signal.pthread_sigmask(signal.SIG_BLOCK, set()))
            script = 'import json,signal; print(json.dumps(sorted(int(x) for x in signal.pthread_sigmask(signal.SIG_BLOCK,set()))))'
            with self.observation_scripts(script, script):
                output = self.check.observe_models(timeout=2)
            self.assertEqual([json.loads(text) for text in output], [expected, expected])
            self.assertEqual(sorted(int(value) for value in signal.pthread_sigmask(signal.SIG_BLOCK, set())), expected)
            self.assert_pair_retired()
        finally:
            signal.pthread_sigmask(signal.SIG_SETMASK, original_mask)

    def test_configuration_snapshot_excludes_only_released_vendor_cache_basenames(self):
        self.check.target.mkdir()
        for name in ('models.yml','config.yml','foreign.db','models.db','models.db-wal','agent.db-shm','skill-descriptions.db'):
            (self.check.target/name).write_text('before')
        before=configuration_snapshot(self.check.target)
        self.assertEqual(set(before),{'models.yml','config.yml','foreign.db'})
        (self.check.target/'models.db').write_text('vendor cache change')
        self.assertEqual(configuration_snapshot(self.check.target),before)
        (self.check.target/'foreign.db').write_text('user data change')
        self.assertNotEqual(configuration_snapshot(self.check.target),before)


class DeliveryWiringTest(unittest.TestCase):
    def test_real_backend_gate_excludes_clients_and_wheel_groups_them(self):
        root=Path(__file__).resolve().parents[3]
        model=(root/'.github/workflows/check_model.yml').read_text()
        for required in ('runs-on: linux-gfx1151-gpu-rocm','name: hrx-solutions-wheel-linux-x86_64',
                         'build_tools/model-smoke.sh'):
            self.assertIn(required,model)
        for excluded in ('LOOM_PI_RUNTIME:','LOOM_OPENCODE_RUNTIME:','LOOM_OMP_RUNTIME:',
                         'build_tools/integrations/omp/stage-omp-runtime.sh'):
            self.assertNotIn(excluded,model)
        wheel=(root/'.github/workflows/build_python_wheel.yml').read_text()
        for required in ('LOOM_OMP_TEST_BUN:','LOOM_OMP_TEST_PACKAGE:',
                         'Check installed UI integrations from the repaired wheel',
                         'build_tools/integrations/check-ui-integrations.sh'):
            self.assertIn(required,wheel)
        self.assertNotIn('from build_tools.integrations.omp.omp_configuration_smoke import qualify',(root/'build_tools/validation/hem_model_smoke.py').read_text())
        shell=(root/'build_tools/model-smoke.sh').read_text()
        self.assertNotIn("for name in ('pi', 'opencode', 'omp'):",shell)
        supervisor=(root/'build_tools/validation/model_smoke.py').read_text()
        self.assertNotIn("for name in ('pi', 'opencode', 'omp'):",supervisor)
        self.assertIn("else ['--profile', 'ci08']", supervisor)
        self.assertIn('*model, *qualification], helper=True)',supervisor)
        self.assertIn("'hem_omp_installer_test'",(root/'build_tools/validation/check_installed_cpu.py').read_text())


if __name__ == '__main__': unittest.main()
