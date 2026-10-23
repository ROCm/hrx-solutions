"""Acceptance must reject broken metadata/requests and omitted CI wiring."""
import argparse
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

from build_tools.integrations.opencode.opencode_configuration_smoke import ConfigurationGate
from build_tools.integrations.common import provider, snapshot


class AcceptanceProofTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        private = root / 'private-runtime'; private.mkdir()
        gate = Mock(root=root, runtime_root=private, state_path=Mock(return_value=private/'namespace-a/state'),
                    args=argparse.Namespace(opencode_runtime=root),
                    states={'a': root / 'venv/bin/python'}, results={})
        self.check = ConfigurationGate(gate)
        self.check.run_client = Mock()
        self.check.url = "http://127.0.0.1:18089"
        self.row = dict(id='a', modelID='a', providerID=provider('local'),
                        package='@opencode/ai/providers/openai-compatible',
                        settings=dict(baseURL='http://127.0.0.1:18080/v1'),
                        capabilities=dict(tools=False, input=['text'], output=['text']),
                        limit=dict(context=32768, output=16384))
        self.expected = {provider('local')+'/a': dict(base_url=self.row['settings']['baseURL'], limit=self.row['limit'])}

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
        self.assertEqual(overrides['OPENCODE_CONFIG_DIR'], str(self.check.target))
        self.assertFalse((gate.root / 'a/state').exists())
        self.assertEqual(self.check.endpoints, gate.state_path('a') / 'clients/endpoints.json')
        self.check.endpoints.parent.mkdir(parents=True)
        self.check.endpoints.write_text('{}')
        self.assertTrue(self.check.endpoints.is_file())
        self.assertFalse((gate.root / 'a/state/clients/endpoints.json').exists())

    def metadata(self):
        self.check.run_client.return_value = json.dumps(dict(data=[self.row]))
        return self.check.models(self.expected)

    def test_exact_native_metadata_required_not_optimistic_client_defaults(self):
        self.assertEqual(self.metadata(), [self.row])
        for key, value in [('modelID','wrong'), ('settings',dict(baseURL='http://127.0.0.1:18081/v1')),
                           ('limit',dict(context=200000,output=32000)),
                           ('capabilities',dict(tools=True,input=['text'],output=['text']))]:
            with self.subTest(key=key):
                before = self.row[key]
                self.row[key] = value
                with self.assertRaises(AssertionError): self.metadata()
                self.row[key] = before

    def test_no_stale_rows_after_shutdown(self):
        self.check.run_client.return_value = json.dumps(dict(data=[self.row]))
        with self.assertRaises(AssertionError): self.check.models({})
        self.check.run_client.return_value = '{"data":[]}'
        self.assertEqual(self.check.models({}), [])

    def test_exit_zero_without_real_text_cannot_pass(self):
        for events in ([], [dict(type='error')], [dict(type='step_finish')]):
            with self.subTest(events=events):
                self.check.run_client.return_value = '\n'.join(map(json.dumps, events))
                with self.assertRaises(AssertionError): self.check.request('p/a')
                self.assertEqual(self.check.evidence['requests'], [])
        self.check.run_client.return_value = '\n'.join(map(json.dumps,[dict(type='text',part=dict(text='hello'))]))
        self.check.request('p/a')
        self.assertEqual(self.check.evidence['requests'][0]['selected'], 'p/a')

    def test_identity_capture_failure_reaps_exact_child_and_preserves_original(self):
        self.check.project.mkdir()
        child = Mock(pid=12345)
        child.poll.return_value = None
        child.wait.return_value = 130
        error = RuntimeError('identity unavailable')
        module = types.ModuleType('hrx_solutions.control.hem_process')
        module.identity = Mock(side_effect=error)
        import socket
        closed = Mock()
        closed.__enter__ = Mock(return_value=closed)
        closed.__exit__ = Mock(return_value=False)
        closed.connect_ex.return_value = 1
        closed.getsockname.return_value = ('127.0.0.1',18089)
        with patch.dict('sys.modules', {'hrx_solutions.control.hem_process': module}), \
             patch('build_tools.integrations.opencode.opencode_configuration_smoke.subprocess.Popen',return_value=child), \
             patch('build_tools.integrations.opencode.opencode_configuration_smoke.socket.socket',return_value=closed):
            with self.assertRaises(RuntimeError) as caught:
                try:
                    self.check.start_client()
                finally:
                    self.check.cleanup_client()
        self.assertIs(caught.exception,error)
        child.terminate.assert_called_once()
        child.wait.assert_called_once_with(timeout=10)
        self.assertIsNone(self.check.child)
        self.assertTrue(self.check.log.closed)
        self.assertTrue(self.check.evidence['client_cleanup'][0]['processes_exited'])
        self.assertFalse(self.check.evidence['client_cleanup'][0]['identity_verified'])

    def test_passive_snapshot_detects_same_length_edit_and_mtime_only_write(self):
        path = self.check.gate.root / 'settings.jsonc'
        path.write_text('abc')
        before = snapshot(path.parent)
        path.write_text('xyz')
        self.assertNotEqual(before, snapshot(path.parent))
        before = snapshot(path.parent)
        import os
        os.utime(path, ns=(path.stat().st_atime_ns, path.stat().st_mtime_ns+1000))
        self.assertNotEqual(before, snapshot(path.parent))


class DeliveryWiringTest(unittest.TestCase):
    def test_repaired_wheel_ui_checks_are_grouped_outside_backend_gate(self):
        source = Path(__file__).resolve().parents[3]
        workflow = (source / '.github/workflows/check_model.yml').read_text()
        for required in ('runs-on: linux-gfx1151-gpu-rocm', 'name: hrx-solutions-wheel-linux-x86_64',
                         'build_tools/model-smoke.sh'):
            self.assertIn(required, workflow)
        for excluded in ('LOOM_PI_RUNTIME:', 'LOOM_OPENCODE_RUNTIME:', 'stage-opencode-runtime.sh'):
            self.assertNotIn(excluded, workflow)
        build = (source / '.github/workflows/build_python_wheel.yml').read_text()
        for required in ('LOOM_OPENCODE_TEST_NODE:', 'build_tools/integrations/opencode/stage-opencode-runtime.sh',
                         'Check installed UI integrations from the repaired wheel',
                         'build_tools/integrations/check-ui-integrations.sh',
                         '-m build_tools.ci_tools.source_tests install'):
            self.assertIn(required, build)
        helper = (source / 'build_tools/validation/hem_model_smoke.py').read_text()
        self.assertNotIn('from build_tools.integrations.opencode.opencode_configuration_smoke import qualify', helper)
        shell = (source / 'build_tools/model-smoke.sh').read_text()
        self.assertNotIn("for name in ('pi', 'opencode', 'omp'):", shell)
        supervisor = (source / 'build_tools/validation/model_smoke.py').read_text()
        self.assertNotIn("for name in ('pi', 'opencode', 'omp'):", supervisor)
        self.assertIn("else ['--profile', 'ci08']", supervisor)
        self.assertIn('*model, *qualification], helper=True)', supervisor)


if __name__ == '__main__':
    unittest.main()
