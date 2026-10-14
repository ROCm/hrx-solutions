"""Managed server identity acquisition across exec and process exit."""

from pathlib import Path
import json
import os
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from binding.python import server_manager


class ProcessIdentityTest(unittest.TestCase):
    def setUp(self):
        self.process = Mock(pid=123)
        self.process.poll.return_value = None
        self.executable = Path("/opt/loom-serve")
        self.argv = [str(self.executable), "--device=amdgpu"]
        self.identity = {
            "start_time": "42", "process_group": self.process.pid,
            "executable": str(self.executable), "command": self.argv,
        }

    def test_retries_transient_exec_snapshot(self):
        before_exec = dict(self.identity, executable="/usr/bin/python")
        mixed_snapshot = dict(self.identity, command=["python", "launcher.py"])
        with (
            patch.object(server_manager, "_process_info", side_effect=[before_exec, mixed_snapshot, self.identity]),
            patch.object(server_manager.time, "monotonic", return_value=0),
            patch.object(server_manager.time, "sleep") as sleep,
        ):
            result = server_manager._wait_for_process_identity(
                self.process, self.executable, self.argv, 1
            )
        self.assertEqual(result, self.identity)
        self.assertEqual(sleep.call_count, 2)

    def test_mismatched_identity_times_out(self):
        for change in ({"executable": "/opt/other"}, {"command": ["other"]}, {"process_group": 456}):
            with (
                self.subTest(change=change),
                patch.object(server_manager, "_process_info", return_value=dict(self.identity, **change)),
                patch.object(server_manager.time, "monotonic", side_effect=[0, 1]),
                patch.object(server_manager.time, "sleep") as sleep,
            ):
                with self.assertRaisesRegex(RuntimeError, "Could not verify"):
                    server_manager._wait_for_process_identity(self.process, self.executable, self.argv, 1)
                sleep.assert_called_once_with(0.01)

    def test_stops_retrying_when_owned_child_exits(self):
        self.process.poll.side_effect = [None, 1]
        with (
            patch.object(server_manager, "_process_info", return_value=None) as info,
            patch.object(server_manager.time, "monotonic", return_value=0),
            patch.object(server_manager.time, "sleep"),
        ):
            with self.assertRaisesRegex(RuntimeError, "Could not verify"):
                server_manager._wait_for_process_identity(self.process, self.executable, self.argv, 1)
        info.assert_called_once_with(self.process.pid)


class StartupCancellationTest(unittest.TestCase):
    def test_cancellation_reaps_child_and_restores_handlers(self):
        # Isolate signals from the test runner. The child is a real detached
        # Python sleeper; no native server, wheel, or model is needed.
        launcher = r"""
import json, os, signal, subprocess, sys, time
from pathlib import Path
from unittest.mock import patch
from binding.python import server_manager

signum, phase = int(sys.argv[1]), sys.argv[2]
root = Path(os.environ['LOOM_SERVE_DIR'])
original = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}
children = []
popen = subprocess.Popen

def spawn(*args, **kwargs):
    child = popen(*args, **kwargs)
    children.append(child)
    if phase == 'spawn':
        os.kill(os.getpid(), signum)
    return child

def probe(*args):
    deadline = time.monotonic() + 3
    while not (root / 'inherited').exists():
        if time.monotonic() > deadline:
            raise RuntimeError('Dummy child did not start')
        time.sleep(.01)
    os.kill(os.getpid(), signum)
    return False

child_code = (
    "import os, time; from pathlib import Path; "
    "p=Path(os.environ['LOOM_SERVE_DIR']); "
    "(p/'inherited').write_text(str(p)); time.sleep(60)"
)
try:
    with patch.object(server_manager.subprocess, 'Popen', side_effect=spawn), \
         patch.object(server_manager, '_probe', side_effect=probe):
        try:
            server_manager.launch('dummy', 'dummy', Path(sys.executable),
                [sys.executable, '-c', child_code], 0, [], {}, ready_timeout=5)
        except SystemExit as exc:
            assert exc.code == 128 + signum, exc.code
        else:
            raise AssertionError('Cancellation did not propagate')
    assert all(signal.getsignal(s) == h for s, h in original.items())
    assert len(children) == 1
    assert children[0].returncode is not None
    assert not Path(f'/proc/{children[0].pid}').exists()
    records = list((root / 'servers').glob('*.json'))
    assert len(records) == 1, records
    record = json.loads(records[0].read_text())
    assert record['state'] == 'failed', record
    if phase == 'probe':
        assert (root / 'inherited').read_text() == str(root)
    print('cleaned')
finally:
    # Prevent a failing regression from leaving its dummy process behind.
    for child in children:
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()
"""
        for signum in (signal.SIGINT, signal.SIGTERM):
            for phase in ('spawn', 'probe'):
                with self.subTest(signal=signum, phase=phase), tempfile.TemporaryDirectory() as directory:
                    result = subprocess.run(
                        [sys.executable, '-c', launcher, str(signum), phase],
                        env=dict(os.environ, LOOM_SERVE_DIR=directory),
                        capture_output=True, text=True, timeout=15,
                    )
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    self.assertEqual(result.stdout.strip(), 'cleaned')


if __name__ == "__main__":
    unittest.main()
