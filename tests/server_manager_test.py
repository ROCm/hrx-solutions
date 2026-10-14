"""Managed server identity acquisition across exec and process exit."""

from pathlib import Path
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


if __name__ == "__main__":
    unittest.main()
