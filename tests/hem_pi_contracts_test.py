"""Pi command boundary before configuration, filesystem and provider workers."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from binding.python import hem_cli, hem_pi_cli
from binding.python.hem_common import DeadlineExpired, HemError, InvalidInput, UncertainOutcome
from binding.python.hem_pi_contracts import PiResult, provider_id


class PiContractTest(unittest.TestCase):
    def invoke(self, suffix=(), *, response=None, error=None):
        out, err = io.StringIO(), io.StringIO()
        with patch.object(hem_pi_cli, "run", return_value=response, side_effect=error) as run:
            with redirect_stdout(out), redirect_stderr(err):
                code = hem_cli.main(["pi", "configure", "--json", *suffix])
        return code, json.loads(out.getvalue()), err.getvalue(), run

    def test_only_one_command_and_exclusive_modes(self):
        for args in (["pi"], ["pi", "sync"], ["pi", "configure", "a-model"],
                     ["pi", "configure", "--check", "--disable"],
                     ["pi", "configure", "--server", "local"],
                     ["pi", "configure", "--ch"],
                     ["pi", "configure", "--timeout", "nan"]):
            with self.subTest(args=args), self.assertRaises(InvalidInput):
                hem_cli.parse_args(args)
        args = hem_cli.parse_args(["pi", "configure", "--pi-dir", "/tmp/pi-target"])
        self.assertEqual(args.pi_dir, Path("/tmp/pi-target"))
        self.assertEqual(args.timeout, 10)
        self.assertFalse(args.check or args.disable)
        for args in (["--json", "pi", "configure"], ["pi", "--json", "configure"],
                     ["pi", "configure", "--json"]):
            self.assertTrue(hem_cli.parse_args(args).json)

    def test_preview_nonzero_result_is_preserved_and_deadline_passed_once(self):
        response = PiResult("changes_required", configured=False, changed=False,
                            completed=True, exit_code=1, affected_paths=[])
        code, body, err, run = self.invoke(["--check", "--timeout", "2.5"], response=response)
        self.assertEqual(code, 1)
        self.assertEqual(body["state"], "changes_required")
        self.assertTrue(body["completed"])
        self.assertNotIn("accepted", body)
        self.assertEqual(err, "")
        self.assertGreater(run.call_args.args[1].remaining(), 2)
        self.assertLessEqual(run.call_args.args[1].remaining(), 2.5)

    def test_offline_is_independent_of_durable_configuration_success(self):
        response = PiResult("configured", configured=True, changed=True, completed=True,
                            accepted=True, available=[],
                            servers=[{"server": "local", "state": "offline"}])
        code, body, _, _ = self.invoke(response=response)
        self.assertEqual(code, 0)
        self.assertTrue(body["configured"])
        self.assertEqual(body["available"], [])

    def test_existing_failure_codes_and_acceptance_are_preserved(self):
        cases = [(HemError("known failure", accepted=False), 1),
                 (InvalidInput("ownership conflict", accepted=False), 2),
                 (DeadlineExpired("accepted update pending", accepted=True), 3),
                 (UncertainOutcome("manifest commit unknown"), 4)]
        for error, expected in cases:
            with self.subTest(expected=expected):
                code, body, err, _ = self.invoke(error=error)
                self.assertEqual(code, expected)
                self.assertFalse(body["completed"])
                self.assertIs(body["accepted"], error.accepted)
                self.assertIn(str(error), err)

    def test_skeleton_and_help_never_create_state_or_claim_success(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            env = dict(os.environ, LOOM_SERVE_DIR=str(root / "state"),
                       PI_CODING_AGENT_DIR=str(root / "pi"))
            for flags, expected in ((["--help"], 0), (["--check", "--json"], 1),
                                    (["--disable", "--json"], 1), (["--json"], 1)):
                process = subprocess.run([sys.executable, "-m", "binding.python.hem_cli",
                                          "pi", "configure", *flags], env=env,
                                         text=True, capture_output=True, check=False)
                self.assertEqual(process.returncode, expected, process.stderr)
                if expected:
                    self.assertIn("not implemented", process.stderr)
                    self.assertFalse(json.loads(process.stdout)["completed"])
                else:
                    self.assertIn("--check", process.stdout)
                self.assertFalse((root / "state").exists())
                self.assertFalse((root / "pi").exists())

    def test_provider_encoding_is_stable_and_avoids_case_punctuation_collisions(self):
        names = ["local", "Local", "a-b", "a_b", "a.b", "ab", "a"]
        identifiers = [provider_id(name) for name in names]
        self.assertEqual(len(set(identifiers)), len(names))
        self.assertEqual(provider_id("local"), "loom-hem-bG9jYWw")
        self.assertTrue(all(item.startswith("loom-hem-") for item in identifiers))
        for name in ("", "../a", "a/b", "é", None):
            with self.assertRaises(InvalidInput):
                provider_id(name)


if __name__ == "__main__":
    unittest.main()
