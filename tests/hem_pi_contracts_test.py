"""Pi command boundary before configuration, filesystem and provider workers."""

from contextlib import redirect_stderr, redirect_stdout
import io
import gc
import warnings
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
        # Earlier SQLite fixtures can leave cyclic garbage. Drain it outside
        # this command-only capture so its warnings remain visible without
        # being mistaken for output from the CLI under test.
        gc.collect()
        out, err = io.StringIO(), io.StringIO()
        with patch.object(hem_pi_cli, "run", return_value=response, side_effect=error) as run:
            with redirect_stdout(out), redirect_stderr(err):
                code = hem_cli.main(["pi", "configure", "--json", *suffix])
        return code, json.loads(out.getvalue()), err.getvalue(), run

    def test_only_one_command_and_exclusive_modes(self):
        self.assertIsNone(hem_cli.parse_args(["pi"]).command)
        for args in (["pi", "sync"], ["pi", "configure", "a-model"],
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

    def test_prior_fixture_resource_warning_stays_outside_command_capture(self):
        class PriorFixture:
            def __del__(self):
                warnings.warn("prior fixture resource was not closed", ResourceWarning)
        # Keep the cycle pending until invoke's explicit collection, even when
        # automatic collection happens at an unrelated bytecode allocation.
        enabled = gc.isenabled()
        gc.disable()
        prior = io.StringIO()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("always", ResourceWarning)
                resource = PriorFixture()
                resource.cycle = resource
                del resource
                with redirect_stderr(prior):
                    code, _, err, _ = self.invoke(response=PiResult(
                        "unchanged", configured=True, changed=False, completed=True, exit_code=0))
            self.assertEqual(code, 0)
            self.assertEqual(err, "")
            self.assertIn("ResourceWarning: prior fixture resource was not closed", prior.getvalue())
        finally:
            if enabled:
                gc.enable()

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

    def test_help_is_passive(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            env = dict(os.environ, LOOM_SERVE_DIR=str(root / "state"),
                       PI_CODING_AGENT_DIR=str(root / "pi"))
            process = subprocess.run([sys.executable, "-m", "binding.python.hem_cli",
                                      "pi", "configure", "--help"], env=env,
                                     text=True, capture_output=True, check=False)
            self.assertEqual(process.returncode, 0, process.stderr)
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
