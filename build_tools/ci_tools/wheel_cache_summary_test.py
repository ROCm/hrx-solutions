# Copyright 2026 The HRX Authors
# Licensed under the MIT License.
# SPDX-License-Identifier: MIT
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from build_tools.ci_tools.wheel_cache_summary import collect


class SummaryTest(unittest.TestCase):
    def test_measured_build_reports_identity_admission_actions_and_duration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "manifest.json").write_text(json.dumps(dict(source_commit="abc", sdk_identity="sdk", compiler_fingerprint="fingerprint", native_configuration="not reported")))
            (root / "admit.json").write_text(json.dumps(dict(ready=True, apparent_bytes=100, allocated_bytes=4096)))
            (root / "prepare.json").write_text('{"enabled":true}')
            (root / "build.seconds").write_text("12")
            (root / "result.whl").write_bytes(b"wheel")
            events = root / "events"
            events.write_text(json.dumps({"buildMetrics": {"timingMetrics": {"wallTimeInMs": "11033"}, "actionSummary": {"actionsExecuted": "3495", "runnerCount": [{"name": "disk cache hit", "count": 3495}, {"name": "linux-sandbox", "count": 0}, {"name": "total", "count": 5044}, {"name": "internal", "count": 1549}]}}}) + "\n")
            result = collect(root, root, root, events, {"GITHUB_SHA": "abc", "MATCHED_KEY": "older-key"})
            self.assertEqual(result["provenance"]["sdk_identity"], "sdk")
            self.assertNotIn("native_configuration", result["provenance"])
            self.assertEqual(result["admission"]["allocated_bytes"], 4096)
            self.assertEqual(result["native"]["runners"]["disk cache hit"], 3495)
            self.assertEqual(result["native"]["executed_processes"], 0)
            self.assertEqual(result["stage_seconds"]["build"], 12)
            self.assertEqual(result["wheels"][0]["sha256"], hashlib.sha256(b"wheel").hexdigest())

    def test_missing_and_malformed_optional_data_still_produces_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "manifest.json").write_text("malformed")
            (root / "events").write_text("bad JSON")
            result = collect(root, root, root, root / "events", {})
            self.assertEqual(result["provenance"], "unavailable: JSONDecodeError")
            self.assertEqual(result["native"], "unavailable: JSONDecodeError")
            self.assertEqual(result["wheels"], [])
            self.assertEqual(result["stage_seconds"], {})


if __name__ == "__main__":
    unittest.main()
