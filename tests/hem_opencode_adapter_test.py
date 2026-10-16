"""Bounded OpenCode discovery and copied-generation contract, no model inference."""
import os
from pathlib import Path
import shutil
import subprocess
import unittest


class OpenCodeAdapterTest(unittest.TestCase):
    def test_adapter_contract(self):
        node = os.environ.get("LOOM_OPENCODE_TEST_NODE") or shutil.which("node")
        if not node:
            self.skipTest("Node >=22 required for TypeScript adapter source contract")
        source = Path(__file__).resolve().parents[1]
        result = subprocess.run([node, "--experimental-strip-types", "--no-warnings",
                                 str(source / "tests/opencode_adapter_test.mjs"),
                                 str(source / "binding/python/opencode_plugin")],
                                capture_output=True, text=True, timeout=25, check=False)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('"checks":', result.stdout)


if __name__ == "__main__":
    unittest.main()
