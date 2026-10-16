"""Bounded OMP discovery/copied-generation contract; no model inference."""
import os
from pathlib import Path
import shutil
import subprocess
import unittest


class OMPAdapterTest(unittest.TestCase):
    def test_adapter_contract(self):
        bun = os.environ.get("LOOM_OMP_TEST_BUN") or shutil.which("bun")
        if not bun:
            self.skipTest("Bun >=1.3.14 required for qualified OMP adapter contract")
        source = Path(__file__).resolve().parents[1]
        result = subprocess.run([bun, str(source / "tests/omp_adapter_test.mjs"),
                                 str(source / "binding/python/omp_extension")],
                                capture_output=True, text=True, timeout=25, check=False)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('"checks":', result.stdout)


if __name__ == "__main__":
    unittest.main()
