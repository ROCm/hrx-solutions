"""Optional cache policy and Docker-wrapper failure contract."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

from build_tools.ci_tools import wheel_cache as cache

REPO = Path(__file__).resolve().parents[2]


class WheelCacheTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "snapshot"

    def complete(self):
        self.root.mkdir(exist_ok=True)
        (self.root / "bazel-actions").mkdir(exist_ok=True)
        data = dict(schema=cache.SCHEMA, compiler_contract=cache.CONTRACT, complete=True)
        for name in ("source_commit", "hrx_pin", "bazel_version", "container_digest", "artifact_set", "sdk_identity", "compiler_fingerprint", "native_configuration", "build_run", "build_attempt"):
            data[name] = "fixture"
        cache.write_manifest(self.root, data)

    def test_first_run_and_supported_older_snapshot(self):
        self.assertTrue(cache.prepare(str(self.root))["enabled"])
        # CI preparation and wrapper preparation can both run on the first seed.
        self.assertTrue(cache.prepare(str(self.root))["enabled"])
        self.complete()
        self.assertTrue(cache.prepare(str(self.root), "restored")["enabled"])
        self.assertTrue(cache.admit(str(self.root))["ready"])

    def test_partial_unknown_and_failed_restore_rejected(self):
        self.root.mkdir()
        (self.root / "bazel-actions").mkdir()
        with self.assertRaises(ValueError):
            cache.prepare(str(self.root), "restored")
        self.assertFalse(cache.prepare(str(self.root), "failed")["enabled"])
        self.complete()
        data = json.loads((self.root / "manifest.json").read_text())
        data["schema"] += 1
        cache.write_manifest(self.root, data)
        with self.assertRaises(ValueError):
            cache.prepare(str(self.root))

    def test_unfinished_oversized_and_changing_not_admitted(self):
        self.complete()
        self.assertFalse(cache.admit(str(self.root), 1)["ready"])
        with mock.patch.object(cache, "sizes", side_effect=[(1, 1), (2, 2)]):
            self.assertFalse(cache.admit(str(self.root))["ready"])
        data = json.loads((self.root / "manifest.json").read_text())
        data["complete"] = False
        cache.write_manifest(self.root, data)
        with self.assertRaises(ValueError):
            cache.admit(str(self.root))

    def test_previous_build_is_not_admitted_for_publication(self):
        self.complete()
        with mock.patch.dict(os.environ, {"GITHUB_RUN_ID": "new-run", "GITHUB_RUN_ATTEMPT": "2"}):
            self.assertFalse(cache.admit(str(self.root))["ready"])

    def test_paths_and_symlinks_rejected(self):
        for path in ("relative", str(REPO), str(REPO / "cache"), str(REPO.parent), "/"):
            with self.assertRaises(ValueError):
                cache.prepare(path)
        self.complete()
        (self.root / "bazel-actions" / "external").symlink_to("/tmp")
        with self.assertRaises(ValueError):
            cache.prepare(str(self.root), "restored")
        with self.assertRaises(ValueError):
            cache.admit(str(self.root))

    def test_configuration_repeat_and_disable(self):
        rc = Path(self.temp.name) / "rc"
        rc.write_text("build --compilation_mode=opt\n")
        cache.configure(rc, True)
        cache.configure(rc, True)
        self.assertEqual(rc.read_text().count("--disk_cache"), 1)
        self.assertIn("--remote_cache_async=false", rc.read_text())
        with rc.open("a") as stream:
            stream.write("common --action_env=HRX_WHEEL_COMPILER_INPUTS=old\n")
            stream.write("common --host_action_env=HRX_WHEEL_COMPILER_INPUTS=old\n")
        cache.configure(rc, False)
        self.assertEqual(rc.read_text(), "build --compilation_mode=opt\n")

    def test_failed_configuration_reports_failure_without_claiming_enabled(self):
        rc = Path(self.temp.name) / "rc-directory"
        rc.mkdir()
        result = subprocess.run([os.sys.executable, str(REPO / "build_tools/wheel_cache.py"), "configure", "--bazelrc", str(rc)], check=True, capture_output=True, text=True)
        data = json.loads(result.stdout)
        self.assertFalse(data["enabled"])
        self.assertFalse(data.get("configured", False))

    def test_wrapper_cache_unavailable_does_not_block_docker(self):
        bindir = Path(self.temp.name) / "bin"
        bindir.mkdir()
        docker = bindir / "docker"
        from build_tools.validation.cancellation_test import DOCKER
        docker.write_text(DOCKER.replace('PYTHON', os.sys.executable, 1).replace(
            'while state.exists(): time.sleep(.02)', 'pass').replace(
            "(root/'creating').touch()", "(root/'creating').touch(); Path(os.environ['CAPTURE']).write_text('\\n'.join(a))"))
        docker.chmod(0o755)
        capture = Path(self.temp.name) / "args"
        environment = dict(os.environ, PATH=f"{bindir}:{os.environ['PATH']}", CAPTURE=str(capture), FAKE_DOCKER=str(Path(self.temp.name)))
        for directory, enabled in (("relative", "0"), (str(self.root), "1")):
            environment["HRX_BAZEL_CACHE_DIR"] = directory
            subprocess.run(["bash", str(REPO / "build_tools/build-wheel.sh")], env=environment, check=True, capture_output=True)
            args = capture.read_text()
            self.assertIn(f"HRX_BAZEL_CACHE_ENABLED={enabled}", args)
            self.assertEqual(":/ci-cache" in args, enabled == "1")
        environment["HRX_BAZEL_CACHE_ENABLED"] = "0"
        subprocess.run(["bash", str(REPO / "build_tools/build-wheel.sh")], env=environment, check=True, capture_output=True)
        self.assertIn("HRX_BAZEL_CACHE_ENABLED=0", capture.read_text())
        environment.pop("HRX_BAZEL_CACHE_ENABLED")
        (self.root / "unexpected").touch()
        subprocess.run(["bash", str(REPO / "build_tools/build-wheel.sh")], env=environment, check=True, capture_output=True)
        self.assertIn("HRX_BAZEL_CACHE_ENABLED=0", capture.read_text())

    def test_unwritable_and_malformed_cli_is_optional(self):
        self.complete()
        (self.root / "manifest.json").write_text("{partial")
        result = subprocess.run([os.sys.executable, str(REPO / "build_tools/wheel_cache.py"), "prepare", "--directory", str(self.root)], check=True, capture_output=True, text=True)
        self.assertFalse(json.loads(result.stdout)["enabled"])
        with mock.patch.object(cache.tempfile, "TemporaryFile", side_effect=PermissionError("read-only")):
            empty = Path(self.temp.name) / "unwritable"
            with self.assertRaises(PermissionError):
                cache.prepare(str(empty))


if __name__ == "__main__":
    unittest.main()
