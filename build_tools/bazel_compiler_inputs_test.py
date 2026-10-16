"""Tests for the unmanaged compiler-input cache contract."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from build_tools import bazel_compiler_inputs as compiler


class CompilerInputsTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        for name in ("lib/llvm/bin/clang", "lib/llvm/bin/clang++", "lib/llvm/bin/ld.lld",
                     "lib/llvm/lib/libLLVM.so", "lib/llvm/lib/clang/24/include/stddef.h",
                     "lib/llvm/include/c++/v1/cstddef", "lib/llvm/bin/clang.cfg"):
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(name)
        self.library = self.root / "lib/llvm/lib/libLLVM.so"
        self.resource = self.root / "lib/llvm/lib/clang/24"
        self.run = patch.object(compiler, "_run", side_effect=self.command)
        self.run.start()
        self.addCleanup(self.run.stop)

    def command(self, *args):
        if args[0] == "ldd":
            return f"libLLVM.so => {self.library} (0x1234)\n"
        return str(self.resource) + "\n"

    def identity(self):
        return compiler.fingerprint(self.root, "sha256:fixture")["fingerprint"]

    def test_actual_content_changes_invalidate(self):
        for name in ("lib/llvm/bin/clang", "lib/llvm/bin/ld.lld", "lib/llvm/lib/libLLVM.so",
                     "lib/llvm/lib/clang/24/include/stddef.h", "lib/llvm/include/c++/v1/cstddef",
                     "lib/llvm/bin/clang.cfg"):
            with self.subTest(name=name):
                before = self.identity()
                path = self.root / name
                path.write_text(path.read_text() + "changed")
                self.assertNotEqual(before, self.identity())

    def test_provenance_and_unrelated_runtime_do_not_invalidate(self):
        before = self.identity()
        (self.root / ".hrx-rocm-artifacts.json").write_text(json.dumps({"run": 2}))
        (self.root / "lib/libhsa-runtime64.so.1").write_text("new runtime")
        self.assertEqual(before, self.identity())

    def test_symlinked_resource_directory_contents_are_tracked(self):
        target = self.root / "runtime-libraries"
        target.mkdir()
        library = target / "libbuiltins.a"
        library.write_text("old")
        (self.resource / "lib").symlink_to(target, target_is_directory=True)
        before = self.identity()
        library.write_text("new")
        self.assertNotEqual(before, self.identity())

    def test_resource_symlink_cycle_fails_closed(self):
        (self.resource / "cycle").symlink_to(self.resource, target_is_directory=True)
        with self.assertRaisesRegex(RuntimeError, "cycle"):
            self.identity()

    def test_image_change_invalidates_system_inputs(self):
        self.assertNotEqual(self.identity(),
                            compiler.fingerprint(self.root, "sha256:new")["fingerprint"])

    def test_unresolved_library_fails_closed(self):
        with patch.object(compiler, "_run", return_value="libLLVM.so => not found"):
            with self.assertRaisesRegex(RuntimeError, "Unresolved"):
                self.identity()

    def test_configuration_replaces_both_exec_and_target_environment(self):
        path = self.root / "bazelrc"
        path.write_text("build --config=opt\n")
        compiler.configure(path, "old")
        compiler.configure(path, "new")
        self.assertEqual(path.read_text(), "build --config=opt\n"
                         "common --action_env=HRX_WHEEL_COMPILER_INPUTS=new\n"
                         "common --host_action_env=HRX_WHEEL_COMPILER_INPUTS=new\n")


if __name__ == "__main__":
    unittest.main()
