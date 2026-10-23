"""The native launcher forwards deployment arguments without model policy."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from binding.python import _launcher


class LauncherTest(unittest.TestCase):
    def test_native_deployment_arguments_and_environment_pass_through(self):
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory)
            executable = package / '_native/bin/loom-serve'
            executable.parent.mkdir(parents=True); executable.touch()
            arguments = ['--model=/model/deployment.json', '--device=cpu', '--listen=127.0.0.1:18080', '--unknown-native-option']
            with patch.object(_launcher, '__file__', str(package / '_launcher.py')), \
                 patch('sys.argv', ['loom-serve', *arguments]), \
                 patch.object(_launcher, 'native_environment', return_value={'NATIVE': 'environment'}), \
                 patch.object(_launcher.os, 'execve', side_effect=SystemExit(0)) as execute:
                with self.assertRaises(SystemExit): _launcher.exec_loom_serve()
            execute.assert_called_once_with(str(executable), [str(executable), *arguments], {'NATIVE': 'environment'})

    def test_missing_native_executable_is_actionable(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(_launcher, '__file__', str(Path(directory) / '_launcher.py')):
            with self.assertRaisesRegex(RuntimeError, 'missing native executable'): _launcher.exec_loom_serve()
