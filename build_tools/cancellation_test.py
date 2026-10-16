"""Owned shared-shell cancellation without building or loading model weights."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest

REPO = Path(__file__).resolve().parents[1]

DOCKER = '''#!PYTHON
import json,os,sys,time
from pathlib import Path
root=Path(os.environ['FAKE_DOCKER'])
a=sys.argv[1:]
state=root/'state.json'
if a[0]=='create':
    (root/'creating').touch()
    time.sleep(float(os.environ.get('CREATE_DELAY','0')))
    label=a[a.index('--label')+1].split('=',1)
    state.write_text(json.dumps(dict(name=a[a.index('--name')+1],labels={label[0]:label[1]},args=a)))
    print('fake-id')
elif a[0]=='start':
    (root/'started').touch()
    while state.exists(): time.sleep(.02)
elif a[0]=='inspect':
    if not state.exists():
        print('Error: No such object',file=sys.stderr);sys.exit(1)
    if '--format' in a: print(json.dumps(json.loads(state.read_text())['labels']))
    else: print(state.read_text())
elif a[0]=='rm':
    if os.environ.get('FAIL_CLEANUP'): sys.exit(1)
    state.unlink(missing_ok=True)
else: sys.exit(7)
'''


class ShellCancellationTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.bin = self.root / 'bin'
        self.bin.mkdir()
        docker = self.bin / 'docker'
        docker.write_text(DOCKER.replace('PYTHON', sys.executable, 1))
        docker.chmod(0o700)
        self.cleanup = self.root / 'cleanup.json'
        self.env = dict(os.environ, PATH=str(self.bin) + os.pathsep + os.environ['PATH'],
                        FAKE_DOCKER=str(self.root), HRX_BAZEL_CACHE_ENABLED='0',
                        HRX_LOCAL_CI_CLEANUP_FILE=str(self.cleanup))
        self.env.pop('PYTHONPATH', None)
        self.env.pop('PYTHONHOME', None)
        self.env['PYTHONNOUSERSITE'] = '1'

    def wait_for(self, path, child):
        end = time.monotonic() + 8
        while not path.exists():
            self.assertIsNone(child.poll(), 'wrapper exited before observed stage')
            self.assertLess(time.monotonic(), end)
            time.sleep(.01)

    def cancel_build(self, early=False, failing=False, signum=signal.SIGTERM):
        environment = dict(self.env, CREATE_DELAY='.3' if early else '0')
        if failing:
            environment['FAIL_CLEANUP'] = '1'
        sentinel = subprocess.Popen([sys.executable, '-c', 'import time;time.sleep(60)'])
        child = subprocess.Popen(['bash', str(REPO / 'build_tools/build-wheel.sh')], env=environment,
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            self.wait_for(self.root / ('creating' if early else 'started'), child)
            child.send_signal(signum)
            time.sleep(.04)
            if child.poll() is None:
                child.send_signal(signum)
            stdout, stderr = child.communicate(timeout=12)
            self.assertEqual(child.returncode, 128 + signum, (stdout, stderr))
            result = json.loads(self.cleanup.read_text())
            self.assertEqual(result['status'], 'unresolved' if failing else 'complete')
            self.assertEqual((self.root / 'state.json').exists(), failing)
            self.assertIsNone(sentinel.poll())
        finally:
            if child.poll() is None:
                child.kill()
                child.communicate(timeout=5)
            (self.root / 'state.json').unlink(missing_ok=True)
            sentinel.terminate()
            sentinel.wait(timeout=5)

    def test_build_sigterm_stops_exact_container_and_reaps(self):
        self.cancel_build()

    def test_build_sigint_before_create_id_and_repeated_signal(self):
        self.cancel_build(early=True, signum=signal.SIGINT)

    def test_build_cleanup_failure_preserves_cancellation_and_unresolved(self):
        self.cancel_build(failing=True)

    def test_model_setup_sigterm_reaps_only_owned_setup_session(self):
        python = self.bin / 'python3'
        python.write_text('#!' + sys.executable + '\nimport os,sys,time\nfrom pathlib import Path\n'
                          'if sys.argv[1:3]==["-m","venv"]:\n'
                          ' Path(os.environ["FAKE_DOCKER"],"setup").touch()\n'
                          ' time.sleep(60)\n'
                          'else: os.execv(sys.executable,[sys.executable,*sys.argv[1:]])\n')
        python.chmod(0o700)
        wheel = self.root / 'fake.whl'
        wheel.touch()
        child = subprocess.Popen(['bash', str(REPO / 'build_tools/model-smoke.sh'), str(wheel), str(self.root / 'run')],
                                 env=self.env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            self.wait_for(self.root / 'setup', child)
            child.terminate()
            out, err = child.communicate(timeout=12)
            self.assertEqual(child.returncode, 143, (out, err))
            self.assertEqual(json.loads(self.cleanup.read_text())['status'], 'complete')
        finally:
            if child.poll() is None:
                child.kill()
                child.communicate(timeout=5)


if __name__ == '__main__':
    unittest.main()
