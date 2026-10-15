"""Retained Hugging Face streaming primitive regression checks."""
from contextlib import redirect_stderr, redirect_stdout
import io
import os
import sys
import unittest
from unittest.mock import Mock, patch

from binding.python import model_download


class DownloadTest(unittest.TestCase):
    def test_stream_redacts_secrets_and_leaves_stdout_clean(self):
        command = [sys.executable, "-c", "import os,sys; print('/cache/model.gguf'); print('Downloading '+os.environ['HF_TOKEN'], file=sys.stderr)"]
        with patch.dict(os.environ, {'HF_TOKEN': 'hf_testsecrettoken'}), redirect_stdout(io.StringIO()) as out, redirect_stderr(io.StringIO()) as err:
            metadata = model_download._run_hf(command)
        self.assertEqual(metadata, '/cache/model.gguf\n')
        self.assertEqual(out.getvalue(), '')
        self.assertIn('Downloading [REDACTED]', err.getvalue())
        self.assertNotIn('hf_testsecrettoken', err.getvalue())

    def test_hf_stream_preserves_metadata_and_forces_visible_progress(self):
        process = Mock(stdout=io.StringIO('/cache/model.gguf\n'), stderr=io.StringIO('weights: 50% 10G/20G\r'))
        process.wait.return_value = 0
        with patch('subprocess.Popen', return_value=process) as run, redirect_stderr(io.StringIO()) as updates, patch.dict(os.environ, {'TQDM_POSITION': '0'}):
            metadata = model_download._run_hf(['hf', 'download', 'repo', 'weights.gguf'])
        self.assertEqual(metadata, '/cache/model.gguf\n')
        self.assertIn('weights: 50% 10G/20G', updates.getvalue())
        self.assertEqual(run.call_args.kwargs['env']['TQDM_POSITION'], '-1')
        self.assertEqual(run.call_args.kwargs['env']['HF_HUB_DISABLE_PROGRESS_BARS'], '0')

    def test_download_failure_is_actionable(self):
        with patch('subprocess.Popen', side_effect=FileNotFoundError), redirect_stderr(io.StringIO()):
            with self.assertRaisesRegex(model_download.model_store.ModelImportError, 'hf auth login'):
                model_download._run_hf(['hf', 'download', 'repo', 'model.gguf'])
