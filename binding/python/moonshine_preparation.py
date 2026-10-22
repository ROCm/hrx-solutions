"""Invoke the fixed HRX-owned Moonshine preparer; Control owns only its I/O."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading

from .hem_common import HemError
from . import model_download
from .model_store import readable_file
from .moonshine_preparer_identity import FILES as TOOL_FILES, RELATIVE, REVISION

TIMEOUT = 300
GRACE = 5
GUIDANCE = ('Moonshine acquisition requires Control running on Python >=3.12 with '
            'hrx-solutions[moonshine] installed. Install that extra in your Python '
            'environment, or import a complete prepared set with --assets-dir.')


def tool_directory():
    return Path(__file__).resolve().parent / '_native' / RELATIVE


def environment():
    value = os.environ.copy()
    for name in ('PYTHONPATH', 'PYTHONHOME'):
        value.pop(name, None)
    value.update(HF_HUB_CACHE=str(model_download.cache_root()), HF_HUB_OFFLINE='1',
                 HF_HUB_DISABLE_TELEMETRY='1')
    return value


def _run(arguments, *, timeout, stdout, stderr):
    """Own and reap this fixed preparer's group even on timeout/interruption."""
    main_thread = threading.current_thread() is threading.main_thread()
    previous = signal.getsignal(signal.SIGTERM) if main_thread else None
    def terminated(signum, frame):
        raise HemError('HRX Moonshine preparation interrupted by SIGTERM')
    if main_thread:
        signal.signal(signal.SIGTERM, terminated)
    process = None
    try:
        process = subprocess.Popen(arguments, env=environment(), start_new_session=True,
                                   cwd=tool_directory(), stdout=stdout, stderr=stderr)
        return process.wait(timeout=timeout)
    except BaseException:
        if main_thread:
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
        if process is not None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=GRACE)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
        raise
    finally:
        if main_thread:
            signal.signal(signal.SIGTERM, previous)


def preflight():
    if sys.version_info < (3, 12):
        raise HemError(GUIDANCE)
    root = tool_directory()
    try:
        for name, checksum in TOOL_FILES.items():
            if hashlib.sha256((root / name).read_bytes()).hexdigest() != checksum:
                raise HemError('Installed Moonshine preparation tool integrity differs; reinstall the wheel')
        with tempfile.TemporaryFile() as errors:
            code = _run([sys.executable, '-c',
                         'from huggingface_hub import hf_hub_download; from safetensors import TensorSpec, serialize'],
                        timeout=30, stdout=subprocess.DEVNULL, stderr=errors)
        if code:
            raise HemError(GUIDANCE)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise HemError(f'{GUIDANCE} Preparation preflight failed: {error}') from error


def prepare(registry, record):
    """Return completed private rotary output, never a helper launch document."""
    parent = registry.root / 'preparation-attempts' / record['id']
    try:
        parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        attempt = Path(tempfile.mkdtemp(prefix='attempt-', dir=parent))
    except OSError as error:
        raise HemError(f'Could not reserve private preparation output; incomplete acquisition retained: {error}') from error
    output = attempt / 'output'  # Upstream requires a nonexistent output path.
    arguments = [sys.executable, str(tool_directory() / 'prepare_assets.py'),
                 '--output', str(output), '--name', 'moonshine-streaming-tiny', '--offline']
    result = {'tool_revision': REVISION, 'completed': False}
    try:
        with (attempt / 'stdout.log').open('wb') as stdout, (attempt / 'stderr.log').open('wb') as stderr:
            code = _run(arguments, timeout=TIMEOUT, stdout=stdout, stderr=stderr)
        if code:
            raise HemError(f'HRX Moonshine preparation exited {code}; incomplete acquisition retained; inspect {attempt}')
        if any(not readable_file(output / name) for name in
               ('model.safetensors', 'tokenizer.json', 'config.json', 'rotary.safetensors',
                'provenance.json', 'model.json')):
            raise HemError(f'HRX preparation did not complete its outputs; inspect {attempt}')
        result['completed'] = True
        return output / 'rotary.safetensors'
    except (OSError, subprocess.TimeoutExpired) as error:
        result['error'] = str(error)
        raise HemError(f'HRX Moonshine preparation failed; incomplete acquisition retained; inspect {attempt}: {error}') from error
    finally:
        (attempt / 'receipt.json').write_text(json.dumps(result, indent=2) + '\n')
