"""Invoke the fixed HRX-owned Moonshine preparer; Control owns only its I/O."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
import tempfile
import threading
from collections.abc import Mapping

from ..hem_common import HemError
from . import model_download
from .model_store import readable_file
from .recipe_types import GeneratedArtifact, HFSource
from .moonshine_preparer_identity import (
    FILES as TOOL_FILES, RELATIVE, REVISION,
    INPUT_REPOSITORY, INPUT_REVISION, INPUT_FILES, GENERATED_FILES,
)

TIMEOUT = 300
GRACE = 5
GUIDANCE = ('Moonshine acquisition requires Control running on Python >=3.12 with '
            'hrx-solutions[moonshine] installed. Install that extra in your Python '
            'environment, or import a complete prepared set with --assets-dir.')


def tool_directory():
    return Path(__file__).resolve().parents[1] / '_native' / RELATIVE


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


def preflight(*, deadline=None):
    if deadline is not None:
        deadline.remaining()
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
                        timeout=30 if deadline is None else min(30, deadline.remaining()),
                        stdout=subprocess.DEVNULL, stderr=errors)
        if code:
            raise HemError(GUIDANCE)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise HemError(f'{GUIDANCE} Preparation preflight failed: {error}') from error


def _check_inputs(inputs):
    if not isinstance(inputs, Mapping) or set(inputs) != set(INPUT_FILES):
        raise HemError('Moonshine preparation inputs differ from its reviewed pinned descriptor')
    for name, path in inputs.items():
        source = HFSource(INPUT_REPOSITORY, INPUT_REVISION, name)
        if (not isinstance(path, Path) or not path.is_absolute()
                or path != model_download._cache_path(source) or not readable_file(path)):
            raise HemError('Moonshine preparation requires its explicit readable pinned cache inputs')


def prepare(inputs, attempt, deadline):
    """Consume explicit inputs in caller scratch; return one complete output set."""
    directory = Path(attempt)
    output = directory / 'output'
    arguments = [sys.executable, str(tool_directory() / 'prepare_assets.py'),
                 '--output', str(output), '--name', 'moonshine-streaming-tiny', '--offline']
    # Refuse unsafe caller scratch before opening diagnostic paths through it.
    try:
        info = directory.lstat()
        if (not directory.is_absolute() or directory.resolve() != directory
                or not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid()
                or info.st_mode & 0o077 or os.path.lexists(output)):
            raise ValueError()
    except (OSError, ValueError, RuntimeError) as error:
        raise HemError('Moonshine preparation requires a private fresh caller-owned attempt') from error
    result = {'tool_revision': REVISION, 'completed': False}
    try:
        _check_inputs(inputs)
        with (directory / 'stdout.log').open('wb') as stdout, (directory / 'stderr.log').open('wb') as stderr:
            code = _run(arguments, timeout=min(TIMEOUT, deadline.remaining()), stdout=stdout, stderr=stderr)
        if code:
            raise HemError(f'HRX Moonshine preparation exited {code}; incomplete acquisition retained; inspect {directory}')
        deadline.remaining()
        if any(not readable_file(output / name) for name in
               ('model.safetensors', 'tokenizer.json', 'config.json', 'rotary.safetensors',
                'provenance.json', 'model.json')):
            raise HemError(f'HRX preparation did not complete its outputs; inspect {directory}')
        generated = []
        for role, format, name in GENERATED_FILES:
            path = output / name
            if output.is_symlink() or path.is_symlink() or not readable_file(path):
                raise HemError('Moonshine generated output is missing or aliased outside its attempt')
            path.resolve(strict=True).relative_to(output.resolve(strict=True))
            output.resolve(strict=True).relative_to(directory)
            generated.append(GeneratedArtifact(role, format, name, path))
        result['completed'] = True
        return tuple(generated)
    except (OSError, ValueError, subprocess.TimeoutExpired) as error:
        result['error'] = str(error)
        raise HemError(f'HRX Moonshine preparation failed; incomplete acquisition retained; inspect {directory}: {error}') from error
    except BaseException as error:
        result['error'] = str(error)
        raise
    finally:
        (directory / 'receipt.json').write_text(json.dumps(result, indent=2) + '\n')
