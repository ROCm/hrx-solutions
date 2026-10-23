"""Stage exact pinned HRX preparation tool bytes, never an upstream checkout path."""
import hashlib
import json
from pathlib import Path
import shutil

from binding.python.models.moonshine_preparer_identity import FILES, RELATIVE, REVISION, UPSTREAM


def stage(native_root, hrx_source):
    source = hrx_source / UPSTREAM
    destination = native_root / RELATIVE
    destination.mkdir(parents=True)
    for name, checksum in FILES.items():
        content = (source / name).read_bytes()
        if hashlib.sha256(content).hexdigest() != checksum:
            raise ValueError('Selected HRX Moonshine preparer differs from reviewed immutable input: ' + name)
        (destination / name).write_bytes(content)
    shutil.copyfile(hrx_source / 'LICENSE', destination / 'LICENSE')
    (destination / 'provenance.json').write_text(json.dumps(dict(
        format=1, repository='https://github.com/ROCm/hrx-system', revision=REVISION,
        upstream_directory=UPSTREAM, files=FILES), indent=2) + '\n')

    return [destination / name for name in (*FILES, 'LICENSE', 'provenance.json')]


def validate_archive(archive):
    prefix = 'hrx_solutions/_native/' + RELATIVE + '/'
    expected = {prefix + name for name in (*FILES, 'LICENSE', 'provenance.json')}
    actual = {name for name in archive.namelist() if name.startswith(prefix) and not name.endswith('/')}
    if actual != expected:
        raise ValueError('Installed Moonshine preparer inventory mismatch')
    for name, checksum in FILES.items():
        if hashlib.sha256(archive.read(prefix + name)).hexdigest() != checksum:
            raise ValueError('Installed Moonshine preparer bytes differ: ' + name)
    metadata = json.loads(archive.read(prefix + 'provenance.json'))
    if metadata != dict(format=1, repository='https://github.com/ROCm/hrx-system',
                        revision=REVISION, upstream_directory=UPSTREAM, files=FILES):
        raise ValueError('Installed Moonshine preparer provenance differs')
    if archive.read(prefix + 'LICENSE') != archive.read('hrx_solutions/_native/share/loom-serve/LICENSE'):
        raise ValueError('Installed Moonshine preparer license differs from selected runtime')
