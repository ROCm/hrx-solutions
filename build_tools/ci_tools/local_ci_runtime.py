# Copyright 2026 The HRX Authors
# Licensed under the MIT License.
# SPDX-License-Identifier: MIT
"""Private local-CI transient state, independent of report-directory permissions."""
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
import uuid

STAGES = ('source', 'tmp', 'installed', 'model', 'openwebui')
MARKER = 'owner.json'


def safe_ancestors(path):
    if not path.is_absolute():
        raise ValueError('Private runtime paths must be absolute')
    for component in reversed((path, *path.parents)):
        info = component.lstat()
        sticky_root = info.st_uid == 0 and bool(info.st_mode & stat.S_ISVTX)
        if (not stat.S_ISDIR(info.st_mode) or info.st_uid not in (0, os.getuid()) or
                info.st_mode & 0o022 and not sticky_root):
            raise ValueError(f'Unsafe private runtime ancestor: {component}')


def validate_stage(path, report_path=None):
    path = Path(path)
    safe_ancestors(path)
    parent = path.parent
    info = parent.lstat()
    current = path.lstat()
    if (path.name not in STAGES or info.st_uid != os.getuid() or info.st_mode & 0o077 or
            current.st_uid != os.getuid() or current.st_mode & 0o077):
        raise ValueError('Private runtime root/stage must be private and same-user')
    marker = parent / MARKER
    marker_info = marker.lstat()
    if (not stat.S_ISREG(marker_info.st_mode) or marker_info.st_uid != os.getuid() or
            marker_info.st_mode & 0o077):
        raise ValueError('Unsafe private runtime owner marker')
    owner = json.loads(marker.read_text())
    if (owner.get('root') != str(parent) or owner.get('uid') != os.getuid() or
            owner.get('device') != info.st_dev or owner.get('inode') != info.st_ino or
            not Path(owner.get('run_dir', '')).is_absolute() or
            uuid.UUID(hex=owner.get('token', '')).hex != owner.get('token')):
        raise ValueError('Private runtime ownership does not match')
    if report_path is not None and not Path(report_path).absolute().is_relative_to(Path(owner['run_dir'])):
        raise ValueError('Private runtime belongs to another run')
    return path


def create(run_dir):
    # Ignore inherited TMPDIR: its ancestors may be group writable, which Pi
    # correctly refuses. Never chmod developer/workspace directory ancestors.
    safe_ancestors(Path('/tmp'))
    root = Path(tempfile.mkdtemp(prefix='hrx-local-ci-', dir='/tmp'))
    info = root.lstat()
    owner = {'token': uuid.uuid4().hex, 'root': str(root), 'uid': os.getuid(),
             'device': info.st_dev, 'inode': info.st_ino, 'run_dir': str(run_dir)}
    with os.fdopen(os.open(root / MARKER, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), 'w') as stream:
        json.dump(owner, stream, sort_keys=True)
        stream.write('\n')
    for name in STAGES:
        (root / name).mkdir(mode=0o700)
        validate_stage(root / name)
    return owner


if __name__ == '__main__':
    if len(sys.argv) != 3:
        raise SystemExit('Usage: local_ci_runtime.py PRIVATE_STAGE_DIRECTORY REPORT_PATH')
    print(validate_stage(Path(sys.argv[1]), Path(sys.argv[2])))
