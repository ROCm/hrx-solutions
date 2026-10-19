# Copyright 2026 The HRX Authors
# Licensed under the MIT License.
# SPDX-License-Identifier: MIT
"""Capture current source bytes with independent, container-visible Git metadata."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess


def git(root, *args):
    return subprocess.check_output(['git', '--no-optional-locks', '-C', str(root), *args])


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def inputs(root):
    tracked = (set(git(root, 'ls-files', '-z', '--cached').decode().split('\0')) |
               set(git(root, 'ls-tree', '-rz', '--name-only', 'HEAD').decode().split('\0'))) - {''}
    untracked = set(git(root, 'ls-files', '-z', '--others', '--exclude-standard').decode().split('\0')) - {''}
    answer = {}
    for name in sorted(tracked | untracked):
        parts = Path(name).parts
        if parts[0] == '.git' or name in ('.bazelrc.local', '.bazelrc.configured'):
            continue
        if name not in tracked and (parts[0] in ('.deps', 'build', 'dist', '.beads', '.agents', '.codex', '.loom-serve',
                                                   '.loom', 'loom-state') or
                                   parts[0].startswith('bazel-') or
                                   Path(name).suffix.lower() in ('.gguf', '.safetensors', '.pt', '.pth', '.ckpt', '.whl')):
            continue
        path = root / name
        try:
            info = path.lstat()
        except FileNotFoundError:
            continue  # Tracked deletion is part of the captured identity.
        if stat.S_ISLNK(info.st_mode):
            target = os.readlink(path)
            resolved = path.resolve(strict=True)
            if os.path.isabs(target) or not resolved.is_relative_to(root):
                raise ValueError(f'Escaping source symlink: {name}')
            answer[name] = {'kind': 'symlink', 'target': target}
        elif stat.S_ISREG(info.st_mode):
            answer[name] = {'kind': 'file', 'mode': stat.S_IMODE(info.st_mode),
                            'executable': bool(info.st_mode & 0o111),
                            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        else:
            raise ValueError(f'Unsupported source input: {name}')
    return answer


def head_inputs(root):
    result = {}
    for entry in git(root, 'ls-tree', '-rz', 'HEAD').split(b'\0'):
        if not entry:
            continue
        meta, raw_name = entry.split(b'\t', 1)
        mode, kind, oid = meta.decode().split()
        name = raw_name.decode()
        if name in ('.bazelrc.local', '.bazelrc.configured'):
            continue
        if kind != 'blob':
            raise ValueError(f'Unsupported Git input: {name} ({kind})')
        content = git(root, 'cat-file', 'blob', oid)
        if mode == '120000':
            result[name] = {'kind': 'symlink', 'target': content.decode()}
        else:
            result[name] = {'kind': 'file', 'mode': 0o755 if mode == '100755' else 0o644,
                            'executable': mode == '100755',
                            'sha256': hashlib.sha256(content).hexdigest()}
    return result


def capture(root, destination):
    root, destination = root.resolve(), destination.resolve()
    head = git(root, 'rev-parse', 'HEAD').decode().strip()
    before = inputs(root)
    status = git(root, 'status', '--porcelain=v1').decode()
    subprocess.run(['git', 'clone', '--local', '--no-hardlinks', '--no-checkout', '--quiet',
                    str(root), str(destination)], check=True)
    subprocess.run(['git', '-C', str(destination), 'checkout', '--quiet', '--detach', head], check=True)
    for path in destination.iterdir():
        if path.name == '.git':
            continue
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink()
    for name, item in before.items():
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if item['kind'] == 'symlink':
            target.symlink_to(item['target'])
        else:
            shutil.copyfile(root / name, target, follow_symlinks=False)
            target.chmod(item['mode'])
    after = inputs(root)
    if before != after or head != git(root, 'rev-parse', 'HEAD').decode().strip():
        raise ValueError('Source changed during capture; retry with a stable checkout')
    # Fresh untracked content is included despite the clone's own ignore rules.
    verify(destination, before)
    # Git's meaningful regular-file mode is the executable bit. Keep complete
    # permissions in the snapshot manifest without treating umask as a source edit.
    identity = lambda manifest: {name: {key: value for key, value in item.items() if key != 'mode'}
                                 for name, item in manifest.items()}
    return {'head': head, 'dirty': identity(before) != identity(head_inputs(root)), 'git_status': status,
            'manifest_digest': digest(before), 'inputs': before}


def verify(root, manifest):
    for name, expected in manifest.items():
        path = root / name
        if expected['kind'] == 'symlink':
            if (not path.is_symlink() or os.readlink(path) != expected['target'] or
                    not path.resolve(strict=True).is_relative_to(root)):
                raise ValueError(f'Captured input changed: {name}')
        elif not path.is_file() or path.is_symlink() or (
                hashlib.sha256(path.read_bytes()).hexdigest() != expected['sha256'] or
                stat.S_IMODE(path.stat().st_mode) != expected['mode']):
            raise ValueError(f'Captured input changed: {name}')
