# Copyright 2026 The HRX Authors
# Licensed under the MIT License.
# SPDX-License-Identifier: MIT
"""Provision the frozen native onboarding inputs; callers own the destination."""
import hashlib
import json
import os
from pathlib import Path
import zipfile

FIXTURES = Path(__file__).resolve().parents[2] / 'tests/fixtures/generic_modalities'


def built_decoder(rocm_root):
    """Locate the decoder with the wheel's exact Bazel build configuration."""
    from build_tools.model_tools.loom_catalog import DECODER_TARGET
    from build_tools.wheel_tools.python_packaging import (
        BazelBuildContext, REPO_ROOT, _native_output, bazel_environment,
        discover_bazel, wheel_build_options,
    )
    context = BazelBuildContext(discover_bazel(), REPO_ROOT,
                                wheel_build_options(rocm_root), bazel_environment())
    return _native_output(context, DECODER_TARGET)


def prepare(destination, decoder):
    decoder = decoder.resolve(strict=True)
    if not decoder.is_file() or not os.access(decoder, os.X_OK):
        raise ValueError('Onboarding requires the real executable Loom decoder')
    manifest = json.loads((FIXTURES / 'native-fixture.json').read_text())
    archive = FIXTURES / manifest['archive']
    if hashlib.sha256(archive.read_bytes()).hexdigest() != manifest['sha256']:
        raise ValueError('Frozen onboarding archive SHA256 mismatch')
    destination = destination.absolute()
    # Refuse existing destinations so a prior run cannot supply hidden inputs.
    destination.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(archive) as bundle:
        if set(bundle.namelist()) != set(manifest['members']):
            raise ValueError('Frozen onboarding archive inventory mismatch')
        for name, expected in manifest['members'].items():
            content = bundle.read(name)
            if hashlib.sha256(content).hexdigest() != expected:
                raise ValueError('Frozen onboarding member SHA256 mismatch: ' + name)
            path = destination / name
            path.resolve().relative_to(destination.resolve())
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
    return {
        'GENERIC_MODALITY_FIXTURE_ARCHIVE': str(archive),
        'GENERIC_MODALITY_FIXTURE_SOURCE_ROOT': str(destination / 'source'),
        'GENERIC_MODALITY_FIXTURE_DECODER': str(decoder),
    }
