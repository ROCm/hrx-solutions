#!/usr/bin/env python3
# Copyright 2026 The HRX Authors
# Licensed under the MIT License.
# SPDX-License-Identifier: MIT
"""Validate the installed wheel assets and bundled runtime without a GPU."""
import importlib.metadata
from pathlib import Path
import hrx_solutions

print(f"hrx-solutions {importlib.metadata.version('hrx-solutions')}")
print(f"package: {hrx_solutions.__file__}")
package_root = Path(hrx_solutions.__file__).resolve().parent
assert (package_root / "_native/bin/loom-serve").is_file()
assert (package_root / "_native/share/loom-serve/LICENSE").is_file()
assert not (package_root / "_native/bin/loom-qwen-epoch-check").exists()
from hrx_solutions.gguf import inspect_gguf
from hrx_solutions.qwen_validation import load_profile
from hrx_solutions.pipeline_validation import resolve_catalog
catalog_root = package_root / "loom-pipeline/llm"
for pipeline, architecture in (("qwen", "qwen_27b"), ("qwen-08b", "qwen_08b")):
    catalog = catalog_root / "qwen" / architecture
    assert resolve_catalog(pipeline) == catalog
    contract = load_profile(catalog)
    assert contract["metadata"]["general.architecture"] == "qwen35"
    manifest = catalog / "sources.txt"
    assert manifest.is_file()
    for line in manifest.read_text().splitlines():
        relative = line.strip()
        if relative and not relative.startswith("#"):
            source = (catalog / relative).resolve(strict=True)
            source.relative_to(catalog_root)
            assert source.is_file()
assert (package_root / "loom-pipeline/llm/LICENSE-APACHE-2.0").is_file()
import struct
import tempfile
from hrx_solutions.pipeline_validation import ModelFiles, validate_model
with tempfile.TemporaryDirectory() as directory:
    weights = Path(directory) / "model.gguf"
    tokenizer = Path(directory) / "tokenizer.json"
    weights.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 0, 0) + bytes(8))
    tokenizer.write_text('{"model": {}}')
    assert inspect_gguf(weights).version == 3
    for pipeline in ("qwen", "qwen-08b"):
        result = validate_model(ModelFiles(weights, tokenizer), pipeline)
        assert not result.compatible
        assert "Missing required metadata" in result.message
# --help does not load HSA. Check the installed runtime API without
# requiring a GPU or allowing host runtime overrides.
import os
import subprocess
import sys
from hrx_solutions._launcher import native_environment
clean_environment = dict(os.environ)
for name in ("IREE_HAL_AMDGPU_LIBHSA_PATH", "IREE_HAL_AMDGPU_LIBAQLPROFILE_PATH", "LD_LIBRARY_PATH"):
    clean_environment.pop(name, None)
subprocess.run(
    [sys.executable, "-c", "import ctypes, os; hsa = ctypes.CDLL(os.environ['IREE_HAL_AMDGPU_LIBHSA_PATH']); hsa.hsa_init; hsa.hsa_amd_queue_create"],
    env=native_environment(clean_environment),
    check=True,
)
