# Generic modality onboarding structural fixtures

Initial catalog: one CHAT executable and one recipe. Addition DATA: another recipe
on that executable, a second CHAT executable, and a nonpreparing ASR executable.
Names and catalog paths intentionally differ from product model names.

Source declarations reuse frozen accepted native source closures. Installed
fixtures reuse actual accepted compiled program bytes from `native-fixture.zip`,
verified by the literal hashes here. The fixture manifests represent normalized
artifact DATA, not a claim that the uniform producer has emitted them. Source and
installed manifests, exact source identities, role/member expectations and recipe
coordinates in the two release JSON files are authored before any producer runs.
The harness never generates its release oracle from staged/produced inventories.

Tiny prepared assets exercise bookkeeping/config/export only. They do not prove
model loading, native ABI behavior, throughput or inference. Real compiler/producer
and genuine artifact qualification remain separate until their actual APIs are
composed. No mocked reader/selector/checker or source family dispatch is used.

The two *-build.bzl files are literal fixture build DATA for the actual uniform
producer. Its separate harness case reuses these accepted compiled inputs and
executes the real Loom decoder built from the current `MODULE.bazel` HRX pin.
It calls real prepare/stage and the strict independent installed checker, without
filtering producer receipts or deriving release expectations from its output.
The archive contains only the five accepted bytecode members, their 74 unique
matching source files, and original licenses. `native-fixture.json` records its
SHA256, per-member identities, original source/HRX commits, and accepted wheel
identity. The bundle preserves historical inputs byte-for-byte; the existing
release, source and compiled-member expectations remain independent and unchanged.
Current model source changes must not refresh these historical expectations.

Run the required eight-case integration lane from the repository root with an
isolated source-test venv and explicit `LOOM_SERVE_DIR`:

```bash
"$python" -m build_tools.ci_tools.source_tests onboarding \
  --decoder "$decoder" --fixture-directory "$new_task_directory"
```

The shared action verifies and extracts the frozen bundle into a new caller-owned
directory and replaces all inherited `GENERIC_MODALITY_FIXTURE_*` variables.
No historical wheel, source checkout, network download or compiler double is
required. The test script fails if any of its eight tests fail or skip, or if the
expected test count changes. The caller owns cleanup of the extracted directory.

`.github/scripts/build_python_wheel.sh` always runs this lane after building the
wheel, in its own venv with the built wheel installed and the actual pinned
decoder. Failure stops the build before wheel repair, upload and cache admission.
Both GitHub CI and default full local CI invoke that script. The ordinary
`*_test.py` source discovery runs before compilation and does not include this
native integration lane. The development-only `build-wheel.sh --fast` command
builds an unrepaired host wheel and does not run these CI checks.
