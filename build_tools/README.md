# Build and validation commands

These tools build the `hrx-solutions` wheel from the HRX revision pinned in
`MODULE.bazel`, check its payload and client assets, and prepare releases.
Application code and source tests live in `binding/python/` and `tests/`;
generic Loom runtime tests remain in HRX. Model weights are external inputs.

The commands below describe the current layout. The cleanup specification is
[`docs/specs/build-tools-cleanup.md`](../docs/specs/build-tools-cleanup.md).

## Choose a gate

| Need | Command | Coverage and owning implementation |
| --- | --- | --- |
| Full local CI | `bash build_tools/local-ci.sh [--model08 DIRECTORY] [--run-dir DIRECTORY]` | Source capture, source client assets, Python tests, wheel build, installed CPU and client checks, GPU backend, Open WebUI; `ci_tools/local_ci.py` |
| Explicit reduced coverage | `bash build_tools/local-ci.sh --cpu-only [--run-dir DIRECTORY]` | Same stages through installed UI checks; skips GPU and Open WebUI, still requires Docker and network |
| Build a repaired wheel | `bash build_tools/build-wheel.sh` | Pinned manylinux container; `.github/scripts/build_python_wheel.sh` and `wheel_tools/` |
| Check an existing wheel | `bash build_tools/check-wheel.sh WHEEL NEW_VENV_DIRECTORY [--native-lifecycle]` | Installed payload/import/console-script checks and CPU lifecycle fixtures; `validation/check_installed_{wheel,cpu}.py`; optional empty AMDGPU server lifecycle |
| Real-model backend | `bash build_tools/model-smoke.sh WHEEL NEW_RUN_DIRECTORY MODEL08_DIRECTORY` | Isolated Qwen 0.8B backend/configuration/lifecycle qualification; `validation/hem_model_smoke.py`, profile `ci08` |
| Real-model Open WebUI | `bash build_tools/model-smoke.sh --openwebui WHEEL NEW_RUN_DIRECTORY MODEL08_DIRECTORY` | Public WebUI wiring against isolated native models; `integrations/openwebui/openwebui_configuration_smoke.py` |

Default local CI runs the full GPU and client gate. CPU-only is explicit reduced
coverage and does not satisfy the repository's pre-landing gate without user
approval. `--model08` is incompatible with `--cpu-only`.

### Invocation, prerequisites, and outputs

All four top-level shell commands locate this checkout from their own path and
can be invoked by absolute path from another directory. Relative wheel, model,
and output arguments resolve from the caller's directory. Examples here assume
the repository root. Use external absolute directories for validation state.

- **Local CI:** Linux x86-64; Bash, Git, Python 3 with venv/pip, Docker daemon,
  curl, tar, timeout, realpath, and sha256sum. Full mode also needs accessible
  `/dev/kfd` and a render node, the supported AMD backend, and model inputs.
  It captures source inputs into a private snapshot and runs checks there.
  `--run-dir` must be new and outside the checkout. By default, evidence is in
  `$CODEX_PROJECT_WORKSPACES/hrx-solutions/gates/local-ci/`, or
  `~/codex/project-workspaces/hrx-solutions/gates/local-ci/` when unset, under
  a unique run directory. Outputs include `summary.json`, stage logs, source
  manifests, the snapshot's wheel, and installed/real-model evidence; the
  private runtime directory is separately reported and retained.
- **Build wheel:** Linux x86-64 with Docker, Python 3, and network access.
  The container acquires pinned Bazelisk, HRX/build inputs, TheRock toolchain,
  Python packages, and build dependencies. Outputs are under `build/wheel/`,
  with the repaired wheel in `build/wheel/wheelhouse/`. The checkout is mounted
  writable. Use a task worktree or local CI's external source snapshot.
  `HRX_BAZEL_CACHE_DIR` enables an optional external cache unless
  `HRX_BAZEL_CACHE_ENABLED=0`; see the root README for the cache policy.
- **Check wheel:** an existing `.whl`, Python 3 with venv/pip, and a destination
  that does not exist. Installs the exact wheel with `--no-index --no-deps`,
  executes outside the checkout, verifies installed import provenance, and
  writes `wheel-check/installed-cpu-evidence.json` inside the new venv. It
  isolates Loom state below the venv, or the private runtime root supplied by
  local CI. CPU fixtures use an isolated state directory. Native lifecycle
  requires an available AMD GPU; it does not run real-model inference.
- **Model smoke:** an existing wheel, Python 3 with venv/pip, network access for
  installation dependencies, and a new or empty external run directory.
  The backend qualification targets gfx1151. Outputs include `install.log`
  and `lifecycle/` evidence; all owned state stays within the instance's
  namespace. Always pass the immutable local fixture for development:
  `Qwen3.5-0.8B-UD-Q5_K_XL.gguf` and adjacent `tokenizer.json` in the supplied
  directory. The default fixture location is
  `$HOME/Downloads/gguf/models/multimodal/qwen/qwen3.5-0.8b`.
  Omitting `MODEL08_DIRECTORY` on the backend route requests acquisition of
  the pinned recipe; do not use that route for cleanup validation. Open WebUI
  requires an explicit model directory and Docker; it inspects or pulls the
  pinned image and records `webui-image.log` plus lifecycle evidence.

Local CI acquires the shared external `gates/qwen08-gpu.lock` around its GPU
and WebUI stages. When invoking real-model checks directly, acquire that same
workspace lock once around the check and its cleanup. Separate state and ports
do not replace GPU coordination. Never nest an extra lock around local CI.
The model supervisor and helpers preserve signal status and report unresolved
owned cleanup; retain that evidence and do not stop unrelated services.

## Source tests and client assets

Run Python source validation from the repository root in a standalone external
venv. Exclude inherited `PYTHONPATH`/`PYTHONHOME`, disable user site packages,
verify interpreter/dependency provenance, and set an absolute instance-specific
`LOOM_SERVE_DIR` before any Loom command. `ci_tools/source_tests.py` owns the shared source-gate setup and discovery
contract used by hosted CI and local CI. Its install action supplies
`setuptools>=77`, `wheel>=0.45`, `packaging>=24.2`, and `textual>=0.47,<2`.
Use the venv's explicit interpreter:

```sh
"$VENV/bin/python" -m build_tools.ci_tools.source_tests install
"$VENV/bin/python" -m build_tools.ci_tools.source_tests verify
"$VENV/bin/python" -m build_tools.ci_tools.source_tests build_tools
"$VENV/bin/python" -m build_tools.ci_tools.source_tests tests
```

These are source tests, including fixtures and mocked ownership checks, rather
than installed-wheel acceptance. Some application tests invoke real client
runtimes; stage them and export the same environment as local CI's `ui_source`
stage to avoid skipping their client coverage.

Client staging uses new external destinations. Pi and OpenCode staging require
curl/tar, checksum tools, timeout, and network to fetch their pinned Node
archives and locked npm packages. OMP staging requires an existing npm and
network, then uses its separately locked Bun runtime and Bun client lock.
Dependency installation scripts are disabled; preserve lockfiles and digests.

| Client | Acquire runtime | Smallest asset check (no model/GPU) |
| --- | --- | --- |
| Pi | `bash build_tools/integrations/pi/stage-pi-runtime.sh NEW_RUNTIME` | `bash build_tools/integrations/pi/check-pi-assets.sh RUNTIME ASSET_DIRECTORY` |
| OpenCode | `bash build_tools/integrations/opencode/stage-opencode-runtime.sh NEW_RUNTIME` | `bash build_tools/integrations/opencode/check-opencode-assets.sh RUNTIME ASSET_DIRECTORY` |
| OMP | `bash build_tools/integrations/omp/stage-omp-runtime.sh NEW_RUNTIME` | `bash build_tools/integrations/omp/check-omp-assets.sh RUNTIME ASSET_DIRECTORY` |

Staging writes only the selected runtime directory (plus normal package-manager
caches). Pi uses Node 22.22.1/Pi 1.3.0, OpenCode uses Node 24.10.0/OpenCode
2.0.31, and OMP uses Bun 1.3.14/OMP 18.11.2. Staged directories must not already
exist. Asset checks accept any working directory via absolute script paths;
outputs are console results plus temporary/runtime scratch fixtures.

For source assets pass `binding/python/pi_extension`,
`binding/python/opencode_plugin`, or `binding/python/omp_extension`.
For installed assets run:

```sh
bash build_tools/integrations/check-ui-integrations.sh INSTALLED_VENV PI_RUNTIME OPENCODE_RUNTIME OMP_RUNTIME
```

This verifies the package is imported from the venv and selects its copied
assets. It writes console results and uses the staged runtimes' scratch space.
The invoking environment must exclude Python path overrides/user site packages.
The same client checks run against source and installed assets as separate CI
stages. They use synthetic protocol fixtures; backend real-model acceptance
and WebUI qualification remain separate.

Pi's gate runs `pi_extension_test.mjs`, `pi_discovery_test.mjs`, and
`pi_budgets_test.mjs`. OpenCode runs `tests/opencode_adapter_test.mjs`; OMP
runs `tests/omp_adapter_test.mjs` and `tests/omp_loader_test.mjs`. The top-level
`pi_provider_test.mjs` has no checked-in gate caller found. OpenCode/OMP
configuration modules contain `qualify()` functions with no production caller
found, while their helpers are unit-tested. These are retirement-audit items;
this guide does not add or remove qualification coverage.

## Helper ownership

Run Python module commands below from the repository root with the task venv
interpreter. They require the source helper package to be importable. Absolute
script paths used by installed checks have dedicated bootstraps; do not replace
those with module commands from an outside-checkout working directory without
preserving installed application import provenance.

| Responsibility | Owning module / invocation | Inputs, prerequisites, and outputs |
| --- | --- | --- |
| Local native build setup | `python -m build_tools.wheel_tools.setup_python --rocm=/path/to/rocm [--bazel EXECUTABLE]` | Existing ROCm/TheRock Clang and pinned Bazel/Bazelisk; writes or validates `.bazelrc.configured`; `--force` replaces incompatible settings |
| Archive acceptance | `python -m build_tools.wheel_tools.validate_wheel WHEEL [--sdist SDIST]` | Existing archives; verifies payload/entrypoints/native constraints and optional reviewed sdist assets; console success/error, no GPU |
| Catalog validation | `python -m build_tools.model_tools.loom_catalog validate --inventory INVENTORY --stamp STAMP` | Catalog inventory/recipes; writes validation stamp; smallest source check is `model_tools/loom_catalog_test.py` |
| Catalog preparation | `python -m build_tools.model_tools.loom_catalog prepare --source SOURCE --output OUTPUT` | Existing bytecode source; writes prepared bytecode source; no weight acquisition |
| Model compilation | `//build_tools:model_package_build` | Bazel action supplies catalog, integrity, module, compiler/linker, declared inputs and output paths; owns native package/metadata/role outputs; `model_tools/model_package_build.py` |
| Catalog Bazel action | `//build_tools:loom_catalog` | Used by `loom-pipeline/bytecode.bzl`; retains runfiles/direct-import handling |
| Cache lifecycle | `ci_tools/wheel_cache.py`, `wheel_cache_summary.py` | Build/container/CI-owned cache directory, fingerprints and diagnostics; writes cache provenance/configuration or summaries; owning unit tests are the smallest check |
| GitHub cache maintenance | `ci_tools/github_wheel_cache.py` | Workflow-owned authenticated GitHub cache metadata and attempted key; may delete eligible remote caches; policy and client unit tests are the smallest check |
| Shared source tests | `python -m build_tools.ci_tools.source_tests {install,verify,build_tools,tests}` | Caller owns venv/environment/runtime fixtures; installs source requirements, verifies provenance, or runs unchanged full unittest discovery; console results |
| Shared version rules | `versioning.py` | Reads `version.json`; release/tag validation; no independent CLI |

Top-level Python forwarding files are compatibility routes into `ci_tools/`,
`wheel_tools/`, `model_tools/`, and `release/`. They remain in use by packaging,
container scripts, workflows, tests, and published documentation. In particular,
the root README documents `python build_tools/setup_python.py`, and
`RELEASING.md` documents `python build_tools/make_release.py`. They cannot be
retired solely because the canonical modules exist.

`MANIFEST.in` includes build-tool Python, shell, and Bazel files. That rule does
not include integration JavaScript, JSON manifests, or lockfiles. The release
contract publishes a wheel, not an sdist. Integration execution from an sdist
is not established by that manifest; preserve existing scope and audit it
separately when moving inputs.

## Releases

Follow [`RELEASING.md`](../RELEASING.md) for authorization and the complete
workflow. From a clean `main` checkout, the standard-library-only helper
`python -m build_tools.release.make_release --version VERSION --bump-dev --dry-run`
checks version/tag/branch/Git identity without changing files. Without
`--dry-run` it creates version commits and a local tag; it never pushes.
The smallest check is `release/release_test.py` in full build-tool discovery.

`release/publish_release.py` owns workflow operations `validate-tag`,
`validate-wheel`, `github`, and `pypi-status`, all taking `--tag TAG` (and
`--wheel-dir`/`--repository` where applicable). It needs `packaging`; GitHub
publication needs authenticated `gh`, and remote publication/status requires
network. Validation prints results; `github` manages release assets/checksums,
and PyPI status inspects the exact release wheel's availability. The release
workflow owns actual PyPI upload. Release preparation/publication are explicit
operations, not validation shortcuts.
