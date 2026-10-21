# Build tools cleanup

## Outcome

Make `build_tools` easy to navigate for wheel maintainers and CI contributors.
A maintainer should be able to find a supported command, understand its inputs
and prerequisites, and locate its implementation and tests without choosing
between duplicate-looking paths. Preserve wheel, catalog, release, cache,
client integration, and process ownership behavior throughout the migration.

This specification authorizes planning only. Implementation, commits, landing,
and publishing require their own authorization. Milestones define deliverables
and dependencies; implementation status belongs in Beads when work is tracked.

Local Beads epic: `proj-ovm`. Its five implementation tasks are open and
unclaimed at baking. This continues the completed package grouping in
`proj-mfb`; the existing functional packages are retained.

## Current structure

The inventory at commit `ef2849249125aa40ee66c121c4b5052f641727c5` contains
89 tracked files, 46 top-level files, and 24 Python or JavaScript test files.
Seventeen top-level Python files forward to implementations already grouped
under `ci_tools`, `wheel_tools`, `model_tools`, and `release`.

| Area | Files | Current responsibility |
| --- | ---: | --- |
| `ci_tools/` | 12 | Local CI, source capture, private runtime directories, wheel and GitHub caches, tests |
| `wheel_tools/` | 11 | Wheel construction, Python setup, compiler inputs, wheel inspection, installed acceptance, tests |
| `model_tools/` | 7 | Catalog validation, package metadata, model compilation, tests |
| `release/` | 4 | Release preparation, publication, tests |
| Top-level compatibility wrappers | 17 | Legacy import and executable paths into those packages |
| Main shell commands | 4 | Build wheel, check wheel, local CI, model smoke |
| Other validation helpers | 5 | Model lifecycle, cancellation, SSE response acceptance |
| Client integration files | 27 | Pi, OpenCode, OMP, Open WebUI, grouped UI checks, staging and locked dependencies |
| Shared metadata | 2 | `BUILD.bazel`, `versioning.py` |

Consumers include `setup.py`, `.github/scripts/`, `.github/workflows/`,
`loom-pipeline/bytecode.bzl`, documentation, and tests. Bazel already uses the
catalog implementation directly through `//build_tools:loom_catalog`.

## Scope

Complete the existing functional grouping, document the command surface,
migrate internal callers to canonical implementations, and remove obsolete
forwarding paths only after validating their replacements. Group client
dependencies and tests beside their checks. Extract the substantial inline
Python supervisor from `model-smoke.sh` as a separate behavior-preserving step.

Keep `ci_tools`, `wheel_tools`, `model_tools`, and `release` named as they are.
Keep application source and application tests in their existing locations;
generic Loom runtime and model tests remain in HRX. Preserve the HRX pin,
model recipes, dependency versions and integrity checks, public wheel entry
points, release version rules, and release/cache trust policy.

Do not introduce a universal task runner, replace Bazel or the packaging
backend, merge distinct Node and Bun acquisition routes, download model
weights for the cleanup, or change qualification coverage incidentally.
Changing dependency pins or adopting additional client qualification is
separate work if the audit finds it necessary.

## Target organization

```text
build_tools/
  README.md
  BUILD.bazel
  versioning.py
  build-wheel.sh
  check-wheel.sh
  local-ci.sh
  model-smoke.sh
  ci_tools/
  wheel_tools/
  model_tools/
  release/
  validation/
    __init__.py
    check_installed_cpu.py
    check_installed_cpu_test.py
    check_installed_wheel.py
    check_installed_wheel_test.py
    hem_model_smoke.py
    hem_model_smoke_test.py
    model_smoke.py
    cancellation_test.py
    assert_response.py
    assert_response_test.py
  integrations/
    __init__.py
    check-ui-integrations.sh
    check_ui_integrations_test.py
    common.py
    pi/
    opencode/
    omp/
    openwebui/
```

Python integration subdirectories must have `__init__.py` where needed for
recursive unittest discovery and canonical imports. `common.py` holds the
existing `provider` and file `snapshot` helpers used by OpenCode and OMP;
avoid making one client import another client's qualification implementation.
Do not add a general integration framework around these helpers.

### File disposition

| Current files | Destination or action |
| --- | --- |
| `ci_tools/*`, `model_tools/*`, `release/*` | Retain their current responsibilities and names |
| `wheel_tools/check_installed_cpu{,_test}.py`, `wheel_tools/check_installed_wheel{,_test}.py` | Move to `validation/`; migrate payload verification and fixture callers too |
| Remaining `wheel_tools/*` | Retain wheel construction, compiler inputs, and archive validation |
| `hem_model_smoke{,_test}.py`, `cancellation_test.py` | Move to `validation/` |
| `smoke/assert_response.py`, `wheel_smoke_test.py` | Move to `validation/assert_response.py` and `validation/assert_response_test.py` |
| `check-ui-integrations.sh`, `check_ui_integrations_test.py` | Move to `integrations/` |
| `stage-pi-runtime.sh`, `check-pi-assets.sh`, all four `pi_*_test.mjs` files | Move to `integrations/pi/`, subject to the retirement audit |
| `pi_runtime/*` | Move to `integrations/pi/runtime/` |
| `stage-opencode-runtime.sh`, `check-opencode-assets.sh`, `opencode_configuration_smoke{,_test}.py` | Move to `integrations/opencode/`, subject to the retirement audit |
| `opencode_runtime/*` | Move to `integrations/opencode/runtime/` |
| `stage-omp-runtime.sh`, `check-omp-assets.sh`, `omp_configuration_smoke{,_test}.py` | Move to `integrations/omp/`, subject to the retirement audit |
| `omp_runtime/client/*`, `omp_runtime/runtime/*` | Move to `integrations/omp/runtime/client/` and `integrations/omp/runtime/runtime/`; retain the two existing acquisition namespaces |
| `openwebui_configuration_smoke{,_test}.py`, `openwebui_wire_capture.py` | Move to `integrations/openwebui/` |
| `BUILD.bazel`, `versioning.py`, four main shell commands | Retain at top level |

The 17 compatibility wrappers map to these canonical modules, with installed
acceptance using its final destination:

| Top-level wrapper stem | Canonical module prefix |
| --- | --- |
| `local_ci`, `local_ci_source`, `local_ci_runtime`, `wheel_cache`, `wheel_cache_summary`, `github_wheel_cache` | `build_tools.ci_tools` |
| `bazel_compiler_inputs`, `python_packaging`, `setup_python`, `validate_wheel` | `build_tools.wheel_tools` |
| `check_installed_cpu`, `check_installed_wheel` | `build_tools.validation` |
| `loom_catalog` | `build_tools.model_tools` |
| `make_release`, `publish_release` | `build_tools.release` |

## Command and execution contracts

The following commands retain their paths, arguments, exit status behavior,
environment configuration, output destinations, and cancellation semantics:

- `bash build_tools/build-wheel.sh`
- `bash build_tools/check-wheel.sh WHEEL NEW_VENV_DIRECTORY [--native-lifecycle]`
- `bash build_tools/local-ci.sh [--cpu-only] [--model08 DIRECTORY] [--run-dir DIRECTORY]`
- `bash build_tools/model-smoke.sh [--openwebui] WHEEL NEW_RUN_DIRECTORY [MODEL08_DIRECTORY]`

Preserve their currently supported working-directory behavior. In particular,
installed-wheel checks execute outside the checkout and import the installed
`hrx_solutions` package from the task venv. Moving build helpers must not make
the source application package shadow the installed package.

Internal imports use canonical package paths. Repository-root module commands
use `python -m`, for example `python -m build_tools.wheel_tools.setup_python
--rocm=/path/to/rocm` and `python -m build_tools.release.make_release ...`.
Update `setup.py`, container scripts, workflows, documentation, imports, mock
patch targets, and test fixture paths together. Preserve Bazel labels and
runfiles execution; Bazel entry points may retain their required import
fallbacks rather than forcing repository-root module execution.

A module command requires the repository root to be importable; it is not a
drop-in replacement for executing an absolute script path outside the checkout.
For that route, use a small explicit bootstrap in the owning shell command or
a private entry module that locates the repository's build helper package
without inherited `PYTHONPATH`. Preserve the outside-checkout working directory
and verify installed application import provenance. Temporary compatibility
wrappers may remain until this route is validated.

The README documents purpose, invocation directory, inputs, prerequisites,
network acquisition, output locations, and smallest relevant check for each
supported command. It distinguishes source tests, installed CPU checks,
source/installed client asset checks, GPU backend checks, Open WebUI checks,
and release operations. It states that default local CI runs the full gate;
CPU-only is explicit reduced coverage.

## Behavior that must survive the cleanup

- Model catalog contents, bytecode inputs, package selection, native library
  expectations, and wheel console scripts retain their existing contracts.
- Cache-key construction, provenance schema, admission rules, fork/branch
  trust policy, and release tag/version checks retain their contracts. Source
  identity changes may legitimately produce new cache entries.
- Backend smoke and installed UI checks remain separate stages. Do not wire
  dormant client `qualify()` functions into the backend gate during relocation.
- Client lockfiles, pinned runtimes, digests, disabled install scripts, timeout
  behavior, and source/installed asset selection remain intact.
- Cancellation retains ownership checks, bounded waits, signal-derived exit
  codes, cleanup evidence, and unresolved-resource reporting. Stopping a check
  never stops an unrelated user daemon or another validation instance.
- Evidence schemas and environment variables consumed across wrappers and CI
  remain compatible; source paths in provenance change only as required by moves.

`MANIFEST.in` currently includes Python, shell, and Bazel build-tool files, but
does not include the integration JavaScript or lockfiles through that rule.
Do not silently broaden or narrow sdist scope. Establish the existing supported
sdist build contract, preserve required inputs at their new locations, and
report any pre-existing inability to run integration checks from an sdist as
a separate finding.

## Milestones

### M1 Document command ownership and migration coverage (`proj-ovm.1`)

**Deliverable:** a `build_tools/README.md` and a reviewed mapping of every
tracked build-tool file to its retained path, destination, or audit category.
Record actual command consumers and the existing source/installed gate split.

**Acceptance:** all 89 baseline files are accounted for; each supported command
has prerequisites, working-directory requirements, inputs, outputs, and owning
implementation. Identify compatibility paths referenced by external-facing
documentation. Record the collected Python test identities and JavaScript gate
membership so subsequent moves cannot silently reduce coverage.

**Validation and stopping state:** static caller and manifest inspection is
sufficient. No behavior changes or deletions; a maintainer can select the
correct command from the README. Dependency: none.

### M2 Group validation and client integrations (`proj-ovm.2`)

**Deliverable:** the target `validation/` and `integrations/` organization,
colocated tests and locked dependencies, and shared OpenCode/OMP helper module.
Update imports, roots derived from `__file__`, subprocess paths, copied scripts,
fixture builders, mock targets, CI invocations, and required package markers.
Retain temporary wrappers where callers have not migrated.

**Acceptance:** source and installed client checks select the same assets as
before; the full Python discovery command still collects all baseline test
identities modulo intentional module renames; every previously gated JavaScript
test remains gated. Backend/UI stage separation and Bazel execution survive.

**Validation and stopping state:** run affected source tests, existing source
and installed UI checks, installed CPU acceptance, and relevant catalog/Bazel
checks. Record wheel/sdist input and payload checks. Layout is usable with
compatibility routes retained; no coverage is intentionally removed.
Dependency: M1.

### M3 Migrate callers to canonical imports and module commands (`proj-ovm.3`)

**Deliverable:** all in-repository consumers use the new canonical paths;
external-facing documentation shows the supported replacements for wrapper
commands. Resolve execution outside the checkout explicitly.

**Acceptance:** packaging imports work in the actual isolated build environment;
container build, release validation, source tests, installed checks, and local
CI invoke canonical implementations. Search finds no executable dependency on
retirement candidates except documented compatibility checks. Bazel labels and
direct/runfiles invocation remain valid.

**Validation and stopping state:** run affected packaging, release, CI/cache,
and model tests; build and inspect the wheel and sdist; run installed-wheel
checks outside the checkout with import provenance verified. Retain wrappers
until the final retirement milestone. Dependency: M2.

### M4 Extract the model smoke supervisor (`proj-ovm.4`)

**Deliverable:** move inline Python supervision from `model-smoke.sh` into
`validation/model_smoke.py`, leaving a small shell launcher. Keep this change
separate from file relocation and wrapper deletion for reviewability.

**Acceptance:** arguments, setup sequencing, model acquisition choice, Open
WebUI route, evidence, return codes, repeated-signal handling, and owned cleanup
match existing behavior. Existing cancellation scenarios still prove an
unrelated sentinel process remains alive and unresolved cleanup is reported.

**Validation and stopping state:** run focused cancellation/supervision tests
and bounded real-model readiness, known query, and shutdown checks. Run the
Open WebUI route because it shares this supervisor. Where isolation is affected,
run two instances concurrently, prove separate inventories, and prove stopping
one leaves the other usable. Hold the existing shared GPU lock throughout
focused real-model checks and their cleanup, as detailed below. Dependency: M3.

### M5 Retire confirmed obsolete paths and complete integration (`proj-ovm.5`)

**Deliverable:** remove eligible compatibility wrappers and obsolete checks
after documenting their disposition and validating replacements.

Audit `pi_provider_test.mjs`, which has no checked-in caller found in the
baseline inspection, and OpenCode/OMP `qualify()` paths, which have no production
caller found while their helper behavior remains unit-tested. Classify each as
an active gate, an intentionally supported manual qualification, or obsolete.
Preserve active contracts and tests. Remove only code whose absence does not
remove an intended contract; extract shared helpers before deleting owners.
Adding new qualification coverage is a separately scoped behavior change.

Wrapper deletion requires migrated in-repository callers, a documented
replacement for documented commands, passing replacement execution checks,
and an explicit compatibility disposition. Retain and document a wrapper when
its external compatibility obligation cannot be resolved; an arbitrary zero
wrapper count is not an acceptance criterion.

**Acceptance:** top-level files are the four stable commands, README, Bazel
metadata, shared version rules, and any explicitly justified compatibility
paths. Every baseline file and test has an accounted-for final disposition.
No stale supported invocation remains in docs, workflows, or packaging.

**Validation and stopping state:** run the full default local CI gate against
the final intended source tree with `--model08` selecting the immutable local
fixture, and require exit 0 before creating or landing a commit on `main`.
Record source identity and external evidence. Missing GPU,
model, Docker, or client dependencies block that gate; do not silently substitute
CPU-only coverage. The final tree is ready for separately authorized landing.
Dependency: M4. Irreversible deletion occurs only after replacement validation.

## Validation environment and evidence

Implementation uses a task branch and worktree under the external project
workspace. Read its `AGENTS.md`, preserve unrelated user changes, and keep logs,
reports, generated runtimes, venvs, state, and weights outside the repository.

Use standalone task-owned venvs created with `python3 -m venv`, system site
packages disabled, inherited `PYTHONPATH` and `PYTHONHOME` excluded, and user
site packages disabled. Use explicit venv interpreter and CLI paths; verify
interpreter and import provenance before accepting results.

Run source tests from the intended checkout, including the existing command
`python -m unittest discover -s build_tools -p '*_test.py' -v` with the task venv
interpreter. Add package markers or fix discovery when moving tests; do not
replace failing collection with a narrower invocation. Run application tests
through the existing source gate when callers or fixtures change.

Install the exact wheel in the instance's venv and run installed checks outside
the checkout. Record the source commit plus dirty-tree identity where relevant,
wheel filename and digest, commands, results, discovered test changes, and
cleanup evidence under the workspace's `gates/` directory. Restart only the
instance's own persistent service after package changes.

Set an absolute per-instance `LOOM_SERVE_DIR` before the first Loom command and
preserve it for child processes; isolate sockets, caches, ports, daemon records,
and runtime state. Preserve normal HOME/XDG, credentials, and shared model
inputs. Use the repository's default Qwen3.5 0.8B Q5_K_XL GGUF and adjacent
tokenizer as immutable external fixtures, imported with `models import --alias`
and exercised through `--pipeline qwen-08b`. Supply their explicit external
directory to smoke checks. Missing fixtures block the real-model gate rather
than triggering a substitute or download for validation.

Direct real-model and Open WebUI checks require the same shared exclusive
`gates/qwen08-gpu.lock` in the external project workspace that local CI uses.
Hold it through the entire owned check and cleanup. Acquire it once around an
intentional two-instance experiment; run both owned instances inside that
protected experiment. Per-instance state and ports do not replace this host
GPU coordination. Do not wrap local CI in another acquisition of its own lock.

Final landing gate: `bash build_tools/local-ci.sh --model08 "$MODEL08_DIRECTORY"`
in its default full GPU/Pi mode, with `MODEL08_DIRECTORY` resolved to the
verified immutable fixture. This option selects input data and retains full
coverage; the launcher's existing no-argument acquisition behavior is unchanged.
Source changes after a passing run require a rerun. CPU-only qualifies for
landing only with explicit user approval of reduced coverage. Retain evidence;
confirm owned processes exited before removing task venv/state.

## Review and implementation decisions

The reviewer must check that the mapping covers the baseline inventory,
milestones preserve executable/import/discovery behavior, and deletion and
landing gates are sufficient. Review must distinguish observed missing callers
from proof of obsolescence.

Implementation checks remain for the exact outside-checkout bootstrap,
sdist-required build inputs, Bazel import fallbacks, compatibility obligations
for documented wrapper paths, and dormant qualification disposition. Resolve
these from code and validation; ask the user only if a compatibility or coverage
decision would change the agreed outcome.
