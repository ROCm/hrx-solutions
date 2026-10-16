# Manage Loom Serve with Loom Control

For direct foreground setup and manual Pi configuration, follow the
[primary Loom Serve walkthrough](llm-serving.md). This companion covers Loom Control
model management and server/instance lifecycle commands.

Loom Control acquires and validates model files, starts empty Loom Serve processes,
and loads models through the native HTTP interface. Its commands use
`loom-control <type> <command>`. A **model** is a local asset set, a **pipeline** is
its packaged implementation, a **server** is an owned native process with saved
launch settings, and an **instance** is a model registration on a server.

| Type | Commands |
| --- | --- |
| `model` | `available`, `download`, `import`, `validate`, `list`, `show` |
| `pipeline` | `list`, `show` |
| `server` | `start`, `list`, `show`, `status`, `logs`, `reconcile`, `shutdown` |
| `instance` | `start`, `stop`, `list`, `show`, `status`, `logs`, `export` |
| `pi` | `configure` (including `--check` and `--disable`) |
| `opencode` | `configure` (including `--check` and `--disable`) |
| `omp` | `configure` (including `--check` and `--disable`) |

Run `loom-control` for a getting-started sequence. Bare resource commands such as
`loom-control model` list their actions and examples. These show the same guidance
as `--help`, exit successfully and do not initialize state or contact a server.
Use `loom-control TYPE COMMAND --help` for arguments, defaults and configuration
constraints. Incomplete actions still report missing inputs and exit 2.
The native
`loom-serve` command remains available for direct foreground execution.
For implementation and extension points, see the
[LLM pipeline guidebook](../../loom-pipeline/llm/README.md).

The management command is now `loom-control`; update scripts that invoke
`loom-hem`. No compatibility alias is installed. Existing state and configuration
remain unchanged.

## Current example

The wheel includes Qwen pipelines and their compatibility contracts, the native
server and its user-space ROCm runtime libraries. Weights remain external.

| Area | Coverage |
| --- | --- |
| Models | Qwen3.5-0.8B UD-Q5_K_XL (`qwen-08b`) and Qwen3.8-27B UD-Q5_K_XL (`qwen`), each with its matching tokenizer. |
| Platform | Linux x86-64, Python 3.9 or newer, a supported AMD GPU and compatible driver. The source pipelines have been exercised on gfx1100 and gfx1151; other devices need validation. |
| Chat | Streaming chat, greedy generation (`stream: true`, `temperature: 0`); the Qwen policy rejects thinking mode. |
| Speculation | MTP defaults to depth three on 27B; 0.8B does not support MTP. |
| Validation | CPU inspection of GGUF metadata, tensor shapes/types and tokenizer configuration. This is structural qualification, without inference or full checkpoint hashing. |

Install the [wheel](../../README.md#install-from-pypi) in a virtual environment.
The inference machine needs GPU-accessible memory for weights, retained state
and execution workspace, plus disk space for the checkpoint. Serving requires
no separate ROCm SDK installation. Generic numerical, HTTP and tool execution
checks belong to the [upstream pipeline checks](../../loom-pipeline/llm/README.md#developing-and-checking-a-change).

## Acquire a model

First inspect the packaged choices:

```bash
loom-control model available
loom-control pipeline list
loom-control pipeline show qwen-08b
```

A recipe pins GGUF and tokenizer download coordinates. Install the external `hf`
CLI if you want downloads (`python -m pip install huggingface_hub`; Python 3.9
users can use `'huggingface_hub[cli]>=0.34,<1'`). Preserve your usual HF login,
cache and credential environment. Loom Control prints resolved `hf download` commands.

```bash
loom-control model download qwen35-08b-q5-k-xl --id qwen08 --dry-run
loom-control model download qwen35-08b-q5-k-xl --id qwen08
```

`--dry-run` previews without acquiring files or publishing model records.
`--offline` requires all recipe files already cached. Downloaded files are
referenced from the HF cache: keep them available and unchanged.

Alternatively import your own checkpoint and its matching tokenizer:

```bash
loom-control model import qwen08 \
  --gguf /srv/models/qwen08/Qwen3.5-0.8B-UD-Q5_K_XL.gguf \
  --tokenizer /srv/models/qwen08/tokenizer.json \
  --alias --pipeline qwen-08b
```

Choose either download or import for a given ID. `--alias` references external
files; omit it to copy files into owned storage. Multiple fragments can follow
`--gguf`. Each model has its own `models/MODEL_ID/` directory, keeping files
such as `tokenizer.json` separate. Inspection shows both scoped and external
locations. Never move, modify or evict an aliased file while it may be in use.

Import and download validate against packaged pipelines before marking a model
ready. A unique compatible pipeline is selected; `--pipeline` constrains that
selection for import and validation. Download defaults to the recipe's declared
pipeline; `--pipeline` overrides that selection. An unknown or ambiguous match
blocks serving. Names alone do not
establish compatibility. Completed assets and an inspectable invalid/incomplete
record survive failure. Repeating the same acquisition can retry; conflicting
assets under an existing ID are refused. Use another ID for a different asset
set. There are no replacement, deletion or shared-cache pruning commands.

```bash
loom-control model list
loom-control model show qwen08 --json
loom-control model validate qwen08 --pipeline qwen-08b
```

Inventory reports accessibility and validation state, including broken aliases;
it does not imply the model is loaded. `show` includes paths, ownership,
provenance and current referencing instances. Every start and export repeats
structural validation rather than trusting a saved result.

## Start a server, then load an instance

Create a named server definition with an explicit device and loopback listener:

```bash
loom-control server start --name local --device amdgpu --listen 127.0.0.1:8080
loom-control server show local
loom-control server status local --json
loom-control instance start qwen08 --name chat-small --server local
```

The server starts empty. Instance start validates, registers with admission
closed, activates parameters, waits for ready/idle/active status without an
error, then opens admission. Success means the instance can accept requests;
there is no start `--no-wait`. A registration name defaults to its model ID.
Use different instance names to serve the same assets with different settings.

```bash
curl --fail-with-body -sS http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"chat-small","messages":[{"role":"user","content":"Say Hello!"}],"stream":true,"temperature":0,"max_tokens":32}'
```

`--name local` refuses an existing definition. After shutdown,
`loom-control server start local` reuses its saved settings and starts empty again;
it does not reload old instances. A verified healthy live server is reused.
A live unhealthy server or uncertain ownership blocks duplicate launch. To
change launch settings, create another server name and choose an unused port.

Instance commands accept `--server NAME` or mutually exclusive `--url URL`.
With neither, exactly one configured server must exist. A URL does not grant
ownership of its process or transfer model files. Registration requires the
local matching build and asset context; remote provisioning is not provided.
Keep lifecycle writers coordinated: avoid concurrent direct curl changes while
Loom Control is changing registrations.

## Configure an instance

Flags override the JSON configuration file, which overrides pipeline defaults.
Server device/listener settings are separate from per-instance capacities.

| Flag / JSON key | Meaning | Default | Range |
| --- | --- | --- | --- |
| `--rows` / `rows` | Maximum simultaneous execution rows; requests can queue beyond this count. | 4 | 1–16 |
| `--context-capacity` / `context_capacity` | Logical context token ceiling per row, not eagerly allocated storage. | 8192 | 1–262144 |
| `--pool-capacity` / `pool_capacity` | Shared KV token positions across rows, divisible by the pipeline's page size; reservations round to pages. | 32768 | 1–4194304 |
| `--prefill-capacity` / `prefill_capacity` | Maximum packed input tokens per model traversal across rows; also limits isolated prefill chunks. | 128 | 1–262144 |
| `--mtp` / `--no-mtp` / `mtp` | Multi-token prediction proposes tokens for verification. | Enabled on 27B; disabled on 0.8B | Boolean; requires pipeline support |
| `--mtp-depth` / `mtp_depth` | Proposal tokens after each anchor. | 3 on 27B; not applicable on 0.8B | 1–3 when enabled/supported |

Capacity limits and capabilities appear in `pipeline show`. Capacity settings
are not a memory-fit guarantee. Prefill capacity must not exceed context
capacity and must fit one anchor plus the MTP depth (at least one token without
MTP). Unknown or duplicate JSON keys, null, wrong types, nonfinite numbers,
unsupported MTP and contradictory resolved settings fail with diagnostics.
For example, explicitly setting depth while MTP resolves disabled is ambiguous
and rejected, even when depth came from the file and `--no-mtp` from flags.
Remove the explicit depth or enable supported MTP. A default depth is simply
suppressed when MTP is disabled.

For an optional 27B profile, create a flat JSON file:

```bash
cat > qwen27.json <<'JSON'
{"rows":4,"context_capacity":8192,"pool_capacity":32768,"prefill_capacity":128,"mtp":true,"mtp_depth":3}
JSON
loom-control model download qwen38-27b-q5-k-xl --id qwen27
loom-control instance start qwen27 --name chat-large --server local \
  --config qwen27.json --context-capacity 16384
```

Here only context capacity changes to 16384. Stop the smaller instance first if
needed; Loom Control reports native resource accounting but cannot promise both fit.
To disable MTP in this profile, remove `mtp_depth` before passing `--no-mtp`.

## Inspect and stop

```bash
loom-control server list --json
loom-control server show local
loom-control server status local
loom-control server logs local
loom-control instance list --server local --json
loom-control instance show chat-small --server local
loom-control instance status chat-small --server local
loom-control instance logs chat-small --server local --follow
```

Server `show` displays saved launch settings. Server `status` and `list` report
observed process/HTTP health, native resources and current launch evidence.
Instance inspection combines native observations with the exact submitted
settings, document and dependencies when a matching tracked submission exists.
Native state is authoritative; an external registration has no invented local
model association. Follow streams write diagnostics to stderr; Ctrl-C ends
following without stopping the server or instance. Server logs also support
`--follow`.

```bash
loom-control instance stop chat-small --server local
loom-control server shutdown local
```

Stop gracefully unregisters and waits for that incarnation to retire, releasing
its current dependency references. `instance stop ... --no-wait` returns once
the native request is accepted; inspect status for retirement. `--abort`
explicitly requests cancellation instead of graceful drain. Server shutdown
drains/unregisters all instances, waits for retirement and only then terminates
the verified owned process. A graceful timeout leaves the server alive with a
pending marker, possibly with some instances already retired. Inspect and
repeat shutdown to continue, or explicitly escalate:

```bash
loom-control server shutdown local --abort
```

Lifecycle commands default to one 10-second budget, including coordination,
validation and completion. `--timeout SECONDS` overrides it for server start,
instance start/stop and server shutdown. Downloads and log following have no
such lifecycle budget. Timeouts preserve evidence; they do not imply rollback.

| Exit | Meaning |
| --- | --- |
| 0 | Completed, or explicitly requested asynchronous stop was accepted. |
| 1 | Known operational failure or unhealthy observation. |
| 2 | Invalid arguments/configuration. |
| 3 | Known accepted action remains pending when the budget expires. |
| 4 | Outcome or ownership is uncertain; inspect before retrying. |

Results use readable text by default: inventories are tables or labeled records,
inspection groups relevant facts, and actions report concise outcomes. Missing
observations remain unknown; structural model validation never claims inference
readiness. Log snapshots are direct text; following writes each observed chunk
to stderr without repeating a final snapshot on human stdout.

**Migration for scripts:** default stdout previously contained indented JSON.
Pass `--json` explicitly to retain the existing result structures and values.
It works before the resource, between resource/action, or after the action:

```bash
loom-control --json model list
loom-control model --json list
loom-control model list --json
```

`--json` produces one JSON action result on stdout. Help always remains text,
including a bare invocation with `--json`. Progress, equivalents and errors
use stderr. Failure JSON includes `accepted`: `true`, `false` or `null` when
acceptance has not been established. Acceptance describes the reported action,
not completion of the entire lifecycle. Uncertainty takes precedence over a
pending timeout. Never blindly replay an uncertain registration or undrain.

## State and recovery

By default Loom Control uses `~/.loom/control.sqlite3`. Set an absolute `LOOM_SERVE_DIR`
before the first command to put the database, models, locks, retained runtime
artifacts and logs in another state namespace. Use the same environment for
all commands in a lifecycle. This does not change HOME/XDG, HF credentials/cache
or other applications' configuration.

SQLite holds only current models, servers and instances with minimal pending
markers, not an operation history or command log. Transactions and durable
staged/atomic file publication protect updates. Inventory is authoritative;
a directory or symlink alone is not a registered model. Keep this state on a
local filesystem supporting SQLite and file locking; do not manually edit its
rows or remove files while instances may still depend on them.

Process identity includes boot/start identity and retained build information,
not just a PID. Loom Control verifies owned processes and their listener association
before lifecycle changes; stale PIDs or an endpoint answering HTTP do not grant
ownership. A new matching process is never silently adopted. Multiple saved
servers can run on distinct loopback ports. There is no background management
daemon or automatic restart/reload.

```bash
loom-control server reconcile local --json
loom-control server status local --json
loom-control instance list --server local --json
```

Reconciliation updates local evidence after observation. It never relaunches,
replays registration, activates/undrains, signals processes or deletes files.
After a crash, an uncertain observation retains dependencies and blocks unsafe
duplicate work. Resolve ownership/native status before retrying. Package/build
changes may require graceful shutdown and a new empty server; a mismatching
live serving build refuses new managed registration.

## Native equivalents and export

Loom Control prints resolved native, HF and curl equivalents to stderr before operations,
including inspectable request-body paths and resolved endpoints. Local validation
and bookkeeping are identified separately. These displays are explanations,
not permission prompts or a dry run. A verified process-handle signal cannot be
faithfully represented by an unchecked `kill PID` command.

Equivalents are always visible, including with `--json` and during repeated
status polling. Commands use shell quoting and backslash continuations at
argument boundaries so paths with spaces remain copyable. Credentials are
redacted and must be supplied again for replay; printing an equivalent does
not establish that the operation succeeded. Export prints a registration example
with `SERVER:PORT`, which must be replaced with a compatible server endpoint.

For a deliberate curl workflow, export a drained registration document:

```bash
loom-control instance export qwen08 --name chat-small --output chat-small.json
curl --fail-with-body -sS http://127.0.0.1:8080/v1/models/register \
  -H 'Content-Type: application/json' --data-binary @chat-small.json
curl -fsS http://127.0.0.1:8080/v1/status
# Inspect preparation before activating; wait for ready/idle/active/no error afterward.
curl --fail-with-body -sS http://127.0.0.1:8080/v1/models/activate \
  -H 'Content-Type: application/json' -d '{"model":"chat-small"}'
curl -fsS http://127.0.0.1:8080/v1/status
# Only after the readiness conditions hold:
curl --fail-with-body -sS http://127.0.0.1:8080/v1/models/undrain \
  -H 'Content-Type: application/json' -d '{"model":"chat-small"}'
curl --fail-with-body -sS http://127.0.0.1:8080/v1/models/unregister \
  -H 'Content-Type: application/json' -d '{"model":"chat-small","mode":"drain"}'
```

Use a fresh instance name if the earlier example is still registered. These
are separate steps with status inspection between them, not an unattended
startup script. Export accepts a model ID, start's configuration flags and
`--name`; it needs no server and performs validation without registration or
invoking a native linker/compiler. Installed-wheel exports reference the packaged
`pipeline.loombc` directly for preparation, control and providers. Its embedded weight policy uses that same module. The exported `name` controls HTTP routing and response
identity independently of the catalog’s internal policy name.

**You own export dependencies and compatibility.** Preserve the referenced
weights, tokenizer and pipeline bytecode for the entire native registration
lifetime, and use a compatible server. Export reserves nothing and records no
history. Paths are server-local; posting JSON does not transfer files. Package
upgrades or external file changes can invalidate paths: revalidate/regenerate
before use. Direct curl does not run Loom Control's Python compatibility validation.

## Check an installed wheel locally

The repository's GPU smoke entry point exercises a fresh installed wheel using
an isolated venv/state namespace on gfx1151:

```bash
bash build_tools/model-smoke.sh /path/to/hrx_solutions.whl /path/to/new-run \
  /srv/models/qwen08
```

The optional last argument must contain the exact
`Qwen3.5-0.8B-UD-Q5_K_XL.gguf` and adjacent `tokenizer.json`; those files are
immutable inputs. Without it, the helper acquires the pinned
`qwen35-08b-q5-k-xl` recipe through Loom Control, preserving normal HF credentials/cache.
Python 3 with venv, Linux shell tools and the supported GPU/driver are required.
The run directory must be new or empty. Serialize overlapping real-model checks
with a shared GPU lock chosen by the caller.

The check starts empty, imports/downloads and validates, registers/activates the
instance, waits for readiness and open admission, checks a bounded `Hello!`
response, retires the instance and shuts down the owned process. Lifecycle
budgets stay at 10 seconds; acquisition has its own 600-second cap and the query
has a 30-second bound. Results live in `lifecycle/results.json`; bounded redacted
evidence is under `lifecycle/diagnostics/`. On failure, cleanup uses only the
owned namespace and public reconcile/abort-shutdown commands, retaining failure
evidence and reporting unresolved cleanup. Confirm owned processes have exited
before removing the run directory. This check does not prove support for other
GPUs or replace the resource-qualified 27B/concurrent-isolation gates.

## Migrating from the setup helpers

`loom-serve-setup` and `loom-serve-start` entry points are removed. Use Loom Control's
`model`, `pipeline`, `server` and `instance` types for acquisition and lifecycle,
and `pi`/`opencode`/`omp` for local client configuration; direct `loom-serve` remains.
Old state files and external checkpoints remain untouched and are not adopted
or replayed. Prefer a fresh state namespace (for example, set
`LOOM_SERVE_DIR="$HOME/.loom-hem"` before running Loom Control), or choose fresh model IDs
whose directories do not already exist. An untracked populated model directory
is refused rather than overwritten. Explicitly import each checkpoint/tokenizer
pair, inspect it, create a new empty server and start its instances. Arrange for
old services to be stopped through their original tooling before uninstalling
that tooling, or use an unused port for Loom Control. Loom Control does not manage legacy or
arbitrary external processes merely because their endpoint responds.

Local Pi integration is covered by [Connect Pi once](#connect-pi-once).
Remote setup/tunnels, general client configuration and automatic migration remain
outside Loom Control. Configure other clients against the streaming chat endpoint.

## Connect Pi once

`loom-control pi configure` installs a bundled, read-only discovery provider for
local Loom Control-managed servers. Pi reads the current model inventory and sends
inference requests; it never downloads models or starts/stops Loom. This
integration is qualified with `@earendil-works/pi-coding-agent` **1.3.0** and
Node **22.22.1**; installing/updating configuration requires Node >=22.19.0. Select that Pi installation
and its Node runtime on `PATH` yourself. Loom Control does not install or upgrade either,
and older `@mariozechner` packages are not interchangeable.

The ordinary serving defaults remain **8192 context / 32768 shared pool**.
Those capacities may not fit Pi's effective compaction and summary budgets.
The real-model configuration check used explicit capacities below, with two
Qwen3.5-0.8B instances and ordinary short prompts. It proves our generated
configuration and routing; it does not qualify long-context performance,
compaction, or Pi's session/tool behavior. Larger defaults have **not** been
promoted.

After importing and validating `qwen08` as above, choose free loopback ports
and sufficient GPU memory. Use a fresh server name if `local` already exists:

```bash
loom-control server start --name pi-local --device amdgpu --listen 127.0.0.1:8081
loom-control instance start qwen08 --name pi-small-a --server pi-local \
  --pipeline qwen-08b --context-capacity 262144 --pool-capacity 262144 \
  --rows 4 --prefill-capacity 128
loom-control instance start qwen08 --name pi-small-b --server pi-local \
  --pipeline qwen-08b --context-capacity 262144 --pool-capacity 262144 \
  --rows 4 --prefill-capacity 128
loom-control pi configure
pi
```

In Pi, open `/model` and select `pi-small-a` or `pi-small-b` under the
`loom-hem-…` provider for `pi-local`, then send a short prompt. Each instance's
pool is shared across its four rows: this does not provide four independent
262144-token windows or promise both instances fit every device. These explicit
overrides do not change pipeline defaults or resize other running instances.
Use `loom-control instance stop NAME --server pi-local` and
`loom-control server shutdown pi-local` when finished.

Pi sees **instance names**, not downloaded model IDs. Provider IDs are stable
`loom-hem-` names followed by the unpadded base64url encoding of the server name.
Discovery joins live `/v1/models` and `/v1/status`; it exposes ready, active,
admitting text instances with supported metadata. Context comes from actual
native metadata; output is capped at the minimum of 8192, native output limit
and actual context. Initial visibility is budget-unverified. Before each
inference, the provider checks effective user/project/exact-model budgets and
current metadata/origin, refusing unsafe or unavailable settings with remediation
instead of rewriting settings, inflating capacity or silently routing elsewhere.

### Persistence, refresh and configuration modes

Configure once, then launch Pi normally. No periodic configure job is needed.
Configuration survives restarts and boots, but it does not start Loom or reload
instances. Restart a saved server with `loom-control server start pi-local`, then
explicitly start the instances you want. Existing-server model changes appear
on startup or Pi's model-picker refresh; an open picker is not a continuous
inventory. Adding a **new server name**, including the first server after an
empty setup, requires Pi `/reload` or restart. Endpoint changes for existing
names refresh without reconfiguration. Offline/unsupported servers report
unavailable; durable setup can succeed with no currently usable models.

```bash
loom-control pi configure --check
loom-control pi configure --json
loom-control pi configure --disable
```

Normal configure idempotently installs/updates or repairs unchanged owned files.
`--check` passively inspects without creating directories, migrating SQLite or
writing configuration, even when Pi or Node is missing or unsupported. It reports
owned-configuration findings separately from runtime compatibility: exit 0 means
no changes required and a qualified runtime; exit 1 means changes are required or
the runtime is unqualified. JSON retains the configuration `state` and adds
`runtime_compatibility` (`state`, `reason`, `next_action`); `runtime` is null when
qualification fails or is skipped. Ownership conflicts and uncertain state retain
their existing stronger exits. `--check` and `--disable` are mutually exclusive. All modes accept
`--pi-dir PATH`, `--json` and `--timeout SECONDS` (default 10). Operational,
invalid-input, pending and uncertain exits follow the existing 1/2/3/4 contract.
Progress and exact discovery GET curl equivalents go to stderr; JSON stdout
stays structured. File installation prints affected paths and explains that
there is no equivalent native Loom command.

Target precedence is `--pi-dir` > `PI_CODING_AGENT_DIR` > `~/.pi/agent`. For an
alternate target, launch Pi with the same directory:

```bash
export PI_CODING_AGENT_DIR="$HOME/.pi-loom/agent"
loom-control pi configure --pi-dir "$PI_CODING_AGENT_DIR"
pi
```

Use the same absolute `LOOM_SERVE_DIR` across Hem commands if overriding its
default. State/target directory ancestors must be trusted, owned by the user
or root, and not writable by others; managed files must not be symlinks.
A deliberately selected symlink agent directory is resolved to its canonical
target. Unsafe existing paths are refused; do not bypass this by weakening
permissions. Each Loom Control root owns at most one Pi target and each target belongs
to one root; another root cannot silently retarget it.

Under the chosen Pi agent directory, Loom Control owns `extensions/loom-hem/` and
`loom-hem/config.json` (identity, versions, owned hashes and endpoint pointer).
Under the Hem root, `integrations/pi/endpoints.json` projects saved local server
definitions; SQLite retains the minimal target identity. These are current
configuration, not a persisted model inventory or inference log. Copied assets
come from the installed wheel, with no runtime package installation. Loom Control uses
durable atomic replacement and detects owned-file drift; multiple files and
SQLite are not one transaction. After an interrupted update, inspect with
`--check`, then rerun configure. If owned files were edited, preserve/export
your edits yourself and restore the expected owned bytes before retrying;
Hem refuses to overwrite or remove conflicting content. An uncoordinated editor
can still race a final ownership check; subsequent checks detect drift.

Pi `settings.json`, `models.json`, authentication, defaults and session choices
remain unchanged. Unmanaged legacy `hrx-loom` listings may coexist and are
reported without deletion; unreachable means unknown, not obsolete. Back up
or export them yourself, inspect default-model references, then manually remove
only entries you know are obsolete. Loom Control does not manage exports or credentials.

Run `loom-control pi configure --disable` **before uninstalling the wheel**. Disable
removes only unchanged owned integration artifacts, reports retained conflicts,
and does not require installed/supported Pi or Node or run runtime qualification.
All ownership, path and edited/foreign-file protections still apply. It does not
cancel an in-flight request; reload/restart Pi afterward. Uninstall
alone may leave copied files. After upgrading the wheel, rerun configure and
reload/restart Pi to update those copies.

This is a same-user loopback trust boundary: schema/profile checks reject
detectable wrong listeners, but cannot authenticate ownership of an
indistinguishable replacement at a recycled port. Pi has no lifecycle authority.
Remote URLs, SSH tunnels and arbitrary external servers are outside this setup.


## Connect OpenCode 2 once

`loom-control opencode configure` installs a bundled local discovery plugin for
**OpenCode 2.0.31** (`@opencode/cli`). The executable is also named `opencode`;
OpenCode 1.x is a different integration and is unsupported. Install/select the
qualified client on `PATH` yourself. Configure never installs or upgrades it.
The standalone client works without Node; pinned Node 24.10.0 in our CI only
runs source tests and stages test dependencies.

After importing the `qwen08` model as described above, for example:

```bash
loom-control server start --name oc-local --device amdgpu --listen 127.0.0.1:8082
loom-control instance start qwen08 --name oc-small-a --server oc-local \
  --pipeline qwen-08b --context-capacity 32768 --pool-capacity 32768 \
  --rows 4 --prefill-capacity 128
loom-control instance start qwen08 --name oc-small-b --server oc-local \
  --pipeline qwen-08b --context-capacity 32768 --pool-capacity 32768 \
  --rows 4 --prefill-capacity 128
loom-control opencode configure
opencode
```

Use the OpenCode model selector to choose the exact instance name under the
stable `loom-hem-…` provider for `oc-local`. Requests use OpenCode's built-in
OpenAI-compatible transport directly to the local Loom `/v1/chat/completions`
endpoint. No inference API key, dummy key, login, proxy or Pi installation is
needed. OpenCode does not start or stop Loom servers/models.

Startup captures ready, active, admitting text instances by joining native
`/v1/models` and `/v1/status`. Context comes from the actual instance; output is
the smaller of native output limit and context. Current validated profile
advertises text input/output and `tools:false`, without reasoning variants.
No tools, agent quality, compaction, session restoration or full context window
is certified by this integration. The example's explicit 32768 capacities fit
our bounded configuration smoke; serving defaults remain **8192 context /
32768 shared pool**. These overrides do not promote larger defaults or promise
that every client workload fits.

Configure once. Ordinary model start/stop and new saved server creation require
client refresh, not another configure. An open picker is a captured snapshot,
not a continuously updated inventory. Refresh the **correct running OpenCode
server**, or restart the client as the fallback:

```bash
opencode reload --server http://127.0.0.1:PORT
```

`PORT` is the OpenCode control-plane port, not a Loom inference port. Use that
OpenCode server's own authentication (for example its `OPENCODE_PASSWORD`).
This control-plane password is separate from keyless local inference. Changing
the config directory does not retarget an already-running default server.
There is no Loom polling service, watcher or additional refresh command.
Unavailable/stopped servers contribute no usable models after refresh; their
saved provider identity can remain in the endpoint projection. A request made
against a stale captured model may be rejected by Loom until refresh.

```bash
loom-control opencode configure --check
loom-control opencode configure --json
loom-control opencode configure --disable
```

`--check` is passive: it does not run OpenCode, contact models or write files.
It reports configuration/ownership separately from runtime qualification and
availability. An absent/unsupported client does not prevent configuration
inspection, but produces an unqualified compatibility finding. Passive binary
qualification currently recognizes the exact qualified Linux x64 2.0.31 build;
other binary identities are reported unqualified without executing them.
`--disable` needs no client and removes only verified unchanged owned files.
Reload/restart OpenCode afterward. Disable before uninstalling the wheel;
rerun configure after a wheel upgrade to update copied assets. Export/backups
of your own configuration are your responsibility.

Destination precedence is `--opencode-dir PATH` > `OPENCODE_CONFIG_DIR` >
`$XDG_CONFIG_HOME/opencode` (normally `~/.config/opencode`). For an alternate
installation, select the same directory when launching and refreshing:

```bash
export OPENCODE_CONFIG_DIR="$HOME/.opencode-loom"
loom-control opencode configure --opencode-dir "$OPENCODE_CONFIG_DIR"
opencode
# For an already-running server launched with this environment:
opencode reload --server http://127.0.0.1:PORT
```

The only owned client directory is `plugins/loom-control-opencode/`, with five
bundled assets plus generated config/manifest. Installation uses bounded
atomic replacements, a pending intent and matching generations. Interrupted
updates fail closed and can be retried; unknown or edited contents are retained
for explicit resolution. Shared `opencode.json`/JSONC, credentials, default
models/providers and other plugins are preserved. Explicit known provider
suppression is reported; effective managed/project policy is evaluated by
OpenCode, so check actual visibility after launch. No general policy evaluator
is added here.

The root-owned `clients/endpoints.json` is a derived projection of saved local
server endpoints, independent of Pi configuration. It stores neither model
inventory nor request history. Lifecycle commands keep an existing projection
current; disable retains it because another client may reference it. Targets
and their existing ancestors must be owned, non-symlink directories without
unsafe write permissions, as with Pi.

GPU CI runs the installed repaired wheel's generated plugin unchanged against
two real named Qwen3.5 0.8B routes, then validates explicit reload, server/client
restart, newly created server routing, zero usable models after shutdown,
idempotence, passive check and client-independent disable. Source and installed
asset contract tests separately cover changed origins/provider deletion and
unsafe/edited files. Oh My Pi uses the separate integration described below.

## Connect Oh My Pi

`loom-control omp configure` supports **Oh My Pi 18.11.2** through its qualified
**Bun 1.3.14 npm route** (`@oh-my-pi/pi-coding-agent`, executable `omp`). It does
not install or upgrade either dependency; other versions and standalone binaries
are not yet qualified. Pi and Oh My Pi are distinct clients with separate assets.

Start your local Loom server and model first. For an ordinary OMP prompt, use an
explicit adequate allocation; the standard client prompt exceeds the unchanged
8192-token default. The configuration gate uses 32768 context and pool capacity:

```sh
loom-control server start --name omp-local --device amdgpu --listen 127.0.0.1:8080
loom-control instance start small --name qwen-local --server omp-local \
  --pipeline qwen-08b --context-capacity 32768 --pool-capacity 32768
loom-control omp configure
omp
```

Choose the `loom-hem-…/qwen-local` route in OMP. The collision-safe provider ID
identifies the saved server; the model ID is the exact native registration ID.
The bundled adapter discovers only ready text models by joining native models
and status, and exposes actual effective context/output limits with tools and
reasoning disabled for the qualified profile. It uses OMP's built-in compatible
transport directly, with native keyless activation, without storing a dummy key
or changing your credentials/default selection. Loom Control owns lifecycle;
OMP never starts or stops servers/models.

Configure installs `extensions/loom-control-omp/` and minimal `auth: none` entries
for each known server in OMP's native models YAML. Existing `models.yml` wins,
otherwise `models.yaml` is used, otherwise `models.yml` is created. Equivalent
user-owned activation entries are borrowed and retained. Unrelated comments,
provider credentials/order, settings and defaults are preserved. Only a narrow
unambiguous block YAML subset is supported: ordinary bare keys with two-space
provider/four-space auth indentation. Empty `providers: {}` is supported; quoted
keys, multiline/flow values, aliases/merges/tags, duplicates and damaged ownership
markers are refused without rewriting. Inspect/export and manually simplify an
unsupported layout; backups/exports are your responsibility. A legacy models.json
requires explicit user migration, never an automatic vendor migration.

Discovery occurs at startup and explicit refresh, with no polling. After starting
or stopping models on an already configured server, exit and restart OMP:

```sh
omp
```

`omp models refresh --json` is an optional inventory diagnostic; a separate refresh
process does not update an existing session's picker. Restart
forces online discovery instead of trusting OMP's 24-hour cache. Until refresh,
visible inventory is a snapshot; request-time checks reject stale origin,
incarnation, readiness or limits without rerouting. Unavailable/invalid endpoints
contribute zero usable models after refresh. F5/live picker reload is not qualified.

**Adding a new saved Loom server requires rerunning `loom-control omp configure`,
then restart OMP.** Missing activation is excluded and reported with that
instruction. Lifecycle and discovery never rewrite client YAML. Ordinary model
changes on previously configured servers require no reconfiguration.

```sh
loom-control omp configure --check
loom-control omp configure --disable
```

Check is passive: no client execution, inference, server start or repair. It
separates owned configuration findings from client compatibility and does not
certify live readiness. Disable needs neither OMP nor Bun and removes only owned,
unchanged assets/stanzas. Borrowed activation, unrelated YAML, settings and the
shared root endpoint document remain. Edited/foreign files and ambiguous partial
updates are preserved with repair guidance; do not force-delete them. Repeating
configure without changed inputs does not rewrite owned bytes.

The default destination follows OMP's active profile: normally `~/.omp/agent`,
or `~/.omp/profiles/NAME/agent` for a named profile. Profile precedence is
`OMP_PROFILE` (including empty) then `PI_PROFILE`; named profiles ignore
`PI_CODING_AGENT_DIR`. The explicit `--omp-dir` avoids inferred targeting and
conflicting explicit selectors are rejected. For an isolated alternate target:

```sh
export PI_CODING_AGENT_DIR="$HOME/.omp-loom-agent"
export OMP_PROFILE= PI_PROFILE=
loom-control omp configure --omp-dir "$PI_CODING_AGENT_DIR"
omp
```

Use a private directory with private ancestors; do not change shared directory
permissions to bypass safety checks. OMP `--config` overlays settings and is not
a models-file selector. Configure prints the resolved launch/refresh invocation
and the truthful configuration action; there is no single native curl equivalent
to installing several owned client files.

The delivery gate runs the installed wheel's public configure against two real
named 0.8B routes and a newly added server, with short genuine client requests,
metadata, model refresh, restart, preservation, passive check and absent-client
disable. These validate generated wiring, not agent tool quality, vendor session
restoration, compaction or a full context window. Defaults remain unchanged.
