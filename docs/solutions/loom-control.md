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
four resource types for acquisition and lifecycle; direct `loom-serve` remains.
Old state files and external checkpoints remain untouched and are not adopted
or replayed. Prefer a fresh state namespace (for example, set
`LOOM_SERVE_DIR="$HOME/.loom-hem"` before running Loom Control), or choose fresh model IDs
whose directories do not already exist. An untracked populated model directory
is refused rather than overwritten. Explicitly import each checkpoint/tokenizer
pair, inspect it, create a new empty server and start its instances. Arrange for
old services to be stopped through their original tooling before uninstalling
that tooling, or use an unused port for Loom Control. Loom Control does not manage legacy or
arbitrary external processes merely because their endpoint responds.

Remote setup/tunnels and application configuration are outside Loom Control's four-type
interface. Configure clients yourself against the streaming chat endpoint;
there is no Pi configuration, remote provisioning or automatic migration command.
