# Remote Loom Serve Design

**Status: Draft for review**

## Goal

Let an agent on a client device discover and start a model on an already
configured Loom Serve host, connect to it through SSH, and configure Pi on the
client to use that model. Loom Serve remains loopback-only. The client uses SSH
port forwarding rather than exposing the server port on the network.

The remote host is assumed to already have:

- The HRX Solutions wheel installed.
- Models imported and validated against their pipelines.
- SSH access configured for the client user.

This design does not install or update the remote wheel, install Hugging Face
tools, import or validate models, or provision the remote operating system.

## Current constraints

- The pinned Loom Serve server binds to `127.0.0.1`; this design does not
  change Loom Serve.
- `loom-serve-start` and `loom-serve-setup serve` now launch a managed
  background server by default and return its ID, endpoint, PID, and log path.
  Pass `--foreground` for interactive debugging.
- `loom-serve-setup models list --json` reports a versioned machine-readable
  inventory for SSH model discovery.
- `loom-serve-setup remote register/list/unregister/models` manages the local
  SSH target registry and discovers models on prepared hosts.
- `loom-serve-setup remote start NAME MODEL_ID` starts through the remote
  managed server CLI. `remote servers NAME list/status/stop` uses its JSON
  interface, leaving server records and logs on the remote host.
- `loom-serve-setup remote connect NAME` manages a detached SSH tunnel on the
  client. `remote connections list/status/disconnect` reconciles and manages
  those tracked connections independently of the remote server lifecycle.
- Managed server and connection commands accept `--json` for full records;
  list commands return arrays and single-record commands return objects.
- Local `servers list/status/stop` commands manage server records and verify
  process identity and the model advertised by the endpoint. These commands
  accept `--json` to return full server records, including endpoint health when
  available; list returns an array and the other commands return one object.
- Local `pi-config` requires a local model manifest. `remote pi-config` uses
  remote model inventory and does not require model files on the client.
- A running server advertises its model ID through `/v1/models`.

## Proposed command interface

Remote definitions and client-side Pi configuration live on the client
device. Remote commands use the registered SSH target to invoke the existing
HRX Solutions commands on the server.

```bash
# On the client: register how to reach the prepared model server.
loom-serve-setup remote register gpu-box \
  --ssh-target gpu-box \
  --server-port 8080 \
  --local-port 18080

# Discover models already installed and validated on the server.
loom-serve-setup remote models gpu-box

# Start a selected model on the server. The command returns a server ID,
# endpoint, and log path while the server runs independently.
loom-serve-setup remote start gpu-box qwen38-27b-q5-k-xl

# Query managed servers and their connection information.
loom-serve-setup remote servers gpu-box list
loom-serve-setup remote servers gpu-box status SERVER_ID

# On the client: create a managed background SSH tunnel and check the model.
loom-serve-setup remote connect gpu-box \
  --model qwen38-27b-q5-k-xl

# Query and disconnect tracked SSH connections.
loom-serve-setup remote connections list
loom-serve-setup remote connections status CONNECTION_ID

# In another client terminal: write/verify Pi's local configuration.
loom-serve-setup remote pi-config gpu-box qwen38-27b-q5-k-xl \
  --check-connection

# Start Pi on the client while the tunnel remains open.
pi --provider hrx-loom \
  --model qwen38-27b-q5-k-xl \
  --no-session

# Disconnect only the tunnel; the remote server stays running.
loom-serve-setup remote connections disconnect CONNECTION_ID

# Stop only the server started by this workflow.
loom-serve-setup remote servers gpu-box stop SERVER_ID
```

`gpu-box` is a local name for a saved connection definition. `remote pi-config`
first confirms the requested ID appears in remote inventory as validated and
ready, then writes Pi's `~/.pi/agent/models.json` on the client using the
registered local tunnel URL, `http://127.0.0.1:18080/v1`. It does not need or
inspect local model weights or manifests and does not edit Pi configuration on
the remote host. It preserves other Pi configuration fields and verifies the
saved provider URL and model ID after writing. `--check-connection` requires
an active tracked tunnel for that remote and probes `/v1/models` for the exact
requested ID before it writes the configuration.

## Command behavior

### Managed server lifecycle (local and remote)

`loom-serve-start MODEL_ID` starts a server as an independent background
process by default. It returns after the server is ready and prints the server
ID, model ID, loopback endpoint, PID, and log path. The explicit `--foreground`
option runs the server in the invoking terminal. Both local starts and remote
starts use this same lifecycle manager; a remote start invokes it over SSH on
the model host.

`--json` on `loom-serve-start MODEL_ID` and `loom-serve-setup serve
MODEL_ID` to print the full managed server record as one JSON object followed
by a newline. `servers list --json` prints a JSON array, while `servers status
ID --json` and `servers stop ID --json` print one full record each. JSON mode
emits no human-readable status text on stdout; errors remain on stderr.
Foreground serve cannot return a managed record and therefore does not support
`--json`.

Add local commands to inspect and stop managed servers:

```bash
loom-serve-setup servers list
loom-serve-setup servers status SERVER_ID
loom-serve-setup servers stop SERVER_ID
```

Remote equivalents address the remote host's server registry:

```bash
loom-serve-setup remote servers NAME list
loom-serve-setup remote servers NAME status SERVER_ID
loom-serve-setup remote servers NAME stop SERVER_ID
```

The Python tool launches the native server detached from the invoking
terminal, with stdin closed and stdout/stderr redirected to a server-specific
log. It records a `starting` entry before launch, waits for readiness, then
marks the entry `running`. On startup failure it captures the log location,
marks the entry `failed`, and cleans up the process it launched.

Store server records under `~/.loom/` on the machine running the server, with
server logs under `~/.loom/logs/`. Store remote registrations and tunnel
records under `~/.loom/` on the client. Use this layout on each machine:

```text
~/.loom/
  state.lock
  remotes/<name>.json
  servers/<server-id>.json
  connections/<connection-id>.json
  logs/<server-id>.log
  logs/<connection-id>.log
```

The remote registry and connection records are client-side. Server records
are written on the server host. Acquire an exclusive OS file lock on
`state.lock` for every read/reconcile/write operation that can change state,
including list and status commands when they reconcile. Hold it only while
reading or updating records; do not hold it during server readiness waits or
network probes. Keep `~/.loom` and its state directories owner-only (`0700`),
and the lock, records, and logs owner-readable/writable only (`0600`). Write
each record through a temporary file and atomic rename.
Retain stopped and failed records for inspection; provide a prune operation
later if state growth becomes a concern. A server record contains at least:

- Server ID, lifecycle status (`starting`, `running`, `stopping`, `stopped`,
  `exited`, `failed`, or `stale`), start time, and stop time.
- Model ID, validated pipeline, and resolved model file paths.
- PID, process group ID, and process start time to detect PID reuse.
- Bind address, port, API endpoint, and launch options (rows, context,
  prefill, pool, and MTP settings).
- Per-server log path and, for remote starts, the registered remote name.

Whenever a command lists, inspects, starts, stops, or changes managed state,
reconcile recorded processes against the live system; do not run a persistent
supervisor. Verify process identity using PID, OS process start time, and the
expected executable/command. A PID alone is not proof that the original
process is alive. If a process that had reached `running` is gone, mark it
`exited`; if it disappears before readiness, mark it `failed`. Record that its
exit code is unknown because no process remains to reap. Explicit stop moves
through `stopping` to `stopped`. If the PID was reused or identity cannot be
confirmed, mark the record `stale` and never signal it. Allocate a `starting`
record before spawning, then persist verified process identity as soon as
available. Do not hold the state lock while waiting for readiness.

Stop sends a graceful termination signal to the verified managed process,
waits for exit, updates the record, and leaves its log available. It must not
stop a server merely because that server owns the requested port. Servers
started manually outside the manager may be detected as listeners, but remain
unmanaged and cannot be stopped by these commands.

### Register and inspect remotes

`remote register NAME --ssh-target TARGET` saves a name and SSH target, the
remote Loom Serve port, and the client's preferred tunnel port. It relies on
the system SSH configuration for usernames, keys, nonstandard SSH ports, and
ProxyJump rules. Do not copy private keys or credentials into the remote
definition.

`remote list` shows registered names. `remote unregister NAME` removes a
definition only when no managed server or connection references it. Store
definitions in `~/.loom/`, not the repository.

### Discover remote models

`remote models NAME` retrieves the remote model inventory over SSH. It reports
model ID, validated pipeline, install mode, and availability. This is a model
store query; it does not require starting the HTTP server or opening a tunnel.

The remote command should request machine-readable output from
`loom-serve-setup models list --json`, rather than parsing human-formatted
columns. Add this output mode if the current CLI does not provide it. Do not
guess compatibility from a model name or filename.

### Start a remote server

`remote start NAME MODEL_ID` uses SSH to invoke `loom-serve-setup serve
MODEL_ID --port PORT --json` on the remote host. The model must already be
installed and validated there. The remote process manager records the server
and returns its server ID, endpoint, PID, and remote log path.
`remote servers NAME list/status/stop` invokes the remote `servers` commands
with `--json`, queries or controls that registry over SSH, and never parses
human output. The server record and log live on the remote host; the local
remote definition only stores connection settings. Shell arguments are quoted
and model/server IDs are validated before execution. SSH keeps the user's
normal host-key verification enabled.

Starting the same model on an already running managed port should report its
existing server record. Starting a different model on an occupied port should
fail clearly without replacing the server. A manually started server may be
discovered by model listing and connection checks, but is unmanaged and cannot
be stopped by these commands.

### Connect through SSH

`remote connect NAME` starts a detached SSH local forward equivalent to:

```bash
ssh -N -L 127.0.0.1:18080:127.0.0.1:8080 gpu-box
```

The first loopback address is on the client. The second is resolved on the
remote host, where Loom Serve listens. Bind the client port to loopback only.
The command redirects SSH output to a connection-specific log, records the
connection, waits for the tunnel to become usable, and returns a connection
ID and local URL. Disconnecting a tunnel does not stop the remote server.

When `--model MODEL_ID` is supplied, wait for the forwarded endpoint and check
that `/v1/models` advertises that exact ID. Report SSH failure, a local port
conflict, endpoint failure, or model mismatch distinctly. If readiness fails,
stop and mark the newly launched tunnel failed. An unexpectedly lost tunnel is
marked failed; it does not stop the server. The client can reconnect.

Manage connections independently from the server lifecycle:

```bash
loom-serve-setup remote connections list
loom-serve-setup remote connections status CONNECTION_ID
loom-serve-setup remote connections disconnect CONNECTION_ID
```

Store tunnel records under `~/.loom/` on the client, with their logs under
`~/.loom/logs/`. Tunnel lifecycle uses `connecting`, `connected`,
`unreachable`, `disconnecting`, `disconnected`, `exited`, `failed`, and
`stale`; explicit disconnect moves through `disconnecting` to `disconnected`,
and a tunnel that exits unexpectedly is `exited`. A record contains:

- Connection ID and state (`connecting`, `connected`, `unreachable`,
  `disconnected`, `failed`, or `stale`), start/end times, and last health-check
  time.
- Registered remote name and SSH target.
- SSH PID, process group ID, and process start time for PID-reuse checks.
- Local loopback URL/port and remote loopback host/port.
- Optional expected model ID and associated managed server ID.
- Per-connection log path.

Status verifies the SSH process identity and probes the local forwarded
endpoint. A live SSH PID alone does not prove the target server is responding;
report `unreachable` when the tunnel process is alive but the HTTP check fails.
Disconnect signals only the verified managed SSH process, waits for it to
exit, marks the connection disconnected, and retains the record and log.
Connection records are stored on the client; server records remain on the
remote host. Use the same `~/.loom/state.lock` to serialize reconciliation and
updates to these records.

### Configure Pi on the client

`remote pi-config NAME MODEL_ID` pulls the remote model identity into the
client-side Pi configuration. Before writing, it queries the remote inventory
and requires the exact model ID to be both validated and available. It writes
the selected ID and the registered loopback tunnel URL to the existing
`~/.pi/agent/models.json`, preserving unrelated configuration and model
extension fields. It does not need a local copy of the weights or a local
model manifest. After writing, it reads the file back and verifies the provider
URL and model ID. `--check-connection` first requires a reconciled active
tracked tunnel for the selected remote, then queries `/v1/models` through that
tunnel and confirms the exact model ID before changing the configuration.

Pi's configured model ID must exactly match the server's advertised ID. A
configuration-only operation can run before connecting; the optional live
check requires the tunnel to be active. Pi remains a separately installed
client.

## SSH and host requirements

The remote host must have a reachable SSH server and the HRX Solutions
commands available in a non-interactive SSH shell's `PATH`. The client needs
an SSH client and the setup CLI. The initial version uses standard SSH keys
and host-key verification; it must not disable host checking or expose the
forwarded port beyond client loopback.

## Failure and cleanup expectations

- A failed start leaves a failed record and its log path, and does not claim
  the model is running.
- A failed connection does not change the server process.
- Stopping a model does not silently alter Pi configuration or remote
  registration.
- Port conflicts, unknown remotes, unavailable model IDs, missing remote
  commands, and SSH failures produce actionable errors.
- SSH commands must use safe argument handling; do not interpolate unchecked
  values into shell command strings.

## Implementation order and next work

The state contract, local managed server lifecycle, remote registration, JSON
model discovery, remote server lifecycle, tracked SSH tunnels, and remote Pi
configuration are implemented. Remaining work is to document and validate the
full client/server workflow, including Pi tool calls through the tunnel.

## Remaining design questions

- Should `remote models` show only models that have passed validation, or also
  unvalidated imports with their state clearly marked?
- Should `remote start` print a remote log path, stream startup logs, or both?
- Should an option later combine `start` and `connect` while keeping their
  separate lifecycle commands available?
