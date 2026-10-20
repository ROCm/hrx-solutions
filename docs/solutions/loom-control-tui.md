# Observe Loom Serve with the Loom Control TUI

`loom-control tui` is an interactive, read-only view of the Loom Serve servers
and model instances managed by Loom Control on the local machine. Use it to
check current state, inspect one service, and follow logs without repeatedly
running status commands.

The TUI observes existing state. It does not start, stop, reconcile, repair, or
otherwise change a server or model instance. Use the regular `loom-control`
commands when an action is required.

## Launch

Install the `hrx-solutions` wheel in a Python 3.9 or newer virtual environment,
then run:

```bash
loom-control tui
```

The command needs an interactive terminal. A GPU is not required to open the
dashboard, although checking a running service naturally requires that service
and its hardware to remain available.

The TUI uses the same local state namespace as the rest of Loom Control:
`~/.loom` by default, or the absolute directory in `LOOM_SERVE_DIR`. Set the
same override used to create the managed services before launching the TUI:

```bash
export LOOM_SERVE_DIR=/absolute/path/to/loom-state
loom-control tui
```

If the selected namespace has no managed services, the dashboard reports
`No services configured`.

## What the dashboard shows

The overview lists each saved server followed by its model instances. Model
rows are indented below their server. Selecting a row updates the summary pane
with its state, kind, endpoint, and a short comparison of recorded and observed
state.

The TUI checks three signals for a managed server:

- the state recorded by Loom Control;
- whether the owned process identity can be verified;
- whether the server's HTTP status is available.

A model row also reports its saved registration and current native readiness
when the owning server can be contacted. These observations may appear as
`running`, `stopped`, `unavailable`, or `unknown`. The dashboard displays
uncertainty rather than changing state to resolve it.

Press **Enter** for the selected row's detail view. Details include the server
and launch identities, endpoint, recorded state, process state, and HTTP state.
Model details also include the model identity, incarnation, and readiness.
Missing values appear as `Unavailable`.

## Keyboard controls

The help strip at the bottom shows the controls that apply to the current view.

| Key | Overview | Details | Logs |
| --- | --- | --- | --- |
| **Up** / **Down** | Change the selected service | | Scroll log output |
| **Enter** | Open details | | |
| **l** | Open logs for the selection | Open logs for the selection | |
| **r** | Refresh observations | Refresh observations | |
| **Space** | | | Pause or resume following |
| **Esc** | | Return to the overview | Return to the previous view |
| **q** | Quit | Quit | Quit |

The dashboard loads observations once at startup. Press **r** to check for
changes; inventory refresh is not automatic. Refresh is disabled while the log
view is open, so press **Esc** before refreshing.

The TUI preserves a selection only while the exact managed launch or model
incarnation still exists. If a refresh finds that it was replaced or removed,
the details pane says that the previous selection or identity is unavailable.
Choose a current row instead of assuming the replacement is the same service.

## Follow logs

Press **l** from the overview or details view to follow the selected service:

- A current server launch reads its retained managed process log.
- A current model instance reads its native model telemetry stream.

Following begins immediately. Press **Space** to pause fetching and press it
again to resume. Press **Esc** to close the log source and return to the view
from which it was opened.

The display retains at most 500 lines and 256 KiB. Explicit markers keep loss
or changes visible:

| Marker | Meaning |
| --- | --- |
| `[gap]` | Earlier output was omitted, truncated, or otherwise could not be read continuously. |
| `[reset]` | The backing log file, stream, or native cursor changed. |
| `[unavailable]` | The selected log source cannot currently be read. |
| `[end]` | The managed server launch or process ended. |

Logs are pinned to the launch identity and, for a model, its incarnation. The
TUI does not silently switch a log view to a replacement process or model.
Stopped servers and observations without a current managed identity cannot open
a live log source; the dashboard reports `Logs unavailable` and remains usable.

## Compact terminals

At more than 60 columns, the service list and summary use a side-by-side
layout. At 60 columns or fewer, they stack vertically and reduce padding. The
same keys and views remain available. Enlarge the terminal if long endpoints or
identity values need more room.

## Visible failures and limitations

Provider, state, process, HTTP, and log failures remain visible in the
dashboard. For example, the state bar can report `Provider unavailable`, and
individual rows can report `unavailable` or `unknown`. A failure does not cause
the TUI to initialize, migrate, reset, reconcile, or repair the Loom namespace.

The current TUI is deliberately small:

- It observes locally managed Loom Control servers and instances, not arbitrary
  remote endpoints or foreground `loom-serve` processes.
- It refreshes inventory only at startup and when **r** is pressed.
- It provides a bounded live log view rather than an archival log browser.
- It does not provide lifecycle buttons, automated remediation, or extended
  diagnostic advice.

## Return to the CLI for actions

Leave the TUI with **q** and use the
[Loom Control command guide](loom-control.md) when you need to change or recover
a service. Common handoffs include:

```bash
# Get a complete status result suitable for scripts or bug reports.
loom-control server status SERVER --json
loom-control instance status INSTANCE --server SERVER --json

# Inspect complete CLI log output or use follow mode.
loom-control server logs SERVER
loom-control instance logs INSTANCE --server SERVER --follow

# Reconcile uncertain saved and process state.
loom-control server reconcile SERVER

# Perform lifecycle actions.
loom-control instance stop INSTANCE --server SERVER
loom-control server shutdown SERVER
```

Use each command's `--help` output for its full arguments and recovery
constraints. After completing an action, relaunch the TUI or press **r** in an
open overview to observe the new state.
