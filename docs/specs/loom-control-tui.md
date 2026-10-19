# Loom Control TUI

## Outcome

Provide a small, read-only terminal interface for the common local operator
flow:

1. Run `loom-control tui`.
2. See locally managed Loom Serve servers and model instances.
3. Select one item and inspect its basic state.
4. Follow its server or model logs.
5. Refresh or quit without affecting the managed services.

The CLI remains the lifecycle and recovery interface. The first TUI reports a
problem when observation or logs fail; it does not diagnose the cause, suggest
repairs, or execute recovery actions.

## First-release scope

- Local Loom Control servers and their registered model instances.
- A Textual application installed with the normal wheel and imported lazily.
- A basic overview, detail view, and two log views.
- Keyboard navigation, visible shortcuts, refresh, and quit.
- Simple states such as running, stopped, unavailable, and unknown.
- Bounded background observation so the interface remains responsive.

## Deferred scope

- Start, stop, restart, registration, activation, draining, or other lifecycle
  actions from the TUI.
- Diagnosis, repair suggestions, recovery commands, or automated cleanup.
- Last-good history, stale-age policy, retries, backoff, and detailed partial
  failure analysis.
- Shareable diagnostic reports and advanced recursive redaction.
- Remote hosts, external endpoints, tunnels, and arbitrary system services.
- Chat, inference requests, GPU telemetry, throughput, and historical metrics.

## Design

### Framework

Use Textual. Keep it as a normal wheel dependency so `loom-control tui` works
after a standard installation. Import it only when the TUI command runs so
other commands do not pay its startup cost.

### Read-only provider

Widgets consume small immutable TUI records through a provider interface. The
provider uses in-process Loom Control state and managed HTTP connections. It
does not shell out to CLI commands, run reconciliation, or expose registry
rows, process records, private paths, native response bodies, commands, or
environments to widgets.

Opening and refreshing the TUI must not create or migrate the Loom namespace,
change application-authored state, or invoke lifecycle operations. A missing
namespace produces an empty screen and remains missing.

Read servers and instances in one existing-state, read-only snapshot. Associate
items with their server launch and model incarnation so a concurrent restart
cannot attach an old observation to a replacement. Run process and HTTP checks
after releasing the database snapshot. One failed server must not hide rows
that were read successfully.

The first release keeps refresh scheduling simple: one refresh is active at a
time, every blocking call has a short deadline, a replaced result cannot update
the screen, and quit releases the provider's owned resources. It retains no
observation history.

### Logs

Keep server process logs and model native logs as separate sources.

Server logs are selected by managed server and launch identity. The source
resolves and validates retained log files internally and never accepts or
returns an arbitrary path. Reads are bounded. Following uses file identity and
offset; rotation, truncation, or a missing old file produces a visible reset or
gap rather than silently joining unrelated bytes.

Model logs use the selected managed server/model identity and native `/logz`
with its opaque cursor. Reads are bounded. A malformed cursor or changed stream
produces a visible reset; a missing source produces an unavailable message.
Output from different model incarnations is never joined.

Only one log worker is active for the selected source. Switching selection,
leaving the log view, or quitting prevents old results from being displayed and
releases owned resources.

### Display safety

Treat names, endpoints, states, and service-provided messages as text before
Rich rendering. Do not put raw commands, environment values, credentials, or
private paths in TUI records. Advanced diagnostics and remediation guidance are
deferred.

## Feature slices

### 1. Textual shell and wheel integration (`proj-bjw.1`)

Provide `loom-control tui`, lazy Textual import, the application shell, loading,
empty and provider-error states, visible shortcuts, terminal restoration, and
a fake provider for headless interaction checks.

This slice is complete.

### 2. Basic local service overview (`proj-bjw.2`)

Add the read-only provider and display configured servers and registered model
instances with basic recorded state, process health, HTTP availability, and a
manual refresh action.

**Acceptance:** an installed wheel displays empty and populated local state;
missing state remains absent; refresh performs no application-authored durable
writes or reconciliation; a concurrent launch change cannot mix identities;
one failed observation does not hide successfully read rows; repeated refresh
and quit keep work and resources bounded.

### 3. Basic server and model details (`proj-bjw.3`)

Show the selected item's endpoint, stable identity, recorded lifecycle state,
process/HTTP availability, model readiness/incarnation where available, and a
plain unavailable message when a field cannot be observed.

**Acceptance:** selection shows only the selected stable identity; missing
fields do not crash or invent values; provider-controlled strings render as
text; a replaced launch or incarnation is not presented as the old selection.

### 4. Basic live logs (`proj-bjw.4`)

Provide separate bounded server and model log views with follow, pause/resume,
scrollback, and return-to-details controls. Show source unavailable, reset, gap,
and end-of-process markers without attempting repair.

**Acceptance:** new output appears while navigation and quit remain responsive;
arbitrary paths and endpoints are rejected; rotation/truncation and native
cursor reset do not splice unrelated streams; scrollback and reads are bounded;
switching views or quitting prevents late output and releases the source.

### 5. Basic navigation and compact layout (`proj-bjw.6`)

Connect overview, details, and logs with consistent keyboard navigation,
visible shortcut help, stable selection, and a usable narrow-terminal layout.
Status meaning must not depend on color.

**Acceptance:** headless interaction covers the complete overview-to-details-
to-logs flow, refresh, back navigation, and quit; replacement of the selected
identity is visible; normal, narrow, and color-disabled layouts retain service
identity and basic state.

## Initial milestones

### M1: Working local dashboard (`proj-bjw.8`)

Includes features 1 and 2. A fresh installed wheel launches the TUI, reads
existing local inventory without changing it, displays basic service state,
refreshes responsively, and exits cleanly.

### M2: Simple operator TUI (`proj-bjw.9`)

Includes features 3, 4, and 6. An operator can navigate from local inventory to
basic details and either bounded log source, observe a clear unavailable/reset
message, and return or quit without affecting services. This is the first
operator-ready release.

## Deferred follow-up

### Resilient refresh and stale-state handling (`proj-bjw.5`)

Later work may add last-good retention, observation age, stale-state policy,
retries/backoff, and richer partial-failure presentation.

### Diagnostic and remediation experience (`proj-bjw.7`)

Later work may add structured diagnosis, recursive sanitization, shareable
summaries, and safe suggested CLI recovery commands.

### Reliability and diagnostics milestone (`proj-bjw.10`)

This milestone is deferred with features 5 and 7. It does not block the initial
operator-ready release.

## Verification

Use fake providers and Textual Pilot for deterministic UI behavior. Use fixture
state, fake managed processes, and a local fake native HTTP service for provider
and log contracts. Each feature has focused checks; M1 and M2 add installed-
wheel launch, interaction, quit, and terminal-restoration gates outside the
checkout. Real model serving is an integration gate for M2 rather than a
requirement for every UI check.
