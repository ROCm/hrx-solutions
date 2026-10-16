# Troubleshooting and connection diagnostics

Follow the [local walkthrough](../solutions/llm-serving.md) first. Diagnose the
failing transition rather than restarting every component.

| Symptom | Inspect and recover |
| --- | --- |
| Missing or unreadable model file | Verify the exact pinned asset set and stable absolute paths; do not substitute a similarly named variant. |
| Server exits during startup | Read foreground diagnostics; check GPU node permissions, driver/wheel support and listener ownership. |
| Registration rejects | Inspect its HTTP error, JSON field names/types and existing registrations. Duplicate names conflict rather than replace. |
| Activation fails | Inspect the matching incarnation's `loom.error`, source closure, assets and capacities; retire before correcting and registering again. |
| Readiness times out | Keep process/files available and observe; accepted preparation can still be running. |
| Inference times out | Check readiness, admission, model identity and capacity; preserve the partial stream. Avoid accumulating unknown requests in retry loops. |
| Pi model absent or request fails | Check JSON syntax, provider key, credentials, model ID, endpoint and effective project settings; retry the bounded curl test first. |
| Retirement times out | Preserve dependencies and observe the original incarnation; cancellation is an explicit separate choice. |

Status and runtime diagnostics are available while the process is alive:

```bash
curl --fail-with-body -sS --max-time 10 http://127.0.0.1:8080/v1/status | jq .
curl --fail-with-body -sS --max-time 10 http://127.0.0.1:8080/logz -H 'X-Loom-Model: qwen3.8-27b'
```

Keep a bounded copy of foreground logs and the deployment when reporting a
failure. Interpreter/import paths or conflicting library environments can be
useful installation diagnostics; they are not routine setup steps. Confirm
which venv supplies `python` and `loom-serve` before mixing installations.

## Check a saved stream

To automate the walkthrough's stream observation, save curl output to `hello.sse`
and require curl success, generated content, a terminal choice and `[DONE]`:

```bash
tr -d '\r' < hello.sse | grep -Fx 'data: [DONE]'
awk '/^data: / && !/\[DONE\]/ {sub(/^data: /, ""); print}' hello.sse |
  jq -se 'any(.[]; any(.choices[]?; (.delta.content // "") | length > 0)) and any(.[]; any(.choices[]?; .finish_reason != null))'
```

Inspect the generated text as well. A role-only event or successful connection
is insufficient. This check assumes the server's one-line JSON SSE records;
it does not claim a generic SSE parser or full OpenAI API conformance.

## Check Pi connectivity

The walkthrough's normal `models.json` entry and budgets configure Pi. The
following **optional restricted diagnostic** tests a short request with a minimal
system prompt and no tools. It does not qualify normal Pi prompts or agent work.
It requires GNU `timeout`; run outside a repository that adds Pi instructions.

```bash
timeout --signal=INT --kill-after=5s 120s pi \
  --provider loom-local --model qwen3.8-27b --thinking off \
  --no-extensions --no-skills --no-prompt-templates --no-tools \
  --no-session --system-prompt 'Answer briefly in plain text.' \
  --print 'Reply with Hello.'
```

Require generated text and a successful exit within the deadline. Keep every
restriction above when interpreting the connection evidence. In particular,
removing the custom prompt or enabling tools changes the tested workload.

A prompt-capacity error requires a smaller serialized prompt/history or a larger
qualified deployment. Reducing output alone does not fix an oversized history.
Check `~/.pi/agent/settings.json` and project `.pi/settings.json`; a resumed
session may restore its earlier model. Opening `/model` reloads the model
configuration; start a new session after changing budget settings.

The 0.8B fixture failed an ordinary tool task and did not complete a read-only
retry task. See [profiles](profiles.md) for this usage boundary. End Pi with
Ctrl-C; it does not retire the model, stop the server or end an SSH tunnel.
