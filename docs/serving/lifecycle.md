# Lifecycle and bounded automation

The [walkthrough](../solutions/llm-serving.md#load-the-model) shows manual HTTP
control. These details expand it for scripts and recovery; they do not replace
its deployment or install steps.

`/readyz` and `/healthz` describe server availability. `/v1/status` includes
registrations that are preparing, serving or retiring. `/v1/models` is an
inference-discovery view: disappearance there does not prove resource retirement.
Status publishes a positive `loom.incarnation` per registration lifetime. A
reused name may refer to a replacement incarnation; coordinate lifecycle writers.

HTTP 202 acknowledges accepted intent, not completion. Registration with drained
admission is metadata-only. Explicit activation initiates preparation and weight
activation while requests remain closed. Undrain opens admission. A lost control
response does not cancel or roll back accepted work; inspect before replaying.

## Readiness automation

The following observes an already registered model on the walkthrough's endpoint.
Record its incarnation before activation or immediately after observing the
accepted drained registration. Use the same shell for subsequent snippets:

```bash
curl --fail-with-body -sS --max-time 10 http://127.0.0.1:8080/v1/status > status.json
INCARNATION=$(jq -er '.data[] | select(.id == "qwen3.8-27b" and .loom.status.admission == "drained" and .loom.error == null) | .loom.incarnation' status.json)
test -n "$INCARNATION" && test "$INCARNATION" -gt 0
```

After the walkthrough's activation request, poll up to 120 seconds. Each
iteration tests state; the interval does not stand in for readiness:

```bash
deadline=$((SECONDS + 120))
ready=false
while (( SECONDS < deadline )); do
  curl --fail-with-body -sS --max-time 5 http://127.0.0.1:8080/v1/status > status.json || break
  if ! jq -e --argjson incarnation "$INCARNATION" \
    '[.data[] | select(.id == "qwen3.8-27b" and .loom.incarnation == $incarnation)] | length == 1 and .[0].loom.error == null' \
    status.json >/dev/null; then
    jq . status.json
    break
  fi
  if jq -e --argjson incarnation "$INCARNATION" \
    '.data[] | select(.id == "qwen3.8-27b" and .loom.incarnation == $incarnation) | .loom | .registration == "ready" and .parameter_operation == "idle" and .parameters.active == true and .status.admission == "drained"' \
    status.json >/dev/null; then
    ready=true
    break
  fi
  sleep 1
done
$ready
```

Only a zero exit from the final command permits undrain. On error, timeout or
identity loss, inspect status and process logs; accepted preparation can still
be running. After undrain, verify the same ready/idle/active/error-free
incarnation now has `status.admission: "open"` before inference.

## Retirement

Graceful unregister (`mode: "drain"`) closes admission, completes admitted work
and waits for resource borrowers to retire. Observe the incarnation recorded
above disappearing, rather than checking discovery alone:

```bash
deadline=$((SECONDS + 120))
retired=false
while (( SECONDS < deadline )); do
  curl --fail-with-body -sS --max-time 5 http://127.0.0.1:8080/v1/status > status.json || break
  if jq -e --argjson incarnation "$INCARNATION" \
    '[.data[] | select(.id == "qwen3.8-27b" and .loom.incarnation == $incarnation)] | length == 0' \
    status.json >/dev/null; then
    retired=true
    break
  fi
  sleep 1
done
$retired
```

A timeout is pending retirement, not permission to delete dependencies. Keep
observing. To explicitly cancel accepted work instead, submit unregister with
`mode: "abort"` after confirming the still-current identity; cancellation also
requires actual retirement observation. Never target a replacement just because
its name matches an old request.

Drain alone (`POST /v1/models/drain` with the model name) is reversible and
leaves the registration alive. New inference can wait within bounded holding
credit; undrain reopens admission. Use unregister when changing capacities or
releasing the registration.

After models retire, Ctrl-C the foreground server and wait for process exit.
The pinned runtime stops admission and joins accepted work before teardown.
Ending Pi or SSH does not stop this process. Starting empty again restores no
registrations; the JSON files remain operator-owned configuration.
