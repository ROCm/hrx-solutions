# Connect from another machine

First complete [the inference check](../solutions/llm-serving.md#send-a-first-request) on the **inference
host**. Keep its server bound to `127.0.0.1`. The model, tokenizer, deployment,
and pipeline paths stay on that host. SSH forwards HTTP traffic; it transfers
none of those files.

On the **inference host**, confirm the listener port used in [server startup](../solutions/llm-serving.md#start-the-server)
and repeat the bounded request. Leave its serving terminal running.

On the **client machine**, choose an unused local loopback port and set the
remote login and listener port. These are examples; replace the host and ports:

```sh
export INFERENCE_HOST='operator@inference-host'
export REMOTE_SERVE_PORT=8080
export LOCAL_SERVE_PORT=18080
ssh -N -T \
  -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
  -L "127.0.0.1:${LOCAL_SERVE_PORT}:127.0.0.1:${REMOTE_SERVE_PORT}" \
  "$INFERENCE_HOST"
```

This terminal stays occupied by SSH. Use normal SSH authentication and host-key
verification. The destination `127.0.0.1` after the local port is interpreted on
the inference host. The listening `127.0.0.1` before it limits the forwarded
port to the client machine. `ExitOnForwardFailure` catches a local bind failure;
it does not prove the remote HTTP server is ready. Keepalives detect a broken
SSH connection rather than restarting it.

In a **second client terminal**, use the same local port:

```sh
export LOCAL_SERVE_PORT=18080
export SERVE_URL="http://127.0.0.1:${LOCAL_SERVE_PORT}"
curl --fail-with-body --connect-timeout 5 --max-time 10 \
  "$SERVE_URL/v1/status"
curl --fail-with-body --no-buffer --connect-timeout 5 --max-time 120 \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"Reply with Hello."}],"stream":true,"temperature":0,"max_tokens":32,"enable_thinking":false}' \
  "$SERVE_URL/v1/chat/completions"
```

Use the exact served model ID from [loading](../solutions/llm-serving.md#load-the-model). Confirm generated
content, a terminal completion event, and `[DONE]`, as in the direct check. A
successful TCP connection or status response alone does not pass inference.
The packaged shared chat policy's ID does not identify the checkpoint size.

Now follow [Pi configuration](../solutions/llm-serving.md#connect-pi) on the **client machine**, setting its
`baseUrl` to `http://127.0.0.1:18080/v1` for this tunnel. Keep the SSH terminal
open throughout the Pi session. Pi's files and tool execution belong to the
client machine; inference uses the remote host.

If SSH reports the local address is in use, choose another local port and update
both the request URL and Pi configuration. If SSH connects but HTTP fails,
repeat the request on the inference host and check its listener, readiness,
admission, and logs before rebuilding the tunnel. A disconnected tunnel does
not imply the model needs registering again.

Stop the tunnel with `Ctrl+C` in the **client SSH terminal**. Confirm a bounded
request to the forwarded client port now fails, then repeat the direct request
on the **inference host** to prove the server is still usable. Retire the model
and stop the server separately using [the shutdown instructions](../solutions/llm-serving.md#stop-and-restart).

This recipe requires an existing reachable SSH host. It does not configure
public HTTP exposure, provision a machine, install a service, or restart either
process automatically.

The forwarding lifecycle was checked using an existing SSH login to `localhost`
on the validation machine: a forwarded streaming request completed, stopping
that tunnel made the local port unreachable, and a direct request still
completed. This checks real SSH forwarding and independent shutdown on one
host; a separate remote inference machine was not qualified.
