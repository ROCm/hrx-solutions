# HRX Solutions Quickstart

Use this guide to install and run a model locally, or connect Pi to a prepared
remote Loom Serve host. Give each checkpoint or fine-tune a unique `MODEL_ID`;
`qwen` is the compatibility pipeline used to validate it. Keep model files
outside the repository and validate before serving.

## Install

On Linux x86-64 with a supported AMD GPU and driver, download a wheel from the
[latest GitHub Release](https://github.com/ROCm/hrx-solutions/releases/latest)
and install it in a virtual environment:

```bash
python3 -m venv "$HOME/.venvs/hrx-solutions"
source "$HOME/.venvs/hrx-solutions/bin/activate"
python -m pip install ./hrx_solutions-*.whl
```

The wheel includes the server and Qwen pipeline sources, not model weights.

## Import and validate

Set paths for the GGUF and matching tokenizer, then choose a unique model ID:

```bash
MODEL_ID="qwen38-27b-q5-k-xl"
GGUF="/path/to/Qwen3.8-27B-UD-Q5_K_XL.gguf"
TOKENIZER="/path/to/tokenizer.json"

loom-serve-setup models import "$MODEL_ID" --gguf "$GGUF" --tokenizer "$TOKENIZER"
loom-serve-setup models validate "$MODEL_ID" --pipeline qwen
loom-serve-setup models list
```

Import copies files into `~/.loom/models/$MODEL_ID/`. Add `--alias` to symlink
from the supplied paths instead; keep those files in place.

To download files from Hugging Face, install/authenticate the `hf` CLI first.
The setup command downloads through its cache; `--alias` symlinks the returned
cache paths into the model store. Pin revisions to commits for repeatability:

```bash
python -m pip install -U 'huggingface_hub[cli]'
hf auth login  # or set HF_TOKEN

loom-serve-setup models import "$MODEL_ID" --alias \
  --hf-repo OWNER/REPOSITORY \
  --gguf-file path/to/model.gguf \
  --tokenizer-file path/to/tokenizer.json \
  --revision COMMIT
```

For files in separate repositories, use `--tokenizer-hf-repo` and optionally
`--tokenizer-revision`. This locally validated Qwen3.8 pair uses separate
[GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) and
[tokenizer](https://huggingface.co/Qwen/Qwen3.8-27B) repositories:

```bash
loom-serve-setup models import "$MODEL_ID" --alias \
  --hf-repo unsloth/Qwen3.8-27B-GGUF \
  --gguf-file Qwen3.8-27B-UD-Q5_K_XL.gguf \
  --revision 408fcc1807ab264d8cd644a7c4c0f58fbd32eebc \
  --tokenizer-hf-repo Qwen/Qwen3.8-27B \
  --tokenizer-file tokenizer.json \
  --tokenizer-revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
```

Alias files depend on the source/cache files remaining available.

## Start and query locally

The managed start runs in the background, waits for readiness, then prints the
server ID, endpoint, PID, and log path:

```bash
loom-serve-start "$MODEL_ID" --port 8080
loom-serve-setup servers list
loom-serve-setup servers status SERVER_ID
```

Check the model and make a streamed chat request (`stream: true` is required
by this Loom Serve build):

```bash
curl -fsS http://127.0.0.1:8080/v1/models
curl -fsSN http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL_ID\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply exactly: HRX READY\"}],\"max_tokens\":16,\"temperature\":0,\"stream\":true}"
```

Use `--foreground` to run interactively or `--json` for machine-readable start
output. Stop the server with `loom-serve-setup servers stop SERVER_ID`. Server
records are in `~/.loom/servers/` and logs in `~/.loom/logs/`.

## Configure and test Pi locally

Pi must already be installed. Configure it to use the local endpoint, verify the
saved config with `--check`, then make a tool call:

```bash
loom-serve-setup pi-config "$MODEL_ID" \
  --base-url http://127.0.0.1:8080/v1
loom-serve-setup pi-config "$MODEL_ID" \
  --base-url http://127.0.0.1:8080/v1 --check

pi --provider hrx-loom --model "$MODEL_ID" \
  --no-session --approve --tools bash --print \
  'Use your bash tool to run: printf PI_TOOL_OK. Then report the exact output.'
```

The response should include `PI_TOOL_OK`.

## Connect Pi to a remote server

The remote host must already have this wheel installed, the model imported and
validated, and SSH access configured. The client uses a local SSH tunnel; the
server stays bound to remote loopback. The remote noninteractive SSH shell must
have `loom-serve-setup` on its `PATH`.

Register the SSH target and ports on the client. The remote port must match the
server's port; the local port is opened on the client:

```bash
REMOTE_NAME="gpu-box"
MODEL_ID="qwen38-27b-q5-k-xl"
loom-serve-setup remote register "$REMOTE_NAME" \
  --ssh-target gpu-box --server-port 8080 --local-port 18080
```

List remote models, start one, and connect the tunnel. Start waits for server
readiness and returns a server ID; connect waits for the requested model to be
advertised at the forwarded endpoint:

```bash
loom-serve-setup remote models "$REMOTE_NAME"
loom-serve-setup remote start "$REMOTE_NAME" "$MODEL_ID"
loom-serve-setup remote connect "$REMOTE_NAME" --model "$MODEL_ID"
loom-serve-setup remote connections list
```

Query through the tunnel, configure Pi on the client, and test a tool call:

```bash
curl -fsSN http://127.0.0.1:18080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL_ID\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply exactly: REMOTE READY\"}],\"max_tokens\":16,\"temperature\":0,\"stream\":true}"

loom-serve-setup remote pi-config "$REMOTE_NAME" "$MODEL_ID" --check-connection
pi --provider hrx-loom --model "$MODEL_ID" \
  --no-session --approve --tools bash --print \
  'Use your bash tool to run: printf REMOTE_PI_OK. Then report the exact output.'
```

The response should include `REMOTE_PI_OK`. Stop the server and tunnel
separately; disconnecting does not stop the remote server:

```bash
loom-serve-setup remote connections disconnect CONNECTION_ID
loom-serve-setup remote servers "$REMOTE_NAME" list
loom-serve-setup remote servers "$REMOTE_NAME" stop SERVER_ID
```

Remote server records and logs live under `~/.loom/` on the remote host;
remote definitions and tunnel records live under `~/.loom/` on the client.
