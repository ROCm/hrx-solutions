# LLM serving

Run a Qwen model on a local AMD GPU, query its streaming chat endpoint, and
connect Pi.

For the implementation and extension points, see the
[LLM pipeline guidebook](../../loom-pipeline/llm/README.md). The steps below
use Bash and the `hrx-solutions` command-line tools.

## Requirements

On the machine running inference:

* Linux x86-64, Python 3.9 or newer, and an installed
  [HRX Solutions wheel](../../README.md#install-from-pypi).
* An AMD GPU accessible through the installed driver. The pinned Qwen sources
  have been exercised upstream on `gfx1100` and `gfx1151`; other devices need
  validation. The wheel includes its ROCm runtime libraries; running it does
  not require a separate ROCm SDK installation.
* Enough GPU-accessible memory for the weights, retained model state, and
  execution workspace, plus disk space for the downloaded checkpoint.
* The GGUF checkpoint and matching `tokenizer.json` below, or another pair
  validated against this pipeline. The current catalog specializes the
  Qwen3.8-27B architecture and UD-Q5_K_XL tensor layouts. A `qwen` pipeline ID
  does not imply support for every Qwen checkpoint or quantization.

Activate the environment containing the wheel before running these commands.
Keep model files outside the source checkout.

## 1. Import a model

A **model ID** names one installed checkpoint or fine-tune and becomes its
HTTP identity. A **pipeline ID** selects the implementation that can execute
it. Here the model ID is `qwen38-27b-q5-k-xl`, and the pipeline is `qwen`.

Choose one of the following import methods. An existing model ID cannot be
imported again; use a new ID for a different checkpoint.

### Download from Hugging Face

Install the `hf` CLI into your environment:

```bash
python -m pip install -U huggingface_hub
```

For repositories requiring authentication, run `hf auth login` or set
`HF_TOKEN` before importing. This example pins the
[GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) and
[tokenizer](https://huggingface.co/Qwen/Qwen3.8-27B) to separate revisions:

```bash
MODEL_ID="qwen38-27b-q5-k-xl"
loom-serve-setup models import "$MODEL_ID" --alias \
  --hf-repo unsloth/Qwen3.8-27B-GGUF \
  --gguf-file Qwen3.8-27B-UD-Q5_K_XL.gguf \
  --revision 408fcc1807ab264d8cd644a7c4c0f58fbd32eebc \
  --tokenizer-hf-repo Qwen/Qwen3.8-27B \
  --tokenizer-file tokenizer.json \
  --tokenizer-revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
```

`--alias` creates symlinks to the Hugging Face cache, avoiding a second copy
of the weights. Keep the referenced cache files available. Omit `--alias` to
copy the downloaded files into the model store.

### Use files already on disk

Supply the GGUF and its matching tokenizer. Replace these example paths:

```bash
MODEL_ID="qwen38-27b-q5-k-xl"
loom-serve-setup models import "$MODEL_ID" --alias \
  --gguf /path/to/Qwen3.8-27B-UD-Q5_K_XL.gguf \
  --tokenizer /path/to/tokenizer.json
```

Omit `--alias` to copy the files. Both methods create a model directory under
`~/.loom/models/$MODEL_ID/` containing `model.gguf`, `tokenizer.json`, and a
manifest recording the import.

## 2. Validate compatibility

```bash
MODEL_ID="qwen38-27b-q5-k-xl"
loom-serve-setup models validate "$MODEL_ID" --pipeline qwen
loom-serve-setup models list
```

Validation runs the bundled Qwen correctness checker on the GPU and records
the pipeline after success. It can take time to load weights and compile
kernels; the default timeout is 1,800 seconds. This checks execution against
the pipeline, beyond the file-format checks performed during import.

## 3. Start the server

```bash
MODEL_ID="qwen38-27b-q5-k-xl"
loom-serve-start "$MODEL_ID" --port 8080
loom-serve-setup servers list
```

The server runs in the background. Start waits for readiness, then prints the
server ID, endpoint, PID, and log path. It uses the pipeline recorded during
validation and advertises the imported model ID at `/v1/models`.

The default settings are four resident rows, an 8,192-token context capacity,
a 32,768-token shared pool, a 128-token prefill capacity, and MTP enabled at
depth three. Use `loom-serve-setup serve --help` for configuration options.
For example, `--no-mtp` disables multi-token prediction, and
`--ready-timeout 600` allows more time for startup. Add `--foreground` to run
interactively; use Ctrl-C to stop that process.

## 4. Send a chat request

This block also works in a fresh terminal. If you chose a different ID, set
`MODEL_ID` to the exact name returned by `/v1/models`:

```bash
MODEL_ID="qwen38-27b-q5-k-xl"
curl -fsS http://127.0.0.1:8080/v1/models
curl -fsSN http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  --data-binary @- <<EOF
{
  "model": "$MODEL_ID",
  "messages": [{"role": "user", "content": "Reply exactly: HRX READY"}],
  "max_tokens": 16,
  "temperature": 0,
  "stream": true
}
EOF
```

Expect server-sent events containing response chunks and a final `[DONE]`.
The current endpoint requires `stream: true` and supports greedy generation
(`temperature: 0`). It implements the chat interface used here; clients must
use those supported options.

### Inspect and stop a server

Replace `SERVER_ID` with the ID printed at startup or by `servers list`:

```bash
loom-serve-setup servers status SERVER_ID
loom-serve-setup servers stop SERVER_ID
```

Server records live in `~/.loom/servers/`, and logs in `~/.loom/logs/`.
Start and server-management commands accept `--json` for automation.

## Connect Pi locally

With Pi already installed and the server running, configure its provider:

```bash
MODEL_ID="qwen38-27b-q5-k-xl"
loom-serve-setup pi-config "$MODEL_ID" \
  --base-url http://127.0.0.1:8080/v1
loom-serve-setup pi-config "$MODEL_ID" \
  --base-url http://127.0.0.1:8080/v1 --check

pi --provider hrx-loom --model "$MODEL_ID"
```

The setup command updates `~/.pi/agent/models.json`. `--check` verifies the
saved configuration without changing it; the chat request above checks the
live server. In Pi, try asking it to use its Bash tool to run
`printf PI_TOOL_OK` and report the output.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| `loom_route_error` | Read `/v1/models` and send that exact model ID. Define `MODEL_ID` in the terminal issuing the request. |
| Missing model files or broken aliases | Check `models list` and restore the original files or Hugging Face cache referenced by the import. |
| Unsupported format with a hash as its name | Preserve a filename ending in `.gguf`. Use the imported model path or a named Hugging Face snapshot path; resolving a snapshot symlink can expose an extensionless cache blob. Update older launchers that resolve these paths. |
| Validation fails | Read the checker output. Check the driver/device access and the exact checkpoint/tokenizer pair before changing the pipeline. |
| Startup fails or times out | Read the log path from startup diagnostics or `servers status SERVER_ID`. Check memory availability, compilation errors, and port ownership. Increase `--ready-timeout` if startup needs more time. |
