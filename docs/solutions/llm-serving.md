# LLM serving

> **Upstream migration in progress:** The setup/start commands and their
> supporting workflows remain part of this solution. The current wheel uses
> upstream's new `loom-serve` executable. Qwen validation needs a compatible
> external checker (`--checker` or `HRX_QWEN_EPOCH_CHECK`), and managed Qwen
> startup still needs its adapter ported to model JSON documents. The Qwen
> validation/start commands below describe that intended workflow; they are
> not yet an end-to-end working path with the new native server.

Run a Qwen model on a local AMD GPU, query its streaming chat endpoint, and
connect Pi.

For the implementation and extension points, see the
[LLM pipeline guidebook](../../loom-pipeline/llm/README.md). The steps below
use Bash and the `hrx-solutions` command-line tools.

## Current example

This worked solution demonstrates source-defined model execution on the shared
HRX runtime. The table describes the packaged example and its checks:

| Area | Current coverage |
| --- | --- |
| Model | Qwen3.8-27B with UD-Q5_K_XL GGUF tensor layouts and the matching tokenizer, pinned below. The `qwen` pipeline ID selects this implementation. |
| Platform | Linux x86-64 and Python 3.9 or newer. The pinned sources have been exercised upstream on `gfx1100` and `gfx1151`; other devices need validation. |
| Chat | Streaming responses with greedy generation (`stream: true`, `temperature: 0`). The current Qwen policy rejects thinking mode. |
| Tools | Function-tool schemas, generated calls, and assistant/tool follow-up history, with Qwen-specific framing and parsing in Loom source. Pi configuration is included. |
| Speculation | The launcher enables multi-token prediction (MTP) at depth three by default; `--no-mtp` disables it. |
| Compatibility check | `models validate` runs the GPU epoch checker with `--epoch=32:8`, without MTP. HTTP, tool-use, and MTP checks are separate [upstream checks](../../loom-pipeline/llm/README.md#developing-and-checking-a-change). |

See the [pinned upstream text guide](https://github.com/ROCm/hrx/blob/6f6176173bb543d0c330a35b78a1837582a76870/experimental/loom_serve/text/README.md)
for hardware notes and the chat protocol contract. Adapting this example to
another architecture or quantization requires reviewing its source and
validating the new checkpoint/tokenizer pair.

## Requirements

On the machine running inference:

* Linux x86-64, Python 3.9 or newer, and an installed
  [HRX Solutions wheel](../../README.md#install-from-pypi).
* An AMD GPU accessible through a compatible installed driver, with coverage
  as described above. The wheel includes its ROCm runtime libraries; running
  it does not require a separate ROCm SDK installation.
* Enough GPU-accessible memory for the weights, retained model state, and
  execution workspace, plus disk space for the downloaded checkpoint.
* The GGUF checkpoint and matching `tokenizer.json` below, or another pair
  validated against this pipeline.

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

Validation runs the configured Qwen correctness checker on the GPU and records
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
a 32,768-token shared pool, and a 128-token prefill capacity, with MTP as
described above. Use `loom-serve-setup serve --help` for configuration options.
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
