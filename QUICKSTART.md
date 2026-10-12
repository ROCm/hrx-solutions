# HRX Solutions Quickstart

This guide is written for an agent setting up one local Loom Serve model. Use a
unique `MODEL_ID` for each checkpoint or fine-tune. The model ID identifies the
installed files; `qwen` is the pipeline that validates and serves them.

Agent checklist: use the GGUF and tokenizer the user selected, keep model files
outside the repository, validate before starting, and leave the server running
while testing clients. Do not infer compatibility from a model name.

## 1. Install the wheel

Use Linux x86-64 with a supported AMD GPU and driver. Download the
`hrx-solutions` wheel from the [latest GitHub Release](https://github.com/ROCm/hrx-solutions/releases/latest),
then install it in a Python environment:

```bash
python3 -m venv "$HOME/.venvs/hrx-solutions"
source "$HOME/.venvs/hrx-solutions/bin/activate"
python -m pip install ./hrx_solutions-*.whl
```

The wheel contains the server and Qwen pipeline sources, but not model weights.
Keep model files outside the repository.

## 2. Import a model

Choose a unique ID for this model. The example below uses the Qwen3.8 27B
`UD-Q5_K_XL` checkpoint and its `tokenizer.json`; this combination passed the
Qwen compatibility check in the local end-to-end run.

```bash
MODEL_ID="qwen38-27b-q5-k-xl"
MODEL_DIR="/path/to/qwen3.8-27b"
GGUF="$MODEL_DIR/Qwen3.8-27B-UD-Q5_K_XL.gguf"
TOKENIZER="$MODEL_DIR/tokenizer.json"
```

### Files already downloaded

By default, import copies the files into `~/.loom/models/$MODEL_ID/`:

```bash
loom-serve-setup models import "$MODEL_ID" \
  --gguf "$GGUF" \
  --tokenizer "$TOKENIZER"
```

Add `--alias` to create symlinks instead of copying the large GGUF. The source
files must stay at their current paths for the alias to work:

```bash
loom-serve-setup models import "$MODEL_ID" --alias \
  --gguf "$GGUF" \
  --tokenizer "$TOKENIZER"
```

### Download from Hugging Face

Install and authenticate the Hugging Face CLI once. The setup tool runs
`hf download` for each requested file and uses the cache paths returned by the
CLI. Pin revisions to commits for repeatable installs. With `--alias`, the
model store symlinks directly to Hugging Face's cache, so it does not make a
second copy of the GGUF.

```bash
python -m pip install -U 'huggingface_hub[cli]'
hf auth login  # optional when HF_TOKEN is already set

loom-serve-setup models import "$MODEL_ID" --alias \
  --hf-repo OWNER/REPOSITORY \
  --gguf-file path/to/model.gguf \
  --tokenizer-file path/to/tokenizer.json \
  --revision COMMIT
```

For separate repositories, set `--tokenizer-hf-repo` and optionally
`--tokenizer-revision`. The known Qwen3.8 GGUF and tokenizer are published in
the [GGUF repository](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) and the
[Qwen tokenizer repository](https://huggingface.co/Qwen/Qwen3.8-27B). These
revisions match the locally tested artifact pair; revalidate if you select
different revisions.

```bash
loom-serve-setup models import "$MODEL_ID" --alias \
  --hf-repo unsloth/Qwen3.8-27B-GGUF \
  --gguf-file Qwen3.8-27B-UD-Q5_K_XL.gguf \
  --revision 408fcc1807ab264d8cd644a7c4c0f58fbd32eebc \
  --tokenizer-hf-repo Qwen/Qwen3.8-27B \
  --tokenizer-file tokenizer.json \
  --tokenizer-revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
```

`hf auth login` credentials and `HF_TOKEN` are available to the CLI. If needed,
the import command also accepts `--token`; avoid placing tokens directly in
committed files.

## 3. Validate and list the model

Validation checks the selected GGUF and tokenizer against the Qwen pipeline.
Only start a model after validation succeeds; a failed validation does not
record a pipeline in the model manifest.

```bash
loom-serve-setup models validate "$MODEL_ID" --pipeline qwen
loom-serve-setup models list
```

The list should show the chosen model ID, `pipeline=qwen`, source, install mode,
and availability. Validation is specific to the checkpoint contents, not just
the model's name. In the local check, `Qwen3.8-27B-UD-Q5_K_XL.gguf` passed;
the tested Q4_K_M and UD-Q4_K_XL files did not match this pipeline.

## 4. Start and query the server

Start the server in a terminal and leave it running while clients connect:

```bash
loom-serve-start "$MODEL_ID" --port 8080
```

The server advertises the installed model ID. In a second terminal, confirm
that it is ready and request a streamed completion:

```bash
curl -fsS http://127.0.0.1:8080/v1/models

curl -sSN http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL_ID\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply exactly: HRX READY\"}],\"max_tokens\":16,\"temperature\":0,\"stream\":true}"
```

Loom Serve expects streaming chat completion requests. Stop the server with
Ctrl-C when finished.

## 5. Configure and test Pi

Pi is an optional client and must already be installed. With the server
running, write Pi's model configuration and verify it:

```bash
loom-serve-setup pi-config "$MODEL_ID" \
  --base-url http://127.0.0.1:8080/v1 \
  --context-window 8192 \
  --max-tokens 512

loom-serve-setup pi-config "$MODEL_ID" \
  --base-url http://127.0.0.1:8080/v1 \
  --context-window 8192 \
  --max-tokens 512 \
  --check
```

Run Pi against the configured model:

```bash
pi --provider hrx-loom --model "$MODEL_ID" \
  --no-session --approve --tools bash --print \
  'Use your bash tool to run: printf PI_TOOL_OK. Then report the exact output.'
```

The response should include `PI_TOOL_OK`. This checks that Pi can reach Loom
and use a coding-agent tool through the configured model.
