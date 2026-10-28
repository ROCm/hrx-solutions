# HRX Solutions

Executable documentation for building accelerated applications on HRX
(HIP Runtime eXtended), a unified compute platform for ROCm. Worked solutions
define application structure and model policy in Loom source, backed by
reusable compiler, execution, and memory services. Developers and agents can
run, inspect, and adapt these examples for their own workloads.

The current package provides Loom Serve for streaming LLM chat and finite
Moonshine speech recognition on AMD GPUs. It includes the native server, Qwen
GGUF compatibility checks and two complete Qwen bytecode pipelines, the selected
Moonshine bytecode package and HRX asset preparer, and the required ROCm libraries.
Model weights are downloaded separately.
For Moonshine acquisition, use Python 3.12 or newer and install
`hrx-solutions[moonshine]`; complete prepared local imports do not need that extra.
See [Moonshine speech recognition](https://github.com/ROCm/hrx-solutions/blob/main/docs/solutions/loom-control.md#moonshine-speech-recognition)
for acquisition, transcription and Open WebUI dictation.

The serving solution makes model computation, weight preparation, and chat
policy explicit in source while sharing native scheduling and execution
infrastructure. Its [implementation guide](https://github.com/ROCm/hrx-solutions/blob/main/loom-pipeline/llm/README.md)
explains the boundaries and how to change them.

Use `loom-serve` directly to start an empty foreground server, register a
handwritten deployment document over HTTP, activate the model, and open admission
after readiness checks. Configure Pi manually against the known endpoint.

`loom-control` also provides model acquisition and managed server/instance lifecycle
commands; see the [Loom Control companion guide](https://github.com/ROCm/hrx-solutions/blob/main/docs/solutions/loom-control.md).
Results are readable text by default. Scripts that previously parsed default JSON
should add `--json`; equivalent native/HF/curl commands remain visible on stderr.
Bare commands display guidance, and action `--help` explains configuration choices.
Run `loom-control tui` for an interactive, read-only view of managed services
and live logs. The [Loom Control TUI guide](https://github.com/ROCm/hrx-solutions/blob/main/docs/solutions/loom-control-tui.md)
documents its views, keyboard controls, and current limitations.

## Install

On Linux x86-64 with Python 3.9 or newer, preferably in a virtual environment:

```bash
python -m pip install hrx-solutions
loom-serve --help
```

Inference requires a supported AMD GPU and driver. Follow the
[LLM serving walkthrough](https://github.com/ROCm/hrx-solutions/blob/main/docs/solutions/llm-serving.md)
for checkpoint acquisition, native JSON configuration, streaming chat, manual
Pi setup, SSH forwarding and graceful shutdown. PyPI publication is pending;
use an available release or build wheel with the walkthrough installation steps.

## Learn more

- [Project overview](https://github.com/ROCm/hrx-solutions#readme)
- [Pipeline implementation guide](https://github.com/ROCm/hrx-solutions/blob/main/loom-pipeline/llm/README.md)
- [Release assets](https://github.com/ROCm/hrx-solutions/releases)
- [HRX platform](https://github.com/ROCm/hrx-system)

The Python launcher and packaging code are MIT licensed. Bundled native code,
pipeline modules, and ROCm libraries retain their upstream licenses and
notices. Model weights are not redistributed.

### Pocket TTS Alba

Prepared Alba assets can be imported and served with the base installation. To
use `model download`, create a Python 3.12 environment and explicitly select CPU
Torch before installing the optional preparation dependencies:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install 'hrx-solutions[pocket]'
.venv/bin/hf auth login
.venv/bin/loom-control model download pocket-tts-alba --id pocket
.venv/bin/loom-control server start --name speech --device amdgpu --listen 127.0.0.1:8080
.venv/bin/loom-control instance start pocket --name reader --server speech
```

Accept the checkpoint repository's access terms at
<https://huggingface.co/kyutai/pocket-tts> before downloading. Use `--offline`
when all three pinned inputs are already in the Hugging Face cache. Preparation
runs the wheel's fixed HRX helper offline; it does not install dependencies or
fetch code. The reference WAV is used only during preparation and is not required
for hosting. Stop with `loom-control instance stop reader --server speech`, then
`loom-control server shutdown speech`.

The recipe pins the full encoder checkpoint from `kyutai/pocket-tts`, the
compatible tokenizer from `kyutai/pocket-tts-without-voice-cloning`, and Alba
MacKenna's Casual reference from `kyutai/tts-voices`. Alba's recording is CC BY
4.0: credit Alba MacKenna and Kyutai, link the source and license, and identify
changes (the helper downmixes, resamples, trims and pads the reference waveform).
The reference is available at
<https://huggingface.co/kyutai/tts-voices/blob/323332d33f997de8394f24a193e1a76df720e01a/alba-mackenna/casual.wav>;
license: <https://creativecommons.org/licenses/by/4.0/>. Model assets stay external
to the wheel; checkpoint/tokenizer repository terms continue to apply.

Pocket read-aloud can use a compatible full Open WebUI deployment with FFmpeg and
pydub preprocessing enabled. Managed configuration requires its explicit TTS
`CACHE_ENABLED` capability; ordinary unpatched 0.12.3 is refused. The locally
reviewed WebUI cache patch is a separate deployment artifact, not installed or
applied by Control. With an administrator token in `OPENWEBUI_TOKEN`:

```sh
loom-control openwebui configure --url http://127.0.0.1:3000 --tts-server speech --tts-instance reader
loom-control openwebui configure --url http://127.0.0.1:3000 --check
loom-control openwebui configure --url http://127.0.0.1:3000 --disable-tts
```

Managed Pocket uses Alba and fixes PCM, speed 1 and empty instructions through the
OpenAI TTS parameters; stored user voice preferences are preserved. Selected TTS
disconnect returns read-aloud to browser speech and preserves chat/STT settings.
The existing `--disable` still disconnects the whole managed integration. A
successful configuration readback does not establish browser playback or hearing.
