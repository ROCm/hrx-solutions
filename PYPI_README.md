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
