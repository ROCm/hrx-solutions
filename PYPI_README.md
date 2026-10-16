# HRX Solutions

Executable documentation for building accelerated applications on HRX
(HIP Runtime eXtended), a unified compute platform for ROCm. Worked solutions
define application structure and model policy in Loom source, backed by
reusable compiler, execution, and memory services. Developers and agents can
run, inspect, and adapt these examples for their own workloads.

The current package provides Loom Serve for streaming LLM chat on AMD GPUs.
It includes the native server, Python GGUF compatibility validation, the Qwen
Loom source catalog, and the required ROCm runtime libraries. Model weights
are downloaded separately.

The serving solution makes model computation, weight preparation, and chat
policy explicit in source while sharing native scheduling and execution
infrastructure. Its [implementation guide](https://github.com/ROCm/hrx-solutions/blob/main/loom-pipeline/llm/README.md)
explains the boundaries and how to change them.

Use `loom-serve` directly to start an empty foreground server, register a
handwritten deployment document over HTTP, activate the model, and open admission
after readiness checks. Configure Pi manually against the known endpoint.

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
pipeline sources, and ROCm libraries retain their upstream licenses and
notices. Model weights are not redistributed.
