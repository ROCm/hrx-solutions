# HRX Solutions

Executable documentation for building accelerated applications on HRX
(HIP Runtime eXtended), a unified compute platform for ROCm. Worked solutions
define application structure and model policy in Loom source, backed by
reusable compiler, execution, and memory services. Developers and agents can
run, inspect, and adapt these examples for their own workloads.

The current package provides Loom Serve for streaming LLM chat on AMD GPUs.
It includes native server and compatibility-checker executables, the Qwen
Loom source catalog, and the required ROCm runtime libraries. Model weights
are downloaded separately.

The serving solution makes model computation, weight preparation, and chat
policy explicit in source while sharing native scheduling and execution
infrastructure. Its [implementation guide](https://github.com/ROCm/hrx-solutions/blob/main/loom-pipeline/llm/README.md)
explains the boundaries and how to change them.

## Install

On Linux x86-64 with Python 3.9 or newer, preferably in a virtual environment:

```bash
python -m pip install hrx-solutions
loom-serve-setup --help
loom-serve-setup models pipelines
loom-serve-start --help
```

Inference requires a supported AMD GPU and driver. Follow the
[LLM serving walkthrough](https://github.com/ROCm/hrx-solutions/blob/main/docs/solutions/llm-serving.md)
for hardware coverage, checkpoint requirements, import and validation,
streaming chat, and Pi configuration.

## Learn more

- [Project overview](https://github.com/ROCm/hrx-solutions#readme)
- [Pipeline implementation guide](https://github.com/ROCm/hrx-solutions/blob/main/loom-pipeline/llm/README.md)
- [Release assets](https://github.com/ROCm/hrx-solutions/releases)
- [HRX platform](https://github.com/ROCm/hrx-system)

The Python launcher and packaging code are MIT licensed. Bundled native code,
pipeline sources, and ROCm libraries retain their upstream licenses and
notices. Model weights are not redistributed.
