# HRX Solutions

HRX Solutions is executable documentation for building accelerated applications
on [HRX (HIP Runtime eXtended)](https://github.com/ROCm/hrx-system), a unified
compute platform for ROCm. Its worked solutions define application structure
and model policy in Loom source, backed by reusable compiler, execution, and
memory services. They provide runnable applications, implementation patterns,
and components that developers and agents can adapt to their own workloads.

Start with [LLM serving](docs/solutions/llm-serving.md) to run a local model
and connect an agent. HRX is a new subsystem being evaluated for inclusion in
ROCm, providing an alternative implementation of the runtime, compiler, and
library stack.

## How the pieces fit

![HRX platform architecture: Loom compiler and C API, libhrx and HIP APIs, shared runtime, and pluggable GPU, NPU, and CPU HAL modules.](docs/assets/hrx-platform.svg)

*HRX platform architecture. The [current serving example](docs/solutions/llm-serving.md#current-example)
documents the hardware, model, and interface coverage of this solution.*

## What the serving solution demonstrates

The LLM solution brings model computation, retained request state, speculative
decoding, and streaming chat together through explicit ownership boundaries:

* **Source-defined policy.** Loom sources own model geometry, weight
  preparation, execution programs, state layouts, and chat/tool conventions.
  Those choices can be inspected and changed alongside the model.
* **Shared execution infrastructure.** Native HRX owns admission, scheduling,
  transport, physical storage, and execution services. Requests share weights,
  compiled commands, and execution infrastructure while retaining independent
  state.
* **Deployment specialization.** The server loads a source catalog and compiles
  for the device and configured capacities. Model-policy changes within the
  existing execution contract can reuse the native server executable.

The [pipeline guidebook](loom-pipeline/llm/README.md) follows these boundaries
from import to generated tokens and shows where to make and check a change.

Existing projects can adopt HRX components, specialized stacks can build on
the substrate, and these reference solutions explore what becomes possible
when application and substrate are developed together. The
[proposed llama.cpp integration](https://github.com/ggml-org/llama.cpp/discussions/27219)
is one example of adopting the technology within an established project.

We use these solutions ourselves and maintain a selective catalog to make
useful platform capabilities accessible. New examples should demonstrate
something worth building on. Some may eventually become products; direct use
and adaptation are encouraged.

## Solution Catalog

| Solution | What you can do |
| --- | --- |
| [LLM serving](docs/solutions/llm-serving.md) | Import and validate a model, serve streaming chat locally, and configure Pi. |

Image serving, audio, NPU workloads, and interactive applications such as [Doom](https://gist.github.com/benvanik/20fbfd298ab1d53588ad12aedade6457)
are candidates for future worked solutions.

## Install from PyPI

PyPI publishing is pending. Once available, install in a Python 3.9 or newer
virtual environment:

```bash
# Installing in a venv is recommended.
python -m pip install hrx-solutions

# Discover setup commands and available pipelines.
loom-serve-setup --help
loom-serve-setup models --help
loom-serve-setup models pipelines

# Explore serving options for an imported model.
loom-serve-start --help
loom-serve-setup serve --help
```

Continue with the [LLM serving walkthrough](docs/solutions/llm-serving.md)
to import a checkpoint, validate it, and start the server.

## Agent Guidebooks

Guidebooks trace execution through the source and explain the design choices
behind each solution. Start here when modifying an implementation:

* [LLM pipelines](loom-pipeline/llm/README.md): model import and validation,
  source JIT, request flow, state ownership, kernels, and extension points.

## HRX design principles

* **Compact and embeddable.** The native stack supports static linking and
  redistribution. The serving wheel bundles its user-space ROCm runtime
  libraries; deployment needs a compatible driver, while building the wheel
  requires the SDK described below.
* **Targets AMD compute hardware.** The platform targets Radeon, Ryzen, and
  Instinct GPUs, XDNA NPUs, and x86 CPUs, including matrix acceleration.
* **Designed for agent authorship.** The [Loom compiler](https://rocm.github.io/hrx-system/loom/)
  accepts [Loom ASM](https://rocm.github.io/hrx-system/loom/reference/) and a
  customized C++ dialect. Explicit state and ownership, inspectable generated
  code, and localized policy changes let agents trace an edit through to its
  behavior and validation.
* **Heterogeneous execution.** The programming model supports pipelined and
  resident co-execution across CPU, GPU, and NPU, including local and remote
  compute units.
* **AI components in one place.** Tokenization, serving JIT, KV-cache management,
  weight preparation and streaming, and virtual memory management share one
  native codebase.
* **Integrated optimization tools.** Compilation reports, profiling,
  sanitizers, tests, and benchmarks connect source choices to generated code
  and observed execution.
* **HIP compatibility.** The HIP implementation enables incremental adoption
  by existing HIP applications.
* **Evolution through use.** We develop and refine the platform with agents
  on real workloads. As approaches mature, we document them and publish
  corresponding solutions.

## Build and Install Locally

From a checkout, with a Python virtual environment activated, install
Bazelisk 1.29.0 on `PATH` and select a ROCm SDK containing the compiler, headers,
and runtime libraries:

```bash
python -m pip install build 'setuptools>=77' 'wheel>=0.45'
python build_tools/setup_python.py --rocm=/path/to/rocm
python -m build --wheel
python -m pip install --force-reinstall dist/hrx_solutions-*.whl
```

Use a `dist/` directory containing just the wheel you want to install, or pass
its exact filename. The setup command checks the Bazel version from
[.bazelversion](.bazelversion) and writes the local SDK selection to the ignored
`.bazelrc.local`. Its `--help` includes commands to install the pinned Bazelisk.

The build fetches the HRX commit pinned in [deps.lock](deps.lock), builds the
native server and compatibility checker, and packages them with this
repository's pipeline sources. The package version comes from
[version.json](version.json) for both local builds and CI. See
[RELEASING.md](RELEASING.md) for release preparation, tags, GitHub assets, and
PyPI trusted publishing.

## License

The Python launcher and packaging code are MIT licensed; see [LICENSE](LICENSE).
Bundled native code, pipeline sources, and ROCm libraries retain their upstream
licenses and notices. Model weights are not redistributed.

Many HRX components were originally derived from [IREE](https://github.com/iree-org/iree)
and use the Apache license with LLVM exception. HRX also borrows or adapts
concepts from MLIR, without using MLIR code. Licenses in individual source
files are authoritative; see the [pipeline notices](loom-pipeline/llm/README.md#license-notices).
