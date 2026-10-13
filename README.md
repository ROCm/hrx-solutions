# HRX Solutions

Worked solutions built on [HRX (HIP Runtime eXtended)](https://github.com/ROCm/hrx-system),
with runnable examples and implementation guidebooks for humans and agents.
Start with [LLM serving](docs/solutions/llm-serving.md) to run a local model,
and connect an agent.

HRX is a new subsystem being evaluated for inclusion in ROCm. It provides an
alternative to the traditional HIP runtime, compiler, and library stack,
organized around these principles:

* **Minimal.** HRX has no build dependencies beyond the C compiler and can
  build in about a minute on a typical desktop. With all features enabled,
  its binaries are roughly 10–15 MiB, support static linking, and can be
  redistributed to systems with device drivers but no ROCm SDK installation.
* **Targets AMD compute hardware.** The compiler and runtime target GPUs
  (Radeon, Ryzen, and Instinct), NPUs (including those in Strix Halo), and x86
  CPUs, including their matrix acceleration facilities.
* **Designed for agent authorship.** The embedded
  [Loom compiler](https://rocm.github.io/hrx-system/loom/) accepts
  [Loom ASM](https://rocm.github.io/hrx-system/loom/reference/) and a customized
  C++ dialect. Both were developed through months of agent-driven iteration
  on real workloads, balancing human legibility, agent authoring and debugging,
  and explicit control over hardware. Constraint-based parameterization supports
  dynamic and multiple-target programs. High-level vector operations and
  machine-level instructions can coexist in the same sources, alongside tests,
  benchmarks, and tuning parameters.
* **Heterogeneous execution.** The programming model supports pipelined and
  resident co-execution across GPU, NPU, and CPU, including combinations of
  local and remote compute units over frontside and backside networks.
* **AI components in one place.** Streaming tokenization, a dedicated LLM
  serving JIT, KV-cache management, parameter prepacking, weight streaming,
  and virtual memory management live in the same monorepo.
* **Integrated optimization tools.** Compilation reports and profiling expose
  generated code and expected or observed behavior to agents. Built-in
  sanitizers provide access checks and other diagnostics through compiler flags.
* **HIP compatibility.** HRX provides a high-fidelity HIP implementation for
  incremental migration of existing HIP applications.
* **Evolution.** We don't just sit down and spec any of these features. We trial
  them and codesign them with agents in the loop over many months. As approaches
  mature, we document them and publish correspnding solutions.

## How the pieces fit

![HRX platform architecture: Loom compiler and C API, libhrx and HIP APIs, shared runtime, and pluggable GPU, NPU, and CPU HAL modules.](docs/assets/hrx-platform.svg)

This repository is executable documentation: complete solutions that agents
can run, inspect, and adapt to new workloads. We use these tools ourselves for
AI serving, and maintain the examples to make more of the platform accessible
without implementing every possible application ourselves. The guidebooks
explain the design decisions and the boundaries where a solution can change.

The approach complements narrower integrations, such as the
[HRX integration proposed for llama.cpp](https://github.com/ggml-org/llama.cpp/discussions/27219).
Some solutions may eventually become products. In the meantime, we encourage
direct use, exploration, and changes for the applications you want to build.

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

## Solution Catalog

| Solution | What you can do |
| --- | --- |
| [LLM serving](docs/solutions/llm-serving.md) | Import and validate a model, serve streaming chat locally, and configure Pi. |

Image serving, audio, NPU workloads, and interactive applications such as [Doom](https://gist.github.com/benvanik/20fbfd298ab1d53588ad12aedade6457)
are candidates for future worked solutions.

## Agent Guidebooks

Guidebooks trace execution through the source and explain the design choices
behind each solution. Start here when modifying an implementation:

* [LLM pipelines](loom-pipeline/llm/README.md): model import and validation,
  source JIT, request flow, state ownership, kernels, and extension points.

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
