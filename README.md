# HRX Solutions

This repository builds and publishes a Python wheel for the experimental Loom
Serve text server from HRX. The wheel contains the native server, its source
catalog, and the ROCm runtime libraries required by the server. Model weights
remain external inputs.

## Release contents

- `loom-serve` starts HRX's text server and defaults to the packaged Qwen source
  catalog. Pass the model weights and tokenizer as command-line arguments.
- HRX source is pinned in [`deps.lock`](deps.lock) to commit
  `6f6176173bb543d0c330a35b78a1837582a76870`.
- GitHub Actions builds a manylinux x86-64 wheel, checks the wheel contents,
  and attaches it to a GitHub Release for a semantic-version tag.

The wheel is published as `hrx-solutions`. It is a platform-specific Linux wheel,
although its Python ABI tag is `py3-none`; it includes native AMDGPU code and
ROCm libraries.

## Use

Download the wheel from a GitHub Release, then install it on a Linux x86-64
machine with a supported AMD GPU and driver:

```bash
python -m pip install ./hrx_solutions-*.whl
loom-serve \
  --weights=/path/to/Qwen3.8-27B-UD-Q5_K_XL.gguf \
  --tokenizer=/path/to/tokenizer.json \
  --port=8080
```

The wheel does not include model weights. The selected checkpoint must match
the model source shipped in the wheel.

## Build locally

Install Bazelisk 1.29.0, select the ROCm SDK, then configure and build the
wheel:

```bash
python build_tools/setup_python.py --rocm=/path/to/rocm
python -m pip install build
python -m build --wheel
```

The build fetches the HRX commit pinned in `deps.lock` and builds the Loom Serve
server with HRX's Bazel workspace. Set `HRX_SOLUTIONS_PACKAGE_VERSION` to override
the default wheel version `0.1.0`.

## License

This repository's Python launcher and packaging code is licensed under MIT;
see [LICENSE](LICENSE). The wheel also bundles HRX's Loom
Serve binary and source catalog and ROCm runtime libraries, which retain their
upstream licenses. Their notices are included with the wheel. Model weights
retain their respective licenses and are not redistributed here.
