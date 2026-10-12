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

See the [quickstart guide](QUICKSTART.md) for wheel installation, local and
Hugging Face model imports, alias mode, serving, and Pi configuration.

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

The `loom-serve-setup` command imports files into `~/.loom/models/<model-id>/`,
validates a selected model against a registered pipeline, starts the server,
and writes Pi's provider configuration. For example:

```bash
# Import local files.
loom-serve-setup models import qwen-7b-instruct \
  --gguf /models/Qwen.gguf --tokenizer /models/tokenizer.json

# Register local files without copying them into the model store.
loom-serve-setup models import qwen-local --alias \
  --gguf /models/Qwen.gguf --tokenizer /models/tokenizer.json

# Download selected files from the Hugging Face cache (requires `hf`).
loom-serve-setup models import qwen-local --hf-repo owner/repository \
  --gguf-file model.gguf --tokenizer-file tokenizer.json --revision main

# Hugging Face aliases symlink the returned cache paths into the model store.
loom-serve-setup models import qwen-7b-instruct --alias --hf-repo owner/repository \
  --gguf-file model.gguf --tokenizer-file tokenizer.json --revision main

loom-serve-setup models list
loom-serve-setup models pipelines
loom-serve-setup models validate qwen-7b-instruct --pipeline qwen
loom-serve-setup serve qwen-7b-instruct --port 8080
loom-serve-setup pi-config qwen-7b-instruct \
  --base-url http://127.0.0.1:8080/v1 \
  --output ~/.pi/agent/models.json
```

Choose a unique model ID for each imported fine-tune, even when several models
use the same architecture. Model IDs identify installed weights; pipeline IDs
select an installed implementation and source catalog. The command workflow is
pipeline agnostic; `qwen` is the initial registered compatibility adapter.
Validation uses the selected adapter's checker and records that pipeline in
the model manifest only after a successful check. The list shows each model's
ID, pipeline (or `unvalidated`), source, storage mode, and availability. Serve
uses the recorded pipeline by default; `--pipeline` explicitly overrides it.
The running endpoint advertises the installed model ID, so Pi and other
OpenAI-compatible clients can address each fine-tune independently.
Older manifests without a recorded pipeline require validation or an explicit
override before serving. Pi remains an optional client installation;
`pi-config` only writes its model configuration.

Hugging Face imports require the Hugging Face CLI. Install it with
`python -m pip install -U 'huggingface_hub[cli]'`. `hf download` obtains each requested file and
returns its path in the standard Hugging Face cache. Alias mode records that
path and symlinks it into the model store. Removing the cached file breaks the
alias; `models list` reports broken aliases, and validation and serving explain
which target is missing.

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
