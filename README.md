# HRX Solutions

HRX Solutions packages Loom Serve as a Python wheel for Linux x86-64 systems
with a supported AMD GPU and driver. The wheel includes the native server,
selected source catalog, and required ROCm libraries. Model weights are
installed separately.

## Release

HRX source is pinned in [`deps.lock`](deps.lock) to
`6f6176173bb543d0c330a35b78a1837582a76870`. GitHub Actions builds and checks a
manylinux wheel and attaches it to GitHub Releases for semantic-version tags.
The wheel has a `py3-none` Python ABI tag but contains platform-specific AMDGPU
code and ROCm libraries.

Install a release wheel, then follow [QUICKSTART.md](QUICKSTART.md) to import,
validate, and serve a model:

```bash
python -m pip install ./hrx_solutions-*.whl
loom-serve-setup models import MODEL_ID --gguf /path/model.gguf \
  --tokenizer /path/tokenizer.json
loom-serve-setup models validate MODEL_ID --pipeline qwen
loom-serve-start MODEL_ID --port 8080
```

`loom-serve-start` and `loom-serve-setup serve` start a managed background
server by default, wait for API readiness, and print its ID, endpoint, PID, and
log path. Use `loom-serve-setup servers list|status|stop` to manage it; pass
`--foreground` for an interactive process or `--json` for machine-readable
output. Records are stored in `~/.loom/servers/` and logs in `~/.loom/logs/`.

The setup tool also supports Hugging Face downloads, alias imports, Pi
configuration, and remote serving through a tracked SSH tunnel. See the
[quickstart](QUICKSTART.md) for these workflows and cleanup commands.

## Build locally

Install Bazelisk 1.29.0, select the ROCm SDK, and build the wheel:

```bash
python build_tools/setup_python.py --rocm=/path/to/rocm
python -m pip install build
python -m build --wheel
```

The build uses the HRX commit pinned in `deps.lock`. Set
`HRX_SOLUTIONS_PACKAGE_VERSION` to override the default version `0.1.0`.

## License

The Python launcher and packaging code are MIT licensed; see [LICENSE](LICENSE).
The bundled Loom Serve binary, source catalog, and ROCm libraries retain their
upstream licenses and notices. Model weights are not redistributed.
