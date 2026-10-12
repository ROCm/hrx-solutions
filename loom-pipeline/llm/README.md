# Qwen Loom pipeline sources

This directory contains the Qwen runtime Loom source catalog maintained by
HRX Solutions. The runtime catalog is rooted at `qwen/`; its `sources.txt`
manifest lists the runtime entrypoints, kernels, programs, and shared GGML
Loom motifs required for Qwen serving. Paths in the manifest are relative to
`qwen/`.

## Provenance

The source files are copied byte-for-byte from the HRX repository at commit
[`6f6176173bb543d0c330a35b78a1837582a76870`](https://github.com/ROCm/hrx/tree/6f6176173bb543d0c330a35b78a1837582a76870/experimental/loom_serve).
Qwen sources originate in `experimental/loom_serve/models/qwen/`; shared
quantization motifs originate in `experimental/loom_serve/motifs/ggml/`.
`qwen/sources.txt` retains the pinned runtime source set, with shared motif
paths adjusted to resolve within this repository layout.

## License notices

The vendored HRX source tree is licensed under Apache License 2.0; a copy is
in [`LICENSE-APACHE-2.0`](LICENSE-APACHE-2.0). Individual Loom files retain
their original copyright and SPDX headers. The Loom sources marked
`Apache-2.0 WITH LLVM-exception` are subject to the LLVM exception text at
<https://llvm.org/LICENSE.txt>. These notices apply to these source files
independently of the repository's own license.
