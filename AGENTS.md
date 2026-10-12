# HRX Solutions Agent Guide

This repository, `hrx-solutions`, builds and publishes the `hrx-solutions` Python
wheel for the Loom Serve text server from the HRX commit pinned in `deps.lock`.

Keep this repository focused on the wheel, its thin Python launcher, pinned
build inputs, release workflows, and the selected model assets shipped with
the wheel. The Qwen `.loom` runtime catalog is intended to be maintained here;
the generic Loom Serve runtime and model tests remain in HRX, and checkpoints
remain external.

Do not commit downloaded model weights, generated serving outputs, or machine-
specific paths. Do not change the HRX source pin without updating `deps.lock`
and reviewing the resulting wheel build.
