# HRX Solutions Agent Guide

This repository, `hrx-solutions`, builds and publishes the `hrx-solutions` Python
wheel for the Loom Serve server from the HRX commit pinned in `MODULE.bazel`.

Keep this repository focused on the wheel, its thin Python launcher, pinned
build inputs, release workflows, and the selected model assets shipped with
the wheel. The Qwen `.loom` runtime catalog is intended to be maintained here;
the generic Loom Serve runtime and model tests remain in HRX, and checkpoints
remain external.

Do not commit downloaded model weights, generated serving outputs, or machine-
specific paths. Do not change the HRX source pin without updating `MODULE.bazel`
and reviewing the resulting wheel build.

## Verification

Before creating a commit on `main` or landing a commit there, run
`bash build_tools/local-ci.sh` against the final intended source tree and require
exit 0. Use the default full GPU/Pi run; CPU-only qualifies only when the user
explicitly approves reduced coverage. Record the exact source identity, command,
result and evidence outside the checkout, and associate that record with the
commit message or status document. Source changes after the run require a rerun.
This verification requirement does not grant commit permission.
