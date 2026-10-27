# Generic modality onboarding structural fixtures

Initial catalog: one CHAT executable and one recipe. Addition DATA: another recipe
on that executable, a second CHAT executable, and a nonpreparing ASR executable.
Names and catalog paths intentionally differ from product model names.

Source declarations reuse accepted current native source closures. Installed
fixtures reuse actual accepted compiled program bytes from a caller-supplied wheel,
verified by the literal hashes here. The fixture manifests represent normalized
artifact DATA, not a claim that the uniform producer has emitted them. Source and
installed manifests, exact source identities, role/member expectations and recipe
coordinates in the two release JSON files are authored before any producer runs.
The harness never generates its release oracle from staged/produced inventories.

Tiny prepared assets exercise bookkeeping/config/export only. They do not prove
model loading, native ABI behavior, throughput or inference. Real compiler/producer
and genuine artifact qualification remain separate until their actual APIs are
composed. No mocked reader/selector/checker or source family dispatch is used.

The two *-build.bzl files are literal fixture build DATA for the actual uniform
producer. Its separate harness case reuses these accepted compiled inputs and
executes the caller-supplied real Loom decoder (`GENERIC_MODALITY_FIXTURE_DECODER`).
It calls real prepare/stage and the strict independent installed checker, without
filtering producer receipts or deriving release expectations from its output.
`GENERIC_MODALITY_FIXTURE_WHEEL` supplies the existing immutable bytecode inputs.
