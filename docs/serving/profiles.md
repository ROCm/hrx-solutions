# Model profiles and qualification

The [walkthrough](../solutions/llm-serving.md) configures a known model and Pi
endpoint. Configuration correctness and model suitability are separate claims.

## Small-model connection profile

Qwen3.5 0.8B UD-Q5_K_XL uses the packaged `qwen-08b` pipeline, one execution
row, context/pool 8192, prefill 128, default output 512 and MTP disabled. Its
manual JSON, drained registration, activation/readiness, admission, bounded
streaming inference and retirement were exercised on Linux x86-64 with an AMD
gfx1151 HSA device. The wheel was `0.4.0.dev0`, Python 3.14.4 and HRX pin
`ed6b79c757c738dea7eaecd6ca55a7df73cf6185`. This is evidence for that tested
configuration, not every AMD device or wheel build.

Pi 1.3.0 static configuration was checked with Node 22.22.1. A short request
passed with tools/extensions/skills/templates disabled and a minimal custom
system prompt. Those restrictions are preserved in the
[connection diagnostic](troubleshooting.md#check-pi-connectivity).

Ordinary Pi coding-agent use is **unqualified**. A normal tool task failed;
a restricted read-only retry repeatedly read without a final answer. Longer
histories, compaction, branch summaries and overflow recovery remain unverified.
Larger context budgets alone did not qualify the small model. Use a capable
model and test the intended prompt/tools/history before relying on agent work.

The [manual Pi provider settings](../solutions/llm-serving.md#connect-pi)
match deployed capacities rather than architectural maxima. Pi's default
compaction reserve 16384 and recent-history budget 20000 exceed this fixture;
the example scopes compaction reserve 1024/recent 2048 to this provider/model.
Branch-summary reserve 1024 is agent-wide. Matching budgets prevents misleading
capacity metadata; it does not prove useful summaries.

Pi configuration references are pinned to 1.3.0:
[models/authentication](https://github.com/earendil-works/pi/blob/v1.3.0/packages/coding-agent/docs/models.md),
[settings](https://github.com/earendil-works/pi/blob/v1.3.0/packages/coding-agent/docs/settings.md),
and [CLI](https://github.com/earendil-works/pi/blob/v1.3.0/packages/coding-agent/docs/cli.md).

## Qwen3.8 27B extension

The packaged `qwen` pipeline lives at `loom-pipeline/llm/qwen/qwen_27b`.
Its `compatibility.json` pins checkpoint/tokenizer independently; use its own
`pipeline.json` descriptor and single `pipeline.loombc`, keeping all preparation,
control and provider definitions together. The small-model example does not qualify 27B memory fit or agent
use on the same hardware.

The example name `qwen3.8-27b` can also be used for 27B: retire the small-model
registration before reusing that name, or choose a distinct name. It supports
`draft_depth` 1–3 for MTP or zero to disable it. Prefill capacity must fit one anchor plus all draft proposals and must not
exceed context capacity. Select rows, context and pool for the target hardware;
then repeat readiness, bounded inference, intended Pi workload and retirement
checks with those exact assets before describing the profile as tested.

For normal agent qualification, retain Pi's normal system prompt and intended
tools. In a disposable directory ask it to read a known file, verify actual tool
execution, its result and a final answer; then exercise longer history,
compaction, summaries, cancellation and capacity errors with bounded requests.
A greeting or tool-shaped text alone is insufficient.
