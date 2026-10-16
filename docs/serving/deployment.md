# Deployment and asset reference

The [local walkthrough](../solutions/llm-serving.md) owns the executable setup.
Its complete, editable [qwen08.json](examples/qwen08.json) uses native JSON with
no include syntax or generation step. Replace its three `ABSOLUTE_...` strings;
keep JSON escaping valid for quotes or backslashes in paths.

## Fields

| Field | Meaning and changes |
| --- | --- |
| `name` | Routing registration name. Keep `qwen3.8-27b` to match the packaged source chat identity, including for the 0.8B fixture. An alias alone does not rename source policy. |
| `device` | Nonnegative index into this process's ordered device group, published in `server.resources.devices`; not necessarily an OS GPU ordinal. |
| `admission` | `drained` registers metadata without source I/O, loading weights or allocating model storage. Activation can prepare it while admission remains closed. |
| `base_directory` | Absolute server-local base for relative source references, including `../` providers. |
| `definition.kind` | `text` selects the text service. |
| `prepare`, `control` | Primary VM programs for cold preparation and warm request/execution policy. |
| `sources` | Shared provider closure from this wheel's `qwen_08b/sources.txt`, excluding `prepare.loom`, `control.loom` and `weights.loom`. Preparation selects its weight policy itself. |
| `weights`, `tokenizer` | Absolute server-local paths to the exact checkpoint and matching tokenizer. |
| `rows` | Concurrent execution rows; example 1, native range 1–16. |
| `context_capacity` | Per-row logical token-position limit, example 8192. Input plus requested completion must fit. |
| `pool_capacity` | Shared retained-state token-position capacity, example 8192. Requests and retained state compete for the pool. |
| `prefill_capacity` | Largest automatically specialized packed token shape, example 128; not a prompt-length limit. Must not exceed context capacity. |
| `max_tokens` | Default request output limit, example 512. Clients can supply their request's bounded output limit. |
| `draft_depth` | Zero disables MTP; required for 0.8B. The packaged 27B pipeline supports depths 1–3. |
| `options` | Source-owned preparation/request defaults; `{}` in this example. Native configuration does not interpret arbitrary source option keys. |

Keep the complete installed `loom-pipeline/llm` tree. Copying only `qwen_08b`
breaks shared providers. Provider order supplies the linker, not an execution
schedule. A different wheel may change this closure: compare its manifest before
reusing an older example. Do not edit installed package sources to rename a model.

Capacities are logical limits, not a guarantee that weights, retained state and
execution workspace fit on a device. Changing capacities requires retirement
and registration again. Server device/listener settings have process lifetime;
model settings have registration lifetime.

Native JSON rejects unknown fields, duplicate keys and wrong types. HTTP paths
are resolved on the server. An absolute `base_directory` is necessary here
because an HTTP document has no containing file directory. Posting JSON does
not copy checkpoint files or sources. Preserve dependencies through retirement;
stop the server before replacing the installed wheel.

## Assets

The `qwen-08b` compatibility profile is
`qwen35-08b-ud-q5-k-xl-v1`, installed beside the sources as
`qwen_08b/compatibility.json`. The pinned download coordinates in the walkthrough
identify its known asset set. Checkpoint and tokenizer repositories/revisions
are independent.

The contract describes architecture metadata, tensor shapes/types and tokenizer
configuration, including a canonical tokenizer hash. A similar filename or
quantization label is insufficient. This manual does not add a structural
validation CLI or claim a full checkpoint payload hash. Native preparation can
catch loading failures; a short successful request does not establish arbitrary
checkpoint support, numerical correctness or agent quality.

`hf download --local-dir models` places files in the selected directory. If you
instead reference symlinks into a shared HF cache, eviction or moving the cache
can invalidate a registration. Retain stable inputs or copy them to independent
storage before loading. Install `huggingface_hub` for the external `hf` CLI;
Python 3.9 users can use `'huggingface_hub[cli]>=0.34,<1'`. Preserve normal HF
credentials and cache settings.

For larger-model capabilities and qualification limits, see [profiles](profiles.md).
