# Deployment and asset reference

The [local walkthrough](../solutions/llm-serving.md) owns the executable setup.
Its complete, editable [qwen08.json](examples/qwen08.json) uses native JSON with
no include syntax or generation step. Replace its three `ABSOLUTE_...` strings;
keep JSON escaping valid for quotes or backslashes in paths.

## Fields

| Field | Meaning and changes |
| --- | --- |
| `name` | Public routing and response model name. `qwen3.8-27b` is the example name; choose another name and use it consistently in HTTP requests and clients. It is independent of the control module’s internal policy name. |
| `device` | Nonnegative index into this process's ordered device group, published in `server.resources.devices`; not necessarily an OS GPU ordinal. |
| `admission` | `drained` registers metadata without source I/O, loading weights or allocating model storage. Activation can prepare it while admission remains closed. |
| `base_directory` | Absolute server-local base for relative module references, including `../` providers. |
| `definition.kind` | `text` selects the text service. |
| `prepare`, `control` | Both select `pipeline.loombc`, the complete module for cold preparation and warm request/execution policy. |
| `sources` | A single entry, `pipeline.loombc`; providers and the self-referencing weight policy reside in this same module. |
| `weights`, `tokenizer` | Absolute server-local paths to the exact checkpoint and matching tokenizer. |
| `rows` | Concurrent execution rows; example 1, native range 1–16. |
| `context_capacity` | Per-row logical token-position limit, example 8192. Input plus requested completion must fit. |
| `pool_capacity` | Shared retained-state token-position capacity, example 8192. Requests and retained state compete for the pool. |
| `prefill_capacity` | Largest automatically specialized packed token shape, example 128; not a prompt-length limit. Must not exceed context capacity. |
| `max_tokens` | Default request output limit, example 512. Clients can supply their request's bounded output limit. |
| `draft_depth` | Zero disables MTP; required for 0.8B. The packaged 27B pipeline supports depths 1–3. |
| `options` | Source-owned preparation/request defaults; `{}` in this example. Native configuration does not interpret arbitrary source option keys. |

The wheel contains exactly one complete `pipeline.loombc` per declared model
specification: Qwen 0.8B and 27B. The adjacent `pipeline.json` descriptor identifies the selected variant and
checks the compiled module hash. All three
executable locators in this example select the same artifact; preparation also
loads its embedded weight policy from that artifact. Preserve the selected
catalog and external weights/tokenizer for the registration lifetime. No
separate linker command is needed on the deployment host; device and capacity
specialization and native compilation still happen at runtime. Model aliases
come from `name`; keep installed modules immutable.

Capacities are logical limits, not a guarantee that weights, retained state and
execution workspace fit on a device. Changing capacities requires retirement
and registration again. Server device/listener settings have process lifetime;
model settings have registration lifetime.

Native JSON rejects unknown fields, duplicate keys and wrong types. HTTP paths
are resolved on the server. An absolute `base_directory` is necessary here
because an HTTP document has no containing file directory. Posting JSON does
not copy checkpoint files or pipeline modules. Preserve dependencies through
retirement; stop the server before replacing the installed wheel.

## Assets

The authored recipe at
`qwen_08b/recipes/qwen35-08b-q5-k-xl.json` pins checkpoint and tokenizer
coordinates independently. Its pipeline binding supplies program integrity and
supported configuration. Local format/index and asset checks are not tensor-shape
qualification. Native preparation/loading reports incompatibility; a successful
short request does not establish arbitrary checkpoint support, numerical
correctness or coding-agent quality.

`hf download --local-dir models` places files in the selected directory. If you
instead reference symlinks into a shared HF cache, eviction or moving the cache
can invalidate a registration. Retain stable inputs or copy them to independent
storage before loading. Install `huggingface_hub` for the external `hf` CLI;
Python 3.9 users can use `'huggingface_hub[cli]>=0.34,<1'`. Preserve normal HF
credentials and cache settings.

For larger-model capabilities and qualification limits, see [profiles](profiles.md).
