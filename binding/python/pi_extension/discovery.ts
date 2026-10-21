import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import type { Model, Provider, RefreshModelsContext } from "@earendil-works/pi-ai";
import { readBounded, validateInstallation } from "./index.ts";
import { createStreamingHooks } from "./stream.ts";

type Manifest = ReturnType<typeof validateInstallation>;
export type Endpoint = { server: string; provider: string; base_url: string; native_profile: string };
export type DiscoveredModel = Model<"openai-completions">;
export type Observation = { endpoint: Endpoint; models: DiscoveredModel[]; reason?: string; budgetVerified: boolean };
/** .3.2 owns budget policy and transport. Absence never implies post-bind safety. */
export type DiscoveryHooks = {
  filterModels?: (models: DiscoveredModel[], endpoint: Endpoint) => DiscoveredModel[];
  settingsBound?: () => boolean;
  stream?: Provider<"openai-completions">["stream"];
  streamSimple?: Provider<"openai-completions">["streamSimple"];
};
const PROFILE = "loom-text-serve-61417137-v1";
const TIMEOUT = 3000, MAX_BYTES = 1024 * 1024, PARALLEL = 4;
const repair = "Run loom-control pi configure to repair the owned integration";
function invalid(reason: string): never { throw new Error(`Loom Hem Pi: ${reason}. ${repair}`); }
function object(value: unknown): value is Record<string, any> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}
function positive(value: unknown): value is number { return Number.isSafeInteger(value) && Number(value) > 0; }

export function validateBaseUrl(value: unknown): string {
  if (typeof value !== "string") invalid("invalid endpoint origin");
  let url: URL;
  try { url = new URL(value); } catch { invalid("invalid endpoint origin"); }
  // Compare canonical bytes too: URL parsing must not normalize away unsafe input.
  const loopback = url.hostname === "[::1]" || /^127(?:\.(?:0|[1-9]\d{0,2})){3}$/.test(url.hostname);
  if (url.protocol !== "http:" || !loopback ||
      url.username || url.password || url.search || url.hash || url.pathname !== "/v1" ||
      (url.href !== value && url.href !== value.replace(":80/v1", "/v1"))) invalid("unsupported endpoint origin");
  return url.href;
}

export function readEndpoints(manifest: Manifest): Endpoint[] {
  let document;
  try { document = JSON.parse(readBounded(manifest.endpoint_file).toString("utf8")); }
  catch { invalid("cannot read the endpoint projection"); }
  if (!object(document) || document.schema_version !== 1 || document.integration_id !== manifest.integration_id ||
      document.hem_root !== manifest.hem_root || !Array.isArray(document.servers)) {
    invalid("incompatible endpoint projection identity or schema");
  }
  let previous = "";
  const providers = new Set<string>();
  return document.servers.map((endpoint: unknown) => {
    if (!object(endpoint) || typeof endpoint.server !== "string" ||
        !/^[A-Za-z0-9][A-Za-z0-9_.-]*$/.test(endpoint.server) || endpoint.server <= previous ||
        endpoint.provider !== "loom-hem-" + Buffer.from(endpoint.server).toString("base64url") ||
        providers.has(endpoint.provider) || endpoint.native_profile !== PROFILE) invalid("invalid endpoint definition");
    previous = endpoint.server;
    providers.add(endpoint.provider);
    return { server: endpoint.server, provider: endpoint.provider, base_url: validateBaseUrl(endpoint.base_url), native_profile: PROFILE };
  });
}

async function getDocument(base: string, path: string, signal: AbortSignal): Promise<unknown> {
  const response = await fetch(base + path, { method: "GET", signal, redirect: "error", credentials: "omit",
    headers: { Accept: "application/json" } });
  if (!response.ok) throw new Error("discovery HTTP failure");
  const length = response.headers.get("content-length");
  if (length !== null && (!/^\d+$/.test(length) || Number(length) > MAX_BYTES)) throw new Error("oversized discovery response");
  if (!response.body) throw new Error("missing discovery response");
  const reader = response.body.getReader();
  const chunks: Uint8Array[] = [];
  let size = 0;
  try {
    while (true) {
      const next = await reader.read();
      if (next.done) break;
      size += next.value.byteLength;
      if (size > MAX_BYTES) throw new Error("oversized discovery response");
      chunks.push(next.value);
    }
  } catch (error) { await reader.cancel().catch(() => {}); throw error; }
  return JSON.parse(Buffer.concat(chunks).toString("utf8"));
}

class SnapshotChanged extends Error {}
function rows(document: unknown): Map<string, Record<string, any>> {
  if (!object(document) || document.object !== "list" || !Array.isArray(document.data)) {
    throw new Error("unsupported discovery schema");
  }
  const result = new Map<string, Record<string, any>>();
  for (const row of document.data) {
    if (!object(row) || row.object !== "model" || typeof row.id !== "string" ||
        !/^[A-Za-z0-9][A-Za-z0-9_.-]*$/.test(row.id) || result.has(row.id)) throw new Error("invalid discovery identity");
    result.set(row.id, row);
  }
  return result;
}
function metadata(row: Record<string, any>): string {
  return JSON.stringify([row.context_length, row.limits?.max_output_tokens, row.input_modalities,
    row.output_modalities, row.loom?.incarnation, row.loom?.registration, row.loom?.kind, row.loom?.endpoint, row.loom?.tokenize]);
}
function compatible(row: Record<string, any>): boolean {
  return positive(row.context_length) && positive(row.limits?.max_output_tokens) &&
    JSON.stringify(row.input_modalities) === '["text"]' && JSON.stringify(row.output_modalities) === '["text"]' &&
    object(row.loom) && positive(row.loom.incarnation) && row.loom.kind === "text" &&
    row.loom.endpoint === "/v1/chat/completions" && row.loom.tokenize === true;
}
// Native residency is component-scoped; retain flat older external status.
// Never use auxiliary image residency as proof the primary model is active.
function primaryParameters(loom: Record<string, any>): Record<string, any> {
  const parameters = loom.parameters;
  if (!object(parameters)) return {};
  if (Object.prototype.hasOwnProperty.call(parameters, "primary")) {
    return object(parameters.primary) ? parameters.primary : {};
  }
  return parameters;
}
export function joinInventory(catalog: unknown, status: unknown, endpoint: Endpoint): DiscoveredModel[] {
  const models = rows(catalog), states = rows(status);
  if (models.size !== states.size || [...models.keys()].some(id => !states.has(id))) throw new SnapshotChanged("native snapshot changed");
  const result: DiscoveredModel[] = [];
  for (const [id, model] of models) {
    const state = states.get(id)!;
    if (metadata(model) !== metadata(state)) throw new SnapshotChanged("native snapshot changed");
    if (!compatible(model) || !compatible(state)) continue;
    const loom = state.loom;
    if (model.loom.error || model.loom.status?.error ||
        loom.registration !== "ready" || loom.parameter_operation !== "idle" || primaryParameters(loom).active !== true ||
        loom.error || loom.status?.error || loom.status?.admission !== "open" ||
        typeof loom.status.phase !== "string" || !["idle", "execute"].includes(loom.status.phase)) continue;
    result.push({ id, name: id, api: "openai-completions", provider: endpoint.provider, baseUrl: endpoint.base_url,
      reasoning: false, input: ["text"], cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
      contextWindow: model.context_length, maxTokens: Math.min(8192, model.limits.max_output_tokens, model.context_length),
      compat: { supportsStore: false, supportsDeveloperRole: false, supportsReasoningEffort: false,
        supportsUsageInStreaming: true, maxTokensField: "max_tokens" } });
  }
  return result;
}
async function probe(endpoint: Endpoint, signal: AbortSignal): Promise<Observation> {
  try {
    for (let attempt = 0; attempt < 2; attempt++) {
      const [catalog, status] = await Promise.all([getDocument(endpoint.base_url, "/models", signal), getDocument(endpoint.base_url, "/status", signal)]);
      try {
        const models = joinInventory(catalog, status, endpoint);
        const withheld = models.length !== rows(catalog).size;
        return { endpoint, models, budgetVerified: false,
          ...(withheld ? { reason: "some native entries are unavailable or unsupported; check readiness and text-profile metadata" } : {}) };
      }
      catch (error) { if (!(error instanceof SnapshotChanged) || attempt) throw error; }
    }
  } catch {
    return { endpoint, models: [], budgetVerified: false, reason: "offline or unsupported/inconsistent native inventory; refresh when available" };
  }
  throw new Error("unreachable discovery state");
}

/** In-memory only. One sweep shares a total deadline and bounded endpoint concurrency. */
export class Discovery {
  private generation = 0;
  private running?: {
    abort: AbortController; promise: Promise<Map<string, Observation>>; settled: boolean;
    callers: Map<unknown, () => void>;
  };
  readonly observations = new Map<string, Observation>();
  readonly providers: Endpoint[];
  constructor(readonly manifest: Manifest, readonly validate = validateInstallation) {
    this.providers = readEndpoints(manifest);
  }
  currentEndpoints(): Endpoint[] {
    const current = this.validate();
    if (current.integration_id !== this.manifest.integration_id || current.hem_root !== this.manifest.hem_root ||
        current.target !== this.manifest.target) invalid("integration ownership changed; reload Pi");
    return readEndpoints(current);
  }
  clear(reason: string) {
    for (const endpoint of this.providers) this.observations.set(endpoint.provider, { endpoint, models: [], budgetVerified: false, reason });
  }
  async refresh(signal: AbortSignal, owner: unknown = signal): Promise<Map<string, Observation>> {
    if (signal.aborted) return new Map();
    // Pi supplies a distinct combined AbortSignal per provider in one registry
    // refresh. Signals identify callers, not global refresh generations.
    this.running?.callers.get(owner)?.();
    if (!this.running || this.running.abort.signal.aborted) {
      const abort = new AbortController();
      const promise = this.sweep(AbortSignal.any([abort.signal, AbortSignal.timeout(TIMEOUT)]), ++this.generation);
      const scan = { abort, promise, settled: false, callers: new Map<unknown, () => void>() };
      this.running = scan;
      const finish = () => {
        scan.settled = true;
        if (this.running === scan) this.running = undefined;
      };
      void promise.then(finish, finish);
    }
    const scan = this.running;
    let cancel!: () => void;
    const cancelled = new Promise<Map<string, Observation>>(resolve => { cancel = () => resolve(new Map()); });
    const release = () => {
      signal.removeEventListener("abort", release);
      if (scan.callers.get(owner) === release) scan.callers.delete(owner);
      cancel();
      // Cancelling one provider must not terminate its siblings' shared GETs.
      if (!scan.settled && !scan.callers.size) {
        if (this.running === scan) { this.running = undefined; ++this.generation; }
        scan.abort.abort();
      }
    };
    scan.callers.set(owner, release);
    signal.addEventListener("abort", release, { once: true });
    try { return await Promise.race([scan.promise, cancelled]); }
    finally { release(); }
  }
  private async sweep(signal: AbortSignal, generation: number): Promise<Map<string, Observation>> {
    let endpoints: Endpoint[];
    try { endpoints = this.currentEndpoints(); }
    catch { this.clear(`integration invalid or disabled. ${repair}`); return new Map(this.observations); }
    const fingerprint = JSON.stringify(endpoints), result = new Map<string, Observation>();
    let cursor = 0;
    const known = new Set(this.providers.map(endpoint => endpoint.provider));
    await Promise.all(Array.from({ length: Math.min(PARALLEL, endpoints.length) }, async () => {
      while (cursor < endpoints.length && !signal.aborted) {
        const endpoint = endpoints[cursor++];
        if (known.has(endpoint.provider)) result.set(endpoint.provider, await probe(endpoint, signal));
      }
    }));
    if (generation !== this.generation) return result;
    try { if (fingerprint !== JSON.stringify(this.currentEndpoints())) throw new Error("endpoint projection changed during refresh"); }
    catch { this.clear(`integration changed during refresh. ${repair}`); return new Map(this.observations); }
    for (const endpoint of this.providers) {
      const observation = result.get(endpoint.provider) || { endpoint, models: [], budgetVerified: false,
        reason: signal.aborted ? "discovery cancelled or timed out; refresh when available" : "server removed; reload Pi" };
      this.observations.set(endpoint.provider, observation);
    }
    return new Map(this.observations);
  }
}

export async function registerProviders(pi: ExtensionAPI, manifest: Manifest, hooks: DiscoveryHooks = {}): Promise<Discovery> {
  const discovery = new Discovery(manifest);
  await discovery.refresh(new AbortController().signal);
  hooks = { ...createStreamingHooks(pi, discovery), ...hooks };
  for (const endpoint of discovery.providers) {
    let models = discovery.observations.get(endpoint.provider)?.models || [];
    let generation = 0;
    const provider: Provider<"openai-completions"> = {
      id: endpoint.provider, name: `Loom Hem (${endpoint.server})`, baseUrl: endpoint.base_url,
      auth: { apiKey: { name: "Configured local Loom endpoint", resolve: async () => ({ auth: { apiKey: "loom-local" }, source: "owned loopback integration" }) } },
      getModels: () => models,
      async refreshModels(context: RefreshModelsContext) {
        if (!context.allowNetwork) {
          // Pi runs a cache-only phase after async factory initialization. Do
          // not erase native-ready initial selection merely because settings
          // are not yet bound, but still honor disabled/changed installations.
          let valid = false;
          try { valid = discovery.currentEndpoints().some(current => current.provider === endpoint.provider &&
            current.base_url === discovery.observations.get(endpoint.provider)?.endpoint.base_url); }
          catch {
            discovery.clear(`integration invalid or disabled. ${repair}`);
            console.error(`Loom Hem Pi (${endpoint.server}): integration invalid or disabled. ${repair}`);
          }
          if (!valid) await context.publish({ update: () => { models = []; } });
          else if (hooks.settingsBound?.()) {
            const next = hooks.filterModels!(models, discovery.observations.get(endpoint.provider)?.endpoint || endpoint);
            await context.publish({ update: () => { if (!context.signal.aborted) models = next; } });
          }
          return;
        }
        const ownGeneration = ++generation;
        const result = await discovery.refresh(context.signal, endpoint.provider);
        let next = result.get(endpoint.provider)?.models || [];
        let reason = result.get(endpoint.provider)?.reason;
        try {
          if (!hooks.filterModels) throw new Error("budget validation unavailable; complete the provider installation");
          next = hooks.filterModels(next, result.get(endpoint.provider)?.endpoint || endpoint);
        } catch { next = []; reason = "effective token budgets are unknown or incompatible; check Pi settings and refresh"; }
        await context.publish({ update: () => {
          if (ownGeneration !== generation || context.signal.aborted) return;
          models = next;
          if (reason) console.error(`Loom Hem Pi (${endpoint.server}): ${reason}`);
        } });
      },
      stream: hooks.stream!,
      streamSimple: hooks.streamSimple!,
    };
    pi.registerProvider(provider);
    const reason = discovery.observations.get(endpoint.provider)?.reason;
    if (reason) console.error(`Loom Hem Pi (${endpoint.server}): ${reason}`);
  }
  return discovery;
}
