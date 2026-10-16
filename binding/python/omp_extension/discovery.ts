import { MAX_BYTES, PROFILE, fail, keys, object, parseDocument, readBounded } from "./installation.ts";
import type { Installation } from "./installation.ts";

export type Endpoint = { server: string; provider: string; base_url: string; native_profile: string };
export type Model = { incarnation: number; id: string; modelID: string; providerID: string; name: string; capabilities: { tools: false; input: string[]; output: string[] }; limit: { context: number; output: number } };
export type Observation = { endpoint: Endpoint; models: Model[]; reason?: string };
const ID = /^[A-Za-z0-9][A-Za-z0-9_.-]*$/;
const positive = (value: unknown): value is number => Number.isSafeInteger(value) && Number(value) > 0;
const TIMEOUT = 3000, PARALLEL = 4, MAX_ENDPOINTS = 256;
export function validateBaseUrl(value: unknown): string {
  if (typeof value !== "string") fail("invalid endpoint origin");
  const match = /^http:\/\/(127(?:\.(?:0|[1-9]\d{0,2})){3}|\[::1\]):([1-9]\d{0,4})\/v1$/.exec(value);
  if (!match || Number(match[2]) > 65535 || (match[1] !== "[::1]" && match[1].split('.').some(part => Number(part) > 255))) fail("unsupported endpoint origin");
  return value;
}
export function readEndpoints(installation: Installation): Endpoint[] {
  const document = parseDocument(readBounded(installation.endpoint_file).toString("utf8"));
  if (!keys(document, ["schema_version", "owner", "hem_root", "servers"]) || document.schema_version !== 1 ||
      document.owner !== "loom-control-clients" || document.hem_root !== installation.hem_root ||
      !Array.isArray(document.servers) || document.servers.length > MAX_ENDPOINTS) fail("invalid endpoint projection identity, schema or count");
  let previous = "";
  return document.servers.map((row: unknown) => {
    if (!keys(row, ["server", "provider", "base_url", "native_profile"]) || typeof row.server !== "string" ||
        !ID.test(row.server) || row.server <= previous || row.provider !== "loom-hem-" + Buffer.from(row.server).toString("base64url") ||
        row.native_profile !== PROFILE) fail("invalid endpoint definition");
    previous = row.server;
    return { server: row.server, provider: row.provider, base_url: validateBaseUrl(row.base_url), native_profile: PROFILE };
  });
}
function rows(document: unknown): Map<string, Record<string, any>> {
  if (!object(document) || document.object !== "list" || !Array.isArray(document.data)) throw new Error("unsupported inventory schema");
  const result = new Map<string, Record<string, any>>();
  for (const row of document.data) {
    if (!object(row) || row.object !== "model" || typeof row.id !== "string" || !ID.test(row.id) || result.has(row.id)) throw new Error("invalid inventory identity");
    result.set(row.id, row);
  }
  return result;
}
class SnapshotChanged extends Error {}
function metadata(row: Record<string, any>): string {
  return JSON.stringify([row.context_length, row.limits?.max_output_tokens, row.input_modalities, row.output_modalities,
    row.loom?.incarnation, row.loom?.registration, row.loom?.kind, row.loom?.endpoint, row.loom?.tokenize]);
}
function compatible(row: Record<string, any>): boolean {
  return positive(row.context_length) && object(row.limits) && positive(row.limits.max_output_tokens) &&
    JSON.stringify(row.input_modalities) === '["text"]' && JSON.stringify(row.output_modalities) === '["text"]' &&
    object(row.loom) && positive(row.loom.incarnation) && row.loom.kind === "text" &&
    row.loom.endpoint === "/v1/chat/completions" && row.loom.tokenize === true;
}
export function joinInventory(catalog: unknown, status: unknown, endpoint: Endpoint): Model[] {
  const models = rows(catalog), states = rows(status);
  if (models.size !== states.size || [...models.keys()].some(id => !states.has(id))) throw new SnapshotChanged("native inventory changed");
  const result: Model[] = [];
  for (const [id, model] of models) {
    const state = states.get(id)!;
    if (metadata(model) !== metadata(state)) throw new SnapshotChanged("native identity or limits changed");
    if (!compatible(model) || !compatible(state)) continue;
    const loom = state.loom;
    if (model.loom.error || model.loom.status?.error || loom.registration !== "ready" ||
        loom.parameter_operation !== "idle" || loom.parameters?.active !== true || loom.error || loom.status?.error ||
        loom.status?.admission !== "open" || !["idle", "execute"].includes(loom.status.phase)) continue;
    result.push({ incarnation: model.loom.incarnation, id, modelID: id, providerID: endpoint.provider, name: id,
      // Conservative supported native profile: never infer tools/reasoning from names.
      capabilities: { tools: false, input: ["text"], output: ["text"] },
      limit: { context: model.context_length, output: Math.min(model.context_length, model.limits.max_output_tokens) } });
  }
  return result;
}
async function getDocument(base: string, path: string, signal: AbortSignal): Promise<unknown> {
  const response = await fetch(base + path, { method: "GET", signal, redirect: "error", credentials: "omit", headers: { Accept: "application/json" }, cache: "no-store" });
  if (!response.ok) throw new Error("discovery HTTP failure");
  const length = response.headers.get("content-length");
  if (length !== null && (!/^\d+$/.test(length) || Number(length) > MAX_BYTES)) { await response.body?.cancel(); throw new Error("oversized response"); }
  if (!response.body) throw new Error("missing discovery response");
  const reader = response.body.getReader(), chunks: Uint8Array[] = [];
  let size = 0;
  try {
    while (true) {
      const next = await reader.read(); if (next.done) break;
      size += next.value.byteLength; if (size > MAX_BYTES) throw new Error("oversized response"); chunks.push(next.value);
    }
  } catch (error) { await reader.cancel().catch(() => {}); throw error; }
  return parseDocument(Buffer.concat(chunks).toString("utf8"));
}
async function probe(endpoint: Endpoint, signal: AbortSignal): Promise<Observation> {
  try {
    for (let attempt = 0; attempt < 2; attempt++) {
      const [catalog, status] = await Promise.all([getDocument(endpoint.base_url, "/models", signal), getDocument(endpoint.base_url, "/status", signal)]);
      try {
        const models = joinInventory(catalog, status, endpoint);
        return { endpoint, models, ...(models.length !== rows(catalog).size ? { reason: "unready or unsupported native entries excluded; check Loom readiness/profile" } : {}) };
      } catch (error) { if (!(error instanceof SnapshotChanged) || attempt) throw error; }
    }
  } catch { return { endpoint, models: [], reason: "offline or malformed/inconsistent native inventory; refresh/restart OMP when ready" }; }
  throw new Error("unreachable discovery state");
}
/** Single bounded startup/reload sweep; no persistent state, timer or lifecycle control. */
export async function discover(installation: Installation, validate: () => Installation, provider?: string): Promise<Observation[]> {
  const allEndpoints = readEndpoints(installation), fingerprint = JSON.stringify(allEndpoints);
  const endpoints = provider === undefined ? allEndpoints : allEndpoints.filter(endpoint => endpoint.provider === provider);
  const signal = AbortSignal.timeout(TIMEOUT), result: Observation[] = new Array(endpoints.length);
  let cursor = 0;
  await Promise.all(Array.from({ length: Math.min(PARALLEL, endpoints.length) }, async () => {
    while (cursor < endpoints.length) {
      const index = cursor++, endpoint = endpoints[index];
      result[index] = signal.aborted ? { endpoint, models: [], reason: "discovery deadline reached; reload when ready" } : await probe(endpoint, signal);
    }
  }));
  const current = validate();
  if (JSON.stringify(current) !== JSON.stringify(installation) || JSON.stringify(readEndpoints(current)) !== fingerprint) fail("installation or endpoint projection changed during discovery");
  return result;
}
