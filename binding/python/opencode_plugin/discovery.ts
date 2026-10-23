import { fail } from "./installation.ts";
import type { Installation } from "./installation.ts";
import { joinInventory as nativeInventory, readEndpoints, sweep } from "./native_protocol.ts";
import type { Endpoint, TextModel } from "./native_protocol.ts";
export { readEndpoints, validateBaseUrl } from "./native_protocol.ts";
export type { Endpoint } from "./native_protocol.ts";

export type Model = { id: string; modelID: string; providerID: string; name: string; capabilities: { tools: false; input: string[]; output: string[] }; limit: { context: number; output: number } };
export type Observation = { endpoint: Endpoint; models: Model[]; reason?: string };
function project(model: TextModel, endpoint: Endpoint): Model {
  return { id: model.id, modelID: model.id, providerID: endpoint.provider, name: model.id,
    capabilities: { tools: false, input: ["text"], output: ["text"] },
    limit: { context: model.context, output: model.output } };
}
export function joinInventory(catalog: unknown, status: unknown, endpoint: Endpoint): Model[] {
  return nativeInventory(catalog, status).map(model => project(model, endpoint));
}
const reasons = {
  excluded: "unready or unsupported native entries excluded; check Loom readiness/profile",
  unavailable: "offline or malformed/inconsistent native inventory; reload OpenCode when ready",
  deadline: "discovery deadline reached; reload when ready",
};
/** Single startup/reload snapshot with adapter-owned binding and projection. */
export async function discover(installation: Installation, validate: () => Installation): Promise<Observation[]> {
  const allEndpoints = readEndpoints(installation), fingerprint = JSON.stringify(allEndpoints);
  const endpoints = allEndpoints;
  const snapshot = await sweep(endpoints, AbortSignal.timeout(3000));
  const current = validate();
  if (JSON.stringify(current) !== JSON.stringify(installation) || JSON.stringify(readEndpoints(current)) !== fingerprint) fail("installation or endpoint projection changed during discovery");
  return snapshot.map(row => ({ endpoint: row.endpoint, models: row.models.map(model => project(model, row.endpoint)),
    ...(row.category ? { reason: reasons[row.category] } : {}) }));
}
/** Transform a fresh host inventory. Never merge cached owned model rows. */
export function applySnapshot(editor: any, snapshot: Observation[]): void {
  for (const { endpoint, models } of snapshot) {
    if (editor.get(endpoint.provider)) fail("provider identity conflicts with another configured source");
    if (!models.length) continue;
    editor.update(endpoint.provider, (provider: any) => Object.assign(provider, {
      name: `Loom Control (${endpoint.server})`, activation: "enabled", package: "@opencode/ai/providers/openai-compatible",
      settings: { baseURL: endpoint.base_url },
    }));
    for (const model of models) editor.models.update(endpoint.provider, model.id, (target: any) => Object.assign(target, model, {
      variants: [], settings: {}, headers: {}, body: {}, compatibility: { maxTokensField: "max_tokens", requireReasoning: false },
    }));
  }
}
