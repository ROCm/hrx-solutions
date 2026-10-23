import { fail } from "./installation.ts";
import type { Installation } from "./installation.ts";
import { joinInventory as nativeInventory, readEndpoints, sweep } from "./native_protocol.ts";
import type { Endpoint, TextModel } from "./native_protocol.ts";
export { readEndpoints, validateBaseUrl } from "./native_protocol.ts";
export type { Endpoint } from "./native_protocol.ts";

export type Model = { incarnation: number; id: string; modelID: string; providerID: string; name: string; capabilities: { tools: false; input: string[]; output: string[] }; limit: { context: number; output: number } };
export type Observation = { endpoint: Endpoint; models: Model[]; reason?: string };
function project(model: TextModel, endpoint: Endpoint): Model {
  return { incarnation: model.incarnation, id: model.id, modelID: model.id, providerID: endpoint.provider, name: model.id,
    capabilities: { tools: false, input: ["text"], output: ["text"] },
    limit: { context: model.context, output: model.output } };
}
export function joinInventory(catalog: unknown, status: unknown, endpoint: Endpoint): Model[] {
  return nativeInventory(catalog, status).map(model => project(model, endpoint));
}
const reasons = {
  excluded: "unready or unsupported native entries excluded; check Loom readiness/profile",
  unavailable: "offline or malformed/inconsistent native inventory; refresh/restart OMP when ready",
  deadline: "discovery deadline reached; reload when ready",
};
/** Single startup/reload snapshot with adapter-owned binding and projection. */
export async function discover(installation: Installation, validate: () => Installation, provider?: string): Promise<Observation[]> {
  const allEndpoints = readEndpoints(installation), fingerprint = JSON.stringify(allEndpoints);
  const endpoints = provider === undefined ? allEndpoints : allEndpoints.filter(endpoint => endpoint.provider === provider);
  const snapshot = await sweep(endpoints, AbortSignal.timeout(3000));
  const current = validate();
  if (JSON.stringify(current) !== JSON.stringify(installation) || JSON.stringify(readEndpoints(current)) !== fingerprint) fail("installation or endpoint projection changed during discovery");
  return snapshot.map(row => ({ endpoint: row.endpoint, models: row.models.map(model => project(model, row.endpoint)),
    ...(row.category ? { reason: reasons[row.category] } : {}) }));
}
