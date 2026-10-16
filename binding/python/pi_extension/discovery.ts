import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

// Explicit staging boundary: never advertise models before discovery is implemented.
export async function registerProviders(_pi: ExtensionAPI, _manifest: unknown): Promise<void> {
  throw new Error("Loom Hem Pi discovery is not implemented in this build");
}
