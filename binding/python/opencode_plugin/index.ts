import { Plugin } from "@opencode/plugin";
import { fileURLToPath } from "node:url";
import { validateInstallation } from "./installation.ts";

export default Plugin.define({ id: "loom-control-opencode", setup: async (context) => {
  const entry = fileURLToPath(import.meta.url);
  const validate = () => validateInstallation(entry);
  const installation = validate();
  // Evaluate the discovery companion only after all copied bytes/generation match.
  const { applySnapshot, discover } = await import("./discovery.ts");
  const snapshot = await discover(installation, validate);
  for (const row of snapshot) if (row.reason) console.error(`Loom Control OpenCode (${row.endpoint.server}): ${row.reason}`);
  await context.provider.transform(editor => {
    // A later transform invocation must not use bytes from a replaced installation.
    if (JSON.stringify(validate()) !== JSON.stringify(installation)) throw new Error("Loom Control installation changed; reload OpenCode");
    applySnapshot(editor, snapshot);
  });
} });
