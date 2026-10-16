import { createHash } from "node:crypto";
import { closeSync, constants, fstatSync, openSync, readSync, realpathSync } from "node:fs";
import { dirname, isAbsolute, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

const names = ["index.ts", "discovery.ts", "budgets.ts", "stream.ts", "profile.json", "package.json"];
const extension = "extensions/loom-hem/";
const repair = "Run loom-hem pi configure to repair the owned integration";

function fail(reason: string): never { throw new Error(`Loom Hem Pi: ${reason}. ${repair}`); }

function readBounded(path: string): Buffer {
  if (realpathSync(path) !== path) fail("aliased managed file");
  const fd = openSync(path, constants.O_RDONLY | constants.O_NOFOLLOW | constants.O_NONBLOCK);
  try {
    const stat = fstatSync(fd);
    if (!stat.isFile() || stat.size > 1024 * 1024) fail("invalid or oversized managed file");
    const bytes = Buffer.alloc(1024 * 1024 + 1);
    let length = 0;
    while (length < bytes.length) {
      const count = readSync(fd, bytes, length, bytes.length - length, null);
      if (!count) break;
      length += count;
    }
    if (length > 1024 * 1024) fail("oversized managed file");
    return bytes.subarray(0, length);
  } finally { closeSync(fd); }
}

function inspectInstallation(entryPath: string) {
  const directory = dirname(realpathSync(entryPath));
  const target = realpathSync(resolve(directory, "../.."));
  if (directory !== join(target, "extensions", "loom-hem")) fail("unexpected extension location");
  const manifest = JSON.parse(readBounded(join(target, "loom-hem", "config.json")).toString("utf8"));
  if (manifest.schema_version !== 1 || manifest.integration_version !== 1 ||
      manifest.enabled !== true || manifest.extension_version !== "1" ||
      manifest.target !== target || manifest.native_profile !== "loom-text-serve-ed6b79c-v1" ||
      manifest.native_commit !== "ed6b79c757c738dea7eaecd6ca55a7df73cf6185" ||
      manifest.pi_package !== "@earendil-works/pi-coding-agent" ||
      JSON.stringify(manifest.pi_versions) !== '["1.3.0"]' ||
      typeof manifest.integration_id !== "string" ||
      !/^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/.test(manifest.integration_id) ||
      typeof manifest.hem_root !== "string" || !isAbsolute(manifest.hem_root) ||
      realpathSync(manifest.hem_root) !== manifest.hem_root ||
      manifest.endpoint_file !== join(manifest.hem_root, "integrations", "pi", "endpoints.json")) {
    fail("disabled or incompatible ownership manifest");
  }
  const artifacts = manifest.artifacts;
  const expected = names.map(name => extension + name).sort();
  if (!artifacts || typeof artifacts !== "object" || Array.isArray(artifacts) ||
      JSON.stringify(Object.keys(artifacts).sort()) !== JSON.stringify(expected)) {
    fail("incomplete ownership manifest");
  }
  for (const relative of expected) {
    if (typeof artifacts[relative] !== "string" || !/^[0-9a-f]{64}$/.test(artifacts[relative])) {
      fail("invalid managed digest");
    }
    const path = join(target, relative);
    const digest = createHash("sha256").update(readBounded(path)).digest("hex");
    if (digest !== artifacts[relative]) fail(`changed managed file ${relative}`);
  }
  return manifest;
}

/** Refresh and dispatch owners must call this again before using copied code. */
export function validateInstallation(entryPath = fileURLToPath(import.meta.url)) {
  try {
    return inspectInstallation(entryPath);
  } catch (error) {
    if (error instanceof Error && error.message.startsWith("Loom Hem Pi:")) throw error;
    // Filesystem/JSON failures can include paths or untrusted file contents.
    fail("cannot read or parse the complete ownership manifest and managed resources");
  }
}

export default async function loomHem(pi: ExtensionAPI): Promise<void> {
  const manifest = validateInstallation();
  // Do not evaluate dependency modules until the entire installed set is verified.
  const { registerProviders } = await import("./discovery.ts");
  await registerProviders(pi, manifest);
}
