import { createHash } from "node:crypto";
import { closeSync, constants, fstatSync, lstatSync, openSync, readSync, realpathSync } from "node:fs";
import { dirname, isAbsolute, join, resolve } from "node:path";

export const MAX_BYTES = 1024 * 1024;
export const PROFILE = "loom-text-serve-ed6b79c-v1";
export const ASSETS = ["index.ts", "discovery.ts", "installation.ts", "profile.json", "package.json"];
const uuid = /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/;
export function fail(reason: string): never {
  throw new Error(`Loom Control OpenCode: ${reason}; run loom-control opencode configure to inspect/repair, then opencode reload or restart`);
}
export function object(value: unknown): value is Record<string, any> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}
export function keys(value: unknown, expected: string[]): value is Record<string, any> {
  return object(value) && JSON.stringify(Object.keys(value).sort()) === JSON.stringify([...expected].sort());
}
export function readBounded(path: string): Buffer {
  if (realpathSync(path) !== path) fail("aliased managed path");
  const fd = openSync(path, constants.O_RDONLY | constants.O_NOFOLLOW | constants.O_NONBLOCK);
  try {
    const stat = fstatSync(fd);
    if (!stat.isFile() || stat.size > MAX_BYTES) fail("nonregular or oversized managed file");
    const buffer = Buffer.alloc(MAX_BYTES + 1);
    let length = 0;
    while (length < buffer.length) {
      const count = readSync(fd, buffer, length, buffer.length - length, null);
      if (!count) break;
      length += count;
    }
    if (length > MAX_BYTES) fail("oversized managed file");
    return buffer.subarray(0, length);
  } finally { closeSync(fd); }
}
/** Reject duplicate keys before JSON.parse can silently choose one identity. */
export function parseDocument(text: string): unknown {
  let offset = 0;
  function white() { while (/\s/.test(text[offset] || "") && offset < text.length) offset++; }
  function string(): string {
    const start = offset++;
    while (offset < text.length) {
      const char = text[offset++];
      if (char === "\\") offset++;
      else if (char === '"') return JSON.parse(text.slice(start, offset));
    }
    throw new Error("unterminated string");
  }
  function value(depth: number): void {
    if (depth > 64) throw new Error("JSON nesting limit");
    white();
    if (text[offset] === '{') {
      offset++; white(); const seen = new Set<string>();
      if (text[offset] !== '}') while (true) {
        white(); if (text[offset] !== '"') throw new Error("invalid object key");
        const key = string(); if (seen.has(key)) throw new Error("duplicate JSON key"); seen.add(key);
        white(); if (text[offset++] !== ':') throw new Error("invalid separator"); value(depth + 1); white();
        if (text[offset] !== ',') break; offset++;
      }
      if (text[offset++] !== '}') throw new Error("invalid object");
    } else if (text[offset] === '[') {
      offset++; white();
      if (text[offset] !== ']') while (true) { value(depth + 1); white(); if (text[offset] !== ',') break; offset++; }
      if (text[offset++] !== ']') throw new Error("invalid array");
    } else if (text[offset] === '"') string();
    else { const start = offset; while (offset < text.length && !/[\s,\]}]/.test(text[offset])) offset++; if (start === offset) throw new Error("invalid value"); }
  }
  value(0); white(); if (offset !== text.length) throw new Error("trailing JSON");
  return JSON.parse(text);
}
export type Installation = { schema_version: 1; generation: string; integration_id: string; target: string; hem_root: string; endpoint_file: string };
export function validateInstallation(entryPath: string): Installation {
  try {
    const directory = dirname(entryPath), target = resolve(directory, "../..");
    if (realpathSync(entryPath) !== entryPath || directory !== join(target, "plugins", "loom-control-opencode")) fail("unexpected plugin location");
    try { lstatSync(join(directory, "pending.json")); fail("incomplete installation"); }
    catch (error: any) { if (error.code !== "ENOENT") throw error; }
    const manifest = parseDocument(readBounded(join(directory, "manifest.json")).toString("utf8"));
    const configBytes = readBounded(join(directory, "config.json"));
    const config = parseDocument(configBytes.toString("utf8"));
    if (!keys(manifest, ["schema_version", "integration_version", "enabled", "generation", "integration_id", "target", "artifacts"]) ||
        manifest.schema_version !== 1 || manifest.integration_version !== 1 || manifest.enabled !== true ||
        !uuid.test(manifest.generation) || !uuid.test(manifest.integration_id) || manifest.target !== target ||
        !keys(config, ["schema_version", "generation", "integration_id", "target", "hem_root", "endpoint_file"]) ||
        config.schema_version !== 1 || config.generation !== manifest.generation || config.integration_id !== manifest.integration_id ||
        config.target !== target || typeof config.hem_root !== "string" || !isAbsolute(config.hem_root) ||
        realpathSync(config.hem_root) !== config.hem_root || config.endpoint_file !== join(config.hem_root, "clients", "endpoints.json") ||
        !keys(manifest.artifacts, [...ASSETS, "config.json"])) fail("incompatible installation identity or generation");
    for (const name of [...ASSETS, "config.json"]) {
      if (typeof manifest.artifacts[name] !== "string" || !/^[0-9a-f]{64}$/.test(manifest.artifacts[name]) ||
          createHash("sha256").update(readBounded(join(directory, name))).digest("hex") !== manifest.artifacts[name]) fail("changed managed asset");
    }
    const profile = parseDocument(readBounded(join(directory, "profile.json")).toString("utf8"));
    if (JSON.stringify(profile) !== JSON.stringify({ native_profile: PROFILE, native_commit: "ed6b79c757c738dea7eaecd6ca55a7df73cf6185", client_package: "@opencode/cli", client_versions: ["2.0.31"], capabilities: { tools: false, input: ["text"], output: ["text"] } })) fail("unsupported native/client capability profile");
    return config as Installation;
  } catch { fail("invalid, disabled, edited or incomplete managed installation"); }
}
