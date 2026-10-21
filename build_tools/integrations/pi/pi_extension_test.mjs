// Copy this runner beside the qualified Pi installation's package.json to use
// its ordinary public package resolution. Set LOOM_PI_ASSET_DIR to wheel assets.
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { mkdtempSync, mkdirSync, readFileSync, rmSync, symlinkSync, writeFileSync } from "node:fs";
import { spawnSync } from "node:child_process";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const { discoverAndLoadExtensions } = await import("@earendil-works/pi-coding-agent");
const assets = process.env.LOOM_PI_ASSET_DIR || join(dirname(fileURLToPath(import.meta.url)), "../binding/python/pi_extension");
const names = ["index.ts", "discovery.ts", "budgets.ts", "stream.ts", "profile.json", "package.json"];
const profile = JSON.parse(readFileSync(join(assets, "profile.json")));
const scratch = mkdtempSync(join(process.env.LOOM_PI_TEST_SCRATCH || tmpdir(), "loom-pi-assets-"));
const results = [];
try {
  const scenarios = process.env.LOOM_PI_FIFO_CASE ? [process.env.LOOM_PI_FIFO_CASE] :
    ["valid-unavailable", "missing-manifest", "malformed-manifest", "null-manifest", "disabled", "wrong-profile", "missing-hash", "changed-self", "changed-dependency", "missing-dependency", "symlink-manifest", "symlink-dependency"];
  for (const scenario of scenarios) {
    const target = join(scratch, scenario), directory = join(target, "extensions", "loom-hem");
    const root = join(target, "state");
    mkdirSync(directory, { recursive: true });
    mkdirSync(root, { recursive: true });
    mkdirSync(join(target, "loom-hem"));
    const artifacts = {};
    for (const name of names) {
      const data = readFileSync(join(assets, name));
      writeFileSync(join(directory, name), data);
      artifacts["extensions/loom-hem/" + name] = createHash("sha256").update(data).digest("hex");
    }
    const manifest = {
      schema_version: profile.manifest_schema_version, integration_version: profile.integration_version,
      integration_id: "12345678-1234-4234-8234-123456789abc", enabled: true, hem_root: root,
      target, extension_version: profile.extension_version, native_profile: profile.native_profile,
      native_commit: profile.native_commit, pi_package: profile.pi_package, pi_versions: profile.pi_versions,
      endpoint_file: join(root, "integrations", "pi", "endpoints.json"), artifacts,
    };
    if (scenario === "disabled") manifest.enabled = false;
    if (scenario === "wrong-profile") manifest.native_profile = "unsupported";
    if (scenario === "missing-hash") delete artifacts["extensions/loom-hem/stream.ts"];
    if (scenario !== "missing-manifest") writeFileSync(join(target, "loom-hem/config.json"), JSON.stringify(manifest));
    if (scenario === "malformed-manifest") writeFileSync(join(target, "loom-hem/config.json"), '{"secret":"DO_NOT_ECHO_FILE_CONTENT"');
    if (scenario === "null-manifest") writeFileSync(join(target, "loom-hem/config.json"), "null");
    if (scenario === "changed-self") writeFileSync(join(directory, "index.ts"), readFileSync(join(directory, "index.ts")) + "\n// changed\n");
    if (scenario === "changed-dependency") writeFileSync(join(directory, "discovery.ts"), 'throw new Error("UNVERIFIED_DEPENDENCY_EXECUTED");');
    if (scenario === "missing-dependency") rmSync(join(directory, "stream.ts"));
    if (scenario.startsWith("symlink-") || scenario.startsWith("fifo-")) {
      const managed = scenario.endsWith("manifest") ? join(target, "loom-hem/config.json") : join(directory, "stream.ts");
      const original = readFileSync(managed);
      rmSync(managed);
      if (scenario.startsWith("fifo-")) {
        const fifo = spawnSync("mkfifo", [managed], { timeout: 1000 });
        assert.equal(fifo.status, 0, String(fifo.stderr));
      } else {
        const outside = join(target, "unmanaged");
        writeFileSync(outside, original);
        symlinkSync(outside, managed);
      }
    }
    const loaded = await discoverAndLoadExtensions([directory], target, target);
    assert.equal(loaded.errors.length, 1, JSON.stringify(loaded.errors));
    const error = String(loaded.errors[0].error);
    if (scenario === "valid-unavailable") assert.match(error, /cannot read the endpoint projection/);
    else {
      assert.match(error, /Loom Hem Pi:/);
      assert.match(error, /Run loom-control pi configure/);
    }
    assert.doesNotMatch(error, /UNVERIFIED_DEPENDENCY_EXECUTED/);
    assert.doesNotMatch(error, /DO_NOT_ECHO_FILE_CONTENT/);
    assert.equal(loaded.extensions.length, 0);
    results.push({ scenario, status: "passed", error });
  }
  if (!process.env.LOOM_PI_FIFO_CASE) {
    for (const scenario of ["fifo-manifest", "fifo-dependency"]) {
      const child = spawnSync(process.execPath, [fileURLToPath(import.meta.url)], {
        env: { ...process.env, LOOM_PI_FIFO_CASE: scenario, LOOM_PI_TEST_SCRATCH: scratch },
        encoding: "utf8", timeout: 5000,
      });
      assert.equal(child.error, undefined, `bounded FIFO loader failed: ${child.error}`);
      assert.equal(child.status, 0, child.stderr);
      results.push(...JSON.parse(child.stdout).results);
    }
  }
  console.log(JSON.stringify({ node: process.version, profile, results }, null, 2));
} finally { rmSync(scratch, { recursive: true, force: true }); }
