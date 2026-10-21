// Test our budget guard directly; no vendor session/product certification.
import assert from 'node:assert/strict';
import { createJiti } from 'jiti';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
const assets = process.env.LOOM_PI_ASSET_DIR;
assert.ok(assets);
const jiti = createJiti(import.meta.url, { alias: {
  '@earendil-works/pi-ai': fileURLToPath(import.meta.resolve('@earendil-works/pi-ai')),
  '@earendil-works/pi-coding-agent': fileURLToPath(import.meta.resolve('@earendil-works/pi-coding-agent')),
} });
const { validateBudgets } = await jiti.import(join(assets, 'budgets.ts'));
const model = { id: 'ours', provider: 'loom-hem-test', contextWindow: 262144, maxTokens: 8192 };
assert.doesNotThrow(() => validateBudgets({}, model));
for (const settings of [undefined, null, { compaction: false }, { branchSummary: { reserveTokens: -1 } },
  { compaction: { reserveTokens: 1 } }, { compaction: { keepRecentTokens: 262144 } }]) {
  assert.throws(() => validateBudgets(settings, model), /Loom Hem Pi/);
}
assert.throws(() => validateBudgets({}, { ...model, contextWindow: 8192 }), /incompatible effective budgets/);
assert.throws(() => validateBudgets({}, { ...model, maxTokens: 16384 }), /unsupported native context/);
console.log('Our effective-budget guards passed');

// Denied paths in our streaming wrapper must stop before adapter/network work.
const { createStreamingHooks } = await jiti.import(join(assets, 'stream.ts'));
const selected = { ...model, api: 'openai-completions', baseUrl: 'http://127.0.0.1:1/v1' };
const endpoint = { provider: selected.provider, base_url: selected.baseUrl,
  native_profile: 'loom-text-serve-61417137-v1' };
const discovery = { currentEndpoints: () => [endpoint], observations: new Map([
  [selected.provider, { endpoint: { ...endpoint }, models: [selected] }],
]) };
let settings = {};
const hooks = createStreamingHooks({ getSettings: () => settings }, discovery);
for (const stream of [hooks.stream, hooks.streamSimple]) {
  assert.throws(() => stream(selected, {}, { maxTokens: 8193 }), /request maxTokens/);
  settings = undefined;
  assert.throws(() => stream(selected, {}), /effective token budgets/);
  settings = {};
  assert.throws(() => stream({ ...selected, contextWindow: 262145 }, {}), /metadata changed/);
  endpoint.base_url = 'http://127.0.0.1:2/v1';
  assert.throws(() => stream(selected, {}), /origin changed/);
  endpoint.base_url = selected.baseUrl;
}
console.log('Our streaming pre-dispatch guards passed without network requests');
