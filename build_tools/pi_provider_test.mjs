// Run beside an isolated locked Pi 1.3.0 installation. Uses public package APIs,
// production copied assets, loopback native/SSE fixtures; never the user's Pi.
import assert from 'node:assert/strict';
import { createHash, randomUUID } from 'node:crypto';
import { createServer } from 'node:http';
import { mkdirSync, mkdtempSync, readFileSync, writeFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

const assets = process.env.LOOM_PI_ASSET_DIR;
assert.ok(assets, 'explicit source or installed-wheel asset directory required');
const scratch = mkdtempSync(join(process.env.LOOM_PI_TEST_SCRATCH || tmpdir(), 'pi-provider-'));
process.env.PI_CODING_AGENT_DIR = join(scratch, 'unused-agent');
process.env.PI_OFFLINE = '1';
const { createAgentSessionServices, createAgentSessionFromServices, SettingsManager, SessionManager } =
  await import('@earendil-works/pi-coding-agent');
const profile = JSON.parse(readFileSync(join(assets, 'profile.json')));
const provider = 'loom-hem-bG9jYWw', id = 'arbitrary-new-instance';
const files = ['index.ts', 'discovery.ts', 'budgets.ts', 'stream.ts', 'profile.json', 'package.json'];
const requests = [], servers = [], sessions = [], originals = new Map(), results = [];
const context = { systemPrompt: 'Fixture.', messages: [{ role: 'user', content: [{ type: 'text', text: 'Request' }], timestamp: Date.now() }] };
function row(state, status) {
  return { object: 'model', id, context_length: state.context, limits: { max_output_tokens: state.output },
    input_modalities: ['text'], output_modalities: ['text'], loom: { kind: 'text', endpoint: '/v1/chat/completions', tokenize: true,
      incarnation: 1, registration: 'ready', ...(status ? { parameter_operation: 'idle', parameters: { active: true },
        status: { phase: state.phase, admission: 'open' } } : {}) } };
}
async function serve(name) {
  const state = { context: 262144, output: 16384, phase: 'idle', available: true };
  const server = createServer(async (req, res) => {
    const record = { name, method: req.method, path: req.url, closed: false };
    requests.push(record);
    if (req.method === 'GET') {
      assert.ok(['/v1/models', '/v1/status'].includes(req.url));
      res.setHeader('content-type', 'application/json');
      res.end(JSON.stringify({ object: 'list', data: state.available ? [row(state, req.url === '/v1/status')] : [] }));
      return;
    }
    assert.equal(req.method, 'POST'); assert.equal(req.url, '/v1/chat/completions');
    let raw = ''; for await (const data of req) raw += data;
    record.body = JSON.parse(raw); record.headers = req.headers;
    res.writeHead(200, { 'content-type': 'text/event-stream' });
    const event = (delta, finish = null) => res.write('data: ' + JSON.stringify({ id: 'fixture', object: 'chat.completion.chunk', created: 1,
      model: id, choices: [{ index: 0, delta, finish_reason: finish }] }) + '\n\n');
    event({ role: 'assistant', content: '' });
    if (raw.includes('ABORT_FIXTURE')) {
      event({ content: 'PARTIAL' }); req.socket.once('close', () => { record.closed = true; }); return;
    }
    event({ content: 'SAFE_STREAM_OK' }); event({}, 'stop');
    if (raw.includes('AUTO_FIXTURE')) res.write('data: ' + JSON.stringify({ id: 'fixture', choices: [],
      usage: { prompt_tokens: 250000, completion_tokens: 1, total_tokens: 250001 } }) + '\n\n');
    res.end('data: [DONE]\n\n');
  });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve)); servers.push(server);
  return { state, base: `http://127.0.0.1:${server.address().port}/v1` };
}
const a = await serve('A'), b = await serve('B');
let sequence = 0;
async function make(label, user = {}, project = {}, { unbound = false, seed = false } = {}) {
  a.state.context = 262144; a.state.output = 16384; a.state.phase = 'idle'; a.state.available = true;
  const target = join(scratch, `${++sequence}-${label}`), directory = join(target, 'extensions/loom-hem'), root = join(target, 'state');
  mkdirSync(directory, { recursive: true }); mkdirSync(join(target, 'loom-hem')); mkdirSync(join(target, 'cwd/.pi'), { recursive: true });
  mkdirSync(join(root, 'integrations/pi'), { recursive: true });
  const artifacts = {};
  for (const file of files) {
    const bytes = readFileSync(join(assets, file)); writeFileSync(join(directory, file), bytes);
    artifacts['extensions/loom-hem/' + file] = createHash('sha256').update(bytes).digest('hex');
  }
  const manifest = { schema_version: 1, integration_version: 1, integration_id: randomUUID(), enabled: true, hem_root: root, target,
    extension_version: '1', native_profile: profile.native_profile, native_commit: profile.native_commit,
    pi_package: profile.pi_package, pi_versions: profile.pi_versions, endpoint_file: join(root, 'integrations/pi/endpoints.json'), artifacts };
  const endpoints = { schema_version: 1, integration_id: manifest.integration_id, hem_root: root,
    servers: [{ server: 'local', provider, base_url: a.base, native_profile: profile.native_profile }] };
  const saveManifest = () => writeFileSync(join(target, 'loom-hem/config.json'), JSON.stringify(manifest));
  const saveEndpoints = () => writeFileSync(manifest.endpoint_file, JSON.stringify(endpoints));
  saveManifest(); saveEndpoints();
  const base = { defaultProvider: provider, defaultModel: id, cacheWarming: 'off', retry: { enabled: false, maxRetries: 0 }, compaction: { enabled: true } };
  writeFileSync(join(target, 'settings.json'), JSON.stringify({ ...base, ...user }));
  writeFileSync(join(target, 'cwd/.pi/settings.json'), JSON.stringify(project));
  for (const path of [join(target, 'settings.json'), join(target, 'cwd/.pi/settings.json')]) originals.set(path, readFileSync(path));
  const settings = SettingsManager.create(join(target, 'cwd'), target, { projectTrusted: true });
  const services = await createAgentSessionServices({ cwd: join(target, 'cwd'), agentDir: target, settingsManager: settings,
    modelRuntimeSignal: AbortSignal.timeout(3000), resourceLoaderOptions: { noExtensions: true, additionalExtensionPaths: [directory],
      disabledBuiltinExtensions: ['mcp'], noSkills: true, noPromptTemplates: true, noThemes: true, noContextFiles: true, systemPrompt: 'Reply briefly.' } });
  assert.deepEqual(services.resourceLoader.getExtensions().errors, []);
  const model = services.modelRuntime.getModel(provider, id);
  assert.ok(model, 'native-ready arbitrary ID initially selectable before budgets bind');
  const fixture = { services, model, settings, manifest, endpoints, saveManifest, saveEndpoints, directory };
  if (unbound) return fixture;
  const sm = SessionManager.inMemory(join(target, 'cwd'));
  if (seed) for (let i = 0; i < 14; i++) {
    sm.appendMessage({ role: 'user', content: [{ type: 'text', text: 'history ' + i + ' ' + 'x'.repeat(12000) }], timestamp: Date.now() });
    sm.appendMessage({ role: 'assistant', content: [{ type: 'text', text: 'Historical answer ' + i }], api: 'openai-completions', provider, model: id,
      stopReason: 'stop', timestamp: Date.now(), usage: { input: 10000, output: 20, cacheRead: 0, cacheWrite: 0, totalTokens: 10020,
        cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } } });
  }
  const { session } = await createAgentSessionFromServices({ services, sessionManager: sm, model, noTools: 'all' });
  await session.bindExtensions({ mode: 'print' }); sessions.push(session);
  return { ...fixture, session, sm };
}
const posts = () => requests.filter(r => r.method === 'POST');
async function refresh(fixture) { return fixture.services.modelRuntime.refresh({ allowNetwork: true, signal: AbortSignal.timeout(3000) }); }
async function action(fixture, kind = 'ordinary') {
  const { session, sm } = fixture;
  if (kind === 'ordinary') {
    await session.prompt('Reply SAFE');
    const answer = session.messages.findLast(m => m.role === 'assistant');
    if (answer?.stopReason === 'error') throw Error(answer.errorMessage);
    assert.equal(session.getLastAssistantText(), 'SAFE_STREAM_OK'); return;
  }
  if (kind === 'compact') return session.compact();
  if (kind === 'branch') return session.navigateTree(sm.getEntries().find(e => e.type === 'message' && e.message.role === 'user').id, { summarize: true });
  return session.summarizeForBugReport({ hint: 'Fixture summary', signal: AbortSignal.timeout(3000) });
}
async function denied(run, pattern) {
  const start = posts().length; let error;
  try { const answer = await run(); if (answer?.stopReason === 'error') error = answer.errorMessage; } catch (e) { error = String(e); }
  assert.match(error || '', pattern); assert.equal(posts().length, start, 'denied request reached HTTP transport'); return error;
}
async function test(name, run) {
  const start = posts().length;
  await run(); results.push({ name, status: 'passed', posts: posts().length - start });
}
try {
  for (const kind of ['ordinary', 'compact', 'branch', 'summary']) await test('safe-public-' + kind, async () => {
    const fixture = await make('safe-' + kind, {}, {}, { seed: kind !== 'ordinary' });
    const start = posts().length; await action(fixture, kind); assert.equal(posts().length, start + 1);
    assert.equal(fixture.model.contextWindow, 262144); assert.equal(fixture.model.maxTokens, 8192);
  });
  const unsafe = [
    ['user', { compaction: { reserveTokens: 300000 } }, {}],
    ['project', {}, { compaction: { reserveTokens: 300000 } }],
    ['exact', { compaction: { modelOverrides: { [provider + '/' + id]: { reserveTokens: 500000 } } } }, {}],
    ['unknown', { branchSummary: { reserveTokens: 'UNKNOWN_SECRET' } }, {}],
  ];
  for (const [name, user, project] of unsafe) for (const kind of ['ordinary', 'compact', 'branch', 'summary']) {
    await test('deny-' + name + '-' + kind, async () => {
      const fixture = await make('deny-' + name + '-' + kind, user, project, { seed: kind !== 'ordinary' });
      const error = await denied(() => action(fixture, kind), /effective.*budgets|effective branchSummary/);
      assert.match(error, /context=262144/); assert.match(error, /settings/); assert.doesNotMatch(error, /UNKNOWN_SECRET/);
    });
  }
  for (const [name, user, project] of [
    ['user', { compaction: { reserveTokens: 24000, keepRecentTokens: 10000 }, branchSummary: { reserveTokens: 24000 } }, {}],
    ['project', { compaction: { reserveTokens: 300000 } }, { compaction: { reserveTokens: 20000, keepRecentTokens: 12000 } }],
    ['exact', { compaction: { reserveTokens: 300000, modelOverrides: { [provider + '/' + id]: { reserveTokens: 16384, keepRecentTokens: 10000 } } } }, {}],
  ]) await test('safe-' + name + '-override', async () => { await action(await make('safe-override-' + name, user, project)); });
  await test('unbound-direct-streamSimple-denied', async () => {
    const f = await make('unbound', {}, {}, { unbound: true });
    await denied(() => f.services.modelRuntime.streamSimple(f.model, context, { apiKey: 'DO_NOT_FORWARD' }).result(), /effective token budgets unavailable/);
  });
  for (const method of ['stream', 'streamSimple']) await test('direct-' + method + '-safe-and-unsafe', async () => {
    const f = await make(method);
    const answer = await f.services.modelRuntime[method](f.model, context, { apiKey: 'DO_NOT_FORWARD', headers: { Authorization: 'Bearer SECRET', Cookie: 'SECRET' }, maxTokens: 128 }).result();
    assert.equal(answer.stopReason, 'stop'); assert.equal(answer.content[0].text, 'SAFE_STREAM_OK');
    assert.equal(posts().at(-1).body.max_tokens, 128); assert.equal(posts().at(-1).headers.authorization, 'Bearer loom-local');
    assert.equal(posts().at(-1).headers.cookie, undefined);
    f.settings.applyOverrides({ compaction: { reserveTokens: 300000 } });
    await denied(() => f.services.modelRuntime[method](f.model, context, { apiKey: 'fixture' }).result(), /incompatible effective budgets/);
  });
  await test('postbind-refresh-withholds-unsafe-selected-request-still-denied', async () => {
    const f = await make('refresh'); f.settings.applyOverrides({ compaction: { reserveTokens: 300000 } }); await refresh(f);
    assert.equal(f.services.modelRuntime.getModel(provider, id), undefined);
    assert.equal(f.session.model.id, id); await denied(() => action(f), /incompatible effective budgets/);
    f.settings.applyOverrides({ compaction: { reserveTokens: 16384 } }); await refresh(f);
    assert.ok(f.services.modelRuntime.getModel(provider, id)); await action(f);
  });
  await test('postbind-cache-only-refresh-filters-without-GET', async () => {
    const f = await make('cache-only'); f.settings.applyOverrides({ branchSummary: { reserveTokens: 300000 } });
    const before = requests.length;
    await f.services.modelRuntime.refresh({ allowNetwork: false, signal: AbortSignal.timeout(3000) });
    assert.equal(requests.length, before); assert.equal(f.services.modelRuntime.getModel(provider, id), undefined);
    await denied(() => action(f), /incompatible effective budgets/);
  });
  await test('selected-old-origin-rebinds-current-only-after-refresh', async () => {
    const f = await make('rebind'); f.endpoints.servers[0].base_url = b.base; f.saveEndpoints();
    await denied(() => action(f), /origin changed/);
    await refresh(f); const start = posts().length; await action(f);
    assert.equal(f.session.model.baseUrl, a.base); assert.equal(posts().length, start + 1); assert.equal(posts().at(-1).name, 'B');
    await f.services.modelRuntime.refresh({ allowNetwork: false, signal: AbortSignal.timeout(3000) });
    assert.equal(f.services.modelRuntime.getModel(provider, id).baseUrl, b.base);
  });
  for (const [name, user] of [
    ['null-compaction', { compaction: null }], ['array-branch', { branchSummary: [] }],
    ['null-branch-reserve', { branchSummary: { reserveTokens: null } }],
    ['negative-reserve', { compaction: { reserveTokens: -1 } }],
    ['fractional-recent', { compaction: { keepRecentTokens: 1.5 } }],
    ['bad-exact', { compaction: { modelOverrides: { [provider + '/' + id]: { reserveTokens: 'DO_NOT_ECHO' } } } }],
  ]) await test('invalid-effective-' + name, async () => {
    const f = await make(name, user);
    // Invalid compaction values can also be rejected by Pi's own prompt
    // preflight. Enter the actual public runtime directly to test our guard.
    const error = await denied(() => f.services.modelRuntime.streamSimple(f.model, context, {}).result(), /invalid effective/);
    assert.doesNotMatch(error, /DO_NOT_ECHO/);
  });
  for (const [name, change] of [
    ['context', () => { a.state.context = 131072; }], ['output', () => { a.state.output = 4096; }],
    ['profile', f => { f.endpoints.servers[0].native_profile = 'unsupported'; f.saveEndpoints(); }],
    ['unavailable', () => { a.state.available = false; }], ['retiring', () => { a.state.phase = 'retiring'; }],
  ]) await test('stale-' + name + '-denied', async () => {
    const f = await make('stale-' + name); change(f); await refresh(f);
    await denied(() => action(f), /Refresh\/reselect|repair/);
  });
  await test('output-request-and-payload-overrides-cannot-bypass-cap', async () => {
    const f = await make('payload');
    await denied(() => f.services.modelRuntime.stream(f.model, context, { maxTokens: 8193 }).result(), /maxTokens/);
    for (const onPayload of [payload => ({ ...payload, max_tokens: 9000 }), payload => { payload.max_tokens = 9000; },
      payload => ({ ...payload, model: 'another-instance' }), payload => ({ ...payload, max_completion_tokens: 9000 })]) {
      await denied(() => f.services.modelRuntime.streamSimple(f.model, context, { onPayload }).result(), /request payload/);
    }
    const answer = await f.services.modelRuntime.streamSimple(f.model, context, { onPayload: payload => ({ ...payload, max_tokens: 64 }) }).result();
    assert.equal(answer.stopReason, 'stop'); assert.equal(posts().at(-1).body.max_tokens, 64);
  });
  await test('settings-change-during-payload-denied-before-POST', async () => {
    const f = await make('payload-settings');
    await denied(() => f.services.modelRuntime.streamSimple(f.model, context, { onPayload: async () => {
      f.settings.applyOverrides({ branchSummary: { reserveTokens: 300000 } });
    } }).result(), /incompatible effective budgets/);
  });
  await test('abort-propagates-with-built-in-output-conventions', async () => {
    const f = await make('abort'), controller = new AbortController();
    const stream = f.services.modelRuntime.streamSimple(f.model, { ...context, messages: [{ role: 'user', content: [{ type: 'text', text: 'ABORT_FIXTURE' }], timestamp: Date.now() }] }, { signal: controller.signal });
    const timer = setTimeout(() => controller.abort(), 2000); let partial = false;
    for await (const event of stream) if (event.type === 'text_delta') { partial = true; controller.abort(); }
    clearTimeout(timer); const answer = await stream.result(); await new Promise(resolve => setTimeout(resolve, 50));
    assert.equal(partial, true); assert.equal(answer.stopReason, 'aborted'); assert.equal(posts().at(-1).closed, true);
  });
  await test('automatic-compaction-safe-public-session', async () => {
    const f = await make('auto-safe', {}, {}, { seed: true }), events = [], start = posts().length;
    f.session.subscribe(event => events.push(event.type)); await f.session.prompt('AUTO_FIXTURE');
    assert.ok(events.includes('compaction_start')); assert.equal(posts().length, start + 2);
  });
  await test('automatic-compaction-rechecks-effective-settings', async () => {
    const f = await make('auto-unsafe', {}, {}, { seed: true }), events = [], start = posts().length;
    let compactionError;
    f.session.subscribe(event => { events.push(event.type); if (event.type === 'compaction_end') compactionError = event.errorMessage;
      if (event.type === 'message_end' && event.message?.role === 'assistant')
      f.settings.applyOverrides({ branchSummary: { reserveTokens: 'UNKNOWN' } }); });
    await f.session.prompt('AUTO_FIXTURE'); assert.ok(events.includes('compaction_start')); assert.equal(posts().length, start + 1);
    assert.match(compactionError, /invalid effective branchSummary.reserveTokens/);
  });
  await test('small-native-context-output-cap-and-manual-budget-safety', async () => {
    const f = await make('small'); a.state.context = 8192; a.state.output = 1024;
    f.settings.applyOverrides({ compaction: { reserveTokens: 1536, keepRecentTokens: 2048 }, branchSummary: { reserveTokens: 1536 } });
    await refresh(f); const model = f.services.modelRuntime.getModel(provider, id);
    assert.equal(model.contextWindow, 8192); assert.equal(model.maxTokens, 1024);
    const answer = await f.services.modelRuntime.streamSimple(model, context, {}).result(); assert.equal(answer.stopReason, 'stop');
    assert.equal(posts().at(-1).body.max_tokens, 1024);
    f.settings.applyOverrides({ branchSummary: { reserveTokens: 16384 } });
    await denied(() => f.services.modelRuntime.streamSimple(model, context, {}).result(), /context=8192/);
  });
  for (const [name, edit] of [
    ['disabled-manifest', f => { f.manifest.enabled = false; f.saveManifest(); }],
    ['changed-dependency', f => writeFileSync(join(f.directory, 'budgets.ts'), '// edited')],
    ['removed-endpoint', f => { f.endpoints.servers = []; f.saveEndpoints(); }],
  ]) await test('dispatch-revalidates-' + name, async () => {
    const f = await make(name); edit(f); await denied(() => action(f), /Loom Hem Pi:/);
  });
  for (const [path, bytes] of originals) assert.deepEqual(readFileSync(path), bytes, 'settings file mutated: ' + path);
  assert.ok(posts().every(r => r.body.max_tokens <= 8192 && r.body.max_tokens > 0));
  assert.ok(requests.every(r => r.method === 'GET' || r.path === '/v1/chat/completions'));
  console.log(JSON.stringify({ node: process.version, profile, results, inferencePosts: posts().length, settingsFilesUnchanged: originals.size }, null, 2));
} finally {
  for (const session of sessions) session.dispose();
  for (const server of servers) { server.closeAllConnections(); await new Promise(resolve => server.close(resolve)); }
  rmSync(scratch, { recursive: true, force: true });
}
