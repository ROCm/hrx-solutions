/** Inspect generated, unmodified assets through the genuine public OMP host. */
import assert from 'node:assert/strict';
import {join} from 'node:path';
import {pathToFileURL} from 'node:url';
const {AuthStorage,ModelRegistry,loadExtensions}=await import(pathToFileURL(join(process.env.LOOM_OMP_TEST_PACKAGE,'src/index.ts')));
const target=process.env.PI_CODING_AGENT_DIR;
const auth=await AuthStorage.create(join(process.env.XDG_DATA_HOME,'acceptance-auth.db'));
const registry=new ModelRegistry(auth,join(target,'models.yml'),{cacheDbPath:join(process.env.XDG_CACHE_HOME,'acceptance-models.db')});
const loaded=await loadExtensions([join(target,'extensions/loom-control-omp/index.ts')],process.cwd());
assert.deepEqual(loaded.errors,[]);assert.equal(loaded.extensions.length,1);
for(const pending of loaded.runtime.pendingProviderRegistrations)registry.registerProvider(pending.name,pending.config,pending.sourceId);
await registry.refreshRuntimeProviders('online');
for(const handler of loaded.extensions[0].handlers.get('session_start')??[])await handler({},{modelRegistry:registry});
const providers=new Set(loaded.runtime.pendingProviderRegistrations.map(row=>row.name));
const models=registry.getAvailableForProviders(providers);
console.log(JSON.stringify({models:models.map(model=>({provider:model.provider,id:model.id,baseUrl:model.baseUrl,
  contextWindow:model.contextWindow,maxTokens:model.maxTokens,supportsTools:model.supportsTools,
  reasoning:model.reasoning,input:model.input}))}));
process.exit(0);
