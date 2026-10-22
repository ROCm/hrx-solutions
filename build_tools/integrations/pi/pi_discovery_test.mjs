// Run beside an isolated, locked Pi 1.3.0 package.json. No GPU or user config.
import assert from 'node:assert/strict';
import { createHash, randomUUID } from 'node:crypto';
import { createServer } from 'node:http';
import { mkdtempSync, mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createJiti } from 'jiti';
const { discoverAndLoadExtensions } = await import('@earendil-works/pi-coding-agent');
const { createModels } = await import('@earendil-works/pi-ai');
const assets = process.env.LOOM_PI_ASSET_DIR;
assert.ok(assets, 'explicit source/installed asset directory required');
const scratch = mkdtempSync(join(process.env.LOOM_PI_TEST_SCRATCH || tmpdir(), 'pi-discovery-'));
const names = ['index.ts', 'discovery.ts', 'budgets.ts', 'stream.ts', 'profile.json', 'package.json'];
// Direct module seam tests need the same public package resolution as the
// supported Pi loader; copied fixtures deliberately live outside node_modules.
const jiti = () => createJiti(import.meta.url, { alias: Object.fromEntries(
  ['@earendil-works/pi-ai', '@earendil-works/pi-coding-agent'].map(name => [name, fileURLToPath(import.meta.resolve(name))])) });
const profile = JSON.parse(readFileSync(join(assets, 'profile.json')));
const results = [], requests = [], servers = [], sockets = new Set();
let response = (path) => ({ object:'list', data:[row(path === '/v1/status')] });
let handler;
function row(status=false) {
  return { object:'model', id:'model-a', context_length:262144, limits:{max_output_tokens:16384},
    input_modalities:['text'], output_modalities:['text'], loom:{kind:'text',endpoint:'/v1/chat/completions',tokenize:true,
      incarnation:1,registration:'ready',...(status ? {parameter_operation:'idle',parameters:{active:true},status:{phase:'idle',admission:'open'}} : {})} };
}
async function server() {
  const s = createServer(async (req,res)=> {
    requests.push({method:req.method,path:req.url,headers:req.headers});
    if (handler) return handler(req,res);
    res.setHeader('content-type','application/json'); res.end(JSON.stringify(response(req.url)));
  });
  s.on('connection', socket=>{sockets.add(socket);socket.on('close',()=>sockets.delete(socket));});
  await new Promise(resolve=>s.listen(0,'127.0.0.1',resolve)); servers.push(s);
  return `http://127.0.0.1:${s.address().port}/v1`;
}
const base = await server();
function setup(label, endpoints=[{server:'local',provider:'loom-hem-bG9jYWw',base_url:base,native_profile:profile.native_profile}]) {
  const target = join(scratch,label), dir = join(target,'extensions/loom-hem'), root=join(target,'state');
  mkdirSync(dir,{recursive:true}); mkdirSync(join(target,'loom-hem')); mkdirSync(join(root,'integrations/pi'),{recursive:true});
  const artifacts={};
  for(const name of names){ const bytes=readFileSync(join(assets,name));writeFileSync(join(dir,name),bytes);artifacts['extensions/loom-hem/'+name]=createHash('sha256').update(bytes).digest('hex'); }
  const manifest={schema_version:1,integration_version:1,integration_id:randomUUID(),enabled:true,hem_root:root,target,
    extension_version:'1',native_profile:profile.native_profile,native_commit:profile.native_commit,pi_package:profile.pi_package,
    pi_versions:profile.pi_versions,endpoint_file:join(root,'integrations/pi/endpoints.json'),artifacts};
  const endpointDoc={schema_version:1,integration_id:manifest.integration_id,hem_root:root,servers:endpoints};
  const save=()=>writeFileSync(join(target,'loom-hem/config.json'),JSON.stringify(manifest));
  const endpointsSave=()=>writeFileSync(manifest.endpoint_file,JSON.stringify(endpointDoc));
  save();endpointsSave();return{target,dir,root,manifest,endpointDoc,save,endpointsSave};
}
async function load(fixture) {
  return discoverAndLoadExtensions([fixture.dir],fixture.target,fixture.target);
}
async function test(label, body){await body();results.push({scenario:label,status:'passed'});}
try {
  await test('public-loader-initial-ready-budget-unverified',async()=>{
    const fixture=setup('ready'), loaded=await load(fixture);
    assert.deepEqual(loaded.errors,[]);
    const provider=loaded.runtime.pendingNativeProviderRegistrations[0].provider;
    const models=provider.getModels();assert.equal(models.length,1);assert.equal(models[0].contextWindow,262144);assert.equal(models[0].maxTokens,8192);
    assert.equal(models[0].reasoning,false);assert.deepEqual(models[0].input,['text']);
    const before=requests.length;await provider.refreshModels({allowNetwork:false,signal:new AbortController().signal,publish:async({update})=>{update();return true;}});
    assert.equal(requests.length,before);assert.equal(provider.getModels().length,1);
    assert.throws(()=>provider.stream(models[0],{messages:[]}),/effective token budgets unavailable/);
    await provider.refreshModels({allowNetwork:true,signal:new AbortController().signal,publish:async({update})=>{update();return true;}});
    assert.equal(provider.getModels().length,0,'post-bind validation missing must not advertise safe models');
  });
  await test('mixed-text-recognition-keeps-chat-only',async()=>{
    response=path=>{
      const text=row(path==='/v1/status'),speech=row(path==='/v1/status');speech.id='dictation';
      speech.loom.kind='recognition';speech.loom.endpoint='/v1/audio/transcriptions';speech.input_modalities=['audio'];
      return {object:'list',data:[text,speech]};
    };
    const loaded=await load(setup('mixed-text-recognition'));
    assert.deepEqual(loaded.errors,[]);
    assert.deepEqual(loaded.runtime.pendingNativeProviderRegistrations[0].provider.getModels().map(r=>r.id),['model-a']);
  });
  const cases = [
    ['primary-ready',r=>r.loom.parameters={primary:{active:true},image:{active:false}},1],
    ['primary-inactive-image-active',r=>r.loom.parameters={primary:{active:false},image:{active:true}},0],
    ['malformed-primary-no-flat-fallback',r=>r.loom.parameters={primary:null,active:true},0],
    ['image-only',r=>r.loom.parameters={image:{active:true}},0],
    ['busy-accepting',r=>{r.loom.status.phase='execute';r.loom.status.active_requests=2;},1],
    ['drained',r=>r.loom.status.admission='drained',0],['loading',r=>r.loom.registration='loading',0],
    ['retiring',r=>r.loom.status.phase='retiring',0],['terminal-error',r=>r.loom.error={message:'failed'},0],
    ['status-error',r=>r.loom.status.error='failed',0],['inactive',r=>r.loom.parameters.active=false,0],
    ['operation-busy',r=>r.loom.parameter_operation='activating',0],['missing-parameters',r=>delete r.loom.parameters,0],
    ['missing-admission',r=>delete r.loom.status.admission,0],['unknown-phase',r=>r.loom.status.phase='unknown',0],
    ['wrong-modality',r=>r.input_modalities=['image'],0],['missing-context',r=>delete r.context_length,0],
    ['invalid-output',r=>r.limits.max_output_tokens=1.5,0],['missing-incarnation',r=>delete r.loom.incarnation,0],
    ['unsupported-kind',r=>r.loom.kind='audio',0],['no-tokenize',r=>r.loom.tokenize=false,0],
  ];
  for(const [label,change,count] of cases) await test(label,async()=>{
    response=path=>{const r=row(path==='/v1/status');if(path==='/v1/status')change(r);return{object:'list',data:[r]};};
    const loaded=await load(setup(label));assert.deepEqual(loaded.errors,[]);assert.equal(loaded.runtime.pendingNativeProviderRegistrations[0].provider.getModels().length,count);
  });
  response=()=>({object:'list',data:[]});
  await test('confirmed-empty',async()=>{const loaded=await load(setup('empty'));assert.deepEqual(loaded.errors,[]);assert.equal(loaded.runtime.pendingNativeProviderRegistrations[0].provider.getModels().length,0);});
  response=path=>({object:'list',data:[row(path==='/v1/status')]});
  await test('duplicate-model-names-separate-providers',async()=>{
    const fixture=setup('duplicates',[{server:'A',provider:'loom-hem-QQ',base_url:base,native_profile:profile.native_profile},{server:'B',provider:'loom-hem-Qg',base_url:base,native_profile:profile.native_profile}]);
    const loaded=await load(fixture);assert.deepEqual(loaded.errors,[]);const providers=loaded.runtime.pendingNativeProviderRegistrations.map(item=>item.provider);
    assert.equal(providers.length,2);assert.equal(providers[0].getModels()[0].id,providers[1].getModels()[0].id);assert.notEqual(providers[0].id,providers[1].id);
  });
  await test('more-than-128-servers-within-byte-bound',async()=>{
    const endpoints=Array.from({length:129},(_,i)=>{const name='s'+String(i).padStart(3,'0');return{server:name,provider:'loom-hem-'+Buffer.from(name).toString('base64url'),base_url:base,native_profile:profile.native_profile};});
    const loaded=await load(setup('many-servers',endpoints));assert.deepEqual(loaded.errors,[]);
    const providers=loaded.runtime.pendingNativeProviderRegistrations.map(item=>item.provider);
    assert.equal(providers.length,129);assert.ok(providers.every(provider=>provider.getModels().length===1));
  });
  await test('long-server-and-instance-names-within-byte-bound',async()=>{
    const name='server-'+('a'.repeat(300)),id='instance-'+('b'.repeat(300));
    response=path=>{const item=row(path==='/v1/status');item.id=id;return{object:'list',data:[item]};};
    const loaded=await load(setup('long-names',[{server:name,provider:'loom-hem-'+Buffer.from(name).toString('base64url'),base_url:base,native_profile:profile.native_profile}]));
    assert.deepEqual(loaded.errors,[]);assert.equal(loaded.runtime.pendingNativeProviderRegistrations[0].provider.getModels()[0].id,id);
    response=path=>({object:'list',data:[row(path==='/v1/status')]});
  });
  await test('one-snapshot-retry',async()=>{let statuses=0;const before=requests.length;
    response=path=>{const r=row(path==='/v1/status');if(path==='/v1/status' && statuses++===0)r.loom.incarnation=2;return{object:'list',data:[r]};};
    const loaded=await load(setup('retry'));assert.equal(loaded.runtime.pendingNativeProviderRegistrations[0].provider.getModels().length,1);assert.equal(requests.length-before,4);
  });
  await test('repeated-inconsistent-withheld',async()=>{const before=requests.length;
    response=path=>{const r=row(path==='/v1/status');if(path==='/v1/status')r.loom.incarnation=2;return{object:'list',data:[r]};};
    const loaded=await load(setup('inconsistent'));assert.equal(loaded.runtime.pendingNativeProviderRegistrations[0].provider.getModels().length,0);assert.equal(requests.length-before,4);
  });
  for(const [label,body] of [['wrong-schema','{}'],['malformed','{"secret":"DO_NOT_ECHO"'],['oversized','x'.repeat(1024*1024+1)]]) await test(label,async()=>{
    handler=(_req,res)=>res.end(body);const loaded=await load(setup(label));assert.deepEqual(loaded.errors,[]);assert.equal(loaded.runtime.pendingNativeProviderRegistrations[0].provider.getModels().length,0);handler=undefined;
  });
  await test('redirect-refused',async()=>{const before=requests.length;handler=(_req,res)=>{res.writeHead(302,{location:base+'/secret'});res.end();};
    const loaded=await load(setup('redirect'));assert.equal(loaded.runtime.pendingNativeProviderRegistrations[0].provider.getModels().length,0);assert.equal(requests.length-before,2);handler=undefined;
  });
  for(const [label,url] of [['external','http://example.com:80/v1'],['userinfo',base.replace('http://','http://name:secret@')],['query',base+'?secret=value'],['normalized-host','http://2130706433:1234/v1']]) await test(label,async()=>{
    const before=requests.length;const f=setup(label);f.endpointDoc.servers[0].base_url=url;f.endpointsSave();const loaded=await load(f);assert.equal(loaded.errors.length,1);assert.match(loaded.errors[0].error,/loom-control pi configure/);assert.equal(requests.length,before);
  });
  await test('identity-rejected-before-network',async()=>{const before=requests.length;const f=setup('identity');f.endpointDoc.integration_id=randomUUID();f.endpointsSave();assert.equal((await load(f)).errors.length,1);assert.equal(requests.length,before);});
  await test('total-three-second-bound',async()=>{handler=()=>{};const start=performance.now();const f=setup('timeout');const loaded=await load(f);
    assert.ok(performance.now()-start<3800);assert.ok(performance.now()-start>=2800);assert.equal(loaded.runtime.pendingNativeProviderRegistrations[0].provider.getModels().length,0);handler=undefined;});
  await test('bounded-parallel-endpoints-share-deadline',async()=>{
    let active=0,peak=0;handler=async(req,res)=>{active++;peak=Math.max(peak,active);await new Promise(resolve=>setTimeout(resolve,35));active--;res.end(JSON.stringify({object:'list',data:[row(req.url==='/v1/status')]}));};
    const endpoints=Array.from({length:12},(_,i)=>{const name='s'+String(i).padStart(2,'0');return{server:name,provider:'loom-hem-'+Buffer.from(name).toString('base64url'),base_url:base,native_profile:profile.native_profile};});
    const loaded=await load(setup('parallel',endpoints));assert.deepEqual(loaded.errors,[]);assert.equal(loaded.runtime.pendingNativeProviderRegistrations.length,12);assert.ok(peak<=8);assert.ok(peak>2);handler=undefined;
  });
  await test('projection-change-during-network-withholds-old-origin',async()=>{
    const f=setup('mid-refresh');let changed=false;handler=(req,res)=>{if(!changed){changed=true;f.endpointDoc.servers[0].base_url=base.replace('/v1','')+'/v1';f.endpointDoc.servers=[];f.endpointsSave();}res.end(JSON.stringify({object:'list',data:[row(req.url==='/v1/status')]}));};
    const loaded=await load(f);assert.deepEqual(loaded.errors,[]);assert.equal(loaded.runtime.pendingNativeProviderRegistrations[0].provider.getModels().length,0);handler=undefined;
  });
  response=path=>({object:'list',data:[row(path==='/v1/status')]});
  // Directly import the same reviewed module for its explicit .3.2 filter hook,
  // with no Pi-private imports or fabricated budget implementation claim.
  const f=setup('refresh');const module=await jiti().import(join(f.dir,'discovery.ts'));
  await test('more-than-1024-model-rows-within-byte-bound',async()=>{
    const document=status=>({object:'list',data:Array.from({length:1025},(_,i)=>({...row(status),id:'model-'+i}))});
    const catalog=document(false),status=document(true);
    assert.ok(Buffer.byteLength(JSON.stringify(catalog))<=1024*1024);assert.ok(Buffer.byteLength(JSON.stringify(status))<=1024*1024);
    assert.equal(module.joinInventory(catalog,status,f.endpointDoc.servers[0]).length,1025);
  });
  await test('canonical-loopback-and-default-port',async()=>{
    assert.equal(module.validateBaseUrl('http://127.0.0.1:80/v1'),'http://127.0.0.1/v1');
    assert.equal(module.validateBaseUrl('http://127.0.0.1/v1'),'http://127.0.0.1/v1');
    assert.equal(module.validateBaseUrl('http://[::1]:80/v1'),'http://[::1]/v1');
    assert.equal(module.validateBaseUrl('http://127.0.0.2:1234/v1'),'http://127.0.0.2:1234/v1');
    assert.throws(()=>module.validateBaseUrl('not a URL'),/loom-control pi configure/);
    assert.throws(()=>module.validateBaseUrl('http://localhost:1234/v1'),/loom-control pi configure/);
  });
  const providers=[];const discovery=await module.registerProviders({registerProvider:p=>providers.push(p)},f.manifest,{filterModels:models=>models});
  const multi=setup('registry-refresh',[{server:'A',provider:'loom-hem-QQ',base_url:base,native_profile:profile.native_profile},{server:'B',provider:'loom-hem-Qg',base_url:base,native_profile:profile.native_profile}]);
  const multiModule=await jiti().import(join(multi.dir,'discovery.ts'));
  const multiProviders=[];await multiModule.registerProviders({registerProvider:p=>multiProviders.push(p)},multi.manifest,{filterModels:models=>models});
  const registry=createModels();for(const provider of multiProviders)registry.setProvider(provider);
  await test('public-registry-concurrent-providers-share-one-scan',async()=>{
    const before=requests.length;const refreshed=await registry.refresh({allowNetwork:true});
    assert.equal(refreshed.errors.size,0);assert.equal(refreshed.aborted,false);
    assert.deepEqual(multiProviders.map(p=>p.getModels().length),[1,1]);
    assert.equal(requests.length-before,4,'distinct Pi provider signals must coalesce into one paired-endpoint sweep');
  });
  await test('public-registry-one-provider-abort-preserves-sibling',async()=>{
    const before=requests.length;handler=async(req,res)=>{await new Promise(resolve=>setTimeout(resolve,65));res.end(JSON.stringify({object:'list',data:[row(req.url==='/v1/status')]}));};
    const cancelA=new AbortController();
    const a=registry.refresh({providers:[multiProviders[0].id],signal:cancelA.signal});
    const b=registry.refresh({providers:[multiProviders[1].id]});
    await new Promise(resolve=>setTimeout(resolve,25));cancelA.abort();
    const [cancelled,healthy]=await Promise.all([a,b]);handler=undefined;
    assert.equal(cancelled.aborted,true);assert.equal(healthy.aborted,false);assert.equal(healthy.errors.size,0);
    assert.deepEqual(multiProviders.map(p=>p.getModels().length),[1,1]);assert.equal(requests.length-before,4);
  });
  const provider=providers[0];
  const refresh=async(signal=new AbortController().signal)=>provider.refreshModels({allowNetwork:true,signal,publish:async({update})=>{update();return true;}});
  await test('offline-clears-in-memory-only',async()=>{handler=(_req,res)=>{res.writeHead(503);res.end();};await refresh();assert.equal(provider.getModels().length,0);assert.equal(f.endpointDoc.servers.length,1);handler=undefined;});
  await test('recovery-refresh',async()=>{await refresh();assert.equal(provider.getModels().length,1);});
  await test('port-rebind',async()=>{const newBase=await server();f.endpointDoc.servers[0].base_url=newBase;f.endpointsSave();await refresh();assert.equal(provider.getModels()[0].baseUrl,newBase);assert.equal(discovery.currentEndpoints()[0].base_url,newBase);});
  await test('new-server-requires-reload',async()=>{f.endpointDoc.servers.push({server:'new',provider:'loom-hem-bmV3',base_url:base,native_profile:profile.native_profile});f.endpointsSave();await refresh();assert.equal(providers.length,1);});
  await test('empty-refresh-removes-catalog',async()=>{response=()=>({object:'list',data:[]});await refresh();assert.equal(provider.getModels().length,0);response=path=>({object:'list',data:[row(path==='/v1/status')]});});
  await test('superseded-generation-cannot-publish',async()=>{let release;handler=async(req,res)=>{await new Promise(resolve=>release=resolve);res.end(JSON.stringify({object:'list',data:[row(req.url==='/v1/status')]}));};
    const old=new AbortController();const first=refresh(old.signal);await new Promise(resolve=>setTimeout(resolve,30));handler=undefined;const second=refresh();await Promise.all([first,second]);release?.();assert.equal(provider.getModels().length,1);
  });
  await test('disabled-refresh-clears-and-repair',async()=>{f.manifest.enabled=false;f.save();await refresh();assert.equal(provider.getModels().length,0);assert.match(discovery.observations.get(provider.id).reason,/loom-control pi configure/);f.manifest.enabled=true;f.save();});
  await test('disabled-cache-only-refresh-clears',async()=>{await refresh();assert.equal(provider.getModels().length,1);f.manifest.enabled=false;f.save();const before=requests.length;
    await provider.refreshModels({allowNetwork:false,signal:new AbortController().signal,publish:async({update})=>{update();return true;}});assert.equal(requests.length,before);assert.equal(provider.getModels().length,0);f.manifest.enabled=true;f.save();});
  await test('torn-dependency-refresh-clears',async()=>{writeFileSync(join(f.dir,'budgets.ts'),'throw Error("DO_NOT_EXECUTE")');await refresh();assert.equal(provider.getModels().length,0);});
  assert.ok(requests.every(r=>r.method==='GET' && ['/v1/models','/v1/status'].includes(r.path)));
  assert.ok(requests.every(r=>r.headers.authorization===undefined && r.headers.cookie===undefined));
  console.log(JSON.stringify({node:process.version,profile,results,requests:requests.length},null,2));
} finally {
  for(const socket of sockets)socket.destroy();
  await Promise.all(servers.map(s=>new Promise(resolve=>s.close(resolve))));
  rmSync(scratch,{recursive:true,force:true});
}
