import assert from 'node:assert/strict';
import { createHash, randomUUID } from 'node:crypto';
import { mkdtempSync, mkdirSync, readFileSync, writeFileSync, copyFileSync, rmSync, symlinkSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { createServer } from 'node:http';

const assets = resolve(process.argv[2]);
const { validateInstallation, ASSETS, parseDocument } = await import(pathToFileURL(join(assets, 'installation.ts')));
const { joinInventory, readEndpoints, validateBaseUrl, discover, applySnapshot } = await import(pathToFileURL(join(assets, 'discovery.ts')));
const root = mkdtempSync(join(tmpdir(), 'loom-oc2-adapter-'));
const target = join(root,'client'), directory = join(target,'plugins','loom-control-opencode'), state = join(root,'state');
mkdirSync(directory,{recursive:true}); mkdirSync(join(state,'clients'),{recursive:true});
const config = {schema_version:1,generation:randomUUID(),integration_id:randomUUID(),target,hem_root:state,endpoint_file:join(state,'clients','endpoints.json')};
const manifest = {schema_version:1,integration_version:1,enabled:true,generation:config.generation,integration_id:config.integration_id,target,artifacts:{}};
function writeManifest() {
  for(const name of [...ASSETS,'config.json']) manifest.artifacts[name]=createHash('sha256').update(readFileSync(join(directory,name))).digest('hex');
  writeFileSync(join(directory,'manifest.json'),JSON.stringify(manifest));
}
function projection(servers=[]) {writeFileSync(config.endpoint_file,JSON.stringify({schema_version:1,owner:'loom-control-clients',hem_root:state,servers}));}
function endpoint(server='local',port=18080) {return {server,provider:'loom-hem-'+Buffer.from(server).toString('base64url'),base_url:`http://127.0.0.1:${port}/v1`,native_profile:'loom-text-serve-61417137-v1'};}
function row(id='qwen-a') {return {object:'model',id,context_length:32768,limits:{max_output_tokens:16384},input_modalities:['text'],output_modalities:['text'],loom:{incarnation:1,registration:'ready',kind:'text',endpoint:'/v1/chat/completions',tokenize:true,parameter_operation:'idle',parameters:{active:true},status:{admission:'open',phase:'idle'}}};}
const list=(...data)=>({object:'list',data});
let checks=0;
function test(name,fn) {fn();checks++;console.log('PASS '+name);}
let server, replacementServer;
try {
  for(const name of ASSETS) copyFileSync(join(assets,name),join(directory,name));
  writeFileSync(join(directory,'config.json'),JSON.stringify(config)); writeManifest(); projection();
  test('valid owned generation and exact manifest',()=>assert.deepEqual(validateInstallation(join(directory,'index.ts')),config));
  test('duplicate JSON fields and excessive nesting rejected',()=>{assert.throws(()=>parseDocument('{"owner":1,"owner":2}'));assert.throws(()=>parseDocument('['.repeat(65)+'0'+']'.repeat(65)));assert.deepEqual(parseDocument('{"a\\u0062":[null,true,"x\\\"y",-2.4]}'),{ab:[null,true,'x"y',-2.4]});});
  test('partial generation/pending blocks',()=>{writeFileSync(join(directory,'pending.json'),'{}');assert.throws(()=>validateInstallation(join(directory,'index.ts')));rmSync(join(directory,'pending.json'));const original=manifest.generation;manifest.generation=randomUUID();writeManifest();assert.throws(()=>validateInstallation(join(directory,'index.ts')));manifest.generation=original;writeManifest();});
  test('edited companion refused',()=>{writeFileSync(join(directory,'discovery.ts'),'edited');assert.throws(()=>validateInstallation(join(directory,'index.ts')));copyFileSync(join(assets,'discovery.ts'),join(directory,'discovery.ts'));writeManifest();});
  test('foreign asset and integration identity rejected',()=>{manifest.artifacts['foreign.txt']='0'.repeat(64);writeFileSync(join(directory,'manifest.json'),JSON.stringify(manifest));assert.throws(()=>validateInstallation(join(directory,'index.ts')));delete manifest.artifacts['foreign.txt'];const id=manifest.integration_id;manifest.integration_id=randomUUID();writeManifest();assert.throws(()=>validateInstallation(join(directory,'index.ts')));manifest.integration_id=id;writeManifest();});
  test('managed symlink refused',()=>{const path=join(directory,'config.json'),data=readFileSync(path);rmSync(path);writeFileSync(join(root,'alias.json'),data);symlinkSync(join(root,'alias.json'),path);assert.throws(()=>validateInstallation(join(directory,'index.ts')));rmSync(path);writeFileSync(path,data);});
  test('numeric exact loopback origins',()=>{for(const url of ['http://127.0.0.1:80/v1','http://[::1]:18080/v1','http://127.255.2.3:65535/v1'])assert.equal(validateBaseUrl(url),url);for(const url of ['http://localhost:80/v1','https://127.0.0.1:80/v1','http://127.0.0.1:0/v1','http://127.256.0.1:80/v1','http://127.0.0.1:65536/v1','http://127.0.0.1/v1','http://127.0.0.1:80/v1?x','http://user@127.0.0.1:80/v1','http://127.0.0.1:080/v1'])assert.throws(()=>validateBaseUrl(url));});
  test('projection ownership/ordering/profile and count',()=>{projection([endpoint()]);assert.equal(readEndpoints(config).length,1);for(const rows of [[endpoint(),endpoint()],[{...endpoint(),provider:'wrong'}],[{...endpoint(),native_profile:'unknown'}],Array.from({length:257},(_,i)=>endpoint('n'+i))]){projection(rows);assert.throws(()=>readEndpoints(config));}projection();const text=readFileSync(config.endpoint_file,'utf8');writeFileSync(config.endpoint_file,text.replace('loom-control-clients','foreign'));assert.throws(()=>readEndpoints(config));projection();});
  test('native exact IDs truthful output without Pi cap',()=>{const models=joinInventory(list(row()),list(row()),endpoint());assert.equal(models[0].id,'qwen-a');assert.deepEqual(models[0].capabilities,{tools:false,input:['text'],output:['text']});assert.deepEqual(models[0].limit,{context:32768,output:16384});const small=row();small.context_length=4096;assert.equal(joinInventory(list(small),list(small),endpoint())[0].limit.output,4096);});
  test('primary residency with legacy flat compatibility and no auxiliary fallback',()=>{
    for (const [parameters,count] of [
      [{primary:{active:true},image:{active:false}},1],
      [{active:true},1],
      [{primary:{active:false},image:{active:true}},0],
      [{primary:null,active:true},0],
      [{image:{active:true}},0],
    ]) {
      const r=row();r.loom.parameters=parameters;
      assert.equal(joinInventory(list(r),list(r),endpoint()).length,count);
    }
  });
  test('malformed duplicate identities and mismatched snapshots',()=>{assert.throws(()=>joinInventory(list(row(),row()),list(row()),endpoint()));assert.throws(()=>joinInventory(list(row()),list(),endpoint()));const changed=row();changed.loom.incarnation=2;assert.throws(()=>joinInventory(list(row()),list(changed),endpoint()));});
  test('unready incompatible and invalid required limits excluded',()=>{for(const modify of [r=>r.loom.registration='pending',r=>r.loom.parameters.active=false,r=>r.loom.status.admission='closed',r=>r.loom.status.phase='load',r=>r.loom.error='failure',r=>r.loom.parameter_operation='load',r=>r.context_length=0,r=>r.limits.max_output_tokens=NaN,r=>r.input_modalities=['image'],r=>r.loom.tokenize=false]){const r=row();modify(r);assert.deepEqual(joinInventory(list(r),list(r),endpoint()),[]);}});
  let mode='ready',requests=[];
  server=createServer((req,res)=>{requests.push({url:req.url,auth:req.headers.authorization});if(mode==='hang')return;if(mode==='large'){res.writeHead(200,{'content-length':1048577});res.end();return;}if(mode==='redirect'){res.writeHead(302,{location:'http://example.com/'});res.end();return;}res.setHeader('content-type','application/json');res.end(mode==='malformed'?'{"object":"list","data":[],"data":[]}':JSON.stringify(mode==='empty'?list():list(row())));});
  await new Promise(r=>server.listen(0,'127.0.0.1',r));
  const ep=endpoint('local',server.address().port); projection([ep]);
  let snapshot=await discover(config,()=>validateInstallation(join(directory,'index.ts')));assert.equal(snapshot[0].models.length,1);assert(requests.every(r=>r.auth===undefined));assert.deepEqual(new Set(requests.map(r=>r.url)),new Set(['/v1/models','/v1/status']));checks++;console.log('PASS bounded keyless native joined discovery');
  for(mode of ['empty','malformed','large','redirect']){snapshot=await discover(config,()=>validateInstallation(join(directory,'index.ts')));assert.equal(snapshot[0].models.length,0);checks++;console.log('PASS no cached rows after '+mode);}
  mode='hang';const before=Date.now();snapshot=await discover(config,()=>validateInstallation(join(directory,'index.ts')));assert.equal(snapshot[0].models.length,0);assert(Date.now()-before<5000);checks++;console.log('PASS total bounded discovery deadline');server.closeAllConnections();
  mode='ready';await assert.rejects(discover(config,()=>({...config,generation:randomUUID()})));checks++;console.log('PASS generation change during refresh refused');
  projection([ep]);await assert.rejects(discover(config,()=>{projection([]);return config;}));checks++;console.log('PASS changed projection cannot mix old rows with new origin');
  const providerRows=new Map(),modelRows=new Map();
  const editor={get:id=>providerRows.get(id),update:(id,fn)=>{const target={};fn(target);providerRows.set(id,target);},models:{update:(p,id,fn)=>{const target={capabilities:{tools:true,input:['image']},limit:{context:200000,output:32000},variants:['optimistic']};fn(target);modelRows.set(p+'/'+id,target);}}};
  const ready={endpoint:ep,models:joinInventory(list(row()),list(row()),ep)};applySnapshot(editor,[ready]);assert.equal(providerRows.get(ep.provider).settings.baseURL,ep.base_url);assert.equal(providerRows.get(ep.provider).settings.apiKey,undefined);assert.deepEqual(modelRows.get(ep.provider+'/qwen-a').variants,[]);assert.equal(modelRows.get(ep.provider+'/qwen-a').compatibility.maxTokensField,'max_tokens');assert.throws(()=>applySnapshot(editor,[ready]));checks++;console.log('PASS public transform exact snapshot binding and foreign conflict refusal');
  // Each public host reload reconstructs a fresh source editor. Exercise the
  // actual discovery/transform pair, including successful URL/provider changes.
  function freshEditor() {
    const providers=new Map(), models=new Map();
    return {providers,models,editor:{get:id=>providers.get(id),update:(id,fn)=>{const value={};fn(value);providers.set(id,value);},models:{update:(provider,id,fn)=>{const value={};fn(value);models.set(provider+'/'+id,value);}}}};
  }
  replacementServer=createServer((req,res)=>{res.setHeader('content-type','application/json');res.end(JSON.stringify(list(row('qwen-b'))));});
  await new Promise(r=>replacementServer.listen(0,'127.0.0.1',r));
  const changed=endpoint('local',replacementServer.address().port), added=endpoint('new',server.address().port);
  projection([ep]);
  const originalEditor=freshEditor();applySnapshot(originalEditor.editor,await discover(config,()=>validateInstallation(join(directory,'index.ts'))));
  assert.deepEqual([...originalEditor.models.keys()],[ep.provider+'/qwen-a']);
  projection([changed,added]);
  const replacedEditor=freshEditor();applySnapshot(replacedEditor.editor,await discover(config,()=>validateInstallation(join(directory,'index.ts'))));
  assert.deepEqual([...replacedEditor.providers.keys()],[changed.provider,added.provider]);
  assert.equal(replacedEditor.providers.get(changed.provider).settings.baseURL,changed.base_url);
  assert(!replacedEditor.models.has(ep.provider+'/qwen-a'));
  assert(replacedEditor.models.has(changed.provider+'/qwen-b'));
  assert(replacedEditor.models.has(added.provider+'/qwen-a'));
  // Previously captured in-flight routing is unchanged; only the new source
  // snapshot gets the new origin and roster.
  assert.equal(originalEditor.providers.get(ep.provider).settings.baseURL,ep.base_url);
  projection([added]);
  const removedEditor=freshEditor();applySnapshot(removedEditor.editor,await discover(config,()=>validateInstallation(join(directory,'index.ts'))));
  assert.deepEqual([...removedEditor.providers.keys()],[added.provider]);
  assert.deepEqual([...removedEditor.models.keys()],[added.provider+'/qwen-a']);
  mode='empty';
  const unavailableEditor=freshEditor();applySnapshot(unavailableEditor.editor,await discover(config,()=>validateInstallation(join(directory,'index.ts'))));
  assert.equal(unavailableEditor.models.size,0);assert.equal(unavailableEditor.providers.size,0);
  checks++;console.log('PASS complete fresh snapshots replace origin/add/remove providers without stale rows or rerouting prior snapshot');
  console.log(JSON.stringify({checks,assets,owned_directory:directory}));
} finally {if(replacementServer){replacementServer.closeAllConnections();await new Promise(r=>replacementServer.close(r));}if(server){server.closeAllConnections();await new Promise(r=>server.close(r));}rmSync(root,{recursive:true,force:true});}
