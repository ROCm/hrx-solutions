/** Genuine pinned public OMP loader/API contract. Synthetic HTTP only; real acceptance is separate. */
import assert from 'node:assert/strict';
import { createHash,randomUUID } from 'node:crypto';
import { mkdtempSync,mkdirSync,readFileSync,writeFileSync,copyFileSync,rmSync } from 'node:fs';
import { join,resolve } from 'node:path';
import { tmpdir } from 'node:os';
import { pathToFileURL } from 'node:url';
const pkg=process.env.LOOM_OMP_TEST_PACKAGE;
if(!pkg)throw Error('Set LOOM_OMP_TEST_PACKAGE to qualified @oh-my-pi/pi-coding-agent package');
const {AuthStorage,ModelRegistry,loadExtensions}=await import(pathToFileURL(join(pkg,'src/index.ts')));
const {streamSimple}=await import(pathToFileURL(resolve(pkg,'../pi-ai/src/index.ts')));
const assets=resolve(process.argv[2]), root=mkdtempSync(join(tmpdir(),'loom-omp-loader-'));
const target=join(root,'agent'),directory=join(target,'extensions','loom-control-omp'),state=join(root,'state');
mkdirSync(directory,{recursive:true});mkdirSync(join(state,'clients'),{recursive:true});mkdirSync(join(root,'cwd'));
process.env.PI_CODING_AGENT_DIR=target;process.env.OMP_PROFILE='';process.env.PI_PROFILE='';
const provider='loom-hem-bG9jYWw',id='qwen-a';let incarnation=1,contextWindow=32768,available=true,wire=[],mode='ready';
const model=()=>({object:'model',id,context_length:contextWindow,limits:{max_output_tokens:16384},input_modalities:['text'],output_modalities:['text'],loom:{incarnation,registration:'ready',kind:'text',endpoint:'/v1/chat/completions',tokenize:true,parameter_operation:'idle',parameters:{active:true},status:{admission:'open',phase:'idle'}}});
const leaked=[];
const destination=Bun.serve({hostname:'127.0.0.1',port:0,async fetch(req){leaked.push({url:req.url,headers:Object.fromEntries(req.headers),body:await req.text()});return new Response('unexpected disclosure');}});
const server=Bun.serve({hostname:'127.0.0.1',port:0,async fetch(req){
 if(req.method==='GET')return new Response(mode==='malformed'?'bad':JSON.stringify({object:'list',data:available?[model()]:[]}),{headers:{'content-type':'application/json'}});
 if(mode==='redirect307'||mode==='redirect308')return new Response(null,{status:mode==='redirect307'?307:308,headers:{location:`http://127.0.0.1:${destination.port}/stolen`}});
 const body=await req.json();wire.push({headers:Object.fromEntries(req.headers),body});
 const chunk={id:'fixture',object:'chat.completion.chunk',created:1,model:id,choices:[{index:0,delta:{role:'assistant',content:'protocol-ok'},finish_reason:null}]};
 const done={...chunk,choices:[{index:0,delta:{},finish_reason:'stop'}],usage:{prompt_tokens:3,completion_tokens:2,total_tokens:5}};
 return new Response([chunk,done].map(c=>'data: '+JSON.stringify(c)+'\n\n').join('')+'data: [DONE]\n\n',{headers:{'content-type':'text/event-stream'}});
}});
try{
 const config={schema_version:1,generation:randomUUID(),integration_id:randomUUID(),target,hem_root:state,endpoint_file:join(state,'clients','endpoints.json'),native_models_file:join(target,'models.yml')};
 const stanza=`  # loom-control omp begin ${config.integration_id} ${provider}\n  ${provider}:\n    auth: none\n  # loom-control omp end ${config.integration_id} ${provider}\n`;
 const manifest={schema_version:1,integration_version:1,enabled:true,generation:config.generation,integration_id:config.integration_id,target,artifacts:{},native_activation:{heading:null,path:config.native_models_file,providers:{[provider]:{ownership:'owned',sha256:createHash('sha256').update(stanza).digest('hex')}}}};
 const assetNames=['index.ts','installation.ts','discovery.ts','activation.ts','stream.ts','profile.json','package.json'];
 for(const name of assetNames)copyFileSync(join(assets,name),join(directory,name));
 writeFileSync(join(directory,'config.json'),JSON.stringify(config));
 for(const name of [...assetNames,'config.json'])manifest.artifacts[name]=createHash('sha256').update(readFileSync(join(directory,name))).digest('hex');
 const {validateInstallation}=await import(pathToFileURL(join(directory,'installation.ts')));
 for(const heading of [
   {original:'providers: {}\r\n',replacement:'providers: \r\n',created_file:false},
   {original:'providers: {} # preserve literal {} here\r\n',replacement:'providers: # preserve literal {} here\r\n',created_file:false}
 ]) {
   manifest.native_activation.heading=heading;
   writeFileSync(join(directory,'manifest.json'),JSON.stringify(manifest));
   writeFileSync(config.native_models_file,'# user comment\n'+heading.replacement+stanza);
   assert.deepEqual(validateInstallation(join(directory,'index.ts')),config);
 }
 console.log('PASS copied TS loader accepts comment and plain CRLF heading provenance');
 const endpoint={server:'local',provider,base_url:`http://127.0.0.1:${server.port}/v1`,native_profile:'loom-text-serve-61417137-v1'};
 writeFileSync(config.endpoint_file,JSON.stringify({schema_version:1,owner:'loom-control-clients',hem_root:state,servers:[endpoint]}));
 const originalYaml=readFileSync(config.native_models_file,'utf8');
 const auth=await AuthStorage.create(join(root,'auth.db')),registry=new ModelRegistry(auth,config.native_models_file,{cacheDbPath:join(root,'models.db')});
 const loaded=await loadExtensions([join(directory,'index.ts')],join(root,'cwd'));assert.deepEqual(loaded.errors,[]);assert.equal(loaded.extensions.length,1);
 for(const pending of loaded.runtime.pendingProviderRegistrations)registry.registerProvider(pending.name,pending.config,pending.sourceId);
 await registry.refreshRuntimeProviders('online');
 const get=()=>registry.getAvailableForProviders(new Set([provider]));let selected=get()[0];assert(selected);assert.equal(selected.contextWindow,32768);assert.equal(selected.maxTokens,16384);assert.equal(selected.supportsTools,false);assert((await selected.resolveHeaders())['x-loom-control-snapshot']);
 const key=await registry.getApiKey(selected);assert.equal(key,'N/A');
 const context={messages:[{role:'user',content:'Synthetic transport fixture',timestamp:Date.now()}]};
 const request=(model=selected,options={})=>streamSimple(model,context,{apiKey:key,maxTokens:16,signal:AbortSignal.timeout(5000),...options}).result();
 async function rejected(promise,pattern){try{const answer=await promise;assert.equal(answer.stopReason,'error',JSON.stringify(answer));assert.match(answer.errorMessage||'',pattern);}catch(error){assert.match(error.message,pattern);}}
 let answer=await request();assert.equal(answer.stopReason,'stop',JSON.stringify(answer));assert.equal(answer.content[0].text,'protocol-ok');
 assert.equal(wire.length,1);assert.equal(wire[0].headers.authorization,undefined);assert.equal(wire[0].headers['x-loom-control-snapshot'],undefined);assert.equal(wire[0].body.model,id);assert.equal(wire[0].body.max_tokens,16);
 console.log('PASS actual copied loader, normalized snapshot, public guarded keyless builtin transport');
 await request(selected,{headers:{Authorization:'secret-ambient','x-foreign':'secret-header'}});assert.equal(wire.length,2);assert.equal(wire[1].headers.authorization,undefined);assert.equal(wire[1].headers['x-foreign'],undefined);
 incarnation=2;await registry.refreshRuntimeProviders('online');const fresh=get()[0];assert.notEqual((await fresh.resolveHeaders())['x-loom-control-snapshot'],(await selected.resolveHeaders())['x-loom-control-snapshot']);
 await rejected(request(selected),/snapshot\/origin\/incarnation/);assert.equal(wire.length,2);selected=fresh;await request();assert.equal(wire.length,3);
 contextWindow=24576;
 for(const handler of loaded.extensions[0].handlers.get('session_start'))await handler({},{modelRegistry:registry});
 assert.equal(get()[0].contextWindow,24576);await rejected(request(selected),/snapshot\/origin\/incarnation/);assert.equal(wire.length,3);selected=get()[0];
 await rejected(request(selected,{maxTokens:20000}),/output exceeds/);assert.equal(wire.length,3);
 await rejected(request(selected,{onPayload:payload=>({...payload,model:'other'})}),/payload changes/);assert.equal(wire.length,3);
 for(const field of ['tools','tool_choice','parallel_tool_calls','functions','function_call','reasoning_effort','reasoning','thinking','enable_thinking','include_reasoning']){
   await rejected(request(selected,{onPayload:payload=>({...payload,[field]:field==='tools'?[{type:'function',function:{name:'unsupported'}}]:'unsupported'})}),/unsupported tools or reasoning/);
   assert.equal(wire.length,3);
 }
 console.log('PASS final payload cannot reintroduce unsupported tool/reasoning fields');
 mode='malformed';await registry.refreshRuntimeProviders('online');assert.deepEqual(get(),[]);await rejected(request(selected),/unavailable/);assert.equal(wire.length,3);
 mode='ready';available=true;await registry.refreshRuntimeProviders('online');assert.equal(get().length,1);
 writeFileSync(config.native_models_file,'providers: {}\n');await registry.refreshRuntimeProviders('online');assert.deepEqual(get(),[]);await rejected(request(selected),/unavailable/);assert.equal(wire.length,3);
 writeFileSync(config.native_models_file,originalYaml);await registry.refreshRuntimeProviders('online');assert.equal(get().length,1);
 assert.equal(readFileSync(config.native_models_file,'utf8'),originalYaml);
 selected=get()[0];
 for(mode of ['redirect307','redirect308']){
   await rejected(request(selected,{streamFirstEventTimeoutMs:1000,streamIdleTimeoutMs:1000}),/redirect|fetch|network|timeout|timed out|abort/i);
   assert.deepEqual(leaked,[]);assert.equal(wire.length,3);
 }
 mode='ready';await request(selected);assert.equal(wire.length,4);assert.deepEqual(leaked,[]);
 console.log('PASS 307/308 redirect origin confinement, no destination request/body/header disclosure, genuine transport remains usable');
 console.log('PASS stale selected snapshot, output/payload, malformed authority, native activation and startup online guard');
 console.log('OMP_PRODUCTION_LOADER_PASS');
}finally{server.stop(true);destination.stop(true);rmSync(root,{recursive:true,force:true});}
