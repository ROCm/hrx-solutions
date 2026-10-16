import type { ExtensionAPI } from "@oh-my-pi/pi-coding-agent";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { validateInstallation, parseDocument, readBounded, fail } from "./installation.ts";

export default async function(pi: ExtensionAPI) {
  const entry=fileURLToPath(import.meta.url), validate=()=>validateInstallation(entry), installation=validate();
  // Companions execute only after their entire copied generation has been verified.
  const {activated}=await import("./activation.ts");
  const {discover,readEndpoints}=await import("./discovery.ts");
  const {API,guardedStream,modelDefinition}=await import("./stream.ts");
  const binding=JSON.stringify(installation);
  function activation(provider:string) {
    const manifest=parseDocument(readBounded(join(dirname(entry),"manifest.json")).toString("utf8")) as any;
    return activated(installation,manifest.native_activation,provider);
  }
  async function snapshot(provider?:string) {
    if(JSON.stringify(validate())!==JSON.stringify(installation))fail("installation changed during session");
    const observations=await discover(installation,validate,provider);
    for(const row of observations){
      if(!activation(row.endpoint.provider)){row.models=[];row.reason="native keyless activation is missing, edited or conflicting; run loom-control omp configure, then omp models refresh --json and restart OMP";}
      if(row.reason)console.error(`Loom Control OMP (${row.endpoint.server}): ${row.reason}`);
    }
    return observations;
  }
  const current=async(provider:string,id:string)=>{
    const observations=await snapshot(provider), row=observations.find(row=>row.endpoint.provider===provider), model=row?.models.find(row=>row.id===id);
    if(!row||!model)throw Error("model is unavailable at its current activated endpoint");
    return {endpoint:row.endpoint,model};
  };
  for(const endpoint of readEndpoints(installation)) {
    if(!activation(endpoint.provider)){console.error(`Loom Control OMP (${endpoint.server}): native activation missing/conflicting; run loom-control omp configure, then refresh/restart OMP`);continue;}
    pi.registerProvider(endpoint.provider,{baseUrl:endpoint.base_url,api:API,streamSimple:guardedStream(current,binding),
      async fetchDynamicModels(){
        try{const row=(await snapshot(endpoint.provider)).find(row=>row.endpoint.provider===endpoint.provider);
          if(!row || JSON.stringify(row.endpoint)!==JSON.stringify(endpoint)){console.error("Loom Control OMP: endpoint changed; restart OMP to register its current origin");return [];}
          return row.models.map(model=>modelDefinition(row.endpoint,model,binding));}
        catch(error){console.error(`Loom Control OMP: discovery unavailable; ${error instanceof Error?error.message:"invalid snapshot"}`);return [];}
      }});
  }
  pi.on("session_start",async(_event,context)=>{await context.modelRegistry.refreshRuntimeProviders("online");});
}
