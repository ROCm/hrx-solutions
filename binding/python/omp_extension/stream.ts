import { createHash } from "node:crypto";
import { createAssistantMessageEventStream, streamSimple } from "@oh-my-pi/pi-ai";
import type { Model, Api, Context, SimpleStreamOptions } from "@oh-my-pi/pi-ai";
import type { Endpoint, Model as NativeModel } from "./discovery.ts";

export const SNAPSHOT_HEADER = "x-loom-control-snapshot";
export const API = "loom-control-omp-text-v1";
export function fingerprint(endpoint: Endpoint, model: NativeModel, binding: string): string {
  return createHash("sha256").update(JSON.stringify([binding,endpoint,model])).digest("hex");
}
export function modelDefinition(endpoint: Endpoint, model: NativeModel, binding: string) {
  return {id:model.id,name:model.name,api:API,baseUrl:endpoint.base_url,reasoning:false,supportsTools:false,input:["text"],
    cost:{input:0,output:0,cacheRead:0,cacheWrite:0},contextWindow:model.limit.context,maxTokens:model.limit.output,
    headers:{[SNAPSHOT_HEADER]:fingerprint(endpoint,model,binding)},
    compat:{maxTokensField:"max_tokens",supportsDeveloperRole:false,supportsStore:false,supportsReasoningEffort:false,supportsToolChoice:false}};
}
/** Public wrapper only. Freshness is checked before delegating to native transport. */
export function guardedStream(current: (provider: string, id: string) => Promise<{endpoint:Endpoint;model:NativeModel}>, binding: string) {
  return (selected: Model<Api>, context: Context, options: SimpleStreamOptions = {}) => {
    const output=createAssistantMessageEventStream();
    void (async()=>{
      try {
        const validationDeadline=Date.now()+10000;
        const observe=async()=>{
          if(Date.now()>=validationDeadline)throw Error("request validation deadline exceeded");
          const observed=await current(selected.provider,selected.id);
          if(Date.now()>=validationDeadline)throw Error("request validation deadline exceeded");
          return observed;
        };
        const captured=await observe(), expected=fingerprint(captured.endpoint,captured.model,binding);
        const selectedHeaders={...selected.headers,...await selected.resolveHeaders?.(options.signal)};
        if(selected.api!==API || selected.baseUrl!==captured.endpoint.base_url || selectedHeaders[SNAPSHOT_HEADER]!==expected ||
          selected.contextWindow!==captured.model.limit.context || selected.maxTokens!==captured.model.limit.output)
          throw Error("selected model snapshot/origin/incarnation/limits changed; refresh and reselect the model");
        const maxTokens=options.maxTokens??captured.model.limit.output;
        if(!Number.isSafeInteger(maxTokens)||maxTokens<=0||maxTokens>captured.model.limit.output)throw Error("request output exceeds native capacity");
        const native={...selected,...modelDefinition(captured.endpoint,captured.model,binding),api:"openai-completions",provider:selected.provider,headers:undefined,resolveHeaders:undefined} as Model<"openai-completions">;
        // Keyless transport only: neither selected headers nor ambient credentials survive.
        if(options.apiKey!=="N/A")throw Error("native keyless authentication was not resolved; repair auth: none activation");
        const ownedFetch: NonNullable<SimpleStreamOptions["fetch"]> = async (input, init) => {
          const request = new Request(input, init);
          if(request.url!==captured.endpoint.base_url+"/chat/completions" || request.method!=="POST")
            throw Error("transport attempted an unexpected native URL or method");
          const fresh=await observe();
          if(fingerprint(fresh.endpoint,fresh.model,binding)!==expected)throw Error("native identity changed before transport send");
          // Rebuild each send/retry from the standard Request. Never forward SDK,
          // ambient or caller headers/credentials, fetch extensions, or redirects.
          const headers = new Headers({Accept:"text/event-stream", "Content-Type":"application/json"});
          return globalThis.fetch(new Request(request,{headers,redirect:"error",credentials:"omit"}));
        };
        const inner=streamSimple(native,context,{...options,apiKey:options.apiKey,headers:undefined,fetch:ownedFetch,oauthIdentity:undefined,credentialId:undefined,maxTokens,
          async onPayload(payload,effective){
            const replaced=await options.onPayload?.(payload,effective), body=replaced===undefined?payload:replaced;
            if(!body||typeof body!=="object"||Array.isArray(body))throw Error("unsupported payload override");
            const request=body as Record<string,unknown>;
            if(native.baseUrl!==captured.endpoint.base_url || native.id!==captured.model.id || request.model!==captured.model.id||request.max_tokens!==maxTokens||request.max_completion_tokens!==undefined)throw Error("payload changes model/output identity");
            if(["tools","tool_choice","parallel_tool_calls","functions","function_call","reasoning_effort","reasoning","thinking","enable_thinking","include_reasoning"].some(key=>request[key]!==undefined))
              throw Error("payload requests unsupported tools or reasoning capabilities");
            const fresh=await observe();
            if(fingerprint(fresh.endpoint,fresh.model,binding)!==expected)throw Error("native identity changed while preparing request");
            return body;
          }});
        output.forwardLocalWorkFrom(inner);
        for await(const event of inner)output.push(event);
        output.end(await inner.result());
      }catch(error){output.fail(new Error(`Loom Control OMP: ${error instanceof Error?error.message:"model unavailable"}; run omp models refresh --json, then restart OMP`));}
    })();
    return output;
  };
}
