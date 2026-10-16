import { createHash } from "node:crypto";
import { lstatSync } from "node:fs";
import { join } from "node:path";
import { fail, keys, object, readBounded } from "./installation.ts";
import type { Installation } from "./installation.ts";

export type Activation = { ownership: "owned" | "borrowed"; sha256: string };
export type NativeActivation = { path: string; providers: Record<string, Activation>; heading: null | { original:string; replacement:string; created_file:boolean } };
export type Stanza = { text: string; owner?: string; valid: boolean };
const ID = /^[A-Za-z0-9][A-Za-z0-9_.-]*$/;
const digest = (text: string) => createHash("sha256").update(text).digest("hex");
/** Deliberately narrow block YAML. Reject ambiguous syntax instead of guessing boundaries. */
export function scanActivation(text: string): Map<string, Stanza> {
  const lines = text.match(/[^\n]*\n|[^\n]+$/g) || [];
  const result = new Map<string, Stanza>(), roots = new Set<string>();
  let inProviders = false, provider: string | undefined, start = 0, valid = false, auth = false;
  let marker: { owner: string; provider: string; start: number } | undefined;
  function finish(end: number) {
    if (!provider) return;
    if (result.has(provider)) throw Error("duplicate provider");
    result.set(provider, { text: lines.slice(start, end).join(""), valid: valid && auth });
    provider = undefined; valid = false; auth = false;
  }
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i].replace(/\r?\n$/, "");
    const begin = /^  # loom-control omp begin ([0-9a-f-]{36}) ([A-Za-z0-9_.-]+)$/.exec(line);
    const end = /^  # loom-control omp end ([0-9a-f-]{36}) ([A-Za-z0-9_.-]+)$/.exec(line);
    if (begin) { if (!inProviders || marker) throw Error("invalid activation marker"); finish(i); marker = {owner:begin[1],provider:begin[2],start:i}; continue; }
    if (end) {
      if (!marker || !provider || end[1] !== marker.owner || end[2] !== provider || marker.provider !== provider) throw Error("mismatched activation marker");
      const id=provider, owner=marker.owner, first=marker.start; finish(i+1);
      const stanza=result.get(id)!; stanza.text=lines.slice(first,i+1).join(""); stanza.owner=owner; marker=undefined; continue;
    }
    if (line.includes("loom-control omp begin") || line.includes("loom-control omp end")) throw Error("damaged activation marker");
    if (!line.trim() || /^\s*#/.test(line)) continue;
    // Restrict all structural keys: quoted/multiline/scalar/alias constructs cannot hide a heading.
    if (/\t|[&*!|>]|(?:^|\s)<<\s*:/.test(line)) throw Error("unsupported YAML syntax");
    if (!/^ /.test(line)) {
      if (marker) throw Error("unterminated activation marker"); finish(i); inProviders=false;
      const top=/^([A-Za-z_][A-Za-z0-9_-]*):(?:\s+(.*))?$/.exec(line);
      if (!top || (top[2] && /[\[\]{}"\']/.test(top[2]) && top[1]!=="providers") || roots.has(top[1])) throw Error("unsupported or duplicate root key"); roots.add(top[1]);
      if (top[1] === "providers") { if(top[2] && !/^\{\}(?:\s*#.*)?$/.test(top[2]) && !/^#/.test(top[2])) throw Error("unsupported providers mapping"); inProviders=!top[2] || /^#/.test(top[2]); }
      continue;
    }
    if (!inProviders) {
      // Foreign simple block content is preserved, but complex scalars/quoted keys are unsupported.
      if (!/^ +[A-Za-z0-9_.-]+:(?:\s+[^\[\]{}"']*)?$/.test(line)) throw Error("unsupported foreign YAML layout");
      continue;
    }
    const key=/^  ([A-Za-z0-9][A-Za-z0-9_.-]*):\s*(?:#.*)?$/.exec(line);
    if (key) { if (marker && provider) throw Error("nested activation marker"); finish(i); provider=key[1];start=i;valid=true;auth=false; if(marker && marker.provider!==provider)throw Error("marker provider mismatch");continue; }
    if (!provider || !/^    [A-Za-z_][A-Za-z0-9_-]*:/.test(line)) throw Error("unsupported provider layout");
    if (/^    auth: none\s*(?:#.*)?$/.test(line)) {if(auth)throw Error("duplicate auth key");auth=true;} else valid=false;
  }
  if(marker)throw Error("unterminated activation marker"); finish(lines.length); return result;
}
export function selectedModelsFile(target: string): string {
  function exists(path: string) {try{lstatSync(path);return true;}catch(error:any){if(error.code==='ENOENT')return false;throw error;}}
  if(exists(join(target,"models.json"))) fail("legacy models.json requires explicit user migration");
  return exists(join(target,"models.yml")) ? join(target,"models.yml") : exists(join(target,"models.yaml")) ? join(target,"models.yaml") : join(target,"models.yml");
}
export function readActivation(installation: Installation): Map<string, Stanza> {
  if(selectedModelsFile(installation.target)!==installation.native_models_file) fail("selected native models file changed");
  return scanActivation(readBounded(installation.native_models_file).toString("utf8"));
}
export function activated(installation: Installation, native: NativeActivation, provider: string): boolean {
  if(!keys(native,["path","providers","heading"]) || native.path!==installation.native_models_file || !object(native.providers)) return false;
  const record=native.providers[provider], stanza=readActivation(installation).get(provider);
  return !!(ID.test(provider) && keys(record,["ownership","sha256"]) && ["owned","borrowed"].includes(record.ownership) &&
    typeof record.sha256==='string' && /^[0-9a-f]{64}$/.test(record.sha256) && stanza?.valid && digest(stanza.text)===record.sha256 &&
    (record.ownership==='owned' ? stanza.owner===installation.integration_id : stanza.owner===undefined));
}
