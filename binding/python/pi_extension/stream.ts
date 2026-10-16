import { openAICompletionsApi } from "@earendil-works/pi-ai";
import type { Model, Provider, StreamOptions, SimpleStreamOptions } from "@earendil-works/pi-ai";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import type { Discovery, DiscoveryHooks } from "./discovery.ts";
import { validateBudgets } from "./budgets.ts";

type TextModel = Model<"openai-completions">;
const profile = "loom-text-serve-ed6b79c-v1";
function unavailable(reason: string): never {
  throw new Error(`Loom Hem Pi: ${reason}. Refresh/reselect the model; run loom-control pi configure if the owned integration needs repair`);
}

/** One guard shared by ordinary inference and every Pi summary dispatch. */
export function createStreamingHooks(pi: ExtensionAPI, discovery: Discovery): DiscoveryHooks {
  // Discovery does not initialize transport; the supported Pi loader supplies
  // the public built-in adapter when an allowed request actually needs it.
  let api: ReturnType<typeof openAICompletionsApi> | undefined;
  const builtIn = () => api ??= openAICompletionsApi();
  let settingsObserved = false;
  function effectiveSettings(model: TextModel) {
    let settings: unknown;
    try { settings = pi.getSettings(); settingsObserved = true; }
    catch { unavailable(`effective token budgets unavailable for ${model.provider}/${model.id} (context=${model.contextWindow}, output cap=${model.maxTokens}); allow Pi initialization to finish and inspect effective user/project/exact-model settings`); }
    return validateBudgets(settings, model);
  }
  function currentModel(selected: TextModel): TextModel {
    const endpoint = discovery.currentEndpoints().find(item => item.provider === selected.provider);
    if (!endpoint || endpoint.native_profile !== profile) unavailable("current configured endpoint/profile is unavailable or changed");
    const observation = discovery.observations.get(selected.provider);
    if (!observation || observation.endpoint.base_url !== endpoint.base_url || observation.endpoint.native_profile !== endpoint.native_profile) {
      unavailable("configured origin changed since the latest native observation");
    }
    const current = observation.models.find(item => item.id === selected.id && item.provider === selected.provider);
    if (!current || current.baseUrl !== endpoint.base_url) unavailable("selected model is no longer native-ready at the current endpoint");
    if (selected.api !== current.api || selected.contextWindow > current.contextWindow || selected.maxTokens > current.maxTokens ||
        !Number.isSafeInteger(selected.contextWindow) || !Number.isSafeInteger(selected.maxTokens) ||
        selected.contextWindow <= 0 || selected.maxTokens <= 0) unavailable("selected context/output/profile metadata changed or is invalid");
    // Use only freshly observed capabilities and the currently configured origin,
    // never a selected object's stale headers/baseUrl/compatibility fields.
    return current;
  }
  function prepare(selected: TextModel, options: StreamOptions | SimpleStreamOptions = {}) {
    const model = currentModel(selected);
    effectiveSettings(model);
    if (options.maxTokens !== undefined && (!Number.isSafeInteger(options.maxTokens) || options.maxTokens <= 0 || options.maxTokens > model.maxTokens)) {
      unavailable("request maxTokens must be a positive safe integer within the native output cap");
    }
    // Local endpoints need no credentials. Preserve cancellation and ordinary Pi
    // request options, but never forward ambient/selected-provider credentials.
    return { model, options: { ...options, headers: undefined, apiKey: "loom-local",
      maxTokens: options.maxTokens ?? model.maxTokens,
      async onPayload(payload: unknown, effective: TextModel) {
        const replacement = await options.onPayload?.(payload, effective);
        const finalPayload = replacement === undefined ? payload : replacement;
        // Pi supports payload callbacks; they cannot bypass our output/identity
        // guard, including by modifying the existing payload in place.
        if (!finalPayload || typeof finalPayload !== "object" || Array.isArray(finalPayload)) unavailable("unsupported request payload override");
        const request = finalPayload as Record<string, unknown>;
        if (request.model !== model.id || !Number.isSafeInteger(request.max_tokens) ||
            Number(request.max_tokens) <= 0 || Number(request.max_tokens) > model.maxTokens ||
            request.max_completion_tokens !== undefined) unavailable("request payload changes the selected model or exceeds the native output cap");
        const latest = currentModel(selected);
        if (latest.baseUrl !== model.baseUrl || latest.contextWindow !== model.contextWindow || latest.maxTokens !== model.maxTokens) {
          unavailable("endpoint or native limits changed while preparing inference");
        }
        effectiveSettings(latest);
        return finalPayload;
      } } };
  }
  const stream: Provider<"openai-completions">["stream"] = (selected, context, options) => {
    const effective = prepare(selected, options);
    return builtIn().stream(effective.model, context, effective.options);
  };
  const streamSimple: Provider<"openai-completions">["streamSimple"] = (selected, context, options) => {
    const effective = prepare(selected, options);
    return builtIn().streamSimple(effective.model, context, effective.options);
  };
  return {
    stream, streamSimple,
    settingsBound() {
      try { pi.getSettings(); settingsObserved = true; } catch { /* Initial selection precedes binding. */ }
      return settingsObserved;
    },
    filterModels(models, endpoint) {
      return models.filter(model => {
        try { effectiveSettings(model); return true; }
        catch (error) {
          console.error(`Loom Hem Pi (${endpoint.server}): ${error instanceof Error ? error.message : "effective token budgets unavailable"}`);
          return false;
        }
      });
    },
  };
}
