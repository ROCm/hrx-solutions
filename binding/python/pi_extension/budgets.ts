import { SettingsManager } from "@earendil-works/pi-coding-agent";
import type { Model } from "@earendil-works/pi-ai";

function object(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}
function tokens(value: unknown): value is number {
  return typeof value === "number" && Number.isSafeInteger(value) && value >= 0;
}
function deny(model: Model<"openai-completions">, reason: string): never {
  throw new Error(`Loom Hem Pi (${model.provider}/${model.id}): ${reason}; actual context=${model.contextWindow}, output cap=${model.maxTokens}. ` +
    "Inspect effective Pi user/project settings and exact-model compaction overrides; adjust the conflicting budgets manually or select a larger-context instance, then refresh/reselect. No settings were changed");
}

/** Resolve Pi's already merged, trusted settings. No second file reader or writes. */
export function validateBudgets(settings: unknown, model: Model<"openai-completions">) {
  if (!object(settings)) deny(model, "effective token budgets are unavailable");
  for (const section of ["compaction", "branchSummary"]) {
    if (settings[section] !== undefined && !object(settings[section])) deny(model, `invalid effective ${section} settings`);
  }
  const compactionInput = settings.compaction as Record<string, unknown> | undefined;
  if (compactionInput?.modelOverrides !== undefined && !object(compactionInput.modelOverrides)) {
    deny(model, "invalid effective compaction.modelOverrides settings");
  }
  // Pi's branch getter uses nullish defaults. Reject explicitly invalid values
  // instead of treating an unknown user budget as a safe default.
  const branchInput = settings.branchSummary as Record<string, unknown> | undefined;
  if (branchInput?.reserveTokens !== undefined && !tokens(branchInput.reserveTokens)) deny(model, "invalid effective branchSummary.reserveTokens");
  let compaction, branch;
  try {
    const resolver = SettingsManager.inMemory(settings);
    compaction = resolver.getCompactionSettings(model);
    branch = resolver.getBranchSummarySettings();
  } catch { deny(model, "invalid effective compaction reserveTokens/keepRecentTokens or exact-model override"); }
  const reserve = compaction.reserveTokens, recent = compaction.keepRecentTokens, branchReserve = branch.reserveTokens;
  if (!tokens(reserve) || !tokens(recent) || !tokens(branchReserve)) deny(model, "unknown or invalid effective compaction/branch token budgets");
  const context = model.contextWindow, output = model.maxTokens;
  if (!tokens(context) || context <= 0 || context > 262144 || !tokens(output) || output <= 0 || output > Math.min(8192, context)) {
    deny(model, "unknown or unsupported native context/output limits");
  }
  if (reserve < output || branchReserve < output || reserve + recent + output >= context || branchReserve + output >= context) {
    deny(model, `incompatible effective budgets: compaction.reserveTokens=${reserve}, compaction.keepRecentTokens=${recent}, branchSummary.reserveTokens=${branchReserve}`);
  }
  return { compaction, branch };
}
