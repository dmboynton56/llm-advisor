import clsx from "clsx";
import { Panel, PanelHead } from "@/components/ui";
import { dateEtIso, fmtUsd, formatOccLabel } from "@/lib/format";
import { asJsonRecord, firstJsonString, jsonNumber } from "@/lib/json";
import type { DecisionEvent } from "@/lib/types";

export function StockTrial({ decisions }: { decisions: DecisionEvent[] }) {
  const startDate = firstJsonString(decisions[0]?.details?.paper_start_date);
  const paperEligible = startDate != null && dateEtIso() >= startDate;
  return (
    <Panel>
      <PanelHead title="Stock experiments" aside={paperEligible ? "Paper eligible" : startDate ? "Decision previews" : "First cohort"} />
      <p className="num text-[11px] text-ink-2">AAPL · MSFT · GOOG · TSLA</p>
      {startDate ? (
        <p className="mt-2 text-[11px] text-ink-3">Paper entries eligible from {startDate} ET.</p>
      ) : null}
      {decisions.length ? (
        <ul className="mt-4 divide-y divide-line">
          {decisions.map((decision, index) => {
            const action = firstJsonString(decision.details?.action);
            const plan = asJsonRecord(decision.details?.option_plan);
            const validation = asJsonRecord(decision.details?.validation);
            const contract = firstJsonString(plan?.option_symbol);
            const quantity = jsonNumber(plan?.qty);
            const reason = firstJsonString(decision.details?.reason, decision.details?.error);
            return (
              <li key={`${decision.event_ts}-${decision.symbol}-${index}`} className="py-3 first:pt-0">
                <div className="flex items-center justify-between gap-3 text-[12px]">
                  <span className="font-medium">{decision.symbol} <span className="text-ink-3">{decision.setup_type}</span></span>
                  <span className={clsx("num text-[10px]", action === "would_buy" ? "text-gain" : action === "error" ? "text-loss" : "text-ink-3")}>
                    {action === "would_buy" ? "Would buy" : action === "error" ? "Error" : "Skipped"}
                  </span>
                </div>
                <p className="num mt-1 text-[10px] text-ink-3">{decision.run_date} ET</p>
                {contract ? (
                  <p className="mt-2 text-[11px] text-ink-2">{quantity ?? "—"} × {formatOccLabel(contract)} · {fmtUsd(jsonNumber(plan?.estimated_premium))}</p>
                ) : null}
                <details className="mt-2 text-[11px] leading-relaxed text-ink-3">
                  <summary className="cursor-pointer text-ink-2">View decision</summary>
                  <p className="mt-2">{reason?.replaceAll("_", " ") ?? "No reason recorded."}</p>
                  {validation ? <p className="mt-2">Model: {firstJsonString(validation.verdict) ?? "unknown"} · {jsonNumber(validation.confidence) ?? "—"}% confidence</p> : null}
                </details>
              </li>
            );
          })}
        </ul>
      ) : (
        <p className="mt-4 text-[12px] leading-relaxed text-ink-3">No trial decisions recorded yet. Proposed trades and reasons to skip them will appear here.</p>
      )}
      <p className="mt-4 border-t border-line pt-3 text-[11px] leading-relaxed text-ink-3">
        Previews sync after each session. They use real quotes and account checks, without placing orders or simulating fills. These stocks use news and technical signals while ML bias is unavailable.
      </p>
    </Panel>
  );
}
