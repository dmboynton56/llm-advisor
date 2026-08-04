import clsx from "clsx";
import type { DecisionLog } from "@/lib/types";
import { fmtTimeEt } from "@/lib/format";

/**
 * The signature element: each signal the strategy raised and the verdict the
 * model gave it. Vetoes are drawn as absence — hatched node, struck symbol —
 * so what the gate removed is as visible as what it let through.
 */
export function DecisionLedger({ log }: { log: DecisionLog }) {
  if (log.decisions.length === 0) {
    return (
      <p className="text-[12.5px] text-ink-3">
        No validation decisions recorded yet. The live loop writes one per signal
        it raises.
      </p>
    );
  }

  return (
    <>
      {/* The list scrolls inside the panel so the sticky rail can never grow
          past the viewport and strand the summary below it. */}
      <ol className="flex max-h-[22rem] flex-col overflow-y-auto">
        {log.decisions.map((decision, index) => {
          const vetoed = decision.verdict === "vetoed";
          const last = index === log.decisions.length - 1;
          return (
            <li
              key={decision.key}
              className={clsx(
                "grid grid-cols-[12px_1fr_auto] items-start gap-x-3 py-2.5",
                index > 0 && "border-t border-line",
              )}
            >
              <span aria-hidden className="relative h-full min-h-4">
                {!last ? (
                  <span className="absolute left-[5px] top-[5px] -bottom-[11px] w-px bg-line" />
                ) : null}
                <span
                  className={clsx(
                    "absolute left-px top-[3px] size-[9px] rounded-full border-[1.5px]",
                    vetoed
                      ? "border-dashed border-ink-3 bg-[repeating-linear-gradient(-45deg,var(--ink-3)_0_1px,transparent_1px_3px)]"
                      : "border-ink bg-ink",
                  )}
                />
              </span>

              <span className="flex flex-wrap items-baseline gap-x-2 gap-y-1">
                <span className="num text-[11px] text-ink-3">
                  {fmtTimeEt(decision.eventTs)}
                </span>
                <span
                  className={clsx(
                    "num text-[12.5px] font-medium",
                    vetoed && "text-ink-2 line-through decoration-line-2",
                  )}
                >
                  {decision.symbol}
                </span>
                {decision.setupType ? (
                  <span className="num rounded border border-line-2 px-1.5 text-[9.5px] font-medium tracking-[0.07em] text-ink-3">
                    {decision.setupType}
                  </span>
                ) : null}
              </span>

              <span className="pt-0.5 text-right">
                <span
                  className={clsx(
                    "num text-[10px] font-semibold uppercase tracking-[0.08em]",
                    vetoed ? "text-ink-3" : "text-ink",
                  )}
                >
                  {vetoed ? "Vetoed" : "Taken"}
                </span>
                {decision.confidence != null ? (
                  <span className="num mt-0.5 block text-[11px] text-ink-3">
                    {Math.round(decision.confidence)}%
                  </span>
                ) : null}
              </span>

              {/* The model writes a paragraph; the rail shows the opening of it
                  and hands over the rest on hover. */}
              {decision.reason ? (
                <p
                  title={decision.reason}
                  className="col-start-2 col-end-4 mt-1 line-clamp-2 text-[11.5px] leading-relaxed text-ink-2"
                >
                  {decision.reason}
                </p>
              ) : null}
            </li>
          );
        })}
      </ol>

      <p className="num mt-1 flex items-center gap-2 border-t border-line pt-3.5 text-[11px] text-ink-3">
        <b className="font-medium text-ink">{log.signals}</b> signals
        <span className="text-line-2">→</span>
        <b className="font-medium text-ink">{log.approved}</b> approved
        <span className="text-line-2">→</span>
        <b className="font-medium text-ink">{log.filled}</b> filled
      </p>
    </>
  );
}
