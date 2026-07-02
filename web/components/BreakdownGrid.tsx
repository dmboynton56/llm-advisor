import clsx from "clsx";
import type { CellStats } from "@/lib/types";
import { fmtNum, fmtPct, fmtSignedUsd, pnlColor } from "@/lib/format";

const MIN_SAMPLE = 10;

export function BreakdownGrid({
  title,
  subtitle,
  cells,
  keyOrder,
}: {
  title: string;
  subtitle?: string;
  cells: Record<string, CellStats>;
  keyOrder?: string[];
}) {
  const keys = keyOrder
    ? keyOrder.filter((k) => k in cells).concat(
        Object.keys(cells).filter((k) => !keyOrder.includes(k)),
      )
    : Object.keys(cells);

  return (
    <section className="rounded-xl border border-zinc-800 bg-zinc-900/50 p-4">
      <div className="mb-3">
        <h2 className="text-sm font-semibold text-zinc-200">{title}</h2>
        {subtitle ? <p className="mt-0.5 text-xs text-zinc-500">{subtitle}</p> : null}
      </div>
      {keys.length === 0 ? (
        <p className="py-6 text-center text-sm text-zinc-500">No data yet.</p>
      ) : (
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
          {keys.map((key) => {
            const cell = cells[key];
            const lowSample = cell.closed_trades < MIN_SAMPLE;
            return (
              <div
                key={key}
                className={clsx(
                  "rounded-lg border p-3",
                  lowSample
                    ? "border-zinc-800/60 bg-zinc-950/40 opacity-55"
                    : "border-zinc-800 bg-zinc-950/60",
                )}
              >
                <div className="flex items-baseline justify-between">
                  <span className="text-sm font-semibold">{key}</span>
                  <span className="text-[10px] uppercase tracking-wide text-zinc-500">
                    n={cell.closed_trades}
                    {lowSample ? " (low)" : ""}
                  </span>
                </div>
                <dl className="mt-2 grid grid-cols-2 gap-x-3 gap-y-1 text-xs">
                  <dt className="text-zinc-500">Win rate</dt>
                  <dd className="text-right tabular-nums text-zinc-200">
                    {fmtPct(cell.win_rate)}
                  </dd>
                  <dt className="text-zinc-500">PnL</dt>
                  <dd
                    className={clsx(
                      "text-right tabular-nums",
                      pnlColor(cell.total_pnl),
                    )}
                  >
                    {fmtSignedUsd(cell.total_pnl)}
                  </dd>
                  <dt className="text-zinc-500">Avg RR</dt>
                  <dd className="text-right tabular-nums text-zinc-200">
                    {fmtNum(cell.avg_realized_rr)}
                  </dd>
                  <dt className="text-zinc-500">Profit factor</dt>
                  <dd className="text-right tabular-nums text-zinc-200">
                    {fmtNum(cell.profit_factor)}
                  </dd>
                </dl>
              </div>
            );
          })}
        </div>
      )}
    </section>
  );
}
