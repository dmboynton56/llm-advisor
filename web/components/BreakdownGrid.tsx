import clsx from "clsx";
import type { CellStats } from "@/lib/types";
import { fmtNum, fmtSignedUsd, pnlColor } from "@/lib/format";
import { Meter, Panel, Section } from "@/components/ui";

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
    ? keyOrder
        .filter((k) => k in cells)
        .concat(Object.keys(cells).filter((k) => !keyOrder.includes(k)))
    : Object.keys(cells);

  return (
    <Section title={title} subtitle={subtitle}>
      {keys.length === 0 ? (
        <Panel>
          <p className="py-4 text-center text-[13px] text-ink-3">No data yet.</p>
        </Panel>
      ) : (
        <div className="grid gap-3.5 sm:grid-cols-2 lg:grid-cols-3">
          {keys.map((key) => {
            const cell = cells[key];
            const lowSample = cell.closed_trades < MIN_SAMPLE;
            return (
              <Panel
                key={key}
                className={clsx("p-4", lowSample && "opacity-55")}
              >
                <div className="flex items-baseline justify-between gap-2">
                  <span className="text-[14px] font-semibold">{key}</span>
                  <span className="num text-[10px] uppercase tracking-[0.08em] text-ink-3">
                    n={cell.closed_trades}
                    {lowSample ? " · low" : ""}
                  </span>
                </div>
                <dl className="mt-3 grid grid-cols-[auto_1fr] items-center gap-x-3 gap-y-2 text-[12px]">
                  <dt className="text-ink-3">Win rate</dt>
                  <dd className="flex justify-end">
                    <Meter value={cell.win_rate} />
                  </dd>

                  <dt className="text-ink-3">PnL</dt>
                  <dd className={clsx("num text-right", pnlColor(cell.total_pnl))}>
                    {fmtSignedUsd(cell.total_pnl)}
                  </dd>

                  <dt className="text-ink-3">Avg RR</dt>
                  <dd className="num text-right">{fmtNum(cell.avg_realized_rr)}</dd>

                  <dt className="text-ink-3">Profit factor</dt>
                  <dd className="num text-right">{fmtNum(cell.profit_factor)}</dd>
                </dl>
                {lowSample ? (
                  <p className="mt-3 border-t border-line pt-2.5 text-[11px] text-ink-3">
                    Under {MIN_SAMPLE} closed trades — read as a hint, not a result.
                  </p>
                ) : null}
              </Panel>
            );
          })}
        </div>
      )}
    </Section>
  );
}
