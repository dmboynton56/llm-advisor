import Link from "next/link";
import clsx from "clsx";
import { EquityCurve } from "@/components/charts/EquityCurve";
import { EmptyState, Panel } from "@/components/ui";
import { dailyEquityPoints, equityPeriod, EQUITY_RANGES, type EquityPoint, type EquityRange } from "@/lib/equity";
import { fmtPct, fmtSignedUsd, fmtUsd, pnlColor, relativeTime, dateEtIso } from "@/lib/format";

export function EquityOverview({ data, range }: { data: EquityPoint[]; range: EquityRange }) {
  const period = equityPeriod(data, range);
  const latest = data.at(-1);
  const intraday = range === "day" || range === "week";
  const chartPoints = intraday ? period.points : dailyEquityPoints(period.points);
  const currentSession = latest && dateEtIso(latest.capturedAt) === dateEtIso();
  const rangeDays = EQUITY_RANGES.find((item) => item.value === range)?.days ?? 30;
  const first = period.points[0];
  const hasFullWindow = latest && first && latest.timestamp - first.timestamp >= rangeDays * 86_400_000;
  const changeLabel = range === "day"
    ? currentSession ? "Today" : "Last session"
    : first && !hasFullWindow ? `Since ${dateEtIso(first.capturedAt)}` : `Over the ${range}`;
  const dates = period.points.length > 0
    ? `${dateEtIso(period.points[0].capturedAt)} — ${dateEtIso(period.points.at(-1)?.capturedAt)} ET`
    : "Waiting for account history";

  return (
    <section aria-labelledby="equity-heading">
      <div className="flex flex-wrap items-end justify-between gap-5">
        <div>
          <h1 id="equity-heading" className="text-[16px] font-semibold">Paper account</h1>
          <p className="num mt-3 text-[clamp(36px,5vw,52px)] font-medium leading-none tracking-[-0.055em]">
            {fmtUsd(latest?.equity)}
          </p>
          <p className={clsx("num mt-3 text-[13px]", pnlColor(period.change))}>
            {fmtSignedUsd(period.change)}
            {period.changePct != null ? ` (${fmtPct(period.changePct, 2)})` : ""}
            <span className="ml-2 font-sans text-ink-3"> {changeLabel}</span>
          </p>
        </div>
        <nav aria-label="Equity time range" className="flex gap-1 rounded-full border border-line bg-sunk p-1">
          {EQUITY_RANGES.map((item) => (
            <Link
              key={item.value}
              href={`/?range=${item.value}`}
              scroll={false}
              prefetch={false}
              aria-current={range === item.value ? "page" : undefined}
              className={clsx(
                "inline-flex min-h-9 items-center rounded-full px-3.5 text-[12px] transition-colors",
                range === item.value ? "bg-card font-semibold text-ink shadow-raised" : "text-ink-3 hover:text-ink",
              )}
            >
              {item.label}
            </Link>
          ))}
        </nav>
      </div>
      <p className="mt-3 text-[12px] text-ink-3">Cash plus the current value of open positions.</p>
      <Panel className="mt-5 p-3 pb-4 sm:p-5">
        {chartPoints.length > 0 ? (
          <EquityCurve data={chartPoints} baseline={period.baseline} intraday={range === "day"} />
        ) : (
          <EmptyState message="Account snapshots will appear when the paper loop records its balance." />
        )}
        <div className="mt-3 flex flex-wrap justify-between gap-2 border-t border-line pt-3 text-[11px] text-ink-3">
          <span className="num">{dates}</span>
          <span>{latest ? `Updated ${relativeTime(latest.capturedAt)}` : "No snapshot yet"}</span>
        </div>
      </Panel>
      <p className="mt-3 text-[11px] text-ink-3">
        Recorded broker equity, including open-position gains and losses, fees, and cash changes.
        {intraday ? " Each point is an account snapshot." : " The chart keeps the first point and each day’s last recorded balance."}
      </p>
    </section>
  );
}
