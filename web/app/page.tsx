import { Card, EmptyState, MetricCard } from "@/components/MetricCard";
import { EquityCurve } from "@/components/charts/EquityCurve";
import { DailyPnlBars } from "@/components/charts/DailyPnlBars";
import {
  getAccountSnapshots,
  getLatestHeartbeat,
  getRuns,
} from "@/lib/data";
import { supabaseConfigured } from "@/lib/supabase";
import {
  fmtDateTime,
  fmtPct,
  fmtSignedUsd,
  fmtUsd,
  pnlColor,
  relativeTime,
} from "@/lib/format";

export const revalidate = 300;

function heartbeatStatus(heartbeatTs: string | null): {
  label: string;
  tone: "positive" | "negative" | "neutral";
} {
  if (!heartbeatTs) return { label: "No heartbeat", tone: "negative" };
  const ageHours = (Date.now() - new Date(heartbeatTs).getTime()) / 3.6e6;
  // The loop only runs on market days, so anything under ~4 days is normal
  // (weekend + holiday gap).
  if (ageHours <= 30) return { label: "Healthy", tone: "positive" };
  if (ageHours <= 96) return { label: "Idle (non-trading days)", tone: "neutral" };
  return { label: "Stale", tone: "negative" };
}

export default async function OverviewPage() {
  const [snapshots, runs, heartbeat] = await Promise.all([
    getAccountSnapshots(90),
    getRuns(30),
    getLatestHeartbeat(),
  ]);

  const latestSnapshot = snapshots.at(-1) ?? null;
  const latestRun = runs.at(-1) ?? null;
  const hb = heartbeatStatus(heartbeat?.heartbeat_ts ?? null);

  const equityPoints = snapshots
    .filter((s) => s.equity !== null)
    .map((s) => ({ label: s.snapshot_date, equity: Number(s.equity) }));

  const pnlPoints = runs.map((r) => ({
    label: r.run_date.slice(5),
    pnl: Number(r.total_pnl ?? 0),
    trades: r.total_trades,
  }));

  const totalPnl30d = runs.reduce((acc, r) => acc + Number(r.total_pnl ?? 0), 0);

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-xl font-semibold tracking-tight">Overview</h1>
        <p className="mt-1 text-sm text-zinc-500">
          Alpaca paper account health and daily results for the LLM-validated
          options engine.
        </p>
      </div>

      {!supabaseConfigured() ? (
        <EmptyState message="Supabase is not configured. Set NEXT_PUBLIC_SUPABASE_URL and NEXT_PUBLIC_SUPABASE_ANON_KEY." />
      ) : null}

      <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
        <MetricCard
          label="Account equity"
          value={fmtUsd(latestSnapshot?.equity ?? null, 0)}
          hint={
            latestSnapshot
              ? `as of ${fmtDateTime(latestSnapshot.captured_at)}`
              : "no snapshots yet"
          }
        />
        <MetricCard
          label="Daily PnL"
          value={fmtSignedUsd(latestSnapshot?.daily_pnl ?? null)}
          tone={
            (latestSnapshot?.daily_pnl ?? 0) > 0
              ? "positive"
              : (latestSnapshot?.daily_pnl ?? 0) < 0
                ? "negative"
                : "neutral"
          }
          hint={
            latestSnapshot?.daily_pnl_pct != null
              ? fmtPct(Number(latestSnapshot.daily_pnl_pct), 2)
              : undefined
          }
        />
        <MetricCard
          label="Trades (last session)"
          value={latestRun ? latestRun.total_trades : "—"}
          hint={latestRun ? `on ${latestRun.run_date}` : undefined}
        />
        <MetricCard
          label="Live loop"
          value={hb.label}
          tone={hb.tone}
          hint={
            heartbeat
              ? `last heartbeat ${relativeTime(heartbeat.heartbeat_ts)}${
                  heartbeat.symbols_tracked
                    ? ` · ${heartbeat.symbols_tracked} symbols`
                    : ""
                }`
              : undefined
          }
        />
      </div>

      <Card
        title="Equity curve"
        subtitle="Paper account equity snapshots captured at live-loop start and end (90 days)"
      >
        {equityPoints.length >= 2 ? (
          <EquityCurve data={equityPoints} />
        ) : (
          <EmptyState message="Not enough equity snapshots yet — the live loop writes one at session start and end each trading day." />
        )}
      </Card>

      <Card
        title="Daily PnL"
        subtitle={
          <>
            Realized PnL per trading day (30 days) ·{" "}
            <span className={pnlColor(totalPnl30d)}>
              {fmtSignedUsd(totalPnl30d)} total
            </span>
          </>
        }
      >
        {pnlPoints.length > 0 ? (
          <DailyPnlBars data={pnlPoints} />
        ) : (
          <EmptyState message="No run history in the last 30 days." />
        )}
      </Card>

      <Card title="Recent sessions">
        {runs.length > 0 ? (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-zinc-800 text-left text-xs uppercase tracking-wide text-zinc-500">
                  <th className="py-2 pr-4 font-medium">Date</th>
                  <th className="py-2 pr-4 font-medium">Trades</th>
                  <th className="py-2 pr-4 font-medium">Closed</th>
                  <th className="py-2 pr-4 font-medium">Win rate</th>
                  <th className="py-2 pr-4 font-medium">PnL</th>
                  <th className="py-2 font-medium">Equity</th>
                </tr>
              </thead>
              <tbody>
                {[...runs].reverse().slice(0, 10).map((run) => (
                  <tr
                    key={run.run_date}
                    className="border-b border-zinc-900 last:border-0"
                  >
                    <td className="py-2 pr-4 tabular-nums">{run.run_date}</td>
                    <td className="py-2 pr-4 tabular-nums">{run.total_trades}</td>
                    <td className="py-2 pr-4 tabular-nums">{run.closed_trades}</td>
                    <td className="py-2 pr-4 tabular-nums">
                      {fmtPct(run.win_rate)}
                    </td>
                    <td
                      className={`py-2 pr-4 tabular-nums ${pnlColor(
                        Number(run.total_pnl ?? 0),
                      )}`}
                    >
                      {fmtSignedUsd(Number(run.total_pnl ?? 0))}
                    </td>
                    <td className="py-2 tabular-nums">
                      {fmtUsd(run.final_equity, 0)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <EmptyState message="No sessions recorded yet." />
        )}
      </Card>
    </div>
  );
}
