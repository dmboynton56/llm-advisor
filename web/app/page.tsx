import { Card, EmptyState, MetricCard } from "@/components/MetricCard";
import { EquityCurve } from "@/components/charts/EquityCurve";
import { DailyPnlBars } from "@/components/charts/DailyPnlBars";
import {
  getAccountSnapshots,
  getLatestHeartbeat,
  getLiveState,
  getRuns,
  getTradeLifecycles,
} from "@/lib/data";
import { supabaseConfigured, checkSupabaseAccess } from "@/lib/supabase";
import {
  fmtDateTime,
  fmtPct,
  fmtSignedUsd,
  fmtUsd,
  formatOccLabel,
  isRegularSessionEt,
  pnlColor,
  relativeTime,
  dateEtIso,
} from "@/lib/format";
import type { LiveStateRow } from "@/lib/types";

export const dynamic = "force-dynamic";

const LIVE_FRESH_MS = 3 * 60_000;

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

function liveStateFresh(row: LiveStateRow | null): boolean {
  if (!row?.heartbeat_ts) return false;
  const age = Date.now() - new Date(row.heartbeat_ts).getTime();
  return !Number.isNaN(age) && age <= LIVE_FRESH_MS;
}

export default async function OverviewPage() {
  const [snapshots, runs, heartbeat, liveState, lifecycles] = await Promise.all([
    getAccountSnapshots(90),
    getRuns(30),
    getLatestHeartbeat(),
    getLiveState("paper"),
    getTradeLifecycles(30),
  ]);

  const access =
    supabaseConfigured() && runs.length === 0 && !heartbeat
      ? await checkSupabaseAccess()
      : null;

  const latestSnapshot = snapshots.at(-1) ?? null;
  const latestRun = runs.at(-1) ?? null;
  const hb = heartbeatStatus(heartbeat?.heartbeat_ts ?? null);
  const liveAccountCapturedAt =
    liveState?.updated_at ?? liveState?.heartbeat_ts ?? null;
  const snapshotCapturedAt = latestSnapshot?.captured_at ?? null;
  const liveAccountIsNewer =
    liveState?.equity != null &&
    liveAccountCapturedAt != null &&
    (snapshotCapturedAt == null ||
      new Date(liveAccountCapturedAt).getTime() >
        new Date(snapshotCapturedAt).getTime());
  const accountEquity = liveAccountIsNewer
    ? liveState?.equity
    : latestSnapshot?.equity;
  const accountDailyPnl = liveAccountIsNewer
    ? liveState?.daily_pnl
    : latestSnapshot?.daily_pnl;
  const accountDailyPnlPct = liveAccountIsNewer
    ? liveState?.daily_pnl != null && liveState.last_equity
      ? liveState.daily_pnl / liveState.last_equity
      : null
    : latestSnapshot?.daily_pnl_pct;
  const accountCapturedAt = liveAccountIsNewer
    ? liveAccountCapturedAt
    : snapshotCapturedAt;

  const equityPoints = snapshots
    .filter((s) => s.equity !== null)
    .map((s) => ({ label: s.snapshot_date, equity: Number(s.equity) }));

  const exitDaily = new Map<string, { pnl: number; trades: number }>();
  for (const lifecycle of lifecycles) {
    const exitDate = dateEtIso(lifecycle.closed_at);
    if (!exitDate) continue;
    const pnl = Number(lifecycle.realized_pnl ?? 0);
    const current = exitDaily.get(exitDate) ?? { pnl: 0, trades: 0 };
    current.pnl += pnl;
    current.trades += 1;
    exitDaily.set(exitDate, current);
  }
  const pnlPoints = [...exitDaily.entries()]
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([date, value]) => ({
      label: date.slice(5),
      pnl: value.pnl,
      trades: value.trades,
    }));

  const totalPnl30d = pnlPoints.reduce((acc, point) => acc + point.pnl, 0);
  const liveFresh = liveStateFresh(liveState);
  const inSession = isRegularSessionEt();
  const sessionEnded = Boolean(liveState?.session_stats?.session_end_reason);

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

      {access && !access.ok ? (
        <EmptyState
          message={`Supabase query failed (HTTP ${access.status}). On Vercel, verify NEXT_PUBLIC_SUPABASE_URL matches the shared project and NEXT_PUBLIC_SUPABASE_ANON_KEY is the full publishable key (sb_publishable_...) or legacy anon JWT — not a truncated value.`}
        />
      ) : null}

      {liveFresh && liveState ? (
        <Card
          title="Live session"
          subtitle={
            <>
              Loop alive · tick {liveState.loop_count ?? "—"} · last{" "}
              {relativeTime(liveState.heartbeat_ts)}
            </>
          }
        >
          <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
            <MetricCard
              label="Equity"
              value={fmtUsd(liveState.equity, 0)}
              hint="from live_state upsert"
            />
            <MetricCard
              label="Broker daily PnL"
              value={fmtSignedUsd(liveState.daily_pnl)}
              tone={
                (liveState.daily_pnl ?? 0) > 0
                  ? "positive"
                  : (liveState.daily_pnl ?? 0) < 0
                    ? "negative"
                    : "neutral"
              }
            />
            <MetricCard
              label="Open uPnL"
              value={fmtSignedUsd(liveState.unrealized_pnl)}
              tone={
                (liveState.unrealized_pnl ?? 0) > 0
                  ? "positive"
                  : (liveState.unrealized_pnl ?? 0) < 0
                    ? "negative"
                    : "neutral"
              }
              hint={`${liveState.open_position_count} open`}
            />
            <MetricCard
              label="Session stats"
              value={`${liveState.session_stats?.wins ?? 0}W / ${liveState.session_stats?.losses ?? 0}L`}
              hint={
                liveState.session_stats?.realized_pnl != null
                  ? `realized ${fmtSignedUsd(Number(liveState.session_stats.realized_pnl))}`
                  : undefined
              }
            />
          </div>
          {(liveState.open_positions?.length ?? 0) > 0 ? (
            <ul className="mt-4 space-y-1 text-sm text-zinc-300">
              {liveState.open_positions.map((p) => (
                <li key={p.symbol} className="flex justify-between gap-4 font-mono text-xs">
                  <span>{formatOccLabel(p.symbol)}</span>
                  <span className={pnlColor(p.unrealized_plpc)}>
                    {fmtPct(
                      Math.abs(p.unrealized_plpc) > 5
                        ? p.unrealized_plpc / 100
                        : p.unrealized_plpc,
                      1,
                    )}
                  </span>
                </li>
              ))}
            </ul>
          ) : (
            <p className="mt-3 text-sm text-zinc-500">Flat — no open positions.</p>
          )}
        </Card>
      ) : inSession && !sessionEnded ? (
        <Card title="Live session" subtitle="Intraday loop telemetry">
          <p className="text-sm text-zinc-400">
            Loop offline
            {liveState?.heartbeat_ts
              ? ` — last heartbeat ${relativeTime(liveState.heartbeat_ts)}`
              : " — no live_state row yet"}
            . Open the command-center blotter for broker-truth marks.
          </p>
        </Card>
      ) : null}

      <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
        <MetricCard
          label="Account equity"
          value={fmtUsd(accountEquity ?? null, 0)}
          hint={
            accountCapturedAt
              ? `latest stored value · ${fmtDateTime(accountCapturedAt)}`
              : "no account telemetry yet"
          }
        />
        <MetricCard
          label="Broker daily PnL"
          value={fmtSignedUsd(accountDailyPnl ?? null)}
          tone={
            (accountDailyPnl ?? 0) > 0
              ? "positive"
              : (accountDailyPnl ?? 0) < 0
                ? "negative"
                : "neutral"
          }
          hint={
            accountDailyPnlPct != null
              ? `${fmtPct(Number(accountDailyPnlPct), 2)} vs prior close`
              : "equity change vs prior close"
          }
        />
        <MetricCard
          label="Trades opened (last entry date)"
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
        title="Strategy PnL by exit date"
        subtitle={
          <>
            Broker-position lifecycle PnL grouped by the ET date it closed (30 days) ·{" "}
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

      <Card
        title="Entry-date cohorts"
        subtitle="Trades and full-lifecycle PnL grouped by the date each position was opened"
      >
        {runs.length > 0 ? (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-zinc-800 text-left text-xs uppercase tracking-wide text-zinc-500">
                  <th className="py-2 pr-4 font-medium">Entry date</th>
                  <th className="py-2 pr-4 font-medium">Opened</th>
                  <th className="py-2 pr-4 font-medium">Now closed</th>
                  <th className="py-2 pr-4 font-medium">Win rate</th>
                  <th className="py-2 pr-4 font-medium">Lifetime PnL</th>
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
