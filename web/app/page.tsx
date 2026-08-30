import clsx from "clsx";
import { DecisionLedger } from "@/components/DecisionLedger";
import { Disclosure } from "@/components/Disclosure";
import { PnlReconciliationChart } from "@/components/charts/PnlReconciliationChart";
import { DailyPnlBars } from "@/components/charts/DailyPnlBars";
import { PositionRail } from "@/components/PositionRail";
import {
  EmptyState,
  Meter,
  Panel,
  PanelHead,
  Section,
  Stat,
  StatRow,
  toneOf,
} from "@/components/ui";
import {
  getAccountSnapshots,
  getBrokerReconciliations,
  getDecisionLog,
  getLatestHeartbeat,
  getLiveState,
  getRuns,
  getTradeLifecycles,
} from "@/lib/data";
import { getTodayOverviewPositions } from "@/lib/positions";
import { supabaseConfigured, checkSupabaseAccess } from "@/lib/supabase";
import {
  fmtPct,
  fmtSignedUsd,
  fmtUsd,
  isRegularSessionEt,
  pnlColor,
  relativeTime,
  dateEtIso,
} from "@/lib/format";
import type { LiveStateRow } from "@/lib/types";

export const dynamic = "force-dynamic";

const LIVE_FRESH_MS = 3 * 60_000;

type HeartbeatStatus = {
  label: string;
  stale: boolean;
};

function liveStateFresh(row: LiveStateRow | null): boolean {
  if (!row?.heartbeat_ts) return false;
  const age = Date.now() - new Date(row.heartbeat_ts).getTime();
  return !Number.isNaN(age) && age <= LIVE_FRESH_MS;
}

function heartbeatStatus(heartbeatTs: string | null): HeartbeatStatus {
  if (!heartbeatTs) return { label: "No heartbeat", stale: true };
  const ageHours = (Date.now() - new Date(heartbeatTs).getTime()) / 3.6e6;
  // The loop only runs on market days, so a gap under ~4 days is just a
  // weekend or a holiday, not a fault.
  if (ageHours <= 96) return { label: "Idle", stale: false };
  return { label: "Stale", stale: true };
}

export default async function OverviewPage() {
  const [snapshots, reconciliations, runs, heartbeat, liveState, lifecycles, decisionLog] =
    await Promise.all([
      getAccountSnapshots(90),
      getBrokerReconciliations(90),
      getRuns(30),
      getLatestHeartbeat(),
      getLiveState("paper"),
      getTradeLifecycles(30),
      getDecisionLog(8),
    ]);

  const access =
    supabaseConfigured() && runs.length === 0 && !heartbeat
      ? await checkSupabaseAccess()
      : null;

  const latestSnapshot = snapshots.at(-1) ?? null;
  const latestRun = runs.at(-1) ?? null;
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
  const accountCapturedAt = liveAccountIsNewer
    ? liveAccountCapturedAt
    : snapshotCapturedAt;
  // The account API's daily P&L is tied to the session represented by the
  // latest account record, not necessarily the server's current calendar
  // date. This matters after midnight ET and across weekends/holidays.
  const accountSessionDate =
    (liveAccountIsNewer ? liveState?.session_date : latestSnapshot?.snapshot_date) ??
    liveState?.session_date ??
    latestSnapshot?.snapshot_date ??
    dateEtIso();
  const sessionLiveState =
    liveState?.session_date === accountSessionDate ? liveState : null;

  const reconciliationPoints = reconciliations
    .filter((row) => row.broker_daily_pnl != null && row.pnl_gap != null)
    .map((row) => ({
      date: row.reconciliation_date,
      lifecyclePnl: Number(row.booked_realized_pnl),
      brokerMtm: Number(row.broker_daily_pnl),
      gap: Number(row.pnl_gap),
      exits: row.lifecycle_exit_count,
    }));
  const latestReconciliation = reconciliationPoints.at(-1) ?? null;

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
  const closed30d = pnlPoints.reduce((acc, point) => acc + point.trades, 0);
  const won30d = lifecycles.filter(lc => lc.closed_at && Number(lc.realized_pnl ?? 0) > 0).length;
  const winRate30d = closed30d > 0 ? won30d / closed30d : null;
  const cohortPnl = runs.reduce((acc, run) => acc + Number(run.total_pnl ?? 0), 0);
  // BQ-backed run rows often have no final_equity, while the account snapshot
  // captured on the same entry date is the authoritative cohort fallback.
  const snapshotEquityByDate = new Map<string, number>();
  for (const snapshot of snapshots) {
    if (snapshot.equity == null) continue;
    const equity = Number(snapshot.equity);
    if (Number.isFinite(equity)) snapshotEquityByDate.set(snapshot.snapshot_date, equity);
  }

  const liveFresh = liveStateFresh(sessionLiveState);
  const inSession = isRegularSessionEt();
  const sessionEnded = Boolean(sessionLiveState?.session_stats?.session_end_reason);
  const hb = heartbeatStatus(heartbeat?.heartbeat_ts ?? null);

  const status = liveFresh
    ? { label: "Live", tone: "live" as const }
    : inSession && !sessionEnded
      ? { label: "Loop offline", tone: "stale" as const }
      : { label: hb.label, tone: hb.stale ? ("stale" as const) : ("idle" as const) };

  const statusMeta = liveFresh
    ? [
        `tick ${liveState?.loop_count ?? "—"}`,
        `heartbeat ${relativeTime(liveState?.heartbeat_ts)}`,
        heartbeat?.symbols_tracked
          ? `${heartbeat.symbols_tracked} symbols`
          : null,
      ]
        .filter(Boolean)
        .join(" · ")
    : [
        `last heartbeat ${relativeTime(heartbeat?.heartbeat_ts)}`,
        heartbeat?.symbols_tracked
          ? `${heartbeat.symbols_tracked} symbols`
          : null,
      ]
        .filter(Boolean)
        .join(" · ");

  const todayPositions = getTodayOverviewPositions(
    sessionLiveState,
    lifecycles,
    accountSessionDate,
  );

  return (
    <div className="grid items-start gap-9 lg:grid-cols-[minmax(0,1fr)_316px] lg:gap-11">
      {/* ------------------------------------------------------ main column */}
      <div>
        {!supabaseConfigured() ? (
          <div className="mb-8">
            <EmptyState message="Supabase is not configured. Set NEXT_PUBLIC_SUPABASE_URL and NEXT_PUBLIC_SUPABASE_ANON_KEY." />
          </div>
        ) : null}

        {access && !access.ok ? (
          <div className="mb-8">
            <EmptyState
              message={`Supabase query failed (HTTP ${access.status}). On Vercel, verify NEXT_PUBLIC_SUPABASE_URL matches the shared project and NEXT_PUBLIC_SUPABASE_ANON_KEY is the full publishable key (sb_publishable_...) or legacy anon JWT — not a truncated value.`}
            />
          </div>
        ) : null}

        <section aria-labelledby="reconciliation-heading">
          <div className="mb-4 flex flex-wrap items-center gap-2.5">
            <span
              aria-hidden
              className={clsx(
                "relative size-[7px] shrink-0 rounded-full",
                status.tone === "live" ? "bg-gain" : "bg-ink-3",
              )}
            >
              {status.tone === "live" ? (
                <span className="absolute -inset-1 animate-ping rounded-full border border-gain" />
              ) : null}
            </span>
            <span
              className={clsx(
                "num text-[11px] font-semibold uppercase tracking-[0.1em]",
                status.tone === "live" ? "text-gain" : "text-ink-3",
              )}
            >
              {status.label}
            </span>
            <span className="num text-[11.5px] text-ink-3">{statusMeta}</span>
          </div>

          <h1 id="reconciliation-heading" className="tag mb-2.5">
            Latest completed P&amp;L reconciliation
          </h1>
          <p className="num text-[12px] text-ink-3">
            {latestReconciliation?.date ?? "No completed reconciliation"}
          </p>

          <div className="mt-5 grid gap-px overflow-hidden rounded-panel-lg border border-line bg-line sm:grid-cols-3">
            {[
              {
                label: "Lifecycle P&L",
                value: latestReconciliation?.lifecyclePnl ?? null,
                hint: "booked realized exits",
              },
              {
                label: "Broker MTM",
                value: latestReconciliation?.brokerMtm ?? null,
                hint: "equity − prior close",
              },
              {
                label: "Gap",
                value: latestReconciliation?.gap ?? null,
                hint: "broker − lifecycle",
              },
            ].map((metric) => (
              <div key={metric.label} className="bg-card px-5 py-4">
                <p className="tag">{metric.label}</p>
                <p
                  className={clsx(
                    "num mt-2 text-[clamp(25px,4vw,38px)] font-medium leading-none tracking-[-0.04em]",
                    pnlColor(metric.value),
                  )}
                >
                  {fmtSignedUsd(metric.value)}
                </p>
                <p className="mt-2 text-[11.5px] text-ink-3">{metric.hint}</p>
              </div>
            ))}
          </div>

          <Panel className="mt-6 p-5 pb-4">
            {reconciliationPoints.length >= 2 ? (
              <>
                <PnlReconciliationChart data={reconciliationPoints} />
                <p className="mt-3 text-[11.5px] text-ink-3">
                  Raw daily EOD observations. Solid is booked lifecycle P&amp;L;
                  dashed is broker mark-to-market. Straight segments connect recorded
                  dates only; hover for the exact gap.
                </p>
              </>
            ) : (
              <EmptyState message="Not enough completed reconciliation days to compare lifecycle P&L with broker MTM." />
            )}
          </Panel>

          <StatRow className="mt-8">
            <Stat
              label="Broker account equity"
              value={fmtUsd(accountEquity ?? null, 0)}
              hint={accountCapturedAt ? `captured ${relativeTime(accountCapturedAt)}` : "no account snapshot"}
            />
            <Stat
              label="Opened, last entry date"
              value={latestRun ? latestRun.total_trades : "—"}
              hint={latestRun ? `on ${latestRun.run_date}` : "no runs recorded"}
            />
            <Stat
              label="Win rate · 30d"
              value={fmtPct(winRate30d)}
              hint={`${closed30d} closed positions`}
            />
            <Stat
              label="Realized · 30d"
              value={fmtSignedUsd(totalPnl30d)}
              tone={toneOf(totalPnl30d)}
              hint={
                pnlPoints.length > 0
                  ? `across ${pnlPoints.length} exit days`
                  : "no exits in window"
              }
            />
          </StatRow>
        </section>

        <Section
          title="P&L by exit date"
          subtitle="Broker-position lifecycle P&L, grouped by the ET date each position closed."
          figure={
            <span className={pnlColor(totalPnl30d)}>
              {fmtSignedUsd(totalPnl30d)}
            </span>
          }
        >
          <Panel>
            {pnlPoints.length > 0 ? (
              <DailyPnlBars data={pnlPoints} />
            ) : (
              <EmptyState message="No positions closed in the last 30 days." />
            )}
          </Panel>
        </Section>

        <section className="mt-11">
          <Disclosure
            title="Entry-date cohorts"
            subtitle="Trades and full-lifecycle P&L grouped by the day each position opened"
            aggregates={[
              { label: "Sessions", value: runs.length },
              { label: "Win rate", value: fmtPct(winRate30d) },
              {
                label: "Lifetime",
                value: fmtSignedUsd(cohortPnl),
                tone:
                  cohortPnl > 0 ? "positive" : cohortPnl < 0 ? "negative" : "neutral",
              },
            ]}
          >
            {runs.length > 0 ? (
              <div className="overflow-x-auto">
                <table className="w-full text-[13px]">
                  <thead>
                    <tr>
                      {[
                        "Entry date",
                        "Opened",
                        "Closed",
                        "Win rate",
                        "Lifetime P&L",
                        "Equity",
                      ].map((head, i) => (
                        <th
                          key={head}
                          className={clsx(
                            "tag whitespace-nowrap border-b border-line px-[18px] py-3 font-medium",
                            i >= 1 && i !== 3 ? "text-right" : "text-left",
                          )}
                        >
                          {head}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {[...runs]
                      .reverse()
                      .slice(0, 10)
                      .map((run) => (
                        <tr
                          key={run.run_date}
                          className="border-b border-line transition-colors last:border-0 hover:bg-sunk"
                        >
                          <td className="num whitespace-nowrap px-[18px] py-3">
                            {run.run_date}
                          </td>
                          <td className="num whitespace-nowrap px-[18px] py-3 text-right">
                            {run.total_trades}
                          </td>
                          <td className="num whitespace-nowrap px-[18px] py-3 text-right">
                            {run.closed_trades}
                          </td>
                          <td className="whitespace-nowrap px-[18px] py-3">
                            <Meter value={run.win_rate} />
                          </td>
                          <td
                            className={clsx(
                              "num whitespace-nowrap px-[18px] py-3 text-right",
                              pnlColor(Number(run.total_pnl ?? 0)),
                            )}
                          >
                            {fmtSignedUsd(Number(run.total_pnl ?? 0))}
                          </td>
                          <td className="num whitespace-nowrap px-[18px] py-3 text-right text-ink-2">
                            {fmtUsd(
                              run.final_equity ??
                                snapshotEquityByDate.get(run.run_date) ??
                                null,
                              0,
                            )}
                          </td>
                        </tr>
                      ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <div className="p-[18px]">
                <EmptyState message="No sessions recorded yet." />
              </div>
            )}
          </Disclosure>
        </section>
      </div>

      {/* -------------------------------------------------------------- rail */}
      <aside
        aria-label="Current session"
        className="flex flex-col gap-4.5 lg:sticky lg:top-[82px]"
      >
        <PositionRail
          positions={todayPositions}
          liveState={sessionLiveState}
          liveFresh={liveFresh}
          capturedAt={sessionLiveState ? liveAccountCapturedAt : null}
          sessionDate={accountSessionDate}
          brokerDailyPnl={accountDailyPnl == null ? null : Number(accountDailyPnl)}
        />

        <Panel>
          <PanelHead
            title="Decision ledger"
            aside={decisionLog.runDate ?? "—"}
          />
          <DecisionLedger log={decisionLog} />
          <a
            href="/funnel"
            className="mt-3.5 inline-flex items-center gap-1.5 text-[12px] text-ink-2 transition-colors hover:text-ink"
          >
            Full funnel and rejection reasons →
          </a>
        </Panel>
      </aside>
    </div>
  );
}
