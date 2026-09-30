import clsx from "clsx";
import { DecisionLedger } from "@/components/DecisionLedger";
import { Disclosure } from "@/components/Disclosure";
import { EquityOverview } from "@/components/EquityOverview";
import { PnlReconciliationChart } from "@/components/charts/PnlReconciliationChart";
import { DailyPnlBars } from "@/components/charts/DailyPnlBars";
import { PositionRail } from "@/components/PositionRail";
import { StockTrial } from "@/components/StockTrial";
import {
  EmptyState,
  Panel,
  PanelHead,
  Section,
} from "@/components/ui";
import {
  getAccountSnapshots,
  getBrokerReconciliations,
  getDecisionLog,
  getLatestHeartbeat,
  getLiveState,
  getRuns,
  getShadowDecisions,
  getTradeLifecycles,
} from "@/lib/data";
import { getTodayOverviewPositions } from "@/lib/positions";
import { equityHistoryDays, equityRange, equitySeries } from "@/lib/equity";
import { supabaseConfigured, checkSupabaseAccess } from "@/lib/supabase";
import {
  fmtSignedUsd,
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

export default async function OverviewPage({
  searchParams,
}: {
  searchParams: Promise<{ range?: string | string[] }>;
}) {
  const range = equityRange((await searchParams).range);
  const [snapshots, reconciliations, runs, heartbeat, liveState, lifecycles, decisionLog, shadowDecisions] =
    await Promise.all([
      getAccountSnapshots(equityHistoryDays(range)),
      getBrokerReconciliations(90),
      getRuns(30),
      getLatestHeartbeat(),
      getLiveState("paper"),
      getTradeLifecycles(30),
      getDecisionLog(8),
      getShadowDecisions(),
    ]);

  const access =
    supabaseConfigured() && runs.length === 0 && !heartbeat
      ? await checkSupabaseAccess()
      : null;

  const latestSnapshot = snapshots.at(-1) ?? null;
  const liveAccountCapturedAt =
    liveState?.heartbeat_ts ?? null;
  const snapshotCapturedAt = latestSnapshot?.captured_at ?? null;
  const liveAccountIsNewer =
    liveState?.equity != null &&
    liveAccountCapturedAt != null &&
    (snapshotCapturedAt == null ||
      new Date(liveAccountCapturedAt).getTime() >
        new Date(snapshotCapturedAt).getTime());
  const equityPoints = equitySeries(snapshots, liveState);
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

        <div className="mb-7 flex flex-wrap items-center gap-2.5">
          <span
            aria-hidden
            className={clsx("size-[7px] shrink-0 rounded-full", status.tone === "live" ? "bg-gain" : "bg-ink-3")}
          />
          <span className={clsx("num text-[11px] font-semibold uppercase tracking-[0.1em]", status.tone === "live" ? "text-gain" : "text-ink-3")}>
            {status.label}
          </span>
          <span className="num text-[11.5px] text-ink-3">{statusMeta}</span>
        </div>

        <EquityOverview data={equityPoints} range={range} />

        <Section
          title="P&L by exit date"
          subtitle="Last 30 days · gains and losses from closed positions, on the day they exited (ET)."
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
            title="Why P&L numbers differ"
            subtitle={`Latest completed reconciliation · ${latestReconciliation?.date ?? "waiting for EOD"}`}
          >
            <div className="p-5">
              <div className="grid gap-5 sm:grid-cols-3">
                {[
                  { label: "Booked lifecycle P&L", value: latestReconciliation?.lifecyclePnl, hint: "Realized gains and losses from completed position lifecycles." },
                  { label: "Broker daily P&L", value: latestReconciliation?.brokerMtm, hint: "Account equity minus the prior close, including open positions." },
                  { label: "Difference", value: latestReconciliation?.gap, hint: "Broker minus booked. Open positions, fees, and day boundaries can create gaps." },
                ].map((metric) => (
                  <div key={metric.label}>
                    <p className="tag">{metric.label}</p>
                    <p className={clsx("num mt-2 text-[22px] font-medium", pnlColor(metric.value))}>
                      {fmtSignedUsd(metric.value)}
                    </p>
                    <p className="mt-2 text-[11.5px] leading-relaxed text-ink-3">{metric.hint}</p>
                  </div>
                ))}
              </div>
              <div className="mt-7 border-t border-line pt-5">
                {reconciliationPoints.length >= 2 ? (
                  <>
                    <PnlReconciliationChart data={reconciliationPoints} />
                    <p className="mt-3 text-[11.5px] text-ink-3">
                      Daily observations, not account balances. Solid is booked lifecycle P&amp;L;
                      dashed is broker daily P&amp;L. Hover for each recorded difference.
                    </p>
                  </>
                ) : (
                  <EmptyState message="Not enough completed days to compare booked P&L and broker daily P&L." />
                )}
              </div>
            </div>
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
        />

        <StockTrial decisions={shadowDecisions} />

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
