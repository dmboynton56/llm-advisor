import clsx from "clsx";
import { DecisionLedger } from "@/components/DecisionLedger";
import { Disclosure } from "@/components/Disclosure";
import { EquityCurve } from "@/components/charts/EquityCurve";
import { DailyPnlBars } from "@/components/charts/DailyPnlBars";
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
  getDecisionLog,
  getLatestHeartbeat,
  getLiveState,
  getRuns,
  getTradeLifecycles,
} from "@/lib/data";
import { supabaseConfigured, checkSupabaseAccess } from "@/lib/supabase";
import {
  fmtDate,
  fmtPct,
  fmtSignedUsd,
  fmtUsd,
  formatOccLabel,
  isRegularSessionEt,
  normalizePlpc,
  pnlColor,
  relativeTime,
  dateEtIso,
} from "@/lib/format";
import type { LiveStateRow } from "@/lib/types";

export const dynamic = "force-dynamic";

const LIVE_FRESH_MS = 3 * 60_000;

function liveStateFresh(row: LiveStateRow | null): boolean {
  if (!row?.heartbeat_ts) return false;
  const age = Date.now() - new Date(row.heartbeat_ts).getTime();
  return !Number.isNaN(age) && age <= LIVE_FRESH_MS;
}

function heartbeatStatus(heartbeatTs: string | null): {
  label: string;
  stale: boolean;
} {
  if (!heartbeatTs) return { label: "No heartbeat", stale: true };
  const ageHours = (Date.now() - new Date(heartbeatTs).getTime()) / 3.6e6;
  // The loop only runs on market days, so a gap under ~4 days is just a
  // weekend or a holiday, not a fault.
  if (ageHours <= 96) return { label: "Idle", stale: false };
  return { label: "Stale", stale: true };
}

export default async function OverviewPage() {
  const [snapshots, runs, heartbeat, liveState, lifecycles, decisionLog] =
    await Promise.all([
      getAccountSnapshots(90),
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

  // "Since" is measured from the oldest snapshot actually in the window, not
  // from an assumed starting balance.
  const windowStart = equityPoints[0] ?? null;
  const sinceStart =
    windowStart && accountEquity != null
      ? Number(accountEquity) - windowStart.equity
      : null;
  const sinceStartPct =
    sinceStart != null && windowStart && windowStart.equity !== 0
      ? sinceStart / windowStart.equity
      : null;

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
  const closed30d = runs.reduce((acc, run) => acc + (run.closed_trades ?? 0), 0);
  const won30d = runs.reduce((acc, run) => acc + (run.winning_trades ?? 0), 0);
  const winRate30d = closed30d > 0 ? won30d / closed30d : null;
  const cohortPnl = runs.reduce((acc, run) => acc + Number(run.total_pnl ?? 0), 0);

  const liveFresh = liveStateFresh(liveState);
  const inSession = isRegularSessionEt();
  const sessionEnded = Boolean(liveState?.session_stats?.session_end_reason);
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

  const openPositions = liveState?.open_positions ?? [];

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

        <section aria-labelledby="equity-heading">
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

          <h1 id="equity-heading" className="tag mb-2.5">
            Paper account equity
          </h1>
          <span className="num block text-[clamp(40px,6vw,62px)] font-medium leading-none tracking-[-0.04em]">
            {fmtUsd(accountEquity ?? null, 2)}
          </span>

          <div className="mt-3.5 flex flex-wrap items-baseline gap-x-5 gap-y-2">
            <span
              className={clsx(
                "num inline-flex items-baseline gap-1.5 text-[13.5px]",
                pnlColor(accountDailyPnl ?? null),
              )}
            >
              {fmtSignedUsd(accountDailyPnl ?? null)}
              {accountDailyPnlPct != null ? (
                <span>{fmtPct(Number(accountDailyPnlPct), 2)}</span>
              ) : null}
              <span className="font-sans text-[12.5px] text-ink-3">today</span>
            </span>
            {sinceStart != null && windowStart ? (
              <span
                className={clsx(
                  "num inline-flex items-baseline gap-1.5 text-[13.5px]",
                  pnlColor(sinceStart),
                )}
              >
                {fmtSignedUsd(sinceStart)}
                {sinceStartPct != null ? <span>{fmtPct(sinceStartPct, 2)}</span> : null}
                <span className="font-sans text-[12.5px] text-ink-3">
                  since {fmtDate(windowStart.label)}
                </span>
              </span>
            ) : null}
          </div>

          <Panel className="mt-6 p-5 pb-4">
            {equityPoints.length >= 2 ? (
              <>
                <EquityCurve
                  data={equityPoints}
                  baseline={windowStart?.equity ?? null}
                />
                <p className="mt-3 text-[11.5px] text-ink-3">
                  Snapshots captured at live-loop start and end, 90 days. The
                  dashed rule marks equity at the start of the window
                  {accountCapturedAt ? ` · last value ${relativeTime(accountCapturedAt)}` : ""}
                  .
                </p>
              </>
            ) : (
              <EmptyState message="Not enough equity snapshots yet — the live loop writes one at session start and end each trading day." />
            )}
          </Panel>

          <StatRow className="mt-8">
            <Stat
              label="Broker daily P&L"
              value={fmtSignedUsd(accountDailyPnl ?? null)}
              tone={toneOf(accountDailyPnl ?? null)}
              hint={
                accountDailyPnlPct != null
                  ? `${fmtPct(Number(accountDailyPnlPct), 2)} vs prior close`
                  : "equity change vs prior close"
              }
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
                            {fmtUsd(run.final_equity, 0)}
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
        <Panel>
          <PanelHead
            title="Open positions"
            aside={`${liveState?.open_position_count ?? 0} held`}
          />
          <p
            className={clsx(
              "num text-[24px] font-medium tracking-[-0.03em]",
              pnlColor(liveState?.unrealized_pnl ?? null),
            )}
          >
            {fmtSignedUsd(liveState?.unrealized_pnl ?? null)}
          </p>

          {openPositions.length > 0 ? (
            <ul className="mt-3.5 flex flex-col">
              {openPositions.map((position) => (
                <li
                  key={position.symbol}
                  className="-mx-2.5 flex items-baseline justify-between gap-2.5 rounded-lg px-2.5 py-2 transition-colors hover:bg-sunk"
                >
                  <span className="num text-[12px] text-ink-2">
                    {formatOccLabel(position.symbol)}
                  </span>
                  <span
                    className={clsx(
                      "num text-[12.5px] font-medium",
                      pnlColor(position.unrealized_plpc),
                    )}
                  >
                    {fmtPct(normalizePlpc(position.unrealized_plpc), 1)}
                  </span>
                </li>
              ))}
            </ul>
          ) : (
            <p className="mt-3 text-[12.5px] text-ink-3">
              {liveState
                ? "Flat — no open positions."
                : "No live state recorded yet."}
            </p>
          )}

          <div className="mt-3.5 flex gap-6 border-t border-line pt-3.5">
            <div className="flex-1">
              <span className="tag">Session</span>
              <span className="num mt-1.5 block text-[15px] font-medium">
                {liveState?.session_stats?.wins ?? 0}W{" "}
                <span className="text-ink-3">/</span>{" "}
                {liveState?.session_stats?.losses ?? 0}L
              </span>
            </div>
            <div className="flex-1">
              <span className="tag">Realized</span>
              <span
                className={clsx(
                  "num mt-1.5 block text-[15px] font-medium",
                  pnlColor(
                    liveState?.session_stats?.realized_pnl != null
                      ? Number(liveState.session_stats.realized_pnl)
                      : null,
                  ),
                )}
              >
                {liveState?.session_stats?.realized_pnl != null
                  ? fmtSignedUsd(Number(liveState.session_stats.realized_pnl))
                  : "—"}
              </span>
            </div>
          </div>

          {!liveFresh && liveState ? (
            <p className="mt-3 text-[11px] text-ink-3">
              Last session · updated {relativeTime(liveAccountCapturedAt)}
            </p>
          ) : null}
        </Panel>

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
