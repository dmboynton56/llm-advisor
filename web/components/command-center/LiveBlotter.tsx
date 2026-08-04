"use client";

import { useCallback, useEffect, useState } from "react";
import { Activity, AlertTriangle, RefreshCw } from "lucide-react";
import clsx from "clsx";
import {
  dateEtIso,
  fmtDateTime,
  fmtNum,
  fmtPct,
  fmtSignedUsd,
  fmtUsd,
  formatOccLabel,
  isRegularSessionEt,
  pnlColor,
  relativeTime,
} from "@/lib/format";
import type { LiveBlotterPayload, LiveStateRow } from "@/lib/types";
import { Panel } from "@/components/ui";

const POLL_MS = 20_000;
const STALE_MS = 3 * 60_000;

function isLiveStateStale(liveState: LiveStateRow | null): boolean {
  if (!liveState?.heartbeat_ts) return true;
  const age = Date.now() - new Date(liveState.heartbeat_ts).getTime();
  return Number.isNaN(age) || age > STALE_MS;
}

export function LiveBlotter() {
  const [data, setData] = useState<LiveBlotterPayload | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  const fetchLive = useCallback(async () => {
    try {
      const res = await fetch("/command-center/api/live", { cache: "no-store" });
      const body = (await res.json()) as LiveBlotterPayload & { error?: string };
      if (!res.ok && !body.account) {
        throw new Error(body.error || `HTTP ${res.status}`);
      }
      setData(body);
      setError(body.error ?? (res.ok ? null : body.error || `HTTP ${res.status}`));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load live blotter");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    let cancelled = false;
    let timer: ReturnType<typeof setInterval> | null = null;

    const tick = () => {
      if (document.hidden || cancelled) return;
      void fetchLive();
    };

    void fetchLive();
    timer = setInterval(tick, POLL_MS);

    const onVisibility = () => {
      if (!document.hidden) void fetchLive();
    };
    document.addEventListener("visibilitychange", onVisibility);

    return () => {
      cancelled = true;
      if (timer) clearInterval(timer);
      document.removeEventListener("visibilitychange", onVisibility);
    };
  }, [fetchLive]);

  const account = data?.account;
  const positions = data?.positions ?? [];
  const openOrders = data?.openOrders ?? [];
  const todaysOrders = data?.todaysOrders ?? [];
  const liveState = data?.liveState ?? null;
  const openUpl = positions.reduce((acc, p) => acc + (p.unrealized_pl || 0), 0);
  const sessionRealized =
    liveState?.session_date === dateEtIso() &&
    liveState.session_stats?.realized_pnl != null
      ? Number(liveState.session_stats.realized_pnl)
      : null;
  const realizedPnl =
    sessionRealized ??
    (account?.daily_pnl != null ? account.daily_pnl - openUpl : null);
  const reconciliationResidual =
    sessionRealized != null && account?.daily_pnl != null
      ? account.daily_pnl - sessionRealized - openUpl
      : null;
  const closedToday =
    liveState?.session_date === dateEtIso()
      ? (liveState.session_stats?.closed?.length ??
        liveState.session_stats?.fills ??
        0)
      : 0;
  const stale = isLiveStateStale(liveState);
  const sessionActive = isRegularSessionEt();
  const showNoStopBanner =
    positions.length > 0 &&
    stale &&
    sessionActive &&
    !liveState?.session_stats?.session_end_reason;

  return (
    <Panel className="flex flex-col gap-4">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <div className="flex items-center gap-2">
            <Activity className="size-4 text-ink-2" />
            <h2 className="text-[14px] font-semibold">Live blotter</h2>
          </div>
          <p className="mt-1 text-[11.5px] text-ink-3">
            Alpaca paper marks · software stop/TP (
            {fmtPct(data?.exitPolicy.stop_loss_pct ?? 0.35)} /{" "}
            {fmtPct(data?.exitPolicy.profit_target_pct ?? 0.25)}) · polls every 20s
          </p>
        </div>
        <div className="flex items-center gap-3 text-[11.5px] text-ink-3">
          <span className="num">
            {data?.fetchedAt
              ? `refreshed ${relativeTime(data.fetchedAt)}`
              : loading
                ? "loading…"
                : "—"}
          </span>
          <button
            type="button"
            onClick={() => void fetchLive()}
            className="inline-flex items-center gap-1.5 rounded-lg border border-line-2 px-2.5 py-1 text-ink-2 transition-colors hover:bg-sunk hover:text-ink"
          >
            <RefreshCw className="size-3" />
            Refresh
          </button>
        </div>
      </div>

      {/* Money is at risk with no enforcement — the one place a warning colour
          outranks the achromatic rule. */}
      {showNoStopBanner ? (
        <div className="flex items-start gap-2.5 rounded-panel border border-loss bg-loss-wash px-3.5 py-2.5 text-[13px]">
          <AlertTriangle className="mt-0.5 size-4 shrink-0 text-loss" />
          <div>
            <p className="font-medium text-loss">
              Positions open with no stop enforcement — the loop is down
            </p>
            <p className="mt-1 text-[11.5px] leading-relaxed text-ink-2">
              Alpaca still shows open positions but the{" "}
              <code className="num">llm_advisor_live_state</code> heartbeat is
              stale
              {liveState?.heartbeat_ts
                ? ` (last ${relativeTime(liveState.heartbeat_ts)})`
                : " (missing)"}
              . Option SL/TP are software-only — close manually or restart the
              live loop.
            </p>
          </div>
        </div>
      ) : null}

      {error ? (
        <p className="rounded-panel border border-dashed border-line-2 px-3.5 py-2.5 text-[13px] text-ink-2">
          {error}
        </p>
      ) : null}

      <div className="grid grid-cols-2 gap-3 lg:grid-cols-5">
        <BlotterStat label="Equity" value={fmtUsd(account?.equity ?? null, 0)} />
        <BlotterStat
          label="Broker daily PnL"
          value={fmtSignedUsd(account?.daily_pnl ?? null)}
          className={pnlColor(account?.daily_pnl)}
          hint={
            account?.daily_pnl_pct != null
              ? `${fmtPct(account.daily_pnl_pct, 2)} vs prior close`
              : "equity change vs prior close"
          }
        />
        <BlotterStat
          label="Open uPnL"
          value={fmtSignedUsd(openUpl)}
          className={pnlColor(openUpl)}
        />
        <BlotterStat
          label="Strategy realized today"
          value={fmtSignedUsd(realizedPnl)}
          className={pnlColor(realizedPnl)}
          hint={
            sessionRealized != null
              ? `${closedToday} exits · full entry-to-exit PnL`
              : "approx. broker PnL minus open uPnL"
          }
        />
        <BlotterStat
          label="Buying power"
          value={fmtUsd(account?.buying_power ?? null, 0)}
          hint={
            liveState
              ? stale
                ? `loop stale · ${relativeTime(liveState.heartbeat_ts)}`
                : `loop alive · tick ${liveState.loop_count ?? "—"}`
              : "no live_state row"
          }
        />
      </div>

      {reconciliationResidual != null &&
      Math.abs(reconciliationResidual) >= 0.01 ? (
        <p className="text-[11.5px] leading-relaxed text-ink-3">
          Reconciliation: broker PnL uses equity change versus the prior close;
          strategy realized PnL uses full entry-to-exit trade PnL. Overnight mark
          basis, fees, and account adjustments account for{" "}
          <span className={clsx("num", pnlColor(reconciliationResidual))}>
            {fmtSignedUsd(reconciliationResidual)}
          </span>
          .
        </p>
      ) : null}

      <div>
        <h3 className="tag">Open positions</h3>
        {positions.length === 0 ? (
          <p className="mt-2 text-[13px] text-ink-3">Flat — no open positions.</p>
        ) : (
          <div className="mt-2 overflow-x-auto">
            <table className="w-full text-[13px]">
              <thead>
                <tr>
                  {[
                    "Contract",
                    "Qty",
                    "Entry",
                    "Mark",
                    "uPnL",
                    "uPnL%",
                    "Stop / TP",
                    "DTE",
                  ].map((label) => (
                    <th
                      key={label}
                      className="tag whitespace-nowrap border-b border-line py-2.5 pr-3 text-left font-medium"
                    >
                      {label}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {positions.map((p) => (
                  <tr key={p.symbol} className="border-b border-line last:border-0">
                    <td className="py-2.5 pr-3">
                      <div className="num text-[12px] text-ink">
                        {formatOccLabel(p.symbol)}
                      </div>
                      <div className="num text-[10px] text-ink-3">{p.symbol}</div>
                    </td>
                    <td className="num py-2.5 pr-3">{p.qty}</td>
                    <td className="num py-2.5 pr-3">{fmtNum(p.entry_price)}</td>
                    <td className="num py-2.5 pr-3">{fmtNum(p.current_price)}</td>
                    <td className={clsx("num py-2.5 pr-3", pnlColor(p.unrealized_pl))}>
                      {fmtSignedUsd(p.unrealized_pl)}
                    </td>
                    <td
                      className={clsx("num py-2.5 pr-3", pnlColor(p.unrealized_plpc))}
                    >
                      {fmtPct(p.unrealized_plpc, 1)}
                    </td>
                    <td className="py-2.5 pr-3 text-[11.5px] text-ink-2">
                      <div className="num">
                        SL {fmtNum(p.stop_mark)} · TP {fmtNum(p.tp_mark)}
                      </div>
                      <div className="text-[10px] text-ink-3">
                        software (loop-enforced)
                      </div>
                    </td>
                    <td className="num py-2.5">{p.dte ?? "—"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        <OrderTable title="Working orders" orders={openOrders} empty="No open orders." />
        <OrderTable
          title="Fills today"
          orders={todaysOrders}
          empty="No fills today yet."
          showFill
        />
      </div>

      {liveState?.session_stats?.session_end_reason ? (
        <p className="text-[11.5px] text-ink-3">
          Session ended: {liveState.session_stats.session_end_reason}
          {liveState.heartbeat_ts
            ? ` · ${fmtDateTime(liveState.heartbeat_ts)}`
            : null}
        </p>
      ) : null}
    </Panel>
  );
}

function BlotterStat({
  label,
  value,
  hint,
  className,
}: {
  label: string;
  value: string;
  hint?: string;
  className?: string;
}) {
  return (
    <div className="rounded-panel border border-line px-3.5 py-2.5">
      <p className="tag">{label}</p>
      <p className={clsx("num mt-1 text-[17px] font-medium", className)}>{value}</p>
      {hint ? <p className="mt-1 text-[10.5px] text-ink-3">{hint}</p> : null}
    </div>
  );
}

function OrderTable({
  title,
  orders,
  empty,
  showFill = false,
}: {
  title: string;
  orders: LiveBlotterPayload["openOrders"];
  empty: string;
  showFill?: boolean;
}) {
  return (
    <div>
      <h3 className="tag">{title}</h3>
      {orders.length === 0 ? (
        <p className="mt-2 text-[13px] text-ink-3">{empty}</p>
      ) : (
        <div className="mt-2 overflow-x-auto">
          <table className="w-full text-[13px]">
            <thead>
              <tr>
                {["Symbol", "Side", "Type", "Qty", showFill ? "Fill" : "Limit"].map(
                  (label) => (
                    <th
                      key={label}
                      className="tag whitespace-nowrap border-b border-line py-2.5 pr-3 text-left font-medium"
                    >
                      {label}
                    </th>
                  ),
                )}
              </tr>
            </thead>
            <tbody>
              {orders.slice(0, 12).map((o, i) => (
                <tr
                  key={`${o.id ?? o.symbol}-${i}`}
                  className="border-b border-line last:border-0"
                >
                  <td className="num py-2.5 pr-3 text-[12px]">{o.symbol}</td>
                  <td className="py-2.5 pr-3 capitalize">{o.side}</td>
                  <td className="py-2.5 pr-3 capitalize">{o.type}</td>
                  <td className="num py-2.5 pr-3">{o.qty ?? "—"}</td>
                  <td className="num py-2.5">
                    {showFill ? fmtNum(o.filled_avg_price) : fmtNum(o.limit_price)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
