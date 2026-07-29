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
    positions.length > 0 && stale && sessionActive && !liveState?.session_stats?.session_end_reason;

  return (
    <section className="space-y-4 rounded-xl border border-zinc-800 bg-zinc-900/40 p-5">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <div className="flex items-center gap-2">
            <Activity className="size-4 text-emerald-400" />
            <h2 className="font-medium">Live blotter</h2>
          </div>
          <p className="mt-1 text-xs text-zinc-500">
            Alpaca paper marks · software stop/TP ({fmtPct(data?.exitPolicy.stop_loss_pct ?? 0.35)} /{" "}
            {fmtPct(data?.exitPolicy.profit_target_pct ?? 0.25)}) · polls every 20s
          </p>
        </div>
        <div className="flex items-center gap-3 text-xs text-zinc-500">
          <span>
            {data?.fetchedAt ? `refreshed ${relativeTime(data.fetchedAt)}` : loading ? "loading…" : "—"}
          </span>
          <button
            type="button"
            onClick={() => void fetchLive()}
            className="inline-flex items-center gap-1 rounded-md border border-zinc-700 px-2 py-1 text-zinc-300 hover:border-zinc-500"
          >
            <RefreshCw className="size-3" />
            Refresh
          </button>
        </div>
      </div>

      {showNoStopBanner ? (
        <div className="flex items-start gap-2 rounded-lg border border-rose-500/40 bg-rose-500/10 px-3 py-2 text-sm text-rose-200">
          <AlertTriangle className="mt-0.5 size-4 shrink-0" />
          <div>
            <p className="font-medium">Positions open with NO stop enforcement (loop down)</p>
            <p className="mt-0.5 text-xs text-rose-200/80">
              Alpaca still shows open positions but{" "}
              <code className="font-mono">llm_advisor_live_state</code> heartbeat is stale
              {liveState?.heartbeat_ts
                ? ` (last ${relativeTime(liveState.heartbeat_ts)})`
                : " (missing)"}
              . Option SL/TP are software-only — close manually or restart the live loop.
            </p>
          </div>
        </div>
      ) : null}

      {error ? (
        <div className="rounded-lg border border-amber-500/30 bg-amber-500/10 px-3 py-2 text-sm text-amber-200">
          {error}
        </div>
      ) : null}

      <div className="grid grid-cols-2 gap-3 lg:grid-cols-5">
        <Stat label="Equity" value={fmtUsd(account?.equity ?? null, 0)} />
        <Stat
          label="Broker daily PnL"
          value={fmtSignedUsd(account?.daily_pnl ?? null)}
          className={pnlColor(account?.daily_pnl)}
          hint={
            account?.daily_pnl_pct != null
              ? `${fmtPct(account.daily_pnl_pct, 2)} vs prior close`
              : "equity change vs prior close"
          }
        />
        <Stat
          label="Open uPnL"
          value={fmtSignedUsd(openUpl)}
          className={pnlColor(openUpl)}
        />
        <Stat
          label="Strategy realized today"
          value={fmtSignedUsd(realizedPnl)}
          className={pnlColor(realizedPnl)}
          hint={
            sessionRealized != null
              ? `${closedToday} exits · full entry-to-exit PnL`
              : "approx. broker PnL minus open uPnL"
          }
        />
        <Stat
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
        <p className="text-xs leading-relaxed text-zinc-500">
          Reconciliation: broker PnL uses equity change versus the prior close;
          strategy realized PnL uses full entry-to-exit trade PnL. Overnight
          mark basis, fees, and account adjustments account for{" "}
          <span className={pnlColor(reconciliationResidual)}>
            {fmtSignedUsd(reconciliationResidual)}
          </span>
          .
        </p>
      ) : null}

      <div>
        <h3 className="text-xs font-medium uppercase tracking-wide text-zinc-500">
          Open positions
        </h3>
        {positions.length === 0 ? (
          <p className="mt-2 text-sm text-zinc-500">Flat — no open positions.</p>
        ) : (
          <div className="mt-2 overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-zinc-800 text-left text-xs uppercase tracking-wide text-zinc-500">
                  <th className="py-2 pr-3 font-medium">Contract</th>
                  <th className="py-2 pr-3 font-medium">Qty</th>
                  <th className="py-2 pr-3 font-medium">Entry</th>
                  <th className="py-2 pr-3 font-medium">Mark</th>
                  <th className="py-2 pr-3 font-medium">uPnL</th>
                  <th className="py-2 pr-3 font-medium">uPnL%</th>
                  <th className="py-2 pr-3 font-medium">Stop / TP</th>
                  <th className="py-2 font-medium">DTE</th>
                </tr>
              </thead>
              <tbody>
                {positions.map((p) => (
                  <tr key={p.symbol} className="border-b border-zinc-900 last:border-0">
                    <td className="py-2 pr-3">
                      <div className="font-mono text-xs text-zinc-200">
                        {formatOccLabel(p.symbol)}
                      </div>
                      <div className="font-mono text-[10px] text-zinc-600">{p.symbol}</div>
                    </td>
                    <td className="py-2 pr-3 tabular-nums">{p.qty}</td>
                    <td className="py-2 pr-3 tabular-nums">{fmtNum(p.entry_price)}</td>
                    <td className="py-2 pr-3 tabular-nums">{fmtNum(p.current_price)}</td>
                    <td className={clsx("py-2 pr-3 tabular-nums", pnlColor(p.unrealized_pl))}>
                      {fmtSignedUsd(p.unrealized_pl)}
                    </td>
                    <td className={clsx("py-2 pr-3 tabular-nums", pnlColor(p.unrealized_plpc))}>
                      {fmtPct(p.unrealized_plpc, 1)}
                    </td>
                    <td className="py-2 pr-3 text-xs text-zinc-400">
                      <div>
                        SL {fmtNum(p.stop_mark)} · TP {fmtNum(p.tp_mark)}
                      </div>
                      <div className="text-[10px] text-zinc-600">
                        software (loop-enforced)
                      </div>
                    </td>
                    <td className="py-2 tabular-nums">{p.dte ?? "—"}</td>
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
        <p className="text-xs text-zinc-500">
          Session ended: {liveState.session_stats.session_end_reason}
          {liveState.heartbeat_ts
            ? ` · ${fmtDateTime(liveState.heartbeat_ts)}`
            : null}
        </p>
      ) : null}
    </section>
  );
}

function Stat({
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
    <div className="rounded-lg border border-zinc-800 bg-zinc-950/50 px-3 py-2">
      <p className="text-[10px] uppercase tracking-wide text-zinc-500">{label}</p>
      <p className={clsx("mt-0.5 text-lg font-semibold tabular-nums", className)}>
        {value}
      </p>
      {hint ? <p className="mt-0.5 text-[10px] text-zinc-600">{hint}</p> : null}
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
      <h3 className="text-xs font-medium uppercase tracking-wide text-zinc-500">
        {title}
      </h3>
      {orders.length === 0 ? (
        <p className="mt-2 text-sm text-zinc-500">{empty}</p>
      ) : (
        <div className="mt-2 overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-zinc-800 text-left text-xs uppercase tracking-wide text-zinc-500">
                <th className="py-2 pr-3 font-medium">Symbol</th>
                <th className="py-2 pr-3 font-medium">Side</th>
                <th className="py-2 pr-3 font-medium">Type</th>
                <th className="py-2 pr-3 font-medium">Qty</th>
                <th className="py-2 font-medium">{showFill ? "Fill" : "Limit"}</th>
              </tr>
            </thead>
            <tbody>
              {orders.slice(0, 12).map((o, i) => (
                <tr
                  key={`${o.id ?? o.symbol}-${i}`}
                  className="border-b border-zinc-900 last:border-0"
                >
                  <td className="py-2 pr-3 font-mono text-xs">{o.symbol}</td>
                  <td className="py-2 pr-3 capitalize">{o.side}</td>
                  <td className="py-2 pr-3 capitalize">{o.type}</td>
                  <td className="py-2 pr-3 tabular-nums">{o.qty ?? "—"}</td>
                  <td className="py-2 tabular-nums">
                    {showFill
                      ? fmtNum(o.filled_avg_price)
                      : fmtNum(o.limit_price)}
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
