import { Card, EmptyState } from "@/components/MetricCard";
import { TradesTable } from "@/components/TradesTable";
import { getLatestOpsMetrics, getTradeLifecycles } from "@/lib/data";
import { supabaseConfigured } from "@/lib/supabase";
import { fmtSignedUsd } from "@/lib/format";
import type { TradeLifecycleRow, TradeRow } from "@/lib/types";

export const dynamic = "force-dynamic";

function lifecycleAsTrade(row: TradeLifecycleRow): TradeRow {
  return {
    trade_uid: row.lifecycle_uid,
    run_date: (row.opened_at ?? row.closed_at ?? "").slice(0, 10),
    order_id: row.entry_order_id,
    symbol: row.symbol,
    underlying_symbol: row.underlying_symbol,
    asset_class: "option",
    side: "buy",
    setup_type: null,
    option_dte: null,
    qty: row.filled_qty,
    entry_price: row.entry_fill_price,
    exit_price: row.exit_fill_price,
    entry_time: row.opened_at,
    exit_time: row.closed_at,
    exit_reason: row.exit_reason,
    pnl: row.realized_pnl,
    status: row.status,
  };
}

export default async function TradesPage() {
  const [lifecycles, opsMetrics] = await Promise.all([
    getTradeLifecycles(90),
    getLatestOpsMetrics(),
  ]);
  const trades = lifecycles.map(lifecycleAsTrade);

  const losers = opsMetrics?.payload?.biggest_losers ?? [];

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-xl font-semibold tracking-tight">Trades</h1>
        <p className="mt-1 text-sm text-zinc-500">
          Broker-position lifecycles over the last 90 days, using actual fills
          when available and exit-date attribution.
        </p>
      </div>

      {!supabaseConfigured() ? (
        <EmptyState message="Supabase is not configured. Set NEXT_PUBLIC_SUPABASE_URL and NEXT_PUBLIC_SUPABASE_ANON_KEY." />
      ) : null}

      <Card>
        {trades.length > 0 ? (
          <TradesTable trades={trades} />
        ) : (
          <EmptyState message="No trades recorded in the last 90 days." />
        )}
      </Card>

      <Card
        title="Biggest losers"
        subtitle="Legacy entry-lot analysis with the LLM's validation reasoning; canonical PnL is the lifecycle table above"
      >
        {losers.length > 0 ? (
          <div className="space-y-3">
            {losers.map((loser) => (
              <div
                key={loser.trade_uid ?? `${loser.run_date}-${loser.symbol}`}
                className="rounded-lg border border-zinc-800 bg-zinc-950/60 p-3"
              >
                <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-sm">
                  <span className="font-semibold text-rose-400">
                    {fmtSignedUsd(loser.pnl)}
                  </span>
                  <span className="font-medium">
                    {loser.underlying_symbol ?? loser.symbol}
                  </span>
                  <span className="text-zinc-400">
                    {loser.side ?? "?"} · {loser.setup_type ?? "?"}
                    {loser.option_dte != null ? ` · ${loser.option_dte} DTE` : ""}
                  </span>
                  <span className="text-xs text-zinc-500">{loser.run_date}</span>
                  {loser.exit_reason ? (
                    <span className="rounded bg-zinc-800 px-1.5 py-0.5 text-[10px] uppercase tracking-wide text-zinc-400">
                      {loser.exit_reason}
                    </span>
                  ) : null}
                </div>
                {loser.validation_reasoning ? (
                  <p className="mt-2 border-l-2 border-zinc-700 pl-3 text-xs leading-relaxed text-zinc-400">
                    {loser.validation_reasoning}
                  </p>
                ) : (
                  <p className="mt-2 text-xs text-zinc-600">
                    No validation reasoning captured for this trade.
                  </p>
                )}
              </div>
            ))}
          </div>
        ) : (
          <EmptyState message="No losing trades in the current metrics window (or ops metrics haven't been computed yet)." />
        )}
      </Card>
    </div>
  );
}
