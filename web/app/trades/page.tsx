import clsx from "clsx";
import { TradesTable } from "@/components/TradesTable";
import { EmptyState, PageHeader, Panel, Section } from "@/components/ui";
import { getLatestOpsMetrics, getTradeLifecycles } from "@/lib/data";
import { supabaseConfigured } from "@/lib/supabase";
import { fmtSignedUsd } from "@/lib/format";
import type { TradeLifecycleRow, TradeRow } from "@/lib/types";
import { deriveTradeDirection } from "@/lib/tradeDirection";

export const dynamic = "force-dynamic";

function lifecycleAsTrade(row: TradeLifecycleRow): TradeRow {
  const direction = deriveTradeDirection({
    symbol: row.symbol,
    side: "buy",
    details: row.details,
    // Current lifecycle records represent broker-held option positions. The
    // explicit event metadata above wins whenever it is available; this keeps
    // older rows readable while the system remains single-long-premium.
    assumeLongOptionPosition: true,
  });

  return {
    trade_uid: row.lifecycle_uid,
    run_date: (row.opened_at ?? row.closed_at ?? "").slice(0, 10),
    order_id: row.entry_order_id,
    symbol: row.symbol,
    underlying_symbol: row.underlying_symbol,
    asset_class: "option",
    side: direction.position_side,
    position_side: direction.position_side,
    contract_type: direction.contract_type,
    signal_bias: direction.signal_bias,
    entry_action: direction.entry_action,
    setup_type: row.setup_type ?? null,
    option_dte: row.option_dte ?? null,
    qty: row.filled_qty,
    entry_price: row.entry_fill_price,
    exit_price: row.exit_fill_price,
    entry_time: row.opened_at,
    exit_time: row.closed_at,
    exit_reason: row.exit_reason,
    pnl: row.realized_pnl,
    status: row.status,
    daily_bias: row.daily_bias ?? null,
    planned_underlying_rr: row.planned_underlying_rr ?? null,
    realized_r: row.realized_r ?? null,
    validation_summary: row.validation_summary ?? null,
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
    <div>
      <PageHeader title="Trades">
        Broker-position lifecycles over the last 90 days, using actual fills when
        available. Daily bias is the premarket ML reading; any LLM disagreement
        is shown as opinion context rather than replacing the model reading.
      </PageHeader>

      {!supabaseConfigured() ? (
        <div className="mb-8">
          <EmptyState message="Supabase is not configured. Set NEXT_PUBLIC_SUPABASE_URL and NEXT_PUBLIC_SUPABASE_ANON_KEY." />
        </div>
      ) : null}

      <Panel padded={false} className="overflow-hidden">
        {trades.length > 0 ? (
          <TradesTable trades={trades} />
        ) : (
          <div className="p-5">
            <EmptyState message="No trades recorded in the last 90 days." />
          </div>
        )}
      </Panel>

      <Section
        title="Biggest losers"
        subtitle="Legacy entry-lot analysis with the model's validation reasoning. Canonical P&L is the lifecycle table above."
      >
        {losers.length > 0 ? (
          <div className="flex flex-col gap-3">
            {losers.map((loser) => (
              <Panel
                key={loser.trade_uid ?? `${loser.run_date}-${loser.symbol}`}
                className="p-4"
              >
                <div className="flex flex-wrap items-center gap-x-3 gap-y-1.5 text-[13px]">
                  <span className="num font-medium text-loss">
                    {fmtSignedUsd(loser.pnl)}
                  </span>
                  <span className="num font-medium">
                    {loser.underlying_symbol ?? loser.symbol}
                  </span>
                  <span className="text-ink-2">
                    {loser.side ?? "?"} · {loser.setup_type ?? "?"}
                    {loser.option_dte != null ? ` · ${loser.option_dte} DTE` : ""}
                  </span>
                  <span className="num text-[11.5px] text-ink-3">
                    {loser.run_date}
                  </span>
                  {loser.exit_reason ? (
                    <span
                      className={clsx(
                        "num rounded border border-line-2 px-1.5 py-0.5",
                        "text-[9.5px] font-medium uppercase tracking-[0.08em] text-ink-3",
                      )}
                    >
                      {loser.exit_reason}
                    </span>
                  ) : null}
                </div>
                {loser.validation_reasoning ? (
                  <p className="mt-2.5 border-l-2 border-line-2 pl-3 text-[12px] leading-relaxed text-ink-2">
                    {loser.validation_reasoning}
                  </p>
                ) : (
                  <p className="mt-2.5 text-[12px] text-ink-3">
                    No validation reasoning captured for this trade.
                  </p>
                )}
              </Panel>
            ))}
          </div>
        ) : (
          <EmptyState message="No losing trades in the current metrics window (or ops metrics haven't been computed yet)." />
        )}
      </Section>
    </div>
  );
}
