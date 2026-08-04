import { BreakdownGrid } from "@/components/BreakdownGrid";
import { EmptyState, PageHeader, Stat, StatRow, toneOf } from "@/components/ui";
import { getLatestOpsMetrics } from "@/lib/data";
import { supabaseConfigured } from "@/lib/supabase";
import { fmtNum, fmtPct, fmtSignedUsd, fmtUsd } from "@/lib/format";

export const dynamic = "force-dynamic";

export default async function BreakdownsPage() {
  const opsMetrics = await getLatestOpsMetrics();
  const payload = opsMetrics?.payload;
  const overall = payload?.overall;
  const breakdowns = payload?.breakdowns;

  return (
    <div>
      <PageHeader title="Breakdowns">
        Performance sliced by underlying, direction, setup type, and days to
        expiration
        {opsMetrics
          ? ` — window ${payload?.range?.start ?? "?"} → ${
              payload?.range?.end ?? opsMetrics.metric_date
            }`
          : ""}
        . Cells with fewer than 10 closed trades are dimmed.
      </PageHeader>

      {!supabaseConfigured() ? (
        <div className="mb-8">
          <EmptyState message="Supabase is not configured. Set NEXT_PUBLIC_SUPABASE_URL and NEXT_PUBLIC_SUPABASE_ANON_KEY." />
        </div>
      ) : null}

      {!payload ? (
        <EmptyState message="Ops metrics haven't been computed yet — the EOD workflow writes a daily rollup after each session." />
      ) : (
        <>
          <StatRow>
            <Stat
              label="Total P&L"
              value={fmtSignedUsd(overall?.total_pnl ?? null)}
              tone={toneOf(overall?.total_pnl ?? null)}
              hint={`${overall?.closed_trades ?? 0} closed trades`}
            />
            <Stat
              label="Win rate"
              value={fmtPct(overall?.win_rate ?? null)}
              hint={`avg win ${fmtUsd(overall?.average_win ?? null)} / avg loss ${fmtUsd(
                overall?.average_loss ?? null,
              )}`}
            />
            <Stat
              label="Avg risk/reward"
              value={fmtNum(overall?.avg_realized_rr ?? null)}
              hint={`profit factor ${fmtNum(overall?.profit_factor ?? null)}`}
            />
            <Stat
              label="Max drawdown"
              value={
                overall?.max_drawdown != null ? fmtUsd(overall.max_drawdown) : "—"
              }
              hint={`${fmtNum(overall?.trades_per_day ?? null, 1)} trades/day`}
            />
          </StatRow>

          <BreakdownGrid
            title="By underlying"
            subtitle="SPY vs QQQ vs IWM"
            cells={breakdowns?.by_underlying ?? {}}
            keyOrder={["SPY", "QQQ", "IWM"]}
          />
          <BreakdownGrid
            title="Long vs short"
            cells={breakdowns?.by_side ?? {}}
            keyOrder={["long", "short"]}
          />
          <BreakdownGrid
            title="By setup type"
            subtitle="MR = mean reversion, TC = trend continuation"
            cells={breakdowns?.by_setup_type ?? {}}
            keyOrder={["MR", "TC"]}
          />
          <BreakdownGrid
            title="By days to expiration"
            subtitle="0DTE vs longer-dated option entries"
            cells={breakdowns?.by_dte_bucket ?? {}}
            keyOrder={["0", "1-3", "4-7", "8-14", "15+"]}
          />
        </>
      )}
    </div>
  );
}
