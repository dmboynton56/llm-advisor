import { Card, EmptyState, MetricCard } from "@/components/MetricCard";
import { ApprovalRateLine } from "@/components/charts/ApprovalRateLine";
import { FunnelBars } from "@/components/charts/FunnelBars";
import { getLatestOpsMetrics, getValidationEvents } from "@/lib/data";
import { supabaseConfigured } from "@/lib/supabase";
import { fmtPct } from "@/lib/format";

export const revalidate = 300;

const STAGE_LABELS: Record<string, string> = {
  signals: "Signals detected",
  validation_approved: "LLM approved",
  execution_attempted: "Execution attempted",
  executed: "Orders executed",
};

export default async function FunnelPage() {
  const [opsMetrics, validationEvents] = await Promise.all([
    getLatestOpsMetrics(),
    getValidationEvents(30),
  ]);

  const funnel = opsMetrics?.payload?.funnel;

  const stageData = funnel
    ? Object.entries(funnel.stages).map(([stage, count]) => ({
        stage: STAGE_LABELS[stage] ?? stage,
        count,
      }))
    : [];

  const rejections = funnel
    ? Object.entries(funnel.rejection_reasons).slice(0, 12)
    : [];
  const maxRejection = rejections.reduce((m, [, c]) => Math.max(m, c), 0);

  const byDay = new Map<string, { approved: number; rejected: number }>();
  for (const event of validationEvents) {
    const entry = byDay.get(event.run_date) ?? { approved: 0, rejected: 0 };
    if (event.event_type === "validation_approved") entry.approved += 1;
    else entry.rejected += 1;
    byDay.set(event.run_date, entry);
  }
  const approvalSeries = Array.from(byDay.entries())
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([day, { approved, rejected }]) => ({
      label: day.slice(5),
      approvalRate: approved / Math.max(approved + rejected, 1),
      decisions: approved + rejected,
    }));

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-xl font-semibold tracking-tight">
          Execution funnel
        </h1>
        <p className="mt-1 text-sm text-zinc-500">
          How raw z-score signals survive the LLM validation gate and reach
          Alpaca as option orders.
        </p>
      </div>

      {!supabaseConfigured() ? (
        <EmptyState message="Supabase is not configured. Set NEXT_PUBLIC_SUPABASE_URL and NEXT_PUBLIC_SUPABASE_ANON_KEY." />
      ) : null}

      {funnel ? (
        <>
          <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
            <MetricCard label="Signals" value={funnel.stages.signals ?? 0} />
            <MetricCard
              label="LLM approval rate"
              value={fmtPct(funnel.llm_approval_rate)}
              hint="approved / (approved + rejected)"
            />
            <MetricCard
              label="Executed"
              value={funnel.stages.executed ?? 0}
              hint="orders accepted by Alpaca"
            />
            <MetricCard
              label="Signal → fill"
              value={
                funnel.stages.signals
                  ? fmtPct((funnel.stages.executed ?? 0) / funnel.stages.signals)
                  : "—"
              }
            />
          </div>

          <Card
            title="Funnel stages"
            subtitle={`Current metrics window (${
              opsMetrics?.payload?.range?.start ?? "?"
            } → ${opsMetrics?.payload?.range?.end ?? opsMetrics?.metric_date})`}
          >
            <FunnelBars data={stageData} />
          </Card>

          <Card
            title="Rejection reasons"
            subtitle="Why signals didn't become trades (validation + execution failures)"
          >
            {rejections.length > 0 ? (
              <div className="space-y-2">
                {rejections.map(([reason, count]) => (
                  <div key={reason} className="flex items-center gap-3 text-sm">
                    <span className="w-64 shrink-0 truncate text-xs text-zinc-400">
                      {reason}
                    </span>
                    <div className="h-4 flex-1 overflow-hidden rounded bg-zinc-800/60">
                      <div
                        className="h-full rounded bg-rose-400/70"
                        style={{
                          width: `${Math.max((count / Math.max(maxRejection, 1)) * 100, 4)}%`,
                        }}
                      />
                    </div>
                    <span className="w-8 text-right text-xs tabular-nums text-zinc-300">
                      {count}
                    </span>
                  </div>
                ))}
              </div>
            ) : (
              <EmptyState message="No rejections in the current window." />
            )}
          </Card>
        </>
      ) : (
        <EmptyState message="Ops metrics haven't been computed yet — the EOD workflow writes a daily rollup after each session." />
      )}

      <Card
        title="LLM approval rate over time"
        subtitle="Daily validation decisions from order events (30 days)"
      >
        {approvalSeries.length > 0 ? (
          <ApprovalRateLine data={approvalSeries} />
        ) : (
          <EmptyState message="No validation events in the last 30 days." />
        )}
      </Card>
    </div>
  );
}
