import { ApprovalRateLine } from "@/components/charts/ApprovalRateLine";
import { FunnelBars } from "@/components/charts/FunnelBars";
import {
  EmptyState,
  PageHeader,
  Panel,
  Section,
  Stat,
  StatRow,
} from "@/components/ui";
import { getLatestOpsMetrics, getValidationEvents } from "@/lib/data";
import { supabaseConfigured } from "@/lib/supabase";
import { fmtPct } from "@/lib/format";

export const dynamic = "force-dynamic";

const STAGE_LABELS = new Map([
  ["signals", "Signals detected"],
  ["validation_approved", "LLM approved"],
  ["execution_attempted", "Execution attempted"],
  ["executed", "Orders executed"],
]);

export default async function FunnelPage() {
  const [opsMetrics, validationEvents] = await Promise.all([
    getLatestOpsMetrics(),
    getValidationEvents(30),
  ]);

  const funnel = opsMetrics?.payload?.funnel;

  const stageData = funnel
    ? Object.entries(funnel.stages).map(([stage, count]) => ({
        stage: STAGE_LABELS.get(stage) ?? stage,
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
    <div>
      <PageHeader title="Execution funnel">
        How raw z-score signals survive the LLM validation gate and reach Alpaca
        as option orders.
      </PageHeader>

      {!supabaseConfigured() ? (
        <div className="mb-8">
          <EmptyState message="Supabase is not configured. Set NEXT_PUBLIC_SUPABASE_URL and NEXT_PUBLIC_SUPABASE_ANON_KEY." />
        </div>
      ) : null}

      {funnel ? (
        <>
          <StatRow>
            <Stat label="Signals" value={funnel.stages.signals ?? 0} />
            <Stat
              label="LLM approval rate"
              value={fmtPct(funnel.llm_approval_rate)}
              hint="approved / (approved + rejected)"
            />
            <Stat
              label="Executed"
              value={funnel.stages.executed ?? 0}
              hint="orders accepted by Alpaca"
            />
            <Stat
              label="Signal → fill"
              value={
                funnel.stages.signals
                  ? fmtPct((funnel.stages.executed ?? 0) / funnel.stages.signals)
                  : "—"
              }
            />
          </StatRow>

          <Section
            title="Funnel stages"
            subtitle={`Current metrics window (${
              opsMetrics?.payload?.range?.start ?? "?"
            } → ${opsMetrics?.payload?.range?.end ?? opsMetrics?.metric_date})`}
          >
            <Panel>
              <FunnelBars data={stageData} />
            </Panel>
          </Section>

          <Section
            title="Rejection reasons"
            subtitle="Why signals didn't become trades — validation vetoes and execution failures."
          >
            <Panel>
              {rejections.length > 0 ? (
                <ul className="flex flex-col gap-2.5">
                  {rejections.map(([reason, count]) => (
                    <li key={reason} className="flex items-center gap-3">
                      <span className="w-56 shrink-0 truncate text-[11.5px] text-ink-2">
                        {reason}
                      </span>
                      <span className="h-3.5 flex-1 overflow-hidden rounded bg-sunk">
                        <span
                          className="block h-full rounded bg-ink-2"
                          style={{
                            width: `${Math.max(
                              (count / Math.max(maxRejection, 1)) * 100,
                              4,
                            )}%`,
                          }}
                        />
                      </span>
                      <span className="num w-8 text-right text-[11.5px] text-ink">
                        {count}
                      </span>
                    </li>
                  ))}
                </ul>
              ) : (
                <EmptyState message="No rejections in the current window." />
              )}
            </Panel>
          </Section>
        </>
      ) : (
        <EmptyState message="Ops metrics haven't been computed yet — the EOD workflow writes a daily rollup after each session." />
      )}

      <Section
        title="LLM approval rate over time"
        subtitle="Daily validation decisions from order events (30 days)"
      >
        <Panel>
          {approvalSeries.length > 0 ? (
            <ApprovalRateLine data={approvalSeries} />
          ) : (
            <EmptyState message="No validation events in the last 30 days." />
          )}
        </Panel>
      </Section>
    </div>
  );
}
