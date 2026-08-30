"use client";

import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { AXIS_LINE, AXIS_TICK, GRID_STROKE } from "./chartTheme";

export type PnlReconciliationPoint = {
  date: string;
  lifecyclePnl: number;
  brokerMtm: number | null;
  gap: number | null;
  exits: number;
};

function money(value: number | null): string {
  if (value === null || !Number.isFinite(value)) return "—";
  return value.toLocaleString("en-US", {
    style: "currency",
    currency: "USD",
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
    signDisplay: "always",
  });
}

function ReconciliationTooltip({
  active,
  payload,
}: {
  active?: boolean;
  payload?: Array<{ payload: PnlReconciliationPoint }>;
}) {
  const point = payload?.[0]?.payload;
  if (!active || !point) return null;
  return (
    <div className="min-w-52 rounded-lg border border-line-2 bg-card px-3 py-2 shadow-panel">
      <p className="tag">{point.date} · {point.exits} exits</p>
      <dl className="mt-2 space-y-1 text-[11px]">
        <div className="flex items-center justify-between gap-5">
          <dt className="text-ink-3">Lifecycle P&amp;L</dt>
          <dd className="num font-medium">{money(point.lifecyclePnl)}</dd>
        </div>
        <div className="flex items-center justify-between gap-5">
          <dt className="text-ink-3">Broker MTM</dt>
          <dd className="num font-medium">{money(point.brokerMtm)}</dd>
        </div>
        <div className="flex items-center justify-between gap-5 border-t border-line pt-1">
          <dt className="text-ink-3">Gap</dt>
          <dd className="num font-medium">{money(point.gap)}</dd>
        </div>
      </dl>
    </div>
  );
}

export function PnlReconciliationChart({
  data,
}: {
  data: PnlReconciliationPoint[];
}) {
  return (
    <div className="h-64 w-full" aria-label="Lifecycle P&L and broker mark-to-market by reconciliation date">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: 8 }}>
          <CartesianGrid stroke={GRID_STROKE} strokeDasharray="3 4" vertical={false} />
          <XAxis
            dataKey="date"
            tick={AXIS_TICK}
            tickLine={false}
            axisLine={{ stroke: AXIS_LINE }}
            minTickGap={32}
            tickFormatter={(date: string) => date.slice(5)}
          />
          <YAxis
            tick={AXIS_TICK}
            tickLine={false}
            axisLine={false}
            width={72}
            tickFormatter={(value: number) =>
              value.toLocaleString("en-US", {
                style: "currency",
                currency: "USD",
                maximumFractionDigits: 0,
              })
            }
          />
          <ReferenceLine y={0} stroke={AXIS_LINE} />
          <Tooltip content={<ReconciliationTooltip />} cursor={{ stroke: AXIS_LINE }} />
          <Legend
            align="left"
            verticalAlign="top"
            height={32}
            iconType="plainline"
            wrapperStyle={{ fontSize: 11, color: "var(--ink-3)" }}
          />
          <Line
            name="Lifecycle P&L"
            type="linear"
            dataKey="lifecyclePnl"
            stroke="var(--ink)"
            strokeWidth={2}
            dot={{ r: 2.5, fill: "var(--card)", strokeWidth: 1.5 }}
            activeDot={{ r: 4, fill: "var(--card)", strokeWidth: 2 }}
          />
          <Line
            name="Broker MTM"
            type="linear"
            dataKey="brokerMtm"
            stroke="var(--ink-3)"
            strokeWidth={2}
            strokeDasharray="5 4"
            connectNulls={false}
            dot={{ r: 2.5, fill: "var(--card)", strokeWidth: 1.5 }}
            activeDot={{ r: 4, fill: "var(--card)", strokeWidth: 2 }}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
