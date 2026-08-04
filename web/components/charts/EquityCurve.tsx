"use client";

import {
  Area,
  AreaChart,
  CartesianGrid,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import {
  AXIS_LINE,
  AXIS_TICK,
  GRID_STROKE,
  TOOLTIP_CONTENT_STYLE,
  TOOLTIP_LABEL_STYLE,
} from "./chartTheme";

export type EquityPoint = {
  label: string;
  equity: number;
};

export function EquityCurve({
  data,
  baseline,
}: {
  data: EquityPoint[];
  baseline?: number | null;
}) {
  // The curve takes its colour from where the period ended, not from where the
  // last tick went.
  const up = data.length > 1 && data[data.length - 1].equity >= data[0].equity;
  const stroke = up ? "var(--gain)" : "var(--loss)";

  return (
    <div className="h-64 w-full">
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: 8 }}>
          <defs>
            <linearGradient id="equityFill" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor={stroke} stopOpacity={0.16} />
              <stop offset="100%" stopColor={stroke} stopOpacity={0} />
            </linearGradient>
          </defs>
          <CartesianGrid stroke={GRID_STROKE} strokeDasharray="3 4" vertical={false} />
          <XAxis
            dataKey="label"
            tick={AXIS_TICK}
            tickLine={false}
            axisLine={{ stroke: AXIS_LINE }}
            minTickGap={40}
          />
          <YAxis
            tick={AXIS_TICK}
            tickLine={false}
            axisLine={false}
            width={76}
            domain={["auto", "auto"]}
            tickFormatter={(v: number) =>
              v.toLocaleString("en-US", {
                style: "currency",
                currency: "USD",
                maximumFractionDigits: 0,
              })
            }
          />
          {baseline != null ? (
            <ReferenceLine
              y={baseline}
              stroke={AXIS_LINE}
              strokeDasharray="3 5"
              ifOverflow="extendDomain"
            />
          ) : null}
          <Tooltip
            contentStyle={TOOLTIP_CONTENT_STYLE}
            labelStyle={TOOLTIP_LABEL_STYLE}
            formatter={(value) => [
              Number(value).toLocaleString("en-US", {
                style: "currency",
                currency: "USD",
              }),
              "Equity",
            ]}
          />
          <Area
            type="monotone"
            dataKey="equity"
            stroke={stroke}
            strokeWidth={2}
            fill="url(#equityFill)"
            dot={false}
            activeDot={{ r: 4, strokeWidth: 2, fill: "var(--card)", stroke }}
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}
