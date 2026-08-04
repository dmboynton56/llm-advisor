"use client";

import {
  CartesianGrid,
  Line,
  LineChart,
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

export type ApprovalPoint = {
  label: string;
  approvalRate: number;
  decisions: number;
};

export function ApprovalRateLine({ data }: { data: ApprovalPoint[] }) {
  return (
    <div className="h-64 w-full">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: 8 }}>
          <CartesianGrid stroke={GRID_STROKE} strokeDasharray="3 4" vertical={false} />
          <XAxis
            dataKey="label"
            tick={AXIS_TICK}
            tickLine={false}
            axisLine={{ stroke: AXIS_LINE }}
            minTickGap={30}
          />
          <YAxis
            tick={AXIS_TICK}
            tickLine={false}
            axisLine={false}
            width={48}
            domain={[0, 1]}
            tickFormatter={(v: number) => `${Math.round(v * 100)}%`}
          />
          <Tooltip
            contentStyle={TOOLTIP_CONTENT_STYLE}
            labelStyle={TOOLTIP_LABEL_STYLE}
            formatter={(value, name) => {
              if (name === "approvalRate") {
                return [`${(Number(value) * 100).toFixed(0)}%`, "Approval rate"];
              }
              return [String(value), "Decisions"];
            }}
          />
          {/* The gate's own behaviour is not a P&L quantity, so it stays achromatic. */}
          <Line
            type="monotone"
            dataKey="approvalRate"
            stroke="var(--ink)"
            strokeWidth={2}
            dot={{ r: 2.5, fill: "var(--ink)", strokeWidth: 0 }}
            activeDot={{ r: 4, fill: "var(--card)", stroke: "var(--ink)", strokeWidth: 2 }}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
