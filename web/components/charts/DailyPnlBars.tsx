"use client";

import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
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
  TOOLTIP_CURSOR_FILL,
  TOOLTIP_LABEL_STYLE,
} from "./chartTheme";

export type DailyPnlPoint = {
  label: string;
  pnl: number;
  trades: number;
};

export function DailyPnlBars({ data }: { data: DailyPnlPoint[] }) {
  return (
    <div className="h-64 w-full">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: 8 }}>
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
            width={64}
            tickFormatter={(v: number) => `$${v.toFixed(0)}`}
          />
          <Tooltip
            cursor={{ fill: TOOLTIP_CURSOR_FILL }}
            contentStyle={TOOLTIP_CONTENT_STYLE}
            labelStyle={TOOLTIP_LABEL_STYLE}
            formatter={(value, name) => {
              if (name === "pnl") {
                return [
                  Number(value).toLocaleString("en-US", {
                    style: "currency",
                    currency: "USD",
                  }),
                  "PnL",
                ];
              }
              return [String(value), "Trades"];
            }}
          />
          <Bar dataKey="pnl" radius={[4, 4, 0, 0]}>
            {data.map((point) => (
              <Cell
                key={point.label}
                fill={point.pnl >= 0 ? "var(--gain)" : "var(--loss)"}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
