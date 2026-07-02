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
          <CartesianGrid stroke="#27272a" strokeDasharray="3 3" vertical={false} />
          <XAxis
            dataKey="label"
            tick={{ fill: "#71717a", fontSize: 11 }}
            tickLine={false}
            axisLine={{ stroke: "#27272a" }}
            minTickGap={30}
          />
          <YAxis
            tick={{ fill: "#71717a", fontSize: 11 }}
            tickLine={false}
            axisLine={false}
            width={64}
            tickFormatter={(v: number) => `$${v.toFixed(0)}`}
          />
          <Tooltip
            cursor={{ fill: "rgba(63,63,70,0.25)" }}
            contentStyle={{
              backgroundColor: "#18181b",
              border: "1px solid #3f3f46",
              borderRadius: 8,
              fontSize: 12,
            }}
            labelStyle={{ color: "#a1a1aa" }}
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
          <Bar dataKey="pnl" radius={[3, 3, 0, 0]}>
            {data.map((point) => (
              <Cell
                key={point.label}
                fill={point.pnl >= 0 ? "#34d399" : "#fb7185"}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
