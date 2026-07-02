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

export type FunnelStage = {
  stage: string;
  count: number;
};

const STAGE_COLORS = ["#60a5fa", "#a78bfa", "#fbbf24", "#34d399"];

export function FunnelBars({ data }: { data: FunnelStage[] }) {
  return (
    <div className="h-64 w-full">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart
          data={data}
          layout="vertical"
          margin={{ top: 8, right: 24, bottom: 0, left: 8 }}
        >
          <CartesianGrid stroke="#27272a" strokeDasharray="3 3" horizontal={false} />
          <XAxis
            type="number"
            tick={{ fill: "#71717a", fontSize: 11 }}
            tickLine={false}
            axisLine={{ stroke: "#27272a" }}
            allowDecimals={false}
          />
          <YAxis
            type="category"
            dataKey="stage"
            tick={{ fill: "#a1a1aa", fontSize: 12 }}
            tickLine={false}
            axisLine={false}
            width={150}
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
          />
          <Bar dataKey="count" radius={[0, 3, 3, 0]} barSize={26}>
            {data.map((point, i) => (
              <Cell key={point.stage} fill={STAGE_COLORS[i % STAGE_COLORS.length]} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
