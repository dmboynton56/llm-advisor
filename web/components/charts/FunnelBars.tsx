"use client";

import {
  Bar,
  BarChart,
  Cell,
  LabelList,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import {
  AXIS_LINE,
  AXIS_TICK,
  TOOLTIP_CONTENT_STYLE,
  TOOLTIP_CURSOR_FILL,
  TOOLTIP_LABEL_STYLE,
} from "./chartTheme";

export type FunnelStage = {
  stage: string;
  count: number;
};

// An ink ramp rather than four hues: the darker the bar, the further the
// signal survived.
const STAGE_FILLS = ["var(--ink-3)", "var(--ink-3)", "var(--ink-2)", "var(--ink)"];

export function FunnelBars({ data }: { data: FunnelStage[] }) {
  return (
    <div className="h-64 w-full">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart
          data={data}
          layout="vertical"
          margin={{ top: 8, right: 40, bottom: 0, left: 8 }}
        >
          <XAxis
            type="number"
            tick={AXIS_TICK}
            tickLine={false}
            axisLine={{ stroke: AXIS_LINE }}
            allowDecimals={false}
          />
          <YAxis
            type="category"
            dataKey="stage"
            tick={{ ...AXIS_TICK, fill: "var(--ink-2)", fontFamily: undefined, fontSize: 12 }}
            tickLine={false}
            axisLine={false}
            width={150}
          />
          <Tooltip
            cursor={{ fill: TOOLTIP_CURSOR_FILL }}
            contentStyle={TOOLTIP_CONTENT_STYLE}
            labelStyle={TOOLTIP_LABEL_STYLE}
          />
          <Bar dataKey="count" radius={[0, 4, 4, 0]} barSize={26}>
            {data.map((point, i) => (
              <Cell key={point.stage} fill={STAGE_FILLS[i % STAGE_FILLS.length]} />
            ))}
            <LabelList
              dataKey="count"
              position="right"
              style={{
                fill: "var(--ink-2)",
                fontSize: 11,
                fontFamily: "var(--font-plex-mono)",
              }}
            />
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
