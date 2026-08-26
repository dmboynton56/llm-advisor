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
} from "./chartTheme";

export type EquityPoint = {
  timestamp: number;
  capturedAt: string;
  equity: number;
  dailyPnl: number | null;
  deltaFromPrevious: number | null;
};

type EquityChartPoint = EquityPoint & {
  chartEquity: number;
};

/**
 * Smooth only the value used to draw the line. The original equity remains on
 * every point so the tooltip continues to report the broker snapshot exactly.
 * A wider window is useful when the live loop has added many intraday ticks,
 * while the first and last points stay anchored to the real period bounds.
 */
function smoothEquity(data: EquityPoint[]): EquityChartPoint[] {
  if (data.length < 5) {
    return data.map((point) => ({ ...point, chartEquity: point.equity }));
  }

  const radius = Math.min(6, Math.max(2, Math.round(data.length / 80)));
  return data.map((point, index) => {
    if (index === 0 || index === data.length - 1) {
      return { ...point, chartEquity: point.equity };
    }

    const start = Math.max(0, index - radius);
    const end = Math.min(data.length - 1, index + radius);
    const window = data.slice(start, end + 1);
    const chartEquity =
      window.reduce((total, candidate) => total + candidate.equity, 0) / window.length;
    return { ...point, chartEquity };
  });
}

function money(value: number | null): string {
  if (value === null || !Number.isFinite(value)) return "—";
  return value.toLocaleString("en-US", {
    style: "currency",
    currency: "USD",
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  });
}

function EquityTooltip({
  active,
  payload,
}: {
  active?: boolean;
  payload?: Array<{ payload: EquityPoint }>;
}) {
  const point = payload?.[0]?.payload;
  if (!active || !point) return null;
  const captured = new Date(point.capturedAt).toLocaleString("en-US", {
    timeZone: "America/New_York",
    month: "short",
    day: "numeric",
    year: "numeric",
    hour: "numeric",
    minute: "2-digit",
    timeZoneName: "short",
  });
  return (
    <div className="rounded-lg border border-line-2 bg-card px-3 py-2 shadow-panel">
      <p className="tag">{captured}</p>
      <p className="num mt-1 text-[14px] font-medium">{money(point.equity)}</p>
      <p className="num mt-1 text-[10px] text-ink-3">
        Point change {money(point.deltaFromPrevious)}
      </p>
      <p className="num text-[10px] text-ink-3">
        Daily P&amp;L {money(point.dailyPnl)}
      </p>
    </div>
  );
}

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
  const chartData = smoothEquity(data);
  const intraday =
    data.length > 1 && data[data.length - 1].timestamp - data[0].timestamp <= 2 * 86_400_000;

  return (
    <div className="h-64 w-full">
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={chartData} margin={{ top: 8, right: 8, bottom: 0, left: 8 }}>
          <defs>
            <linearGradient id="equityFill" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor={stroke} stopOpacity={0.16} />
              <stop offset="100%" stopColor={stroke} stopOpacity={0} />
            </linearGradient>
          </defs>
          <CartesianGrid stroke={GRID_STROKE} strokeDasharray="3 4" vertical={false} />
          <XAxis
            dataKey="timestamp"
            type="number"
            domain={["dataMin", "dataMax"]}
            tick={AXIS_TICK}
            tickLine={false}
            axisLine={{ stroke: AXIS_LINE }}
            minTickGap={40}
            tickFormatter={(value: number) =>
              new Date(value).toLocaleString("en-US", {
                timeZone: "America/New_York",
                month: "short",
                day: "numeric",
                hour: intraday ? "numeric" : undefined,
                minute: intraday ? "2-digit" : undefined,
              })
            }
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
            content={<EquityTooltip />}
            cursor={{ stroke: AXIS_LINE, strokeDasharray: "3 4" }}
          />
          <Area
            type="monotone"
            dataKey="chartEquity"
            stroke={stroke}
            strokeWidth={2}
            strokeLinecap="round"
            fill="url(#equityFill)"
            dot={false}
            activeDot={{ r: 4, strokeWidth: 2, fill: "var(--card)", stroke }}
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}
