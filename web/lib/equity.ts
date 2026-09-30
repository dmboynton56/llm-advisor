import { dateEtIso } from "./format";
import type { AccountSnapshot, LiveStateRow } from "./types";

export const EQUITY_RANGES = [
  { value: "day", label: "Day", days: 1 },
  { value: "week", label: "Week", days: 7 },
  { value: "month", label: "Month", days: 30 },
  { value: "year", label: "Year", days: 365 },
] as const;

export type EquityRange = (typeof EQUITY_RANGES)[number]["value"];

export type EquityPoint = {
  timestamp: number;
  capturedAt: string;
  equity: number;
  lastEquity: number | null;
  dailyPnl: number | null;
  deltaFromPrevious: number | null;
};

export function equityRange(value: string | string[] | undefined): EquityRange {
  return EQUITY_RANGES.find((range) => range.value === value)?.value ?? "month";
}

export function equityHistoryDays(range: EquityRange): number {
  return (EQUITY_RANGES.find((item) => item.value === range)?.days ?? 30) + 7;
}

export function equitySeries(
  snapshots: AccountSnapshot[],
  live: LiveStateRow | null,
): EquityPoint[] {
  const observations = snapshots.map((snapshot) => ({
    capturedAt: snapshot.captured_at,
    equity: snapshot.equity,
    lastEquity: snapshot.last_equity,
    dailyPnl: snapshot.daily_pnl,
  }));
  if (live?.equity != null && live.heartbeat_ts) {
    observations.push({
      capturedAt: live.heartbeat_ts,
      equity: live.equity,
      lastEquity: live.last_equity,
      dailyPnl: live.daily_pnl,
    });
  }

  const byTimestamp = new Map<number, EquityPoint>();
  for (const observation of observations) {
    const timestamp = new Date(observation.capturedAt).getTime();
    if (observation.equity == null || !Number.isFinite(timestamp)) continue;
    const equity = Number(observation.equity);
    if (!Number.isFinite(equity)) continue;
    byTimestamp.set(timestamp, {
      timestamp,
      capturedAt: observation.capturedAt,
      equity,
      lastEquity: observation.lastEquity,
      dailyPnl: observation.dailyPnl,
      deltaFromPrevious: null,
    });
  }
  const points = [...byTimestamp.values()].sort((a, b) => a.timestamp - b.timestamp);
  return points.map((point, index) => ({
    ...point,
    deltaFromPrevious: index > 0 ? point.equity - points[index - 1].equity : null,
  }));
}

export function equityPeriod(data: EquityPoint[], range: EquityRange) {
  const latest = data.at(-1);
  if (!latest) return { points: [], baseline: null, change: null, changePct: null };

  let points: EquityPoint[];
  if (range === "day") {
    const day = dateEtIso(latest.capturedAt);
    points = data.filter((point) => dateEtIso(point.capturedAt) === day);
  } else {
    const days = EQUITY_RANGES.find((item) => item.value === range)?.days ?? 30;
    const cutoff = latest.timestamp - days * 86_400_000;
    // Keep the last observed balance before the period as its honest baseline.
    const before = data.findLastIndex((point) => point.timestamp <= cutoff);
    points = before >= 0 ? data.slice(before) : data.filter((point) => point.timestamp >= cutoff);
  }
  const baseline = range === "day" ? latest.lastEquity : points[0]?.equity ?? null;
  const change = baseline == null ? null : latest.equity - baseline;
  const changePct = change == null || !baseline ? null : change / baseline;
  return { points, baseline, change, changePct };
}

export function dailyEquityPoints(points: EquityPoint[]): EquityPoint[] {
  const daily = new Map<string, EquityPoint>();
  for (const point of points) daily.set(dateEtIso(point.capturedAt), point);
  const first = points[0];
  return first
    ? [...new Map([first, ...daily.values()].map((point) => [point.timestamp, point])).values()]
    : [];
}
