import test from "node:test";
import assert from "node:assert/strict";
import { dailyEquityPoints, equityPeriod, equityRange, equitySeries } from "./equity";
import type { AccountSnapshot } from "./types";

function snapshot(capturedAt: string, equity: number | null, lastEquity = 100000): AccountSnapshot {
  return {
    captured_at: capturedAt, snapshot_date: capturedAt.slice(0, 10),
    equity, last_equity: lastEquity, daily_pnl: equity == null ? null : equity - lastEquity,
    buying_power: null, daily_pnl_pct: null, source: "paper_live_loop",
  };
}

test("equity uses observed account balances rather than a sum of daily P&L", () => {
  const points = equitySeries([
    snapshot("2026-09-29T15:00:00Z", 98500),
    snapshot("2026-09-28T20:00:00Z", 100000),
    snapshot("2026-09-29T14:00:00Z", 99000),
    snapshot("invalid", 90000),
    snapshot("2026-09-29T16:00:00Z", null),
  ], null);
  assert.deepEqual(points.map(point => point.equity), [100000, 99000, 98500]);
  assert.equal(equityPeriod(points, "day").change, -1500);
  assert.equal(equityPeriod(points, "day").points.length, 2);
  assert.equal(points.at(-1)?.deltaFromPrevious, -500);
});

test("day is the latest recorded ET session even when its UTC date has rolled over", () => {
  const points = equitySeries([
    snapshot("2026-09-29T19:00:00Z", 99000),
    snapshot("2026-09-30T00:30:00Z", 99500),
  ], null);
  const period = equityPeriod(points, "day");
  assert.equal(period.points.length, 2);
  assert.equal(period.baseline, 100000);
  assert.equal(period.change, -500);
});

test("a week uses the last actual observation before its boundary", () => {
  const points = equitySeries([
    snapshot("2026-09-19T20:00:00Z", 100000),
    snapshot("2026-09-22T20:00:00Z", 99000),
    snapshot("2026-09-23T20:00:00Z", 99500),
    snapshot("2026-09-30T19:00:00Z", 100500),
  ], null);
  const period = equityPeriod(points, "week");
  assert.equal(period.baseline, 99000);
  assert.equal(period.change, 1500);
  assert.deepEqual(period.points.map(point => point.equity), [99000, 99500, 100500]);
});

test("long ranges keep the opening observation and each ET day's final balance", () => {
  const points = equitySeries([
    snapshot("2026-09-28T14:00:00Z", 100000),
    snapshot("2026-09-28T15:00:00Z", 99900),
    snapshot("2026-09-29T14:00:00Z", 99500),
    snapshot("2026-09-29T20:00:00Z", 99800),
  ], null);
  assert.deepEqual(dailyEquityPoints(points).map(point => point.equity), [100000, 99900, 99800]);
});

test("missing history stays empty and unknown ranges fall back to month", () => {
  assert.deepEqual(equityPeriod([], "year"), { points: [], baseline: null, change: null, changePct: null });
  assert.equal(equityRange("invalid"), "month");
  assert.equal(equityRange(["year", "day"]), "month");
  const points = equitySeries([snapshot("2026-09-29T15:00:00Z", 100000)], null);
  assert.equal(equityPeriod(points, "year").points.length, 1);
});
