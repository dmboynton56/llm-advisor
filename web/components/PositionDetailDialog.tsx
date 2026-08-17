"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import clsx from "clsx";
import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceDot,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { X } from "lucide-react";
import {
  AXIS_LINE,
  AXIS_TICK,
  GRID_STROKE,
  TOOLTIP_CONTENT_STYLE,
  TOOLTIP_CURSOR_FILL,
  TOOLTIP_LABEL_STYLE,
} from "@/components/charts/chartTheme";
import {
  fmtDateTime,
  fmtNum,
  fmtPct,
  fmtSignedUsd,
  formatOccLabel,
  pnlColor,
} from "@/lib/format";
import {
  asJsonRecord,
  firstJsonString,
  jsonBoolean,
  jsonNumber,
  jsonRecords,
} from "@/lib/json";
import type { JsonValue, OverviewPosition, PositionFill } from "@/lib/types";
import { formatPositionStatus } from "@/lib/positions";

type PositionChartBar = {
  timestamp: string;
  timestampMs: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number | null;
};

type ChartResponse = {
  bars: PositionChartBar[];
  source?: string;
  error?: string;
  error_code?: string;
  request_id?: string;
  retryable?: boolean;
};

function parseChartResponse(value: JsonValue): ChartResponse {
  const root = asJsonRecord(value) ?? {};
  const bars = jsonRecords(root.bars).flatMap((bar) => {
    const timestamp = firstJsonString(bar.timestamp);
    const timestampMs = jsonNumber(bar.timestampMs) ?? (timestamp ? Date.parse(timestamp) : NaN);
    const close = jsonNumber(bar.close);
    if (!timestamp || !Number.isFinite(timestampMs) || close === null) return [];
    return [{
      timestamp,
      timestampMs,
      open: jsonNumber(bar.open) ?? close,
      high: jsonNumber(bar.high) ?? close,
      low: jsonNumber(bar.low) ?? close,
      close,
      volume: jsonNumber(bar.volume),
    }];
  });
  return {
    bars,
    source: firstJsonString(root.source) ?? undefined,
    error: firstJsonString(root.error) ?? undefined,
    error_code: firstJsonString(root.error_code) ?? undefined,
    request_id: firstJsonString(root.request_id) ?? undefined,
    retryable: jsonBoolean(root.retryable) ?? undefined,
  };
}

function fillLabel(fill: PositionFill): string {
  if (fill.kind === "entry") return "Entry";
  if (fill.kind === "partial_exit") return fill.stage ? `${fill.stage} exit` : "Partial exit";
  return "Final exit";
}

function markerColor(fill: PositionFill): string {
  if (fill.kind === "entry") return "var(--ink)";
  return fill.pnl != null && fill.pnl < 0 ? "var(--loss)" : "var(--gain)";
}

function ChartTooltip({
  active,
  payload,
}: {
  active?: boolean;
  payload?: Array<{ payload: PositionChartBar }>;
}) {
  const point = payload?.[0]?.payload;
  if (!active || !point) return null;
  return (
    <div className="rounded-lg border border-line-2 bg-card px-3 py-2 shadow-panel">
      <p className="tag">{fmtDateTime(point.timestamp)}</p>
      <p className="num mt-1 text-[14px] font-medium">{fmtNum(point.close)}</p>
      <p className="num mt-1 text-[10px] text-ink-3">
        H {fmtNum(point.high)} · L {fmtNum(point.low)}
      </p>
    </div>
  );
}

function Stat({ label, value, tone }: { label: string; value: React.ReactNode; tone?: string }) {
  return (
    <div className="min-w-0">
      <p className="tag">{label}</p>
      <p className={clsx("num mt-1.5 truncate text-[15px] font-medium", tone)}>{value}</p>
    </div>
  );
}

export function PositionDetailDialog({
  position,
  onClose,
}: {
  position: OverviewPosition | null;
  onClose: () => void;
}) {
  const dialogRef = useRef<HTMLDialogElement>(null);
  const [chart, setChart] = useState<ChartResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [chartReady, setChartReady] = useState(false);
  const [retryNonce, setRetryNonce] = useState(0);

  useEffect(() => {
    const dialog = dialogRef.current;
    if (!dialog) return;
    let frame: number | null = null;

    if (position) {
      // A native dialog is display:none until showModal(). Mounting Recharts
      // before that point lets ResponsiveContainer cache a zero-width parent.
      setChartReady(false);
      if (!dialog.open) dialog.showModal();
      frame = window.requestAnimationFrame(() => setChartReady(true));
    } else {
      setChartReady(false);
      if (dialog.open) dialog.close();
    }

    return () => {
      if (frame !== null) window.cancelAnimationFrame(frame);
    };
  }, [position]);

  useEffect(() => {
    if (!position) return;
    let cancelled = false;
    setLoading(true);
    setChart(null);
    fetch(`/api/position-chart?position_id=${encodeURIComponent(position.id)}`, {
      cache: "no-store",
    })
      .then(async (response) => {
        const value: JsonValue = await response.json();
        const payload = parseChartResponse(value);
        if (!response.ok && !payload.error) throw new Error("Chart unavailable");
        return payload;
      })
      .then((payload) => {
        if (!cancelled) setChart(payload);
      })
      .catch((error) => {
        if (!cancelled) {
          setChart({
            bars: [],
            error: error instanceof Error ? error.message : "Chart unavailable",
          });
        }
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [position, retryNonce]);

  const chartBars = useMemo(
    () => (chart?.bars ?? []).filter((bar) => Number.isFinite(bar.close)),
    [chart],
  );

  const close = () => {
    dialogRef.current?.close();
    onClose();
  };

  return (
    <dialog
      ref={dialogRef}
      aria-label={position ? `${formatOccLabel(position.option_symbol)} details` : undefined}
      className="position-dialog w-[min(760px,calc(100vw-16px))] overflow-hidden rounded-[18px] border border-panel-border bg-card p-0 text-ink shadow-panel"
      onClose={onClose}
    >
      {position ? (
        <div className="flex max-h-[calc(100dvh-16px)] flex-col overflow-y-auto">
          <div className="sticky top-0 z-10 border-b border-line bg-card/95 px-4 py-4 backdrop-blur-sm sm:px-6">
            <div className="flex items-start justify-between gap-4">
              <div className="min-w-0">
                <p className="tag">{position.status === "open" ? "Open position" : "Closed today"}</p>
                <h2 className="mt-1 truncate text-[17px] font-semibold tracking-[-0.02em]">
                  {formatOccLabel(position.option_symbol)}
                </h2>
                <p className="num mt-1 truncate text-[10.5px] text-ink-3">
                  {position.option_symbol}
                </p>
              </div>
              <button
                type="button"
                aria-label="Close position details"
                onClick={close}
                className="-mr-1 -mt-1 grid size-10 shrink-0 place-items-center rounded-full text-ink-3 transition-colors hover:bg-sunk hover:text-ink"
              >
                <X size={18} strokeWidth={1.8} />
              </button>
            </div>
            <div className="mt-3 flex flex-wrap items-baseline gap-x-3 gap-y-1 text-[11px] text-ink-3">
              <span>{position.status === "open" ? "Opened" : "Held"} {fmtDateTime(position.opened_at)}</span>
              {position.closed_at ? <span>· Closed {fmtDateTime(position.closed_at)}</span> : null}
              {position.exit_reason ? <span>· {formatPositionStatus(position)}</span> : null}
            </div>
          </div>

          <div className="px-4 py-5 sm:px-6">
            <div className="grid grid-cols-2 gap-x-5 gap-y-5 border-b border-line pb-5 sm:grid-cols-4">
              <Stat label="P&L" value={fmtSignedUsd(position.total_pnl)} tone={pnlColor(position.total_pnl)} />
              <Stat label="Return" value={fmtPct(position.return_pct, 1)} tone={pnlColor(position.total_pnl)} />
              <Stat
                label="Quantity"
                value={
                  position.status === "open"
                    ? `${position.remaining_qty ?? "—"} / ${position.initial_qty ?? "—"}`
                    : `${position.initial_qty ?? "—"} / ${position.initial_qty ?? "—"}`
                }
              />
              <Stat
                label={position.status === "open" ? "Mark" : "Exit px"}
                value={fmtNum(position.current_price ?? position.exit_price)}
              />
            </div>

            {position.status === "open" && (position.realized_pnl ?? 0) !== 0 ? (
              <p className="mt-3 text-[11px] text-ink-3">
                <span className={pnlColor(position.realized_pnl)}>{fmtSignedUsd(position.realized_pnl)}</span>{" "}
                realized from partial exits · {fmtSignedUsd(position.unrealized_pnl)} unrealized
              </p>
            ) : null}

            <div className="mt-5 min-w-0 overflow-hidden rounded-xl border border-line bg-paper/35 p-2 sm:p-3">
              <div className="flex items-center justify-between gap-3 px-1 pb-2">
                <p className="tag">Option price · 1 min</p>
                {chart?.source ? <span className="tag">{chart.source}</span> : null}
              </div>
              {loading ? (
                <div className="grid h-64 place-items-center text-[12px] text-ink-3">Loading contract history…</div>
              ) : chartBars.length > 0 && chartReady ? (
                <div className="h-64 w-full min-w-0">
                  <ResponsiveContainer
                    width="100%"
                    height="100%"
                    minWidth={0}
                    minHeight={0}
                    debounce={80}
                  >
                    <LineChart data={chartBars} margin={{ top: 12, right: 10, bottom: 2, left: 2 }}>
                      <CartesianGrid stroke={GRID_STROKE} strokeDasharray="3 4" vertical={false} />
                      <XAxis
                        dataKey="timestampMs"
                        type="number"
                        domain={["dataMin", "dataMax"]}
                        tick={AXIS_TICK}
                        tickLine={false}
                        axisLine={{ stroke: AXIS_LINE }}
                        minTickGap={36}
                        tickFormatter={(value: number) =>
                          new Date(value).toLocaleTimeString("en-US", {
                            timeZone: "America/New_York",
                            hour: "numeric",
                            minute: "2-digit",
                          })
                        }
                      />
                      <YAxis
                        tick={AXIS_TICK}
                        tickLine={false}
                        axisLine={false}
                        width={48}
                        domain={["auto", "auto"]}
                        tickFormatter={(value: number) => `$${value.toFixed(2)}`}
                      />
                      <Tooltip
                        cursor={{ fill: TOOLTIP_CURSOR_FILL }}
                        contentStyle={TOOLTIP_CONTENT_STYLE}
                        labelStyle={TOOLTIP_LABEL_STYLE}
                        content={<ChartTooltip />}
                      />
                      <Line
                        type="monotone"
                        dataKey="close"
                        stroke="var(--ink)"
                        strokeWidth={2}
                        dot={false}
                        activeDot={{ r: 4, fill: "var(--ink)", stroke: "var(--card)", strokeWidth: 2 }}
                        connectNulls
                      />
                      {position.fills.map((fill, index) => {
                        const timestampMs = fill.timestamp ? Date.parse(fill.timestamp) : NaN;
                        if (!Number.isFinite(timestampMs) || fill.price === null) return null;
                        return (
                          <ReferenceDot
                            key={`${fill.kind}-${fill.timestamp}-${index}`}
                            x={timestampMs}
                            y={fill.price}
                            r={fill.kind === "entry" ? 4 : 5}
                            fill={markerColor(fill)}
                            stroke="var(--card)"
                            strokeWidth={2}
                            ifOverflow="extendDomain"
                          />
                        );
                      })}
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              ) : chartBars.length > 0 ? (
                <div className="grid h-64 place-items-center text-[12px] text-ink-3">Preparing chart...</div>
              ) : (
                <div className="flex h-64 flex-col items-center justify-center gap-3 px-6 text-center text-[12px] text-ink-3">
                  <p>{chart?.error ?? "No option bars were available for this holding window."}</p>
                  {chart?.request_id ? (
                    <p className="num text-[10px] text-ink-3">Request {chart.request_id}</p>
                  ) : null}
                  {chart?.retryable || chart?.error_code === "NO_DATA" ? (
                    <button
                      type="button"
                      onClick={() => setRetryNonce((value) => value + 1)}
                      className="rounded-lg border border-line-2 px-3 py-1.5 text-[11px] text-ink-2 transition-colors hover:bg-sunk"
                    >
                      Retry
                    </button>
                  ) : null}
                </div>
              )}
            </div>

            <div className="mt-5">
              <div className="flex items-baseline justify-between gap-3">
                <p className="tag">Execution trail</p>
                <p className="text-[11px] text-ink-3">{position.fills.length} fills</p>
              </div>
              {position.fills.length > 0 ? (
                <ol className="mt-2 divide-y divide-line border-y border-line">
                  {position.fills.map((fill, index) => (
                    <li key={`${fill.kind}-${fill.timestamp}-${index}`} className="flex items-center justify-between gap-3 py-3 text-[12px]">
                      <div className="flex min-w-0 items-center gap-2.5">
                        <span
                          aria-hidden
                          className="size-2 shrink-0 rounded-full border-2 border-card ring-1 ring-line-2"
                          style={{ backgroundColor: markerColor(fill) }}
                        />
                        <div className="min-w-0">
                          <p className="font-medium">{fillLabel(fill)}</p>
                          <p className="num mt-0.5 truncate text-[10.5px] text-ink-3">{fmtDateTime(fill.timestamp)}</p>
                        </div>
                      </div>
                      <div className="shrink-0 text-right">
                        <p className="num">{fill.qty ?? "—"} @ {fmtNum(fill.price)}</p>
                        {fill.kind !== "entry" ? <p className={clsx("num mt-0.5 text-[10.5px]", pnlColor(fill.pnl))}>{fmtSignedUsd(fill.pnl)}</p> : null}
                      </div>
                    </li>
                  ))}
                </ol>
              ) : (
                <p className="mt-2 text-[12px] text-ink-3">No fill trail was recorded for this position.</p>
              )}
            </div>
          </div>
        </div>
      ) : null}
    </dialog>
  );
}
