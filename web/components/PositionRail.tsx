"use client";

import { useMemo, useState } from "react";
import clsx from "clsx";
import { Panel, PanelHead } from "@/components/ui";
import { PositionDetailDialog } from "@/components/PositionDetailDialog";
import {
  fmtPct,
  fmtSignedUsd,
  formatOccLabel,
  pnlColor,
  relativeTime,
} from "@/lib/format";
import type { LiveStateRow, OverviewPosition } from "@/lib/types";
import { formatPositionStatus } from "@/lib/positions";

function PositionRow({
  position,
  onSelect,
}: {
  position: OverviewPosition;
  onSelect: () => void;
}) {
  const displayPnl = position.total_pnl;
  return (
    <li>
      <button
        type="button"
        onClick={onSelect}
        className="group flex min-h-14 w-full items-center justify-between gap-3 rounded-lg px-2.5 py-2 text-left transition-colors hover:bg-sunk focus-visible:bg-sunk"
      >
        <span className="min-w-0">
          <span className="block truncate text-[12px] font-medium text-ink-2">
            {formatOccLabel(position.option_symbol)}
          </span>
          <span className="num mt-1 block truncate text-[10px] text-ink-3">
            {position.status === "open"
              ? `${position.remaining_qty ?? "—"}/${position.initial_qty ?? "—"} left`
              : formatPositionStatus(position)}
          </span>
        </span>
        <span className="shrink-0 text-right">
          <span className={clsx("num block text-[12.5px] font-medium", pnlColor(displayPnl))}>
            {fmtPct(position.return_pct, 1)}
          </span>
          <span className={clsx("num mt-1 block text-[10px]", pnlColor(displayPnl))}>
            {fmtSignedUsd(displayPnl)}
          </span>
        </span>
      </button>
    </li>
  );
}

export function PositionRail({
  positions,
  liveState,
  liveFresh,
  capturedAt,
}: {
  positions: OverviewPosition[];
  liveState: LiveStateRow | null;
  liveFresh: boolean;
  capturedAt: string | null;
}) {
  const [selected, setSelected] = useState<OverviewPosition | null>(null);
  const open = useMemo(() => positions.filter((position) => position.status === "open"), [positions]);
  const closed = useMemo(() => positions.filter((position) => position.status === "closed"), [positions]);

  return (
    <>
      <Panel>
        <PanelHead
          title="Today’s positions"
          aside={`${open.length} open · ${closed.length} closed`}
        />
        <p
          className={clsx(
            "num text-[24px] font-medium tracking-[-0.03em]",
            pnlColor(liveState?.unrealized_pnl ?? null),
          )}
        >
          {fmtSignedUsd(liveState?.unrealized_pnl ?? null)}
        </p>
        <p className="mt-1 text-[11px] text-ink-3">open unrealized P&L · tap a position for its trail</p>

        {positions.length > 0 ? (
          <div className="mt-3.5">
            {open.length > 0 ? (
              <>
                <p className="tag border-b border-line pb-2">Open</p>
                <ul className="mt-1 divide-y divide-line/70">
                  {open.map((position) => (
                    <PositionRow
                      key={position.id}
                      position={position}
                      onSelect={() => setSelected(position)}
                    />
                  ))}
                </ul>
              </>
            ) : null}
            {closed.length > 0 ? (
              <div className={clsx(open.length > 0 && "mt-4")}>
                <p className="tag border-b border-line pb-2">Closed today</p>
                <ul className="mt-1 divide-y divide-line/70">
                  {closed.map((position) => (
                    <PositionRow
                      key={position.id}
                      position={position}
                      onSelect={() => setSelected(position)}
                    />
                  ))}
                </ul>
              </div>
            ) : null}
          </div>
        ) : (
          <p className="mt-3 text-[12.5px] text-ink-3">
            {liveState ? "No open or closed positions today." : "No live state recorded yet."}
          </p>
        )}

        <div className="mt-3.5 flex gap-6 border-t border-line pt-3.5">
          <div className="flex-1">
            <span className="tag">Session</span>
            <span className="num mt-1.5 block text-[15px] font-medium">
              {liveState?.session_stats?.wins ?? 0}W <span className="text-ink-3">/</span>{" "}
              {liveState?.session_stats?.losses ?? 0}L
            </span>
          </div>
          <div className="flex-1">
            <span className="tag">Realized</span>
            <span
              className={clsx(
                "num mt-1.5 block text-[15px] font-medium",
                pnlColor(
                  liveState?.session_stats?.realized_pnl != null
                    ? Number(liveState.session_stats.realized_pnl)
                    : null,
                ),
              )}
            >
              {liveState?.session_stats?.realized_pnl != null
                ? fmtSignedUsd(Number(liveState.session_stats.realized_pnl))
                : "—"}
            </span>
          </div>
        </div>

        {!liveFresh && liveState ? (
          <p className="mt-3 text-[11px] text-ink-3">
            Last session · updated {relativeTime(capturedAt)}
          </p>
        ) : null}
      </Panel>
      <PositionDetailDialog position={selected} onClose={() => setSelected(null)} />
    </>
  );
}

