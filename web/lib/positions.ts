import { dateEtIso, parseOccSymbol } from "@/lib/format";
import {
  asJsonRecord,
  firstJsonString,
  jsonNumber,
  jsonRecords,
  type JsonInput,
} from "@/lib/json";
import type {
  JsonRecord,
  LiveOpenPosition,
  LiveStateRow,
  OverviewPosition,
  PositionFill,
  TradeLifecycleRow,
} from "@/lib/types";

function numberOrNull(value: JsonInput): number | null {
  return jsonNumber(value);
}

function normalizeSessionDate(value: string | Date): string {
  // A bare ISO date is already an ET session date. Parsing it as a JavaScript
  // Date would interpret midnight as UTC and shift it to the prior ET date.
  if (value instanceof Date) return dateEtIso(value);
  const text = String(value);
  return /^\d{4}-\d{2}-\d{2}$/.test(text) ? text : dateEtIso(text);
}

function positive(value: number | null): number | null {
  return value === null ? null : Math.abs(value);
}

function returnPct(
  pnl: number | null,
  entry: number | null,
  initialQty: number | null,
): number | null {
  if (pnl === null || entry === null || initialQty === null || initialQty <= 0) {
    return null;
  }
  const cost = Math.abs(entry) * Math.abs(initialQty) * 100;
  return cost > 0 ? pnl / cost : null;
}

function fillFromRaw(
  raw: JsonRecord | null | undefined,
  kind: PositionFill["kind"],
  fallbackTimestamp: string | null = null,
): PositionFill | null {
  const value = raw ?? {};
  const timestamp = firstJsonString(value.timestamp, value.event_ts) ?? fallbackTimestamp;
  const qty = numberOrNull(value.qty ?? value.filled_qty);
  const price = numberOrNull(value.price ?? value.filled_avg_price);
  const pnl = numberOrNull(value.pnl ?? value.realized_pnl);
  if (!timestamp && qty === null && price === null && pnl === null) return null;
  return {
    kind,
    stage: firstJsonString(value.stage),
    timestamp,
    qty,
    price,
    pnl,
    reason: firstJsonString(value.reason),
  };
}

function dedupeFills(fills: PositionFill[]): PositionFill[] {
  const seen = new Set<string>();
  return fills.filter((fill) => {
    const key = [
      fill.kind,
      fill.stage ?? "",
      fill.timestamp ?? "",
      fill.qty ?? "",
      fill.price ?? "",
    ].join("|");
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

function stateFromDetails(details: JsonRecord | null): JsonRecord {
  const root = details ?? {};
  return asJsonRecord(root.tiered_exit_state ?? root.exit_state) ?? {};
}

function partialFillsFromDetails(details: JsonRecord | null): PositionFill[] {
  const root = details ?? {};
  return jsonRecords(root.tiered_partial_fills)
    .map((fill) => fillFromRaw(fill, "partial_exit"))
    .filter((fill): fill is PositionFill => Boolean(fill));
}

function addEntryFill(
  fills: PositionFill[],
  openedAt: string | null,
  qty: number | null,
  price: number | null,
): PositionFill[] {
  const entry = fillFromRaw(
    { timestamp: openedAt, qty, price },
    "entry",
  );
  return entry ? [entry, ...fills] : fills;
}

function normalizeOpenPosition(position: LiveOpenPosition): OverviewPosition {
  const symbol = position.option_symbol ?? position.symbol;
  const state = position.tiered_exit_state ?? {};
  const initialQty =
    numberOrNull(position.initial_qty) ?? numberOrNull(state.initial_qty) ?? positive(position.qty);
  const remainingQty =
    numberOrNull(position.remaining_qty) ??
    numberOrNull(state.remaining_qty) ??
    positive(position.qty);
  const entryPrice = numberOrNull(position.entry_price) ?? numberOrNull(state.entry_price);
  const realizedPnl =
    numberOrNull(position.realized_pnl) ?? numberOrNull(state.realized_pnl) ?? 0;
  const unrealizedPnl = numberOrNull(position.unrealized_pl) ?? 0;
  const totalPnl = realizedPnl + unrealizedPnl;
  const fills = position.fills ??
    jsonRecords(state.exit_fills)
      .map((fill) => fillFromRaw(fill, "partial_exit"))
      .filter((fill): fill is PositionFill => Boolean(fill));

  return {
    id:
      position.position_id ??
      firstJsonString(state.lifecycle_id) ??
      symbol,
    entry_order_id: position.entry_order_id ?? null,
    status: "open",
    symbol,
    option_symbol: symbol,
    underlying_symbol: position.underlying_symbol ?? parseOccSymbol(symbol)?.underlying ?? null,
    side: position.side ?? null,
    opened_at: position.opened_at ?? null,
    closed_at: null,
    initial_qty: initialQty,
    remaining_qty: remainingQty,
    entry_price: entryPrice,
    current_price: numberOrNull(position.current_price),
    exit_price: null,
    realized_pnl: realizedPnl,
    unrealized_pnl: unrealizedPnl,
    total_pnl: totalPnl,
    return_pct: returnPct(totalPnl, entryPrice, initialQty),
    exit_reason: null,
    setup_type: position.setup_type ?? null,
    dte: numberOrNull(position.dte) ?? parseOccSymbol(symbol)?.dte ?? null,
    fills: addEntryFill(
      dedupeFills(fills),
      position.opened_at ?? null,
      initialQty,
      entryPrice,
    ),
  };
}

function normalizeLifecycle(row: TradeLifecycleRow): OverviewPosition {
  const details = row.details ?? {};
  const state = stateFromDetails(row.details);
  const partialFills = partialFillsFromDetails(row.details);
  const initialQty =
    numberOrNull(details.initial_qty) ??
    numberOrNull(state.initial_qty) ??
    positive(row.filled_qty);
  const partialQty = partialFills.reduce((sum, fill) => sum + (fill.qty ?? 0), 0);
  const totalPnl = numberOrNull(row.realized_pnl);
  const partialPnl = partialFills.reduce((sum, fill) => sum + (fill.pnl ?? 0), 0);
  const finalQty =
    numberOrNull(details.final_exit_qty) ??
    (row.filled_qty !== null && partialQty > 0
      ? Math.max(0, row.filled_qty - partialQty)
      : row.filled_qty);
  const finalPnl =
    totalPnl !== null && partialPnl !== 0 ? totalPnl - partialPnl : totalPnl;
  const finalPrice =
    numberOrNull(details.final_exit_price) ?? numberOrNull(row.exit_fill_price);
  const finalFill = fillFromRaw(
    {
      timestamp: row.closed_at,
      qty: finalQty,
      price: finalPrice,
      pnl: finalPnl,
      reason: row.exit_reason,
      stage: "final",
    },
    "final_exit",
  );
  const fills = addEntryFill(
    dedupeFills([
      ...partialFills,
      ...(finalFill ? [finalFill] : []),
    ]),
    row.opened_at,
    initialQty,
    numberOrNull(row.entry_fill_price),
  );

  return {
    id: row.lifecycle_uid,
    entry_order_id: row.entry_order_id,
    status: "closed",
    symbol: row.symbol,
    option_symbol: row.symbol,
    underlying_symbol: row.underlying_symbol ?? parseOccSymbol(row.symbol)?.underlying ?? null,
    side: "long",
    opened_at: row.opened_at,
    closed_at: row.closed_at,
    initial_qty: initialQty,
    remaining_qty: 0,
    entry_price: numberOrNull(row.entry_fill_price),
    current_price: null,
    exit_price: finalPrice,
    realized_pnl: totalPnl,
    unrealized_pnl: 0,
    total_pnl: totalPnl,
    return_pct: returnPct(totalPnl, numberOrNull(row.entry_fill_price), initialQty),
    exit_reason: row.exit_reason,
    setup_type: row.setup_type ?? null,
    dte: row.option_dte ?? parseOccSymbol(row.symbol)?.dte ?? null,
    fills,
  };
}

function normalizeLiveClosed(
  closed: NonNullable<LiveStateRow["session_stats"]["closed"]>[number],
): OverviewPosition {
  const symbol = closed.option_symbol ?? closed.symbol;
  const initialQty =
    numberOrNull(closed.initial_qty) ?? numberOrNull(closed.remaining_qty);
  const entryPrice = numberOrNull(closed.entry_price);
  const totalPnl = numberOrNull(closed.pnl);
  const finalPrice = numberOrNull(closed.exit_price);
  const finalFill = fillFromRaw(
    {
      timestamp: closed.closed_at,
      qty: initialQty,
      price: finalPrice,
      pnl: totalPnl,
      reason: closed.exit_reason,
      stage: "final",
    },
    "final_exit",
  );
  return {
    id: closed.position_id ?? symbol,
    entry_order_id: closed.entry_order_id ?? null,
    status: "closed",
    symbol,
    option_symbol: symbol,
    underlying_symbol: closed.underlying_symbol ?? parseOccSymbol(symbol)?.underlying ?? null,
    side: closed.side ?? "long",
    opened_at: closed.opened_at ?? null,
    closed_at: closed.closed_at,
    initial_qty: initialQty,
    remaining_qty: 0,
    entry_price: entryPrice,
    current_price: null,
    exit_price: finalPrice,
    realized_pnl: totalPnl,
    unrealized_pnl: 0,
    total_pnl: totalPnl,
    return_pct: returnPct(totalPnl, entryPrice, initialQty),
    exit_reason: closed.exit_reason,
    setup_type: null,
    dte: parseOccSymbol(symbol)?.dte ?? null,
    fills: addEntryFill(
      dedupeFills([
        ...(closed.fills ?? []),
        ...(finalFill ? [finalFill] : []),
      ]),
      closed.opened_at ?? null,
      initialQty,
      entryPrice,
    ),
  };
}

function identityNumber(value: number | null): string {
  return value === null ? "" : value.toFixed(6);
}

function timeBucket(value: string | null): string {
  if (!value) return "";
  const timestamp = Date.parse(value);
  return Number.isFinite(timestamp)
    ? String(Math.floor(timestamp / 60_000))
    : value.trim();
}

/**
 * A live-state close and its EOD lifecycle row do not always share an ID:
 * live tiered state historically used trade_pk while lifecycles use the entry
 * order ID. Keep several conservative aliases so those records collapse
 * without merging separate entries in the same option contract.
 */
function positionAliases(position: OverviewPosition): string[] {
  const aliases: string[] = [];
  const symbol = (position.option_symbol || position.symbol).toUpperCase();
  const id = position.id.trim().toUpperCase();
  const entryOrderId = position.entry_order_id?.trim().toUpperCase();
  const opened = timeBucket(position.opened_at);
  const closed = timeBucket(position.closed_at);

  if (id && id !== symbol) aliases.push(`id:${id}`);
  if (entryOrderId) aliases.push(`entry:${entryOrderId}`);
  if (symbol && opened) aliases.push(`opened:${symbol}:${opened}`);
  if (symbol && closed) aliases.push(`closed:${symbol}:${closed}`);
  if (symbol && opened && position.initial_qty !== null && position.entry_price !== null) {
    aliases.push(
      `trade:${symbol}:${opened}:${identityNumber(position.initial_qty)}:${identityNumber(position.entry_price)}`,
    );
  }
  if (symbol && closed && position.initial_qty !== null && position.exit_price !== null) {
    aliases.push(
      `exit:${symbol}:${closed}:${identityNumber(position.initial_qty)}:${identityNumber(position.exit_price)}`,
    );
  }
  return aliases;
}

type PositionSource = "live-open" | "live-closed" | "lifecycle";

function sourcePriority(source: PositionSource): number {
  if (source === "live-closed") return 3;
  if (source === "lifecycle") return 2;
  return 1;
}

/** Merge live open/session data with durable lifecycles closed on one ET date. */
export function getTodayOverviewPositions(
  liveState: LiveStateRow | null,
  lifecycles: TradeLifecycleRow[],
  sessionDate: string | Date = new Date(),
): OverviewPosition[] {
  const targetDate = normalizeSessionDate(sessionDate);
  const merged = new Map<string, { position: OverviewPosition; priority: number }>();
  const aliasToKey = new Map<string, string>();
  let generatedKey = 0;

  const add = (position: OverviewPosition, source: PositionSource) => {
    const aliases = positionAliases(position);
    const matchingKeys = new Set(
      aliases
        .map((alias) => aliasToKey.get(alias))
        .filter((key): key is string => typeof key === "string" && merged.has(key)),
    );
    const key = matchingKeys.values().next().value ?? `position:${generatedKey++}`;
    const candidates = [
      ...[...matchingKeys]
        .map((matchingKey) => merged.get(matchingKey))
        .filter(
          (candidate): candidate is { position: OverviewPosition; priority: number } =>
            Boolean(candidate),
        ),
      { position, priority: sourcePriority(source) },
    ];
    const winner = candidates.reduce((current, candidate) =>
      candidate.priority > current.priority ? candidate : current,
    );

    for (const matchingKey of matchingKeys) merged.delete(matchingKey);
    merged.set(key, winner);

    for (const candidate of candidates) {
      for (const alias of positionAliases(candidate.position)) aliasToKey.set(alias, key);
    }
  };

  for (const position of liveState?.open_positions ?? []) {
    const normalized = normalizeOpenPosition(position);
    add(normalized, "live-open");
  }
  for (const closed of liveState?.session_stats?.closed ?? []) {
    if (dateEtIso(closed.closed_at) !== targetDate) continue;
    const normalized = normalizeLiveClosed(closed);
    add(normalized, "live-closed");
  }
  for (const lifecycle of lifecycles) {
    if (dateEtIso(lifecycle.closed_at) !== targetDate) continue;
    const normalized = normalizeLifecycle(lifecycle);
    add(normalized, "lifecycle");
  }

  return [...merged.values()].map(({ position }) => position).sort((a, b) => {
    if (a.status !== b.status) return a.status === "open" ? -1 : 1;
    const aTime = Date.parse(a.status === "open" ? a.opened_at ?? "" : a.closed_at ?? "");
    const bTime = Date.parse(b.status === "open" ? b.opened_at ?? "" : b.closed_at ?? "");
    return (Number.isFinite(bTime) ? bTime : 0) - (Number.isFinite(aTime) ? aTime : 0);
  });
}

export type OverviewSessionMetrics = {
  openUnrealizedPnl: number;
  realizedPnl: number;
  wins: number;
  losses: number;
  closedCount: number;
};

function openPositionRealizedPnlForDate(
  position: OverviewPosition,
  targetDate: string,
): number {
  const partialFills = position.fills.filter((fill) => fill.kind === "partial_exit");
  if (partialFills.length === 0) {
    // Legacy live-state rows did not always persist exit fills. Only use the
    // aggregate realized value when the position itself opened in this
    // session; an overnight position may carry prior-session partial P&L.
    return dateEtIso(position.opened_at) === targetDate
      ? numberOrNull(position.realized_pnl) ?? 0
      : 0;
  }

  const datedFills = partialFills.filter(
    (fill) => dateEtIso(fill.timestamp) === targetDate,
  );
  if (datedFills.length > 0) {
    return datedFills.reduce((sum, fill) => sum + (fill.pnl ?? 0), 0);
  }

  // If timestamps are absent in a legacy row, use its aggregate only for a
  // same-session entry; this keeps prior-session partial exits out of today.
  const hasTimestamp = partialFills.some((fill) => Boolean(fill.timestamp));
  return !hasTimestamp && dateEtIso(position.opened_at) === targetDate
    ? numberOrNull(position.realized_pnl) ?? 0
    : 0;
}

/**
 * Calculate the rail's session totals from the exact positions it renders.
 * This is intentionally independent of process-local live_state counters,
 * which are reset when a multi-segment workflow hands off to a new process.
 */
export function getOverviewSessionMetrics(
  positions: OverviewPosition[],
  sessionDate: string | Date = new Date(),
): OverviewSessionMetrics {
  const targetDate = normalizeSessionDate(sessionDate);
  const metrics: OverviewSessionMetrics = {
    openUnrealizedPnl: 0,
    realizedPnl: 0,
    wins: 0,
    losses: 0,
    closedCount: 0,
  };

  for (const position of positions) {
    if (position.status === "open") {
      metrics.openUnrealizedPnl += numberOrNull(position.unrealized_pnl) ?? 0;
      metrics.realizedPnl += openPositionRealizedPnlForDate(position, targetDate);
      continue;
    }

    const pnl = numberOrNull(position.total_pnl) ?? numberOrNull(position.realized_pnl) ?? 0;
    metrics.realizedPnl += pnl;
    metrics.closedCount += 1;
    if (pnl > 0) metrics.wins += 1;
    if (pnl < 0) metrics.losses += 1;
  }

  return metrics;
}

export function formatPositionStatus(position: OverviewPosition): string {
  if (position.status === "open") {
    return `${position.remaining_qty ?? "—"} left`;
  }
  return position.exit_reason?.replaceAll("_", " ") ?? "closed";
}
