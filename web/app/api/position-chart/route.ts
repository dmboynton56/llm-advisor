import { NextResponse } from "next/server";
import { getLiveState } from "@/lib/data";
import { supabaseSelect } from "@/lib/supabase";
import type { JsonRecord, LiveOpenPosition, TradeLifecycleRow } from "@/lib/types";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const DATA_BASE = "https://data.alpaca.markets/v1beta1/options/bars";
const OCC_RE = /^[A-Z0-9.]{1,10}\d{6}[CP]\d{8}$/;

type ResolvedPosition = {
  id: string;
  symbol: string;
  openedAt: string | null;
  closedAt: string | null;
  status: "open" | "closed";
};

type RawBar = {
  t?: string;
  o?: number;
  h?: number;
  l?: number;
  c?: number;
  v?: number;
};

function asRecord(value: unknown): JsonRecord {
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as JsonRecord)
    : {};
}

function positionId(position: LiveOpenPosition): string {
  const state = asRecord(
    (position as LiveOpenPosition & { tiered_exit_state?: JsonRecord })
      .tiered_exit_state,
  );
  return String(position.position_id ?? state.lifecycle_id ?? position.option_symbol ?? position.symbol);
}

function resolveLivePosition(
  positionIdValue: string,
  state: Awaited<ReturnType<typeof getLiveState>>,
): ResolvedPosition | null {
  for (const position of state?.open_positions ?? []) {
    if (positionId(position) !== positionIdValue) continue;
    return {
      id: positionIdValue,
      symbol: String(position.option_symbol ?? position.symbol).toUpperCase(),
      openedAt: position.opened_at ?? null,
      closedAt: null,
      status: "open",
    };
  }
  for (const closed of state?.session_stats?.closed ?? []) {
    const id = String(closed.position_id ?? closed.option_symbol ?? closed.symbol);
    if (id !== positionIdValue) continue;
    return {
      id,
      symbol: String(closed.option_symbol ?? closed.symbol).toUpperCase(),
      openedAt: closed.opened_at ?? null,
      closedAt: closed.closed_at ?? null,
      status: "closed",
    };
  }
  return null;
}

async function resolvePosition(positionIdValue: string): Promise<ResolvedPosition | null> {
  const live = resolveLivePosition(positionIdValue, await getLiveState("paper"));
  if (live) return live;

  const rows = await supabaseSelect<TradeLifecycleRow>(
    "llm_advisor_trade_lifecycles",
    `select=lifecycle_uid,symbol,opened_at,closed_at&lifecycle_uid=eq.${encodeURIComponent(positionIdValue)}&limit=1`,
  );
  const row = rows?.[0];
  if (!row) return null;
  return {
    id: row.lifecycle_uid,
    symbol: row.symbol.toUpperCase(),
    openedAt: row.opened_at,
    closedAt: row.closed_at,
    status: row.closed_at ? "closed" : "open",
  };
}

function windowFor(position: ResolvedPosition): { start: Date; end: Date } {
  const now = new Date();
  const end = position.closedAt
    ? new Date(new Date(position.closedAt).getTime() + 15 * 60_000)
    : now;
  const opened = position.openedAt ? new Date(position.openedAt) : null;
  const fallbackStart = new Date(end.getTime() - 6 * 60 * 60_000);
  const start = opened
    ? new Date(opened.getTime() - 15 * 60_000)
    : fallbackStart;
  const maxLookback = new Date(end.getTime() - 14 * 24 * 60 * 60_000);
  return { start: start < maxLookback ? maxLookback : start, end };
}

function normalizedBars(payload: unknown, symbol: string) {
  const root = asRecord(payload);
  const bars = root.bars;
  const raw = Array.isArray(bars)
    ? bars
    : Array.isArray(asRecord(bars)[symbol])
      ? (asRecord(bars)[symbol] as unknown[])
      : [];
  return raw
    .map((value) => {
      const bar = asRecord(value) as RawBar;
      const timestamp = typeof bar.t === "string" ? bar.t : null;
      const timestampMs = timestamp ? Date.parse(timestamp) : NaN;
      const open = Number(bar.o);
      const high = Number(bar.h);
      const low = Number(bar.l);
      const close = Number(bar.c);
      if (!timestamp || !Number.isFinite(timestampMs) || !Number.isFinite(close)) return null;
      return {
        timestamp,
        timestampMs,
        open: Number.isFinite(open) ? open : close,
        high: Number.isFinite(high) ? high : close,
        low: Number.isFinite(low) ? low : close,
        close,
        volume: Number.isFinite(Number(bar.v)) ? Number(bar.v) : null,
      };
    })
    .filter((bar): bar is NonNullable<typeof bar> => Boolean(bar))
    .sort((a, b) => a.timestampMs - b.timestampMs);
}

export async function GET(request: Request) {
  const url = new URL(request.url);
  const requestedId = url.searchParams.get("position_id")?.trim() ?? "";
  if (!requestedId || requestedId.length > 200) {
    return NextResponse.json({ error: "position not found" }, { status: 404 });
  }

  const position = await resolvePosition(requestedId);
  if (!position || !OCC_RE.test(position.symbol)) {
    return NextResponse.json({ error: "position not found" }, { status: 404 });
  }

  const key = (process.env.ALPACA_API_KEY ?? "").trim();
  const secret = (process.env.ALPACA_SECRET_KEY ?? "").trim();
  if (!key || !secret) {
    return NextResponse.json(
      { bars: [], error: "Option history is not configured on this deployment." },
      { status: 503 },
    );
  }

  const { start, end } = windowFor(position);
  const source = (process.env.ALPACA_OPTIONS_FEED ?? "indicative").toLowerCase() === "opra"
    ? "opra"
    : "indicative";
  const allBars: ReturnType<typeof normalizedBars> = [];
  let pageToken: string | null = null;

  try {
    for (let page = 0; page < 3; page += 1) {
      const params = new URLSearchParams({
        symbols: position.symbol,
        timeframe: "1Min",
        start: start.toISOString(),
        end: end.toISOString(),
        limit: "10000",
        sort: "asc",
      });
      if (pageToken) params.set("page_token", pageToken);
      const response = await fetch(`${DATA_BASE}?${params.toString()}`, {
        headers: {
          "APCA-API-KEY-ID": key,
          "APCA-API-SECRET-KEY": secret,
        },
        next: { revalidate: position.status === "open" ? 30 : 86_400 },
      });
      if (!response.ok) {
        return NextResponse.json(
          { bars: [], error: "Option history is temporarily unavailable." },
          { status: 502 },
        );
      }
      const payload = await response.json();
      allBars.push(...normalizedBars(payload, position.symbol));
      const next = asRecord(payload).next_page_token;
      pageToken = typeof next === "string" && next ? next : null;
      if (!pageToken) break;
    }
  } catch {
    return NextResponse.json(
      { bars: [], error: "Option history is temporarily unavailable." },
      { status: 502 },
    );
  }

  const uniqueBars = [...new Map(allBars.map((bar) => [bar.timestamp, bar])).values()];
  return NextResponse.json(
    { symbol: position.symbol, source, bars: uniqueBars },
    {
      headers: {
        "Cache-Control": position.status === "open"
          ? "public, s-maxage=30, stale-while-revalidate=60"
          : "public, s-maxage=86400, stale-while-revalidate=86400",
      },
    },
  );
}
