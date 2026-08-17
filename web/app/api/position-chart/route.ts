import { NextResponse } from "next/server";
import { randomUUID } from "node:crypto";
import { getLiveState } from "@/lib/data";
import { asJsonRecord, firstJsonString, jsonNumber, jsonRecords } from "@/lib/json";
import { supabaseSelect } from "@/lib/supabase";
import type { JsonValue, LiveOpenPosition, TradeLifecycleRow } from "@/lib/types";

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

type ChartWindow = {
  start: Date;
  end: Date;
};

type ChartErrorCode =
  | "CONFIGURATION_ERROR"
  | "UPSTREAM_AUTH"
  | "UPSTREAM_RATE_LIMIT"
  | "UPSTREAM_UNAVAILABLE"
  | "INVALID_UPSTREAM_RESPONSE"
  | "NO_DATA";

function errorResponse(
  code: ChartErrorCode,
  message: string,
  status: number,
  requestId: string,
  retryable: boolean,
) {
  return NextResponse.json(
    {
      bars: [],
      error: message,
      error_code: code,
      request_id: requestId,
      retryable,
    },
    {
      status,
      headers: { "Cache-Control": "no-store" },
    },
  );
}

function logUpstreamFailure(input: {
  requestId: string;
  positionId: string;
  symbol: string;
  status?: number;
  body?: string;
  error?: unknown;
  elapsedMs: number;
}) {
  const body = input.body
    ? input.body.replace(/[\r\n]+/g, " ").slice(0, 400)
    : undefined;
  console.error(
    JSON.stringify({
      event: "position_chart_upstream_failure",
      request_id: input.requestId,
      position_id: input.positionId,
      symbol: input.symbol,
      upstream_status: input.status ?? null,
      upstream_body_prefix: body ?? null,
      error: input.error instanceof Error ? input.error.message : input.error ?? null,
      elapsed_ms: input.elapsedMs,
    }),
  );
}

function positionId(position: LiveOpenPosition): string {
  const state = position.tiered_exit_state ?? {};
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

function windowFor(position: ResolvedPosition): ChartWindow {
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

function normalizedBars(payload: JsonValue, symbol: string) {
  const root = asJsonRecord(payload) ?? {};
  const bars = root.bars;
  const raw = Array.isArray(bars)
    ? jsonRecords(bars)
    : jsonRecords(asJsonRecord(bars)?.[symbol]);
  return raw
    .map((bar) => {
      const timestamp = firstJsonString(bar.t);
      const timestampMs = timestamp ? Date.parse(timestamp) : NaN;
      const open = jsonNumber(bar.o);
      const high = jsonNumber(bar.h);
      const low = jsonNumber(bar.l);
      const close = jsonNumber(bar.c);
      if (!timestamp || !Number.isFinite(timestampMs) || !Number.isFinite(close)) return null;
      return {
        timestamp,
        timestampMs,
        open: open ?? close,
        high: high ?? close,
        low: low ?? close,
        close,
        volume: jsonNumber(bar.v),
      };
    })
    .filter((bar): bar is NonNullable<typeof bar> => Boolean(bar))
    .sort((a, b) => a.timestampMs - b.timestampMs);
}

export async function GET(request: Request) {
  const requestId = randomUUID();
  const startedAt = Date.now();
  const url = new URL(request.url);
  const requestedId = url.searchParams.get("position_id")?.trim() ?? "";
  if (!requestedId || requestedId.length > 200) {
    return errorResponse("INVALID_UPSTREAM_RESPONSE", "Position not found.", 404, requestId, false);
  }

  const position = await resolvePosition(requestedId);
  if (!position || !OCC_RE.test(position.symbol)) {
    return errorResponse("INVALID_UPSTREAM_RESPONSE", "Position not found.", 404, requestId, false);
  }

  const key = (process.env.ALPACA_API_KEY ?? "").trim();
  const secret = (process.env.ALPACA_SECRET_KEY ?? "").trim();
  if (!key || !secret) {
    return errorResponse(
      "CONFIGURATION_ERROR",
      "Option history is not configured on this deployment.",
      503,
      requestId,
      false,
    );
  }

  const { start, end } = windowFor(position);
  // The options bars endpoint rejects a `feed` query parameter. Keep the
  // response label honest instead of claiming a feed that was never sent.
  const source = "alpaca-options-bars";
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
        signal: AbortSignal.timeout(10_000),
        next: { revalidate: position.status === "open" ? 30 : 86_400 },
      });
      if (!response.ok) {
        const body = await response.text();
        const status = response.status;
        logUpstreamFailure({
          requestId,
          positionId: requestedId,
          symbol: position.symbol,
          status,
          body,
          elapsedMs: Date.now() - startedAt,
        });
        if (status === 401 || status === 403) {
          return errorResponse("UPSTREAM_AUTH", "The option history service rejected its deployment credentials.", 502, requestId, false);
        }
        if (status === 429) {
          return errorResponse("UPSTREAM_RATE_LIMIT", "Option history is rate limited. Try again shortly.", 503, requestId, true);
        }
        return errorResponse("UPSTREAM_UNAVAILABLE", "The option history service is temporarily unavailable.", 502, requestId, true);
      }
      let payload: JsonValue;
      try {
        payload = await response.json();
      } catch (error) {
        logUpstreamFailure({
          requestId,
          positionId: requestedId,
          symbol: position.symbol,
          error,
          elapsedMs: Date.now() - startedAt,
        });
        return errorResponse("INVALID_UPSTREAM_RESPONSE", "The option history service returned invalid data.", 502, requestId, true);
      }
      allBars.push(...normalizedBars(payload, position.symbol));
      pageToken = firstJsonString(asJsonRecord(payload)?.next_page_token);
      if (!pageToken) break;
    }
  } catch (error) {
    logUpstreamFailure({
      requestId,
      positionId: requestedId,
      symbol: position.symbol,
      error,
      elapsedMs: Date.now() - startedAt,
    });
    return errorResponse("UPSTREAM_UNAVAILABLE", "The option history service is temporarily unavailable.", 502, requestId, true);
  }

  const uniqueBars = [...new Map(allBars.map((bar) => [bar.timestamp, bar])).values()];
  if (!uniqueBars.length) {
    return NextResponse.json(
      {
        symbol: position.symbol,
        source,
        bars: [],
        error: "No option bars were available for this holding window.",
        error_code: "NO_DATA" as const,
        request_id: requestId,
        retryable: false,
      },
      { headers: { "Cache-Control": "no-store" } },
    );
  }
  return NextResponse.json(
    { symbol: position.symbol, source, bars: uniqueBars, request_id: requestId },
    {
      headers: {
        "Cache-Control": position.status === "open"
          ? "public, s-maxage=30, stale-while-revalidate=60"
          : "public, s-maxage=86400, stale-while-revalidate=86400",
      },
    },
  );
}
