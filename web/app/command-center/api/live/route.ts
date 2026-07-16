import { cookies } from "next/headers";
import { NextResponse } from "next/server";
import {
  COMMAND_CENTER_COOKIE,
  commandCenterToken,
} from "@/lib/commandCenterAuth";
import { getLiveState } from "@/lib/data";
import { normalizePlpc, parseOccSymbol } from "@/lib/format";
import type {
  LiveBlotterPayload,
  LiveExitPolicy,
  LiveOpenPosition,
  LiveOrder,
} from "@/lib/types";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const PAPER_BASE = "https://paper-api.alpaca.markets";

function envBool(value: string | undefined): boolean {
  return (value ?? "").trim().toLowerCase() === "true";
}

function num(value: unknown): number | null {
  if (value === null || value === undefined || value === "") return null;
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
}

function defaultExitPolicy(liveStatePolicy?: LiveExitPolicy | null): LiveExitPolicy {
  if (liveStatePolicy?.stop_loss_pct != null && liveStatePolicy?.profit_target_pct != null) {
    return {
      stop_loss_pct: Number(liveStatePolicy.stop_loss_pct),
      profit_target_pct: Number(liveStatePolicy.profit_target_pct),
      max_hold_minutes: liveStatePolicy.max_hold_minutes,
      allow_overnight: liveStatePolicy.allow_overnight,
    };
  }
  const stop = Number(process.env.OPTION_STOP_LOSS_PCT ?? "0.35");
  const tp = Number(process.env.OPTION_PROFIT_TARGET_PCT ?? "0.25");
  return {
    stop_loss_pct: Number.isFinite(stop) ? stop : 0.35,
    profit_target_pct: Number.isFinite(tp) ? tp : 0.25,
  };
}

function enrichPosition(
  raw: Record<string, unknown>,
  policy: LiveExitPolicy,
): LiveOpenPosition {
  const symbol = String(raw.symbol ?? "");
  const assetClass = String(raw.asset_class ?? "");
  const isOption = assetClass.toLowerCase().includes("option");
  const qty = Math.abs(num(raw.qty) ?? 0);
  const multiplier = isOption ? 100 : 1;
  const costBasis = Math.abs(num(raw.cost_basis) ?? 0);
  const marketValue = Math.abs(num(raw.market_value) ?? 0);
  const entryFromBasis = qty > 0 ? costBasis / (qty * multiplier) : null;
  const markFromMv = qty > 0 ? marketValue / (qty * multiplier) : null;
  const entry =
    entryFromBasis ??
    num(raw.avg_entry_price) ??
    num(raw.current_price);
  const mark = markFromMv ?? num(raw.current_price) ?? entry;
  const plpc = normalizePlpc(num(raw.unrealized_plpc) ?? 0);
  const stopPct = policy.stop_loss_pct;
  const tpPct = policy.profit_target_pct;
  const stopMark = entry != null ? entry * (1 - stopPct) : null;
  const tpMark = entry != null ? entry * (1 + tpPct) : null;
  const parsed = parseOccSymbol(symbol);
  return {
    symbol,
    option_symbol: isOption ? symbol : null,
    underlying_symbol: parsed?.underlying ?? null,
    asset_class: isOption ? "option" : assetClass || null,
    qty: num(raw.qty) ?? 0,
    side: String(raw.side ?? ""),
    entry_price: entry,
    current_price: mark,
    unrealized_pl: num(raw.unrealized_pl) ?? 0,
    unrealized_plpc: plpc,
    dte: parsed?.dte ?? null,
    stop_mark: stopMark,
    tp_mark: tpMark,
    pct_to_stop: plpc + stopPct,
    pct_to_tp: tpPct - plpc,
  };
}

function mapOrder(raw: Record<string, unknown>): LiveOrder {
  return {
    id: raw.id != null ? String(raw.id) : undefined,
    symbol: String(raw.symbol ?? ""),
    side: String(raw.side ?? ""),
    type: String(raw.type ?? ""),
    qty: num(raw.qty),
    filled_qty: num(raw.filled_qty),
    limit_price: num(raw.limit_price),
    stop_price: num(raw.stop_price),
    filled_avg_price: num(raw.filled_avg_price),
    status: String(raw.status ?? ""),
    submitted_at: raw.submitted_at != null ? String(raw.submitted_at) : null,
    filled_at: raw.filled_at != null ? String(raw.filled_at) : null,
  };
}

function todayAfterEtIso(): string {
  // 09:00 ET on the current ET calendar day → ISO for Alpaca `after` filter.
  const fmt = new Intl.DateTimeFormat("en-US", {
    timeZone: "America/New_York",
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
  });
  const parts = fmt.formatToParts(new Date());
  const year = parts.find((p) => p.type === "year")?.value ?? "1970";
  const month = parts.find((p) => p.type === "month")?.value ?? "01";
  const day = parts.find((p) => p.type === "day")?.value ?? "01";
  // Construct as ET wall clock; Alpaca accepts RFC3339. Use -04:00/-05:00 via
  // a UTC instant approx: noon UTC on that date is safely "after 09:00 ET prior day"
  // — better: encode as America/New_York offset via Date.
  const rough = new Date(`${year}-${month}-${day}T09:00:00`);
  // Treat as ET by formatting back; for filter we just need an early-enough ISO.
  // Use Explicit offset from a known ET formatter trick:
  const probe = new Date(
    new Date().toLocaleString("en-US", { timeZone: "America/New_York" }),
  );
  void probe;
  return new Date(`${year}-${month}-${day}T13:00:00.000Z`).toISOString(); // 09:00 EDT
}

async function alpacaGet(
  path: string,
  key: string,
  secret: string,
): Promise<unknown> {
  const res = await fetch(`${PAPER_BASE}${path}`, {
    headers: {
      "APCA-API-KEY-ID": key,
      "APCA-API-SECRET-KEY": secret,
    },
    cache: "no-store",
  });
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new Error(`Alpaca ${path} → ${res.status} ${body.slice(0, 200)}`);
  }
  return res.json();
}

export async function GET() {
  if (process.env.COMMAND_CENTER_ENABLED !== "true") {
    return NextResponse.json({ error: "not found" }, { status: 404 });
  }

  const password = process.env.COMMAND_CENTER_PASSWORD;
  if (!password) {
    return NextResponse.json({ error: "unauthorized" }, { status: 401 });
  }
  const jar = await cookies();
  const cookie = jar.get(COMMAND_CENTER_COOKIE)?.value;
  if (!cookie || cookie !== commandCenterToken(password)) {
    return NextResponse.json({ error: "unauthorized" }, { status: 401 });
  }

  if (!envBool(process.env.ALPACA_PAPER_TRADING)) {
    return NextResponse.json(
      {
        error:
          "ALPACA_PAPER_TRADING must be true — live blotter refuses non-paper keys",
      },
      { status: 500 },
    );
  }

  const key = (process.env.ALPACA_API_KEY ?? "").trim();
  const secret = (process.env.ALPACA_SECRET_KEY ?? "").trim();
  if (!key || !secret) {
    return NextResponse.json(
      { error: "ALPACA_API_KEY / ALPACA_SECRET_KEY not configured" },
      { status: 500 },
    );
  }

  const liveState = await getLiveState("paper");
  const exitPolicy = defaultExitPolicy(liveState?.exit_policy ?? null);

  try {
    const after = todayAfterEtIso();
    const [accountRaw, positionsRaw, openOrdersRaw, closedOrdersRaw] =
      await Promise.all([
        alpacaGet("/v2/account", key, secret),
        alpacaGet("/v2/positions", key, secret),
        alpacaGet("/v2/orders?status=open&nested=true&limit=100", key, secret),
        alpacaGet(
          `/v2/orders?status=closed&after=${encodeURIComponent(after)}&limit=200&direction=desc`,
          key,
          secret,
        ),
      ]);

    const accountObj = accountRaw as Record<string, unknown>;
    const equity = num(accountObj.equity);
    const lastEquity = num(accountObj.last_equity);
    const dailyPnl =
      equity != null && lastEquity != null ? equity - lastEquity : null;
    const dailyPnlPct =
      dailyPnl != null && lastEquity ? dailyPnl / lastEquity : null;

    const positions = (Array.isArray(positionsRaw) ? positionsRaw : []).map(
      (p) => enrichPosition(p as Record<string, unknown>, exitPolicy),
    );
    const openOrders = (Array.isArray(openOrdersRaw) ? openOrdersRaw : []).map(
      (o) => mapOrder(o as Record<string, unknown>),
    );
    const todaysOrders = (Array.isArray(closedOrdersRaw) ? closedOrdersRaw : [])
      .map((o) => mapOrder(o as Record<string, unknown>))
      .filter((o) => o.status === "filled" || (o.filled_qty ?? 0) > 0);

    const payload: LiveBlotterPayload = {
      account: {
        equity,
        last_equity: lastEquity,
        buying_power: num(accountObj.buying_power),
        daily_pnl: dailyPnl,
        daily_pnl_pct: dailyPnlPct,
      },
      positions,
      openOrders,
      todaysOrders,
      fetchedAt: new Date().toISOString(),
      exitPolicy,
      liveState,
    };
    return NextResponse.json(payload, {
      headers: { "Cache-Control": "no-store" },
    });
  } catch (err) {
    const message = err instanceof Error ? err.message : "alpaca fetch failed";
    return NextResponse.json(
      {
        account: {
          equity: null,
          last_equity: null,
          buying_power: null,
          daily_pnl: null,
          daily_pnl_pct: null,
        },
        positions: [],
        openOrders: [],
        todaysOrders: [],
        fetchedAt: new Date().toISOString(),
        exitPolicy,
        liveState,
        error: message,
      } satisfies LiveBlotterPayload,
      { status: 500 },
    );
  }
}
