import { getLiveState } from "@/lib/data";
import { asJsonRecord, jsonNumber, jsonRecords } from "@/lib/json";
import { normalizePlpc, parseOccSymbol } from "@/lib/format";
import type {
  JsonValue,
  JsonRecord,
  LiveBlotterPayload,
  LiveExitPolicy,
  LiveOpenPosition,
  LiveOrder,
} from "@/lib/types";

const PAPER_BASE = "https://paper-api.alpaca.markets";

function envBool(value: string | undefined): boolean {
  return (value ?? "").trim().toLowerCase() === "true";
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
  raw: JsonRecord,
  policy: LiveExitPolicy,
): LiveOpenPosition {
  const symbol = String(raw.symbol ?? "");
  const assetClass = String(raw.asset_class ?? "");
  const isOption = assetClass.toLowerCase().includes("option");
  const qty = Math.abs(jsonNumber(raw.qty) ?? 0);
  const multiplier = isOption ? 100 : 1;
  const costBasis = Math.abs(jsonNumber(raw.cost_basis) ?? 0);
  const marketValue = Math.abs(jsonNumber(raw.market_value) ?? 0);
  const entryFromBasis = qty > 0 ? costBasis / (qty * multiplier) : null;
  const markFromMv = qty > 0 ? marketValue / (qty * multiplier) : null;
  const entry = entryFromBasis ?? jsonNumber(raw.avg_entry_price) ?? jsonNumber(raw.current_price);
  const mark = markFromMv ?? jsonNumber(raw.current_price) ?? entry;
  const plpc = normalizePlpc(jsonNumber(raw.unrealized_plpc) ?? 0);
  const parsed = parseOccSymbol(symbol);
  const positionId = raw.asset_id != null ? String(raw.asset_id) : null;
  return {
    symbol,
    position_id: positionId,
    option_symbol: isOption ? symbol : null,
    underlying_symbol: parsed?.underlying ?? null,
    asset_class: isOption ? "option" : assetClass || null,
    qty: jsonNumber(raw.qty) ?? 0,
    side: String(raw.side ?? ""),
    entry_price: entry,
    current_price: mark,
    unrealized_pl: jsonNumber(raw.unrealized_pl) ?? 0,
    unrealized_plpc: plpc,
    dte: parsed?.dte ?? null,
    stop_mark: entry != null ? entry * (1 - policy.stop_loss_pct) : null,
    tp_mark: entry != null ? entry * (1 + policy.profit_target_pct) : null,
    pct_to_stop: plpc + policy.stop_loss_pct,
    pct_to_tp: policy.profit_target_pct - plpc,
  };
}

function mapOrder(raw: JsonRecord): LiveOrder {
  return {
    id: raw.id != null ? String(raw.id) : undefined,
    symbol: String(raw.symbol ?? ""),
    side: String(raw.side ?? ""),
    type: String(raw.type ?? ""),
    qty: jsonNumber(raw.qty),
    filled_qty: jsonNumber(raw.filled_qty),
    limit_price: jsonNumber(raw.limit_price),
    stop_price: jsonNumber(raw.stop_price),
    filled_avg_price: jsonNumber(raw.filled_avg_price),
    status: String(raw.status ?? ""),
    submitted_at: raw.submitted_at != null ? String(raw.submitted_at) : null,
    filled_at: raw.filled_at != null ? String(raw.filled_at) : null,
  };
}

function todayAfterEtIso(): string {
  const fmt = new Intl.DateTimeFormat("en-US", {
    timeZone: "America/New_York",
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
  });
  const parts = fmt.formatToParts(new Date());
  const year = parts.find((part) => part.type === "year")?.value ?? "1970";
  const month = parts.find((part) => part.type === "month")?.value ?? "01";
  const day = parts.find((part) => part.type === "day")?.value ?? "01";
  // 13:00 UTC is before 09:00 ET in standard time and equal to 09:00 ET in
  // daylight time, so it is a safe lower bound for a same-day paper blotter.
  return new Date(year + "-" + month + "-" + day + "T13:00:00.000Z").toISOString();
}

async function alpacaGet(path: string, key: string, secret: string): Promise<JsonValue> {
  const response = await fetch(PAPER_BASE + path, {
    headers: {
      "APCA-API-KEY-ID": key,
      "APCA-API-SECRET-KEY": secret,
    },
    cache: "no-store",
  });
  if (!response.ok) {
    const body = await response.text().catch(() => "");
    throw new Error("Alpaca " + path + " → " + response.status + " " + body.slice(0, 160));
  }
  return response.json();
}

export function alpacaPaperErrorPayload(
  liveState: LiveBlotterPayload["liveState"],
  error: string,
): LiveBlotterPayload & { error: string } {
  return {
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
    exitPolicy: defaultExitPolicy(liveState?.exit_policy ?? null),
    liveState,
    error,
  };
}

export async function fetchAlpacaPaperSnapshot(
  liveStateOverride?: LiveBlotterPayload["liveState"],
): Promise<LiveBlotterPayload> {
  if (!envBool(process.env.ALPACA_PAPER_TRADING)) {
    throw new Error("Paper-only access is disabled.");
  }

  const key = process.env.ALPACA_API_KEY?.trim();
  const secret = process.env.ALPACA_SECRET_KEY?.trim();
  if (!key || !secret) {
    throw new Error("Paper account credentials are not configured on the server.");
  }

  const [accountRaw, positionsRaw, openOrdersRaw, closedOrdersRaw] =
    await Promise.all([
      alpacaGet("/v2/account", key, secret),
      alpacaGet("/v2/positions", key, secret),
      alpacaGet("/v2/orders?status=open&nested=true&limit=100", key, secret),
      alpacaGet(
        "/v2/orders?status=closed&after=" + encodeURIComponent(todayAfterEtIso()) + "&limit=200&direction=desc",
        key,
        secret,
      ),
    ]);

  let liveState = liveStateOverride ?? null;
  if (liveStateOverride === undefined) {
    try {
      liveState = await getLiveState("paper");
    } catch {
      // The paper-account view should remain useful if the optional loop-state
      // row is unavailable; Alpaca is still the source of truth for equity,
      // positions, and orders.
    }
  }
  const exitPolicy = defaultExitPolicy(liveState?.exit_policy ?? null);
  const account = asJsonRecord(accountRaw) ?? {};
  const equity = jsonNumber(account.equity);
  const lastEquity = jsonNumber(account.last_equity);
  const dailyPnl = equity != null && lastEquity != null ? equity - lastEquity : null;

  return {
    account: {
      equity,
      last_equity: lastEquity,
      buying_power: jsonNumber(account.buying_power),
      daily_pnl: dailyPnl,
      daily_pnl_pct: dailyPnl != null && lastEquity ? dailyPnl / lastEquity : null,
    },
    positions: jsonRecords(positionsRaw).map((position) => enrichPosition(position, exitPolicy)),
    openOrders: jsonRecords(openOrdersRaw).map(mapOrder),
    todaysOrders: jsonRecords(closedOrdersRaw)
      .map(mapOrder)
      .filter((order) => order.status === "filled" || (order.filled_qty ?? 0) > 0),
    fetchedAt: new Date().toISOString(),
    exitPolicy,
    liveState,
  };
}
