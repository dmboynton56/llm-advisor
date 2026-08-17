import type {
  JsonRecord,
  OptionContractType,
  TradeBias,
  TradeDirection,
  TradeEntryAction,
  TradePosition,
} from "@/lib/types";
import {
  asJsonRecord,
  firstJsonString,
  type JsonInput,
} from "@/lib/json";

function asText(value: JsonInput): string | null {
  const text = firstJsonString(value)?.toLowerCase() ?? null;
  return text || null;
}

function firstNormalized<T>(
  normalize: (value: JsonInput) => T | null,
  ...values: JsonInput[]
): T | null {
  for (const value of values) {
    const normalized = normalize(value);
    if (normalized) return normalized;
  }
  return null;
}

export function normalizePositionSide(value: JsonInput): TradePosition | null {
  const normalized = asText(value);
  if (["long", "buy", "buy_to_open"].includes(normalized ?? "")) return "long";
  if (["short", "sell", "sell_to_open"].includes(normalized ?? "")) return "short";
  // sell_to_close is an exit action, not a short position.
  return null;
}

export function normalizeContractType(value: JsonInput): OptionContractType | null {
  const normalized = asText(value);
  if (normalized === "call" || normalized === "c") return "call";
  if (normalized === "put" || normalized === "p") return "put";
  return null;
}

export function normalizeBias(value: JsonInput): TradeBias | null {
  const normalized = asText(value);
  if (["bullish", "long", "up"].includes(normalized ?? "")) return "bullish";
  if (["bearish", "short", "down"].includes(normalized ?? "")) return "bearish";
  return null;
}

export function normalizeEntryAction(value: JsonInput): TradeEntryAction | null {
  const normalized = asText(value);
  if (normalized === "buy_to_open" || normalized === "buy to open") {
    return "buy_to_open";
  }
  if (normalized === "sell_to_open" || normalized === "sell to open") {
    return "sell_to_open";
  }
  return null;
}

export function optionContractTypeFromSymbol(
  symbol: string | null | undefined,
): OptionContractType | null {
  const compact = String(symbol ?? "")
    .trim()
    .toUpperCase()
    .replace(/\s+/g, "");
  const match = compact.match(/^[A-Z0-9.]{1,10}\d{6}([CP])\d{8}$/);
  return normalizeContractType(match?.[1]);
}

function directionRecord(direction: TradeDirection): JsonRecord {
  return {
    position_side: direction.position_side,
    contract_type: direction.contract_type,
    signal_bias: direction.signal_bias,
    entry_action: direction.entry_action,
  };
}

export function deriveTradeDirection({
  symbol,
  side,
  details,
  assumeLongOptionPosition = false,
}: {
  symbol?: string | null;
  side?: JsonInput;
  details?: JsonInput;
  assumeLongOptionPosition?: boolean;
}): TradeDirection {
  const root = asJsonRecord(details) ?? {};
  const stored = asJsonRecord(root.trade_direction) ?? {};
  const optionPlan = asJsonRecord(root.option_plan) ?? {};
  const order = asJsonRecord(root.order) ?? {};
  const orderPlan = asJsonRecord(order.option_plan) ?? {};
  const position = asJsonRecord(root.position) ?? {};
  const underlyingPlan = asJsonRecord(
    optionPlan.underlying_trade_plan ?? orderPlan.underlying_trade_plan,
  ) ?? {};

  let positionSide = firstNormalized(
    normalizePositionSide,
    stored.position_side,
    root.position_side,
    position.side,
    optionPlan.position_side,
    optionPlan.side,
    optionPlan.position_intent,
    orderPlan.position_side,
    orderPlan.side,
    orderPlan.position_intent,
    order.side,
    order.position_intent,
    side,
  );

  const contractType =
    firstNormalized(
      normalizeContractType,
      stored.contract_type,
      root.contract_type,
      optionPlan.contract_type,
      orderPlan.contract_type,
    ) ?? optionContractTypeFromSymbol(symbol);

  const explicitEntryAction = firstNormalized(
    normalizeEntryAction,
    stored.entry_action,
    root.entry_action,
    optionPlan.position_intent,
    orderPlan.position_intent,
    order.position_intent,
  );

  const signalBias = firstNormalized(
    normalizeBias,
    stored.signal_bias,
    root.signal_bias,
    root.bias,
    root.signal_side,
    optionPlan.signal_side,
    orderPlan.signal_side,
    underlyingPlan.side,
  );

  if (!positionSide && assumeLongOptionPosition) positionSide = "long";

  const entryAction =
    explicitEntryAction ??
    (assumeLongOptionPosition && positionSide === "long" ? "buy_to_open" : null);

  return {
    position_side: positionSide,
    contract_type: contractType,
    signal_bias:
      signalBias ??
      (positionSide === "long" && contractType
        ? contractType === "call"
          ? "bullish"
          : "bearish"
        : null),
    entry_action: entryAction,
  };
}

export function tradeDirectionDetails(direction: TradeDirection): JsonRecord {
  return directionRecord(direction);
}
