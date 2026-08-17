export function fmtUsd(value: number | null | undefined, digits = 2): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return value.toLocaleString("en-US", {
    style: "currency",
    currency: "USD",
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}

export function fmtSignedUsd(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  const sign = value > 0 ? "+" : "";
  return `${sign}${fmtUsd(value)}`;
}

export function fmtPct(value: number | null | undefined, digits = 1): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return `${(value * 100).toFixed(digits)}%`;
}

export function fmtNum(value: number | null | undefined, digits = 2): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return value.toFixed(digits);
}

export function fmtDate(value: string | null | undefined): string {
  if (!value) return "—";
  return value.slice(0, 10);
}

export function fmtDateTime(value: string | null | undefined): string {
  if (!value) return "—";
  try {
    return new Date(value).toLocaleString("en-US", {
      timeZone: "America/New_York",
      month: "short",
      day: "numeric",
      hour: "numeric",
      minute: "2-digit",
      timeZoneName: "short",
    });
  } catch {
    return value;
  }
}

export function fmtTimeEt(value: string | null | undefined): string {
  if (!value) return "—";
  try {
    return new Date(value).toLocaleTimeString("en-US", {
      timeZone: "America/New_York",
      hour: "2-digit",
      minute: "2-digit",
      hour12: false,
    });
  } catch {
    return "—";
  }
}

export function dateEtIso(
  value: Date | string | null | undefined = new Date(),
): string {
  if (!value) return "";
  const date = value instanceof Date ? value : new Date(value);
  if (Number.isNaN(date.getTime())) return "";
  const parts = new Intl.DateTimeFormat("en-US", {
    timeZone: "America/New_York",
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
  }).formatToParts(date);
  const year = parts.find((p) => p.type === "year")?.value;
  const month = parts.find((p) => p.type === "month")?.value;
  const day = parts.find((p) => p.type === "day")?.value;
  return year && month && day ? `${year}-${month}-${day}` : "";
}

export function relativeTime(value: string | null | undefined): string {
  if (!value) return "never";
  const then = new Date(value).getTime();
  if (Number.isNaN(then)) return "unknown";
  const diffMs = Date.now() - then;
  const minutes = Math.floor(diffMs / 60000);
  if (minutes < 1) return "just now";
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 48) return `${hours}h ago`;
  return `${Math.floor(hours / 24)}d ago`;
}

export function pnlColor(value: number | null | undefined): string {
  if (value === null || value === undefined || value === 0) return "text-ink-2";
  return value > 0 ? "text-gain" : "text-loss";
}

/** OCC option symbol: ROOT + YYMMDD + C/P + strike*1000 (8 digits). */
const OCC_RE = /^([A-Z0-9.]{1,10})(\d{6})([CP])(\d{8})$/;

export type ParsedOcc = {
  underlying: string;
  expiry: string; // YYYY-MM-DD
  right: "C" | "P";
  strike: number;
  dte: number | null;
};

export function parseOccSymbol(
  symbol: string,
  onDate: Date = new Date(),
): ParsedOcc | null {
  const cleaned = symbol.toUpperCase().replace(/\s+/g, "");
  const match = OCC_RE.exec(cleaned);
  if (!match) return null;
  const [, underlying, yymmdd, right, strikeRaw] = match;
  const yy = Number(yymmdd.slice(0, 2));
  const mm = Number(yymmdd.slice(2, 4));
  const dd = Number(yymmdd.slice(4, 6));
  const year = 2000 + yy;
  const expiry = `${year.toString().padStart(4, "0")}-${mm
    .toString()
    .padStart(2, "0")}-${dd.toString().padStart(2, "0")}`;
  const strike = Number(strikeRaw) / 1000;
  const expiryUtc = Date.UTC(year, mm - 1, dd);
  const todayUtc = Date.UTC(
    onDate.getUTCFullYear(),
    onDate.getUTCMonth(),
    onDate.getUTCDate(),
  );
  const dte = Math.round((expiryUtc - todayUtc) / 86_400_000);
  return {
    underlying,
    expiry,
    right: right === "C" ? "C" : "P",
    strike,
    dte: Number.isFinite(dte) ? dte : null,
  };
}

export function formatOccLabel(symbol: string): string {
  const parsed = parseOccSymbol(symbol);
  if (!parsed) return symbol;
  const right = parsed.right === "C" ? "Call" : "Put";
  return `${parsed.underlying} ${parsed.expiry} ${parsed.strike} ${right}`;
}

/** Weekday 09:30–16:00 America/New_York. */
export function isRegularSessionEt(now: Date = new Date()): boolean {
  const parts = new Intl.DateTimeFormat("en-US", {
    timeZone: "America/New_York",
    weekday: "short",
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  }).formatToParts(now);
  const weekday = parts.find((p) => p.type === "weekday")?.value ?? "";
  if (weekday === "Sat" || weekday === "Sun") return false;
  const hour = Number(parts.find((p) => p.type === "hour")?.value ?? "0");
  const minute = Number(parts.find((p) => p.type === "minute")?.value ?? "0");
  const mins = hour * 60 + minute;
  return mins >= 9 * 60 + 30 && mins <= 16 * 60;
}

export function normalizePlpc(raw: number | null | undefined): number {
  if (raw === null || raw === undefined || Number.isNaN(raw)) return 0;
  return Math.abs(raw) > 5 ? raw / 100 : raw;
}
