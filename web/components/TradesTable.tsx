"use client";

import { useMemo, useState } from "react";
import clsx from "clsx";
import type { TradePosition, TradeRow } from "@/lib/types";
import {
  dateEtIso,
  fmtDateTime,
  fmtNum,
  fmtSignedUsd,
  pnlColor,
} from "@/lib/format";

function uniqueValues(rows: TradeRow[], key: keyof TradeRow): string[] {
  return Array.from(
    new Set(rows.map((r) => String(r[key] ?? "")).filter(Boolean)),
  ).sort();
}

function dteBucket(dte: number | null): string {
  if (dte === null || dte === undefined) return "";
  if (dte <= 0) return "0";
  if (dte <= 3) return "1-3";
  if (dte <= 7) return "4-7";
  if (dte <= 14) return "8-14";
  return "15+";
}

const DTE_OPTIONS = ["0", "1-3", "4-7", "8-14", "15+"];

function positionForTrade(trade: TradeRow): TradePosition | "unknown" {
  if (trade.position_side) return trade.position_side;
  if (
    trade.entry_action === "buy_to_open" ||
    trade.side === "buy" ||
    trade.side === "long"
  ) {
    return "long";
  }
  if (trade.entry_action === "sell_to_open" || trade.side === "short") return "short";
  return "unknown";
}

function entryActionLabel(action: TradeRow["entry_action"]): string {
  if (action === "buy_to_open") return "Buy to open";
  if (action === "sell_to_open") return "Sell to open";
  return "Unknown entry";
}

/**
 * Badges stay achromatic — colour in this app means money, and a call is not a
 * profit. Direction is carried by the word plus a filled/outlined weight.
 */
function badgeClass(
  kind: "long" | "short" | "call" | "put" | "bullish" | "bearish" | "unknown",
) {
  const solid = kind === "long" || kind === "bullish" || kind === "call";
  return solid
    ? "border-transparent bg-sunk text-ink"
    : kind === "unknown"
      ? "border-dashed border-line-2 text-ink-3"
      : "border-line-2 text-ink-2";
}

function Badge({
  kind,
  children,
  title,
}: {
  kind: "long" | "short" | "call" | "put" | "bullish" | "bearish" | "unknown";
  children: React.ReactNode;
  title?: string;
}) {
  return (
    <span
      title={title}
      className={clsx(
        "inline-block rounded border px-1.5 py-0.5 text-[11px] capitalize",
        badgeClass(kind),
      )}
    >
      {children}
    </span>
  );
}

function FilterSelect({
  label,
  value,
  onChange,
  options,
}: {
  label: string;
  value: string;
  onChange: (v: string) => void;
  options: string[];
}) {
  return (
    <label className="flex items-center gap-2 text-[11.5px] text-ink-3">
      {label}
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="num rounded-lg border border-line-2 bg-card px-2 py-1 text-[11.5px] text-ink transition-colors hover:border-ink-3 focus:outline-none"
      >
        <option value="">All</option>
        {options.map((opt) => (
          <option key={opt} value={opt}>
            {opt}
          </option>
        ))}
      </select>
    </label>
  );
}

export function TradesTable({ trades }: { trades: TradeRow[] }) {
  const [underlying, setUnderlying] = useState("");
  const [position, setPosition] = useState("");
  const [contractType, setContractType] = useState("");
  const [bias, setBias] = useState("");
  const [setup, setSetup] = useState("");
  const [dte, setDte] = useState("");
  const [entryDate, setEntryDate] = useState("");
  const [exitDate, setExitDate] = useState("");

  const underlyings = useMemo(
    () => uniqueValues(trades, "underlying_symbol"),
    [trades],
  );
  const entryDates = useMemo(
    () => uniqueValues(trades, "run_date").reverse(),
    [trades],
  );
  const exitDates = useMemo(
    () =>
      Array.from(
        new Set(
          trades.map((trade) => dateEtIso(trade.exit_time)).filter(Boolean),
        ),
      )
        .sort()
        .reverse(),
    [trades],
  );

  const filtered = useMemo(
    () =>
      trades.filter((t) => {
        if (underlying && (t.underlying_symbol ?? "") !== underlying) return false;
        if (position && positionForTrade(t) !== position) return false;
        if (contractType && (t.contract_type ?? "") !== contractType) return false;
        if (bias && (t.signal_bias ?? "") !== bias) return false;
        if (setup && (t.setup_type ?? "") !== setup) return false;
        if (dte && dteBucket(t.option_dte) !== dte) return false;
        if (entryDate && t.run_date !== entryDate) return false;
        if (exitDate && dateEtIso(t.exit_time) !== exitDate) return false;
        return true;
      }),
    [trades, underlying, position, contractType, bias, setup, dte, entryDate, exitDate],
  );

  const filteredPnl = filtered.reduce((acc, t) => acc + Number(t.pnl ?? 0), 0);

  const head = [
    "Entry",
    "Exit",
    "Symbol",
    "Position",
    "Contract",
    "Bias",
    "Setup",
    "DTE",
    "Qty",
    "Entry px",
    "Exit px",
    "PnL",
    "Exit reason",
  ];

  return (
    <div>
      <div className="border-b border-line p-5">
        <div className="flex flex-wrap items-center gap-x-4 gap-y-2.5">
          <FilterSelect
            label="Entry date"
            value={entryDate}
            onChange={setEntryDate}
            options={entryDates}
          />
          <FilterSelect
            label="Exit date"
            value={exitDate}
            onChange={setExitDate}
            options={exitDates}
          />
          <FilterSelect
            label="Underlying"
            value={underlying}
            onChange={setUnderlying}
            options={underlyings}
          />
          <FilterSelect
            label="Position"
            value={position}
            onChange={setPosition}
            options={["long", "short"]}
          />
          <FilterSelect
            label="Contract"
            value={contractType}
            onChange={setContractType}
            options={["call", "put"]}
          />
          <FilterSelect
            label="Bias"
            value={bias}
            onChange={setBias}
            options={["bullish", "bearish"]}
          />
          <FilterSelect
            label="Setup"
            value={setup}
            onChange={setSetup}
            options={["MR", "TC"]}
          />
          <FilterSelect label="DTE" value={dte} onChange={setDte} options={DTE_OPTIONS} />
          <span className="num ml-auto text-[11.5px] text-ink-3">
            {filtered.length} trades ·{" "}
            <span className={pnlColor(filteredPnl)}>
              {fmtSignedUsd(filteredPnl)} lifetime P&L
            </span>
          </span>
        </div>
        <p className="mt-3 text-[11.5px] text-ink-3">
          Position is whether the option was bought or sold. Contract is call or
          put. Bias is the stock direction the signal expected.
        </p>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-[13px]">
          <thead>
            <tr>
              {head.map((label) => (
                <th
                  key={label}
                  className="tag whitespace-nowrap border-b border-line px-4 py-3 text-left font-medium"
                >
                  {label}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {filtered.slice(0, 200).map((t) => (
              <tr
                key={t.trade_uid}
                className="border-b border-line transition-colors last:border-0 hover:bg-sunk"
              >
                <td className="num whitespace-nowrap px-4 py-3 text-[11.5px] text-ink-2">
                  {fmtDateTime(t.entry_time) !== "—"
                    ? fmtDateTime(t.entry_time)
                    : t.run_date}
                </td>
                <td className="num whitespace-nowrap px-4 py-3 text-[11.5px] text-ink-2">
                  {fmtDateTime(t.exit_time)}
                </td>
                <td className="whitespace-nowrap px-4 py-3">
                  <span className="num font-medium">
                    {t.underlying_symbol ?? t.symbol}
                  </span>
                  {t.asset_class === "option" ? (
                    <span className="num ml-2 rounded border border-line-2 px-1.5 py-0.5 text-[9.5px] uppercase tracking-[0.08em] text-ink-3">
                      opt
                    </span>
                  ) : null}
                </td>
                <td className="whitespace-nowrap px-4 py-3">
                  <Badge
                    kind={positionForTrade(t)}
                    title="Long means the option was bought to open; short means it was sold to open."
                  >
                    {positionForTrade(t)}
                  </Badge>
                  <span className="mt-1 block text-[10px] text-ink-3">
                    {entryActionLabel(t.entry_action)}
                  </span>
                </td>
                <td className="whitespace-nowrap px-4 py-3">
                  <Badge kind={t.contract_type ?? "unknown"}>
                    {t.contract_type ?? "Unknown"}
                  </Badge>
                </td>
                <td className="whitespace-nowrap px-4 py-3">
                  <Badge kind={t.signal_bias ?? "unknown"}>
                    {t.signal_bias ?? "Unknown"}
                  </Badge>
                </td>
                <td className="num whitespace-nowrap px-4 py-3 text-ink-2">
                  {t.setup_type ?? "—"}
                </td>
                <td className="num whitespace-nowrap px-4 py-3">
                  {t.option_dte ?? "—"}
                </td>
                <td className="num whitespace-nowrap px-4 py-3">{t.qty ?? "—"}</td>
                <td className="num whitespace-nowrap px-4 py-3">
                  {fmtNum(t.entry_price)}
                </td>
                <td className="num whitespace-nowrap px-4 py-3">
                  {fmtNum(t.exit_price)}
                </td>
                <td
                  className={clsx(
                    "num whitespace-nowrap px-4 py-3",
                    pnlColor(t.pnl),
                  )}
                >
                  {t.pnl !== null ? fmtSignedUsd(Number(t.pnl)) : t.status ?? "—"}
                </td>
                <td className="whitespace-nowrap px-4 py-3 text-[11.5px] text-ink-2">
                  {t.exit_reason ?? "—"}
                </td>
              </tr>
            ))}
          </tbody>
        </table>

        {filtered.length === 0 ? (
          <p className="py-10 text-center text-[13px] text-ink-3">
            No trades match the current filters.
          </p>
        ) : null}
        {filtered.length > 200 ? (
          <p className="num border-t border-line py-3 text-center text-[11.5px] text-ink-3">
            Showing the first 200 of {filtered.length} trades.
          </p>
        ) : null}
      </div>
    </div>
  );
}
