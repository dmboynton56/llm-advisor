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

function badgeClass(
  kind: "long" | "short" | "call" | "put" | "bullish" | "bearish" | "unknown",
) {
  if (kind === "long" || kind === "bullish") {
    return "bg-emerald-500/10 text-emerald-400";
  }
  if (kind === "short" || kind === "bearish") {
    return "bg-rose-500/10 text-rose-400";
  }
  if (kind === "call") return "bg-sky-500/10 text-sky-400";
  if (kind === "put") return "bg-amber-500/10 text-amber-400";
  return "bg-zinc-800 text-zinc-400";
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
    <label className="flex items-center gap-2 text-xs text-zinc-400">
      {label}
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="rounded-md border border-zinc-800 bg-zinc-900 px-2 py-1 text-xs text-zinc-200 focus:border-zinc-600 focus:outline-none"
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

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-3">
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
        <span className="ml-auto text-xs text-zinc-500">
          {filtered.length} trades ·{" "}
          <span className={pnlColor(filteredPnl)}>
            {fmtSignedUsd(filteredPnl)} lifetime PnL
          </span>
        </span>
      </div>
      <p className="text-xs text-zinc-500">
        Position means whether we bought or sold the option. Contract means call or put. Bias means the expected stock direction.
      </p>

      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-zinc-800 text-left text-xs uppercase tracking-wide text-zinc-500">
              <th className="py-2 pr-4 font-medium">Entry</th>
              <th className="py-2 pr-4 font-medium">Exit</th>
              <th className="py-2 pr-4 font-medium">Symbol</th>
              <th className="py-2 pr-4 font-medium">Position</th>
              <th className="py-2 pr-4 font-medium">Contract</th>
              <th className="py-2 pr-4 font-medium">Bias</th>
              <th className="py-2 pr-4 font-medium">Setup</th>
              <th className="py-2 pr-4 font-medium">DTE</th>
              <th className="py-2 pr-4 font-medium">Qty</th>
              <th className="py-2 pr-4 font-medium">Entry px</th>
              <th className="py-2 pr-4 font-medium">Exit px</th>
              <th className="py-2 pr-4 font-medium">PnL</th>
              <th className="py-2 font-medium">Exit reason</th>
            </tr>
          </thead>
          <tbody>
            {filtered.slice(0, 200).map((t) => (
              <tr
                key={t.trade_uid}
                className="border-b border-zinc-900 last:border-0"
              >
                <td className="whitespace-nowrap py-2 pr-4 text-xs tabular-nums text-zinc-400">
                  {fmtDateTime(t.entry_time) !== "—"
                    ? fmtDateTime(t.entry_time)
                    : t.run_date}
                </td>
                <td className="whitespace-nowrap py-2 pr-4 text-xs tabular-nums text-zinc-400">
                  {fmtDateTime(t.exit_time)}
                </td>
                <td className="py-2 pr-4">
                  <span className="font-medium">{t.underlying_symbol ?? t.symbol}</span>
                  {t.asset_class === "option" ? (
                    <span className="ml-2 rounded bg-zinc-800 px-1.5 py-0.5 text-[10px] uppercase text-zinc-400">
                      opt
                    </span>
                  ) : null}
                </td>
                <td className="py-2 pr-4">
                  <div className="space-y-1">
                    <span
                      className={clsx(
                        "rounded px-1.5 py-0.5 text-xs capitalize",
                        badgeClass(positionForTrade(t)),
                      )}
                      title="Long means the option was bought to open; short means it was sold to open."
                    >
                      {positionForTrade(t)}
                    </span>
                    <div className="text-[10px] text-zinc-500">
                      {entryActionLabel(t.entry_action)}
                    </div>
                  </div>
                </td>
                <td className="py-2 pr-4">
                  <span
                    className={clsx(
                      "rounded px-1.5 py-0.5 text-xs capitalize",
                      badgeClass(t.contract_type ?? "unknown"),
                    )}
                  >
                    {t.contract_type ?? "Unknown"}
                  </span>
                </td>
                <td className="py-2 pr-4">
                  <span
                    className={clsx(
                      "rounded px-1.5 py-0.5 text-xs capitalize",
                      badgeClass(t.signal_bias ?? "unknown"),
                    )}
                  >
                    {t.signal_bias ?? "Unknown"}
                  </span>
                </td>
                <td className="py-2 pr-4 text-zinc-300">{t.setup_type ?? "—"}</td>
                <td className="py-2 pr-4 tabular-nums">{t.option_dte ?? "—"}</td>
                <td className="py-2 pr-4 tabular-nums">{t.qty ?? "—"}</td>
                <td className="py-2 pr-4 tabular-nums">{fmtNum(t.entry_price)}</td>
                <td className="py-2 pr-4 tabular-nums">{fmtNum(t.exit_price)}</td>
                <td className={clsx("py-2 pr-4 tabular-nums", pnlColor(t.pnl))}>
                  {t.pnl !== null ? fmtSignedUsd(Number(t.pnl)) : t.status ?? "—"}
                </td>
                <td className="py-2 text-xs text-zinc-400">{t.exit_reason ?? "—"}</td>
              </tr>
            ))}
          </tbody>
        </table>
        {filtered.length === 0 ? (
          <p className="py-8 text-center text-sm text-zinc-500">
            No trades match the current filters.
          </p>
        ) : null}
        {filtered.length > 200 ? (
          <p className="py-2 text-center text-xs text-zinc-600">
            Showing first 200 of {filtered.length} trades.
          </p>
        ) : null}
      </div>
    </div>
  );
}
