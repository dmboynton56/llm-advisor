"use client";

import { useMemo, useState } from "react";
import clsx from "clsx";
import type { TradeRow } from "@/lib/types";
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
  const [side, setSide] = useState("");
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
        const normalizedSide =
          t.side === "buy" || t.side === "long"
            ? "long"
            : t.side === "sell" || t.side === "short"
              ? "short"
              : (t.side ?? "");
        if (side && normalizedSide !== side) return false;
        if (setup && (t.setup_type ?? "") !== setup) return false;
        if (dte && dteBucket(t.option_dte) !== dte) return false;
        if (entryDate && t.run_date !== entryDate) return false;
        if (exitDate && dateEtIso(t.exit_time) !== exitDate) return false;
        return true;
      }),
    [trades, underlying, side, setup, dte, entryDate, exitDate],
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
          label="Side"
          value={side}
          onChange={setSide}
          options={["long", "short"]}
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

      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-zinc-800 text-left text-xs uppercase tracking-wide text-zinc-500">
              <th className="py-2 pr-4 font-medium">Entry</th>
              <th className="py-2 pr-4 font-medium">Exit</th>
              <th className="py-2 pr-4 font-medium">Symbol</th>
              <th className="py-2 pr-4 font-medium">Side</th>
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
                  <span
                    className={clsx(
                      "rounded px-1.5 py-0.5 text-xs",
                      t.side === "buy" || t.side === "long"
                        ? "bg-emerald-500/10 text-emerald-400"
                        : "bg-rose-500/10 text-rose-400",
                    )}
                  >
                    {t.side === "buy" || t.side === "long" ? "long" : "short"}
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
