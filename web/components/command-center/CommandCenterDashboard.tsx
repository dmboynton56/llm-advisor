"use client";

import { useEffect, useState } from "react";
import { Flag, Plus, Radio, Trash2 } from "lucide-react";
import clsx from "clsx";
import type { Opportunity, WatchlistFlag } from "@/lib/commandCenter";
import { Panel, PanelHead } from "@/components/ui";

const STORAGE_KEY = "llm-advisor-command-center-watchlist";

export function CommandCenterDashboard({
  initialFlags,
  opportunities,
}: {
  initialFlags: WatchlistFlag[];
  opportunities: Opportunity[];
}) {
  const [flags, setFlags] = useState(initialFlags);
  const [newSymbol, setNewSymbol] = useState("");

  useEffect(() => {
    const saved = window.localStorage.getItem(STORAGE_KEY);
    if (!saved) return;
    try {
      setFlags(JSON.parse(saved) as WatchlistFlag[]);
    } catch {
      window.localStorage.removeItem(STORAGE_KEY);
    }
  }, []);

  useEffect(() => {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(flags));
  }, [flags]);

  function update(symbol: string, patch: Partial<WatchlistFlag>) {
    setFlags((current) =>
      current.map((item) => (item.symbol === symbol ? { ...item, ...patch } : item)),
    );
  }

  function addSymbol() {
    const symbol = newSymbol.trim().toUpperCase();
    if (!symbol || flags.some((item) => item.symbol === symbol)) return;
    setFlags((current) => [...current, { symbol, flagged: true, note: "" }]);
    setNewSymbol("");
  }

  return (
    <div className="grid gap-5 lg:grid-cols-[1.1fr_0.9fr]">
      <Panel>
        <PanelHead title="Watchlist flags" aside="This browser only" />

        <div className="flex gap-2">
          <input
            value={newSymbol}
            onChange={(event) => setNewSymbol(event.target.value)}
            onKeyDown={(event) => event.key === "Enter" && addSymbol()}
            placeholder="Add symbol"
            className="num min-w-0 flex-1 rounded-lg border border-line-2 bg-paper px-3 py-2 text-[13px] uppercase outline-none transition-colors focus:border-ink-3"
          />
          <button
            onClick={addSymbol}
            className="rounded-lg border border-line-2 px-3 text-ink-2 transition-colors hover:bg-sunk hover:text-ink"
            aria-label="Add symbol"
          >
            <Plus className="size-4" />
          </button>
        </div>

        <ul className="mt-4 flex flex-col">
          {flags.map((item) => (
            <li
              key={item.symbol}
              className="flex items-center gap-3 border-t border-line py-2.5 first:border-t-0"
            >
              <button
                onClick={() => update(item.symbol, { flagged: !item.flagged })}
                className={clsx(
                  "rounded p-1 transition-colors",
                  item.flagged ? "text-ink" : "text-ink-3 hover:text-ink-2",
                )}
                aria-label={`Toggle ${item.symbol} flag`}
                aria-pressed={item.flagged}
              >
                <Flag className="size-4" fill={item.flagged ? "currentColor" : "none"} />
              </button>
              <span className="num w-14 text-[13px] font-semibold">
                {item.symbol}
              </span>
              <input
                value={item.note}
                onChange={(event) => update(item.symbol, { note: event.target.value })}
                placeholder="Operator note"
                className="min-w-0 flex-1 border-b border-line bg-transparent px-1 py-1 text-[13px] text-ink-2 outline-none transition-colors focus:border-ink-3"
              />
              <button
                onClick={() =>
                  setFlags((current) =>
                    current.filter((row) => row.symbol !== item.symbol),
                  )
                }
                className="text-ink-3 transition-colors hover:text-loss"
                aria-label={`Remove ${item.symbol}`}
              >
                <Trash2 className="size-4" />
              </button>
            </li>
          ))}
        </ul>
      </Panel>

      <div className="flex flex-col gap-5">
        <Panel>
          <PanelHead title="Opportunities" />
          <div className="flex flex-col gap-3">
            {opportunities.map((opportunity) => (
              <article
                key={opportunity.id}
                className="rounded-panel border border-line p-4"
              >
                <div className="flex items-center justify-between gap-3">
                  <span className="num text-[13px] font-semibold">
                    {opportunity.symbol}
                  </span>
                  <span className="num rounded-full border border-line-2 px-2 py-0.5 text-[10px] uppercase tracking-[0.08em] text-ink-3">
                    {opportunity.direction}
                  </span>
                </div>
                <p className="mt-2 text-[13px] leading-relaxed text-ink-2">
                  {opportunity.thesis}
                </p>
                <p className="mt-2 text-[11.5px] text-ink-3">{opportunity.source}</p>
              </article>
            ))}
          </div>
        </Panel>

        <Panel className="border-dashed border-line-2">
          <div className="flex items-center gap-2 text-ink">
            <Radio className="size-4" />
            <h2 className="text-[13px] font-semibold">Robinhood MCP</h2>
          </div>
          <p className="mt-2 text-[13px] text-ink-2">
            Not connected. This cycle uses mock data and cannot place orders.
          </p>
          <a
            href="https://github.com/dmboynton56/llm-advisor/blob/main/docs/robinhood_mcp_execution_plan.md"
            className="mt-3 inline-block border-b border-line-2 pb-px text-[12px] text-ink-2 transition-colors hover:text-ink"
          >
            Read the execution plan →
          </a>
        </Panel>
      </div>
    </div>
  );
}
