"use client";

import { useEffect, useState } from "react";
import { Flag, Plus, Radio, Trash2 } from "lucide-react";
import clsx from "clsx";
import type { Opportunity, WatchlistFlag } from "@/lib/commandCenter";

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
    <div className="grid gap-6 lg:grid-cols-[1.1fr_0.9fr]">
      <section className="rounded-xl border border-zinc-800 bg-zinc-900/40 p-5">
        <div className="flex items-center justify-between gap-4">
          <div>
            <h2 className="font-medium">Watchlist flags</h2>
            <p className="mt-1 text-xs text-zinc-500">Mock rows; edits persist in this browser only.</p>
          </div>
          <Flag className="size-5 text-emerald-400" />
        </div>
        <div className="mt-4 flex gap-2">
          <input
            value={newSymbol}
            onChange={(event) => setNewSymbol(event.target.value)}
            onKeyDown={(event) => event.key === "Enter" && addSymbol()}
            placeholder="Add symbol"
            className="min-w-0 flex-1 rounded-md border border-zinc-700 bg-zinc-950 px-3 py-2 text-sm uppercase outline-none focus:border-emerald-500"
          />
          <button onClick={addSymbol} className="rounded-md bg-zinc-800 px-3 text-zinc-200 hover:bg-zinc-700" aria-label="Add symbol">
            <Plus className="size-4" />
          </button>
        </div>
        <div className="mt-4 space-y-3">
          {flags.map((item) => (
            <div key={item.symbol} className="rounded-lg border border-zinc-800 bg-zinc-950/60 p-3">
              <div className="flex items-center gap-3">
                <button
                  onClick={() => update(item.symbol, { flagged: !item.flagged })}
                  className={clsx("rounded p-1", item.flagged ? "text-emerald-400" : "text-zinc-600")}
                  aria-label={`Toggle ${item.symbol} flag`}
                >
                  <Flag className="size-4" fill={item.flagged ? "currentColor" : "none"} />
                </button>
                <span className="w-14 font-mono text-sm font-semibold">{item.symbol}</span>
                <input
                  value={item.note}
                  onChange={(event) => update(item.symbol, { note: event.target.value })}
                  placeholder="Operator note"
                  className="min-w-0 flex-1 border-b border-zinc-800 bg-transparent px-1 py-1 text-sm text-zinc-300 outline-none focus:border-emerald-500"
                />
                <button
                  onClick={() => setFlags((current) => current.filter((row) => row.symbol !== item.symbol))}
                  className="text-zinc-600 hover:text-rose-400"
                  aria-label={`Remove ${item.symbol}`}
                >
                  <Trash2 className="size-4" />
                </button>
              </div>
            </div>
          ))}
        </div>
      </section>

      <div className="space-y-6">
        <section className="rounded-xl border border-zinc-800 bg-zinc-900/40 p-5">
          <h2 className="font-medium">Opportunities</h2>
          <div className="mt-4 space-y-3">
            {opportunities.map((opportunity) => (
              <article key={opportunity.id} className="rounded-lg border border-zinc-800 bg-zinc-950/60 p-4">
                <div className="flex items-center justify-between">
                  <span className="font-mono text-sm font-semibold text-emerald-300">{opportunity.symbol}</span>
                  <span className="rounded-full bg-zinc-800 px-2 py-0.5 text-xs capitalize text-zinc-400">{opportunity.direction}</span>
                </div>
                <p className="mt-2 text-sm leading-6 text-zinc-300">{opportunity.thesis}</p>
                <p className="mt-2 text-xs text-zinc-600">{opportunity.source}</p>
              </article>
            ))}
          </div>
        </section>

        <section className="rounded-xl border border-amber-500/20 bg-amber-500/5 p-5">
          <div className="flex items-center gap-2 text-amber-300">
            <Radio className="size-4" />
            <h2 className="font-medium">Robinhood MCP</h2>
          </div>
          <p className="mt-2 text-sm text-zinc-400">Not connected. This cycle uses mock data and cannot place orders.</p>
          <a
            href="https://github.com/dmboynton56/llm-advisor/blob/main/docs/robinhood_mcp_execution_plan.md"
            className="mt-3 inline-block text-xs text-amber-300 underline-offset-2 hover:underline"
          >
            Read the execution plan →
          </a>
        </section>
      </div>
    </div>
  );
}
