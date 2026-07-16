import { notFound } from "next/navigation";
import { CommandCenterDashboard } from "@/components/command-center/CommandCenterDashboard";
import { LiveBlotter } from "@/components/command-center/LiveBlotter";
import { getOpportunities, getWatchlistFlags } from "@/lib/commandCenter";

export const dynamic = "force-dynamic";

export default async function CommandCenterPage() {
  if (process.env.COMMAND_CENTER_ENABLED !== "true") notFound();
  const [flags, opportunities] = await Promise.all([
    getWatchlistFlags(),
    getOpportunities(),
  ]);

  return (
    <div className="space-y-6">
      <div>
        <p className="text-xs font-medium uppercase tracking-[0.2em] text-emerald-500">Private operator surface</p>
        <h1 className="mt-2 text-xl font-semibold tracking-tight">Command center</h1>
        <p className="mt-1 text-sm text-zinc-500">
          Live paper blotter (marks, open orders, software stop/TP) plus watchlist scaffolding for a future Robinhood MCP integration.
        </p>
      </div>
      <LiveBlotter />
      <CommandCenterDashboard initialFlags={flags} opportunities={opportunities} />
    </div>
  );
}
