import { notFound } from "next/navigation";
import { CommandCenterDashboard } from "@/components/command-center/CommandCenterDashboard";
import { LiveBlotter } from "@/components/command-center/LiveBlotter";
import { PageHeader } from "@/components/ui";
import { getOpportunities, getWatchlistFlags } from "@/lib/commandCenter";

export const dynamic = "force-dynamic";

export default async function CommandCenterPage() {
  if (process.env.COMMAND_CENTER_ENABLED !== "true") notFound();
  const [flags, opportunities] = await Promise.all([
    getWatchlistFlags(),
    getOpportunities(),
  ]);

  return (
    <div className="flex flex-col gap-5">
      <div>
        <p className="tag">Private operator surface</p>
        <div className="mt-2">
          <PageHeader title="Command center">
            Live paper blotter (marks, open orders, software stop/TP) plus
            watchlist scaffolding for a future Robinhood MCP integration.
          </PageHeader>
        </div>
      </div>
      <LiveBlotter />
      <CommandCenterDashboard initialFlags={flags} opportunities={opportunities} />
    </div>
  );
}
