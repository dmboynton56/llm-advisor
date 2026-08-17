import { mobileJson, requireMobileUser } from "@/lib/mobileAuth";
import { getMobileSnapshot } from "@/lib/mobileSnapshot";

export const dynamic = "force-dynamic";

export async function GET(request: Request) {
  const auth = await requireMobileUser(request);
  if ("response" in auth) return auth.response;

  const url = new URL(request.url);
  const status = url.searchParams.get("status")?.toLowerCase();
  const setup = url.searchParams.get("setup")?.toLowerCase();
  const symbol = url.searchParams.get("symbol")?.toLowerCase();
  const limit = Math.min(Math.max(Number(url.searchParams.get("limit") ?? 50) || 50, 1), 100);
  const snapshot = await getMobileSnapshot();
  const trades = snapshot.trades
    .filter((trade) => !status || trade.status.toLowerCase() === status)
    .filter((trade) => !setup || trade.setup?.toLowerCase() === setup)
    .filter((trade) => !symbol || trade.symbol.toLowerCase() === symbol)
    .slice(0, limit);
  return mobileJson({ schemaVersion: 1, generatedAt: snapshot.generatedAt, trades });
}
