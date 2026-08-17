import { fetchAlpacaPaperSnapshot } from "@/lib/alpacaPaper";
import { mobileJson, requireMobileUser } from "@/lib/mobileAuth";

export const dynamic = "force-dynamic";

export async function GET(request: Request) {
  const auth = await requireMobileUser(request);
  if ("response" in auth) return auth.response;

  try {
    const snapshot = await fetchAlpacaPaperSnapshot();
    return mobileJson({
      schemaVersion: 1,
      provider: "alpaca",
      environment: "paper",
      readOnly: true,
      fetchedAt: snapshot.fetchedAt,
      account: snapshot.account,
      positions: snapshot.positions,
      openOrders: snapshot.openOrders,
      recentOrders: snapshot.todaysOrders,
      liveState: snapshot.liveState,
    });
  } catch {
    return mobileJson(
      {
        error: "alpaca_unavailable",
        message: "The server could not read the Alpaca paper account right now.",
      },
      { status: 503 },
    );
  }
}
