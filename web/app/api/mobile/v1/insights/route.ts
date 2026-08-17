import { mobileJson, requireMobileUser } from "@/lib/mobileAuth";
import { getMobileSnapshot } from "@/lib/mobileSnapshot";

export const dynamic = "force-dynamic";

export async function GET(request: Request) {
  const auth = await requireMobileUser(request);
  if ("response" in auth) return auth.response;
  const snapshot = await getMobileSnapshot();
  return mobileJson({
    schemaVersion: snapshot.schemaVersion,
    generatedAt: snapshot.generatedAt,
    performance: snapshot.performance,
    breakdowns: snapshot.breakdowns,
    funnel: snapshot.funnel,
    equityHistory: snapshot.equityHistory,
  });
}
