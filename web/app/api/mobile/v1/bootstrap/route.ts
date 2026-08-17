import { getMobileSnapshot } from "@/lib/mobileSnapshot";
import { mobileJson, requireMobileUser } from "@/lib/mobileAuth";

export const dynamic = "force-dynamic";

export async function GET(request: Request) {
  const auth = await requireMobileUser(request);
  if ("response" in auth) return auth.response;
  return mobileJson(await getMobileSnapshot());
}
