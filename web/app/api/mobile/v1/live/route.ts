import { mobileJson, requireMobileUser } from "@/lib/mobileAuth";
import { getMobileSnapshot, mobileLivePayload } from "@/lib/mobileSnapshot";

export const dynamic = "force-dynamic";

export async function GET(request: Request) {
  const auth = await requireMobileUser(request);
  if ("response" in auth) return auth.response;
  return mobileJson(mobileLivePayload(await getMobileSnapshot()));
}
