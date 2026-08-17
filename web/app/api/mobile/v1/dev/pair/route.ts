import { issueMobileDemoToken, mobileDemoEnabled, pairingCodeMatches } from "@/lib/mobileDemoAuth";
import { asJsonRecord, firstJsonString } from "@/lib/json";
import { mobileJson } from "@/lib/mobileAuth";
import type { JsonValue } from "@/lib/types";

export const dynamic = "force-dynamic";

export async function POST(request: Request) {
  // This route is deliberately absent unless the server has explicitly opted
  // into the private development path and paper trading is enabled.
  if (!mobileDemoEnabled()) {
    return mobileJson({ error: "not_found" }, { status: 404 });
  }

  let body: JsonValue;
  try {
    body = await request.json();
  } catch {
    return mobileJson(
      { error: "invalid_request", message: "A pairing code is required." },
      { status: 400 },
    );
  }

  const code = firstJsonString(asJsonRecord(body)?.code) ?? "";
  if (!code || !pairingCodeMatches(code)) {
    return mobileJson(
      { error: "unauthorized", message: "That pairing code is not valid." },
      { status: 401 },
    );
  }

  const session = issueMobileDemoToken();
  if (!session) {
    return mobileJson(
      { error: "configuration_error", message: "Private paper access is not configured." },
      { status: 503 },
    );
  }

  return mobileJson({
    accessToken: session.accessToken,
    expiresAt: session.expiresAt.toISOString(),
    provider: "alpaca",
    environment: "paper",
    readOnly: true,
  });
}
