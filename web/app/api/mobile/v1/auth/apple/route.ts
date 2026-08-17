import { mobileJson } from "@/lib/mobileAuth";
import { asJsonRecord, firstJsonString, jsonNumber } from "@/lib/json";
import type { JsonValue } from "@/lib/types";

export const dynamic = "force-dynamic";

function supabaseBaseUrl(): string {
  return (
    process.env.NEXT_PUBLIC_SUPABASE_URL ??
    process.env.SUPABASE_URL ??
    ""
  ).replace(/\/$/, "");
}

function supabaseApiKey(): string | null {
  return (
    process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY ??
    process.env.SUPABASE_ANON_KEY ??
    process.env.SUPABASE_SERVICE_ROLE_KEY ??
    null
  );
}

export async function POST(request: Request) {
  let body: JsonValue;
  try {
    body = await request.json();
  } catch {
    return mobileJson({ error: "invalid_request", message: "A JSON identity token is required." }, { status: 400 });
  }

  const requestBody = asJsonRecord(body) ?? {};
  const identityToken = firstJsonString(requestBody.identityToken) ?? "";
  const nonce = firstJsonString(requestBody.nonce) ?? undefined;
  const url = supabaseBaseUrl();
  const apiKey = supabaseApiKey();
  if (!identityToken || !url || !apiKey) {
    return mobileJson({ error: "unauthorized", message: "Apple sign-in is not configured on this server." }, { status: 401 });
  }

  try {
    const tokenRequest = {
      provider: "apple",
      id_token: identityToken,
      nonce,
    };
    const response = await fetch(`${url}/auth/v1/token?grant_type=id_token`, {
      method: "POST",
      headers: {
        apikey: apiKey,
        "Content-Type": "application/json",
      },
      body: JSON.stringify(tokenRequest),
      cache: "no-store",
    });
    const payload: JsonValue = await response.json();
    const responseBody = asJsonRecord(payload) ?? {};
    const accessToken = firstJsonString(responseBody.access_token);
    if (!response.ok || !accessToken) {
      return mobileJson({ error: "unauthorized", message: "Supabase rejected the Apple identity token." }, { status: 401 });
    }
    return mobileJson({
      accessToken,
      refreshToken: firstJsonString(responseBody.refresh_token),
      expiresIn: jsonNumber(responseBody.expires_in),
    });
  } catch {
    return mobileJson({ error: "service_unavailable", message: "The authentication service is unavailable." }, { status: 503 });
  }
}
