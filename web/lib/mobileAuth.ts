import { NextResponse } from "next/server";
import { asJsonRecord, firstJsonString } from "@/lib/json";
import { verifyMobileDemoToken } from "@/lib/mobileDemoAuth";
import type { JsonValue } from "@/lib/types";

type MobileUser = {
  id: string;
  email?: string | null;
};

type AuthResult =
  | { user: MobileUser; mode: "supabase" | "demo" }
  | { response: NextResponse };

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

function unauthorized(message = "A valid Supabase session is required.") {
  return NextResponse.json(
    { error: "unauthorized", message },
    { status: 401, headers: { "WWW-Authenticate": "Bearer" } },
  );
}

/**
 * Verify the caller's Supabase access token server-side. The native client
 * never receives a service-role key; the API keeps the existing read-only
 * Supabase access behind this boundary.
 */
export async function requireMobileUser(request: Request): Promise<AuthResult> {
  const authorization = request.headers.get("authorization") ?? "";
  const match = /^Bearer\s+(.+)$/i.exec(authorization.trim());
  if (match) {
    const demoSession = verifyMobileDemoToken(match[1]);
    if (demoSession) {
      return {
        user: {
          id: demoSession.subject,
          email: null,
        },
        mode: "demo",
      };
    }
  }
  const url = supabaseBaseUrl();
  const apiKey = supabaseApiKey();
  if (!match || !url || !apiKey) return { response: unauthorized() };

  try {
    const response = await fetch(`${url}/auth/v1/user`, {
      headers: {
        apikey: apiKey,
        Authorization: `Bearer ${match[1]}`,
      },
      cache: "no-store",
    });
    if (!response.ok) return { response: unauthorized() };

    const userPayload: JsonValue = await response.json();
    const userRecord = asJsonRecord(userPayload);
    const id = firstJsonString(userRecord?.id);
    if (!id) return { response: unauthorized() };
    const user: MobileUser = {
      id,
      email: firstJsonString(userRecord?.email),
    };

    const allowList = (
      process.env.MOBILE_ALLOWED_USER_IDS ??
      process.env.MOBILE_ALLOWED_USER_ID ??
      ""
    )
      .split(",")
      .map((value) => value.trim())
      .filter(Boolean);
    if (allowList.length > 0 && !allowList.includes(user.id)) {
      return {
        response: NextResponse.json(
          { error: "forbidden", message: "This account is not enabled for the private mobile app." },
          { status: 403 },
        ),
      };
    }

    return { user, mode: "supabase" };
  } catch {
    return { response: unauthorized("The authentication service is unavailable.") };
  }
}

export function mobileJson<T>(payload: T, init?: ResponseInit): NextResponse<T> {
  const headers = new Headers(init?.headers);
  headers.set("Cache-Control", "no-store");
  return NextResponse.json(payload, {
    ...init,
    headers,
  });
}
