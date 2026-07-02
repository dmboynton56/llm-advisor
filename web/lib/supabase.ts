const SUPABASE_URL =
  process.env.NEXT_PUBLIC_SUPABASE_URL ?? process.env.SUPABASE_URL;
// Server components only — service role is never NEXT_PUBLIC_*.
const SUPABASE_API_KEY =
  process.env.SUPABASE_SERVICE_ROLE_KEY ??
  process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY ??
  process.env.SUPABASE_ANON_KEY;

export function supabaseConfigured(): boolean {
  return Boolean(SUPABASE_URL && SUPABASE_API_KEY);
}

function buildSupabaseHeaders(apiKey: string): HeadersInit {
  const headers: Record<string, string> = { apikey: apiKey };
  // Legacy JWT anon/service_role keys use Bearer; sb_publishable_ keys must not —
  // PostgREST tries to parse Bearer as JWT and returns 401 Invalid JWT.
  if (apiKey.startsWith("eyJ")) {
    headers.Authorization = `Bearer ${apiKey}`;
  }
  return headers;
}

/** Cheap probe when pages render empty but env vars are set. */
export async function checkSupabaseAccess(): Promise<
  { ok: true } | { ok: false; status: number } | null
> {
  if (!SUPABASE_URL || !SUPABASE_API_KEY) return null;
  try {
    const res = await fetch(
      `${SUPABASE_URL}/rest/v1/llm_advisor_backtest_runs?select=run_date&limit=1`,
      { headers: buildSupabaseHeaders(SUPABASE_API_KEY), cache: "no-store" },
    );
    return res.ok ? { ok: true } : { ok: false, status: res.status };
  } catch {
    return { ok: false, status: 0 };
  }
}

/**
 * Read-only PostgREST select. Returns null when Supabase isn't configured or
 * the request fails, so pages can render graceful empty states.
 */
export async function supabaseSelect<T>(
  table: string,
  query: string,
): Promise<T[] | null> {
  if (!SUPABASE_URL || !SUPABASE_API_KEY) return null;
  try {
    const res = await fetch(`${SUPABASE_URL}/rest/v1/${table}?${query}`, {
      headers: buildSupabaseHeaders(SUPABASE_API_KEY),
      cache: "no-store",
    });
    if (!res.ok) return null;
    return (await res.json()) as T[];
  } catch {
    return null;
  }
}
