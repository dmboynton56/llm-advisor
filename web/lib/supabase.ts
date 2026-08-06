const SUPABASE_URL = normalizeSupabaseUrl(
  process.env.NEXT_PUBLIC_SUPABASE_URL ?? process.env.SUPABASE_URL ?? "",
);
// Server components only — service role is never NEXT_PUBLIC_*.
const SUPABASE_API_KEY =
  process.env.SUPABASE_SERVICE_ROLE_KEY ??
  process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY ??
  process.env.SUPABASE_ANON_KEY;

function normalizeSupabaseUrl(url: string): string {
  return url.replace(/\/rest\/v1\/?$/i, "").replace(/\/$/, "");
}

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

/**
 * Read a large ordered result set without relying on PostgREST's default row
 * limit. This is intentionally server-side and read-only, like
 * supabaseSelect().
 */
export async function supabaseSelectPaged<T>(
  table: string,
  query: string,
  pageSize = 1000,
  maxRows = 50_000,
): Promise<T[] | null> {
  if (!SUPABASE_URL || !SUPABASE_API_KEY) return null;
  const rows: T[] = [];
  try {
    for (let offset = 0; offset < maxRows; offset += pageSize) {
      const separator = query ? "&" : "";
      const res = await fetch(
        `${SUPABASE_URL}/rest/v1/${table}?${query}${separator}limit=${pageSize}&offset=${offset}`,
        {
          headers: buildSupabaseHeaders(SUPABASE_API_KEY),
          cache: "no-store",
        },
      );
      if (!res.ok) return null;
      const page = (await res.json()) as T[];
      rows.push(...page);
      if (page.length < pageSize) break;
    }
    return rows;
  } catch {
    return null;
  }
}
