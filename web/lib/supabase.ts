const SUPABASE_URL = normalizeSupabaseUrl(
  process.env.NEXT_PUBLIC_SUPABASE_URL ?? process.env.SUPABASE_URL ?? "",
);
// Server components only — service role is never NEXT_PUBLIC_*.
const SUPABASE_API_KEY =
  process.env.SUPABASE_SERVICE_ROLE_KEY ??
  process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY ??
  process.env.SUPABASE_ANON_KEY;
const SUPABASE_FETCH_TIMEOUT_MS = Number(
  process.env.SUPABASE_FETCH_TIMEOUT_MS ?? 8000,
);

async function fetchWithTimeout(
  input: RequestInfo | URL,
  init?: RequestInit,
): Promise<Response> {
  const controller = new AbortController();
  const timeout = setTimeout(
    () => controller.abort(),
    Number.isFinite(SUPABASE_FETCH_TIMEOUT_MS) && SUPABASE_FETCH_TIMEOUT_MS > 0
      ? SUPABASE_FETCH_TIMEOUT_MS
      : 8000,
  );
  const upstreamSignal = init?.signal;
  const abortUpstream = () => controller.abort();
  upstreamSignal?.addEventListener("abort", abortUpstream, { once: true });

  try {
    return await fetch(input, { ...init, signal: controller.signal });
  } finally {
    clearTimeout(timeout);
    upstreamSignal?.removeEventListener("abort", abortUpstream);
  }
}

function normalizeSupabaseUrl(url: string): string {
  return url.replace(/\/rest\/v1\/?$/i, "").replace(/\/$/, "");
}

export function supabaseConfigured(): boolean {
  return Boolean(SUPABASE_URL && SUPABASE_API_KEY);
}

function buildSupabaseHeaders(apiKey: string): HeadersInit {
  const headers = new Headers({ apikey: apiKey });
  // Legacy JWT anon/service_role keys use Bearer; sb_publishable_ keys must not —
  // PostgREST tries to parse Bearer as JWT and returns 401 Invalid JWT.
  if (apiKey.startsWith("eyJ")) {
    headers.set("Authorization", `Bearer ${apiKey}`);
  }
  return headers;
}

/** Cheap probe when pages render empty but env vars are set. */
export async function checkSupabaseAccess(): Promise<
  { ok: true } | { ok: false; status: number } | null
> {
  if (!SUPABASE_URL || !SUPABASE_API_KEY) return null;
  try {
    const res = await fetchWithTimeout(
      `${SUPABASE_URL}/rest/v1/llm_advisor_backtest_runs?select=run_date&limit=1`,
      {
        headers: buildSupabaseHeaders(SUPABASE_API_KEY),
        cache: "no-store",
      },
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
    const res = await fetchWithTimeout(`${SUPABASE_URL}/rest/v1/${table}?${query}`, {
      headers: buildSupabaseHeaders(SUPABASE_API_KEY),
      cache: "no-store",
    });
    if (!res.ok) return null;
    const rows: T[] = await res.json();
    return rows;
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
      const res = await fetchWithTimeout(
        `${SUPABASE_URL}/rest/v1/${table}?${query}${separator}limit=${pageSize}&offset=${offset}`,
        {
          headers: buildSupabaseHeaders(SUPABASE_API_KEY),
          cache: "no-store",
        },
      );
      if (!res.ok) return null;
      const page: T[] = await res.json();
      rows.push(...page);
      if (page.length < pageSize) break;
    }
    return rows;
  } catch {
    return null;
  }
}
