const SUPABASE_URL =
  process.env.NEXT_PUBLIC_SUPABASE_URL ?? process.env.SUPABASE_URL;
const SUPABASE_ANON_KEY =
  process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY ?? process.env.SUPABASE_ANON_KEY;

export function supabaseConfigured(): boolean {
  return Boolean(SUPABASE_URL && SUPABASE_ANON_KEY);
}

function buildSupabaseHeaders(apiKey: string): HeadersInit {
  const headers: Record<string, string> = { apikey: apiKey };
  // Legacy JWT anon keys use Bearer; sb_publishable_ keys must not — PostgREST
  // tries to parse Bearer as JWT and returns 401 Invalid JWT.
  if (apiKey.startsWith("eyJ")) {
    headers.Authorization = `Bearer ${apiKey}`;
  }
  return headers;
}

/**
 * Read-only PostgREST select. Returns null when Supabase isn't configured or
 * the request fails, so pages can render graceful empty states.
 */
export async function supabaseSelect<T>(
  table: string,
  query: string,
): Promise<T[] | null> {
  if (!SUPABASE_URL || !SUPABASE_ANON_KEY) return null;
  try {
    const res = await fetch(`${SUPABASE_URL}/rest/v1/${table}?${query}`, {
      headers: buildSupabaseHeaders(SUPABASE_ANON_KEY),
      next: { revalidate: 300 },
    });
    if (!res.ok) return null;
    return (await res.json()) as T[];
  } catch {
    return null;
  }
}
