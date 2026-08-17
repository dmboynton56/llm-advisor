-- 011: Private native-app access, device registrations, and notification outbox.
-- The native app is read-only. Only server-side service_role code may use these
-- tables; the mobile API authenticates the user's Supabase access token first.

create table if not exists public.llm_advisor_mobile_access (
  user_id uuid primary key references auth.users(id) on delete cascade,
  enabled boolean not null default true,
  inserted_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table if not exists public.llm_advisor_mobile_devices (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  platform text not null check (platform in ('ios', 'macos')),
  environment text not null default 'production' check (environment in ('sandbox', 'production')),
  device_token text not null,
  notifications_enabled boolean not null default false,
  live_activity_enabled boolean not null default false,
  last_seen_at timestamptz,
  inserted_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  unique (user_id, device_token)
);

create index if not exists idx_llm_advisor_mobile_devices_user
  on public.llm_advisor_mobile_devices(user_id, last_seen_at desc);

create table if not exists public.llm_advisor_mobile_notification_outbox (
  id uuid primary key default gen_random_uuid(),
  event_id text not null unique,
  user_id uuid not null references auth.users(id) on delete cascade,
  category text not null check (category in ('fill', 'safety', 'session')),
  payload jsonb not null default '{}'::jsonb,
  status text not null default 'pending' check (status in ('pending', 'sent', 'failed', 'cancelled')),
  attempts integer not null default 0,
  available_at timestamptz not null default now(),
  delivered_at timestamptz,
  last_error text,
  inserted_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create index if not exists idx_llm_advisor_mobile_outbox_pending
  on public.llm_advisor_mobile_notification_outbox(status, available_at);

alter table public.llm_advisor_mobile_access enable row level security;
alter table public.llm_advisor_mobile_devices enable row level security;
alter table public.llm_advisor_mobile_notification_outbox enable row level security;

revoke all on table public.llm_advisor_mobile_access from anon, authenticated;
revoke all on table public.llm_advisor_mobile_devices from anon, authenticated;
revoke all on table public.llm_advisor_mobile_notification_outbox from anon, authenticated;

grant select, insert, update, delete on table public.llm_advisor_mobile_access to service_role;
grant select, insert, update, delete on table public.llm_advisor_mobile_devices to service_role;
grant select, insert, update, delete on table public.llm_advisor_mobile_notification_outbox to service_role;
