-- PCP Telegram Bot: RLS hardening + RPC access layer
-- Run this entire script in the Supabase SQL Editor after the base schema exists.

-- ---------------------------------------------------------------------------
-- 1. Row Level Security: block direct anon access to tables
-- ---------------------------------------------------------------------------
alter table telegram_user_profile enable row level security;
alter table telegram_agent_memories enable row level security;

drop policy if exists "deny_anon_profile" on telegram_user_profile;
create policy "deny_anon_profile"
  on telegram_user_profile
  for all
  to anon
  using (false);

drop policy if exists "deny_anon_memories" on telegram_agent_memories;
create policy "deny_anon_memories"
  on telegram_agent_memories
  for all
  to anon
  using (false);

-- ---------------------------------------------------------------------------
-- 2. SECURITY DEFINER RPC functions (bot uses anon key + these functions)
-- ---------------------------------------------------------------------------

create or replace function bot_user_profile_exists(p_user_id bigint)
returns boolean
language sql
stable
security definer
set search_path = public
as $$
  select exists(
    select 1
    from telegram_user_profile
    where user_id = p_user_id
    limit 1
  );
$$;

create or replace function bot_get_profile(p_user_id bigint)
returns table (
  key_name text,
  value_data text,
  context_desc text,
  is_sensitive boolean
)
language sql
stable
security definer
set search_path = public
as $$
  select key_name, value_data, context_desc, is_sensitive
  from telegram_user_profile
  where user_id = p_user_id;
$$;

create or replace function bot_upsert_profile_records(p_records jsonb)
returns void
language plpgsql
security definer
set search_path = public
as $$
begin
  insert into telegram_user_profile (user_id, key_name, value_data, context_desc, is_sensitive)
  select
    (r->>'user_id')::bigint,
    r->>'key_name',
    r->>'value_data',
    r->>'context_desc',
    coalesce((r->>'is_sensitive')::boolean, true)
  from jsonb_array_elements(p_records) as r
  on conflict (user_id, key_name) do update set
    value_data = excluded.value_data,
    context_desc = excluded.context_desc,
    is_sensitive = excluded.is_sensitive;
end;
$$;

create or replace function bot_delete_profile_keys(p_user_id bigint, p_key_names text[])
returns void
language sql
security definer
set search_path = public
as $$
  delete from telegram_user_profile
  where user_id = p_user_id
    and key_name = any(p_key_names);
$$;

create or replace function bot_get_recent_memories(p_user_id bigint, p_limit int)
returns table (memory_text text)
language sql
stable
security definer
set search_path = public
as $$
  select tam.memory_text
  from telegram_agent_memories tam
  where tam.user_id = p_user_id
  order by tam.created_at desc
  limit greatest(p_limit, 0);
$$;

create or replace function bot_insert_memory(
  p_user_id bigint,
  p_memory_text text,
  p_memory_vector vector(3072)
)
returns void
language sql
security definer
set search_path = public
as $$
  insert into telegram_agent_memories (user_id, memory_text, memory_vector)
  values (p_user_id, p_memory_text, p_memory_vector);
$$;

create or replace function bot_delete_all_memories(p_user_id bigint)
returns void
language sql
security definer
set search_path = public
as $$
  delete from telegram_agent_memories
  where user_id = p_user_id;
$$;

create or replace function bot_prune_memories(p_user_id bigint, p_max_count int)
returns void
language sql
security definer
set search_path = public
as $$
  delete from telegram_agent_memories
  where id in (
    select id
    from telegram_agent_memories
    where user_id = p_user_id
    order by created_at desc
    offset greatest(p_max_count, 0)
  );
$$;

-- ---------------------------------------------------------------------------
-- 3. Grants for anon role (used by the bot with SUPABASE_KEY)
-- ---------------------------------------------------------------------------
revoke all on function bot_user_profile_exists(bigint) from public;
revoke all on function bot_get_profile(bigint) from public;
revoke all on function bot_upsert_profile_records(jsonb) from public;
revoke all on function bot_delete_profile_keys(bigint, text[]) from public;
revoke all on function bot_get_recent_memories(bigint, int) from public;
revoke all on function bot_insert_memory(bigint, text, vector) from public;
revoke all on function bot_delete_all_memories(bigint) from public;
revoke all on function bot_prune_memories(bigint, int) from public;

grant execute on function bot_user_profile_exists(bigint) to anon;
grant execute on function bot_get_profile(bigint) to anon;
grant execute on function bot_upsert_profile_records(jsonb) to anon;
grant execute on function bot_delete_profile_keys(bigint, text[]) to anon;
grant execute on function bot_get_recent_memories(bigint, int) to anon;
grant execute on function bot_insert_memory(bigint, text, vector) to anon;
grant execute on function bot_delete_all_memories(bigint) to anon;
grant execute on function bot_prune_memories(bigint, int) to anon;

grant execute on function match_telegram_memories(vector, double precision, integer, bigint) to anon;
