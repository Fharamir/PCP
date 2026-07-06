# Personal Copilot Project (Gemini GenAI & Supabase RAG)

Asynchronous Telegram bot that acts as a personal AI assistant with long-term memory. Powered by the Google GenAI SDK and Supabase (PostgreSQL + pgvector).

## Key Features

* **Multimodal input**: text, voice transcription, and photo analysis through the same agent pipeline.
* **Dual structured output**: one Gemini call returns the user-facing reply and profile operations (upsert/delete) as JSON.
* **Two-layer memory**:
  * *Short-term*: last 5 conversation summaries injected into every prompt.
  * *Long-term (RAG)*: vector search on older memories, only when the message likely references the past.
* **On-demand web search**: a separate Gemini call with Google Search runs only when keywords suggest fresh/live data (news, weather, today, etc.). It cannot be combined with structured JSON in the same request, so search results are injected into the main prompt as context.
* **Original-language storage**: memories and profile values are stored in the user's language; embeddings are multilingual (`gemini-embedding-2`).
* **Security & access**: Supabase RLS blocks direct table access; the bot uses the **anon key** and SECURITY DEFINER RPC functions. Optional allowlist and per-user rate limiting.
* **Resilient async pipeline**: reply first, then background embedding and DB writes; timeouts, retries, and graceful shutdown.

## Tech Stack

* **Language**: Python 3.10+
* **AI**: Google GenAI SDK (`gemini-3.5-flash`, `gemini-3.1-flash-lite`, `gemini-embedding-2`)
* **Database**: Supabase (PostgreSQL + pgvector)
* **Libraries**: `google-genai`, `python-telegram-bot`, `supabase`, `pydantic`, `tenacity`, `python-dotenv`

## Getting Started

### 1. Clone and install

```bash
git clone https://github.com/Fharamir/PCP.git
cd PCP/Telegram
pip install google-genai python-telegram-bot supabase pydantic tenacity python-dotenv
```

### 2. Environment variables

Create `Telegram/accessdata.env`:

```dotenv
SUPABASE_URL="<your-supabase-project-id>"
SUPABASE_KEY="<your-supabase-anon-key>"
GEMINI_API_KEY="<your-gemini-api-key>"
TELEGRAM_BOT_TOKEN="<your-telegram-bot-token>"

# Optional
# ALLOWED_USER_IDS="123456789,987654321"
# MAX_MEMORIES_PER_USER=100
# RATE_LIMIT_SECONDS=3
# GEMINI_TIMEOUT_SECONDS=120
# BOT_DEBUG=true
```

The bot no longer requires the service role key at runtime.

### 3. Database setup (Supabase SQL Editor)

Run in order:

1. **Base schema** (tables + `match_telegram_memories` RPC) — see section below if starting from scratch.
2. **`Telegram/supabase_rls.sql`** — enables RLS, creates `bot_*` RPC functions, grants execute to `anon`.

### 4. Run the bot

```bash
cd Telegram
python Bot_Telegram.py
```

## Message Pipeline

```
User message
  → allowlist + rate limit
  → per-user lock
  → fetch profile + last 5 memories (short-term)
  → [optional] RAG vector search (keyword/heuristic gated)
  → [optional] web search snapshot (keyword gated, separate Gemini call)
  → Gemini structured call (reply + profile ops)
  → send reply to user
  → background: embed interaction → insert memory → prune to MAX_MEMORIES_PER_USER
```

## Model Routing

| Task | Model | Notes |
|------|-------|-------|
| Main reply + profile extraction | `gemini-3.5-flash` (temp 0.4) | Structured JSON output |
| Web search snapshot | `gemini-3.1-flash-lite` (temp 0.1) | Separate call with Google Search |
| Voice transcription | `gemini-3.1-flash-lite` | Original spoken language |
| Embeddings | `gemini-embedding-2` | 3072-dim, original language text |

## Base Schema (first-time setup)

```sql
create extension if not exists vector;

create table telegram_user_profile (
  user_id bigint not null,
  key_name text not null,
  value_data text,
  context_desc text,
  is_sensitive boolean default true,
  created_at timestamptz default now(),
  primary key (user_id, key_name)
);

create table telegram_agent_memories (
  id bigserial primary key,
  user_id bigint not null,
  memory_text text,
  memory_vector vector(3072),
  created_at timestamptz default now()
);

create or replace function match_telegram_memories (
  query_embedding vector(3072),
  match_threshold float,
  match_count int,
  p_user_id bigint
)
returns table (
  id bigint,
  memory_text text,
  similarity float
)
language sql stable
as $$
  select
    tam.id,
    tam.memory_text,
    1 - (tam.memory_vector <=> query_embedding) as similarity
  from telegram_agent_memories tam
  where tam.user_id = p_user_id
    and 1 - (tam.memory_vector <=> query_embedding) > match_threshold
  order by similarity desc
  limit match_count;
$$;
```

Then run `Telegram/supabase_rls.sql`.
