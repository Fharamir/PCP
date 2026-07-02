# Personal Copilot Project (Gemini GenAI & Supabase RAG)

This project is a high-performance, asynchronous Telegram bot that acts as a personal AI assistant with long-term memory capabilities. It features an advanced cognitive architecture powered by the latest Google Gemini GenAI SDK and uses Supabase for structured data persistence and multimodal vector search (RAG).

## 🌟 Key Features

*   **Next-Gen AI Core**: Migrated entirely to the official, unified **Google GenAI SDK** using advanced models (`gemini-3.5-flash`, `gemini-3.1-flash-lite`, and `gemini-embedding-2`).
*   **Multimodal Capabilities**: Native support for **Voice Message Transcription** and **Photo/Image Analysis** directly through the bot interface.
*   **Real-Time Web Search (Grounding)**: Dynamic integration with Google Search. The bot can independently browse the internet to provide up-to-date answers when required.
*   **Dual Structured Output**: Utilizes Pydantic schemas and Gemini's native JSON output to generate human-facing text and extract database operations (profile changes/GDPR deletions) simultaneously in a single, atomic API call.
*   **Long-Term Vector Memory (RAG)**: Generates 3072-dimensional embeddings via `gemini-embedding-2` to store summaries of past interactions and retrieve them semantically via PostgreSQL's `pgvector`.
*   **Asynchronous & Resilient Architecture**: Non-blocking network I/O with `asyncio`, isolated database write operations inside parallel background threads (`ThreadPoolExecutor`), and exponential backoff retry-logic via `tenacity` protecting against 503 Overload errors.

## 🛠️ Tech Stack

*   **Language**: Python 3.10+
*   **AI Engine**: Google GenAI SDK (Unified Client API)
*   **Database**: Supabase (PostgreSQL + `pgvector` extension)
*   **Key Python Libraries**:
    *   `google-genai` (Official & Unified Google SDK)
    *   `python-telegram-bot` (v20+ Native Async)
    *   `supabase` (Python Client)
    *   `pydantic` (v2+ Data Validation)
    *   `tenacity` (Advanced Retrying)
    *   `python-dotenv`

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/Fharamir/PCP.git
cd PCP
```

### 2. Install Dependencies
Install the required up-to-date Python packages:
```bash
pip install google-genai python-telegram-bot supabase pydantic tenacity python-dotenv
```

### 3. Configure Environment Variables
Create a file named `accessdata.env` in the Telegram directory and fill it with your credentials:
```dotenv
SUPABASE_URL="<your-supabase-project-id>"
SUPABASE_KEY="<your-supabase-anon-key>"
SUPABASE_SERVICE_KEY="<your-supabase-service-role-key>"
GEMINI_API_KEY="<your-gemini-api-key>"
TELEGRAM_BOT_TOKEN="<your-telegram-bot-token>"
```

### 4. Database Setup (Supabase)
Go to the **SQL Editor** in your Supabase dashboard and run the following script. This configures the schema for structured preferences and maps the 3072-dimensional space required for `gemini-embedding-2`.

```sql
-- 1. Enable the vector extension if not already enabled
create extension if not exists vector;

-- 2. Create the table for Telegram user profile data
create table telegram_user_profile (
  user_id bigint not null,
  key_name text not null,
  value_data text,
  context_desc text,
  is_sensitive boolean default true,
  created_at timestamptz default now(),
  primary key (user_id, key_name)
);

-- 3. Create the table for Telegram long-term vector memory
create table telegram_agent_memories (
  id bigserial primary key,
  user_id bigint not null,
  memory_text text,
  memory_vector vector(3072),
  created_at timestamptz default now()
);

-- 4. Create the RPC function for semantic search (Cosine Distance)
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
as \[   select     tam.id,     tam.memory_text,     1 - (tam.memory_vector <=> query_embedding) as similarity   from telegram_agent_memories tam   where tam.user_id = p_user_id and 1 - (tam.memory_vector <=> query_embedding) > match_threshold   order by similarity desc   limit match_count; \];
```

## 💻 Usage

Run the asynchronous bot interface:
```bash
python Bot_Telegram.py
```

### Cognitive Model Routing Protocol (Internal Logic)
The bot optimizes token usage and costs by routing different tasks to specific Gemini architectures:
1.  **`gemini-3.1-flash-lite`** (Temp 0.1): Handles translation queries, conversation summaries, database operations, and text transcriptions.
2.  **`gemini-3.5-flash`** (Temp 0.4): Handles core interactions, multimodal context analysis (vision), agentic reasoning, and grounding with Google Search.
3.  **`gemini-embedding-2`**: Native multimodal 3072-dimension mapping for semantic long-term memory storage.
