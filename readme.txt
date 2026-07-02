# Personal Memory Bot

This project is a Python-based conversational AI agent that demonstrates long-term memory capabilities. It uses Google's Gemini models for its core intelligence and Supabase as a backend for data persistence, including a vector database for semantic memory retrieval (RAG).

## Features

*   **Telegram Bot Interface**: Interact with the agent directly through Telegram.
*   **Long-Term Memory (RAG)**: The agent can remember past conversations by storing summaries in a vector database and retrieving relevant information semantically.
*   **User Profile Management**: The agent can learn and recall user preferences (e.g., name, favorite color) and stores them in a structured profile.
*   **Structured Dual Output**: Uses Pydantic and a single Gemini call to generate a user response and extract database actions simultaneously, improving performance and reducing API calls.
*   **API Resilience**: Automatically retries API calls to Gemini using an exponential backoff strategy to handle temporary service unavailability.
*   **Secure Credential Management**: Keeps API keys and credentials separate from the source code using a `.env` file.
*   **Asynchronous & Performant**: Built with `asyncio` and `python-telegram-bot`, with database operations running in background threads to ensure the bot remains responsive.

## Tech Stack

*   **Language**: Python 3
*   **AI Model**: Google Gemini (`gemini-3.5-flash`, `gemini-3.1-flash-lite`, `gemini-embedding-2`)
*   **Database**: Supabase (PostgreSQL with `pgvector` extension for RAG)
*   **Key Python Libraries**:
    *   `google-generativeai`
    *   `python-telegram-bot`
    *   `supabase`
    *   `pydantic`
    *   `tenacity`
    *   `python-dotenv`

## 🚀 Getting Started

Follow these steps to get the project running.

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 2. Install Dependencies

Install the required Python packages using pip:

```bash
pip install google-generativeai python-telegram-bot supabase pydantic tenacity python-dotenv
```

### 3. Set up Supabase

This project uses two distinct database schemas: one for the command-line test script (`test_memoria.py`) and one for the Telegram Bot (`Bot_Telegram.py`).

### 4. Configure Environment Variables

1.  Rename the `accessdata.env.example` file (or create a new file) to `accessdata.env`.
2.  Fill it with your credentials. You will need your Supabase Project URL, Key, and your Google Gemini API Key.

    ```dotenv
    SUPABASE_URL="<your-supabase-project-url>"
    SUPABASE_KEY="<your-supabase-anon-key>"
    GEMINI_API_KEY="<your-gemini-api-key>"
    EMAIL_CLIENTE="<your-supabase-auth-email>"
    PASSWORD_SICURA="<your-supabase-auth-password>"
    TELEGRAM_BOT_TOKEN="<your-telegram-bot-token>"
    SUPABASE_SERVICE_KEY="<your-supabase-service-role-key>"
    ```

## 5. Usage

### Running the Telegram Bot

1.  **Database Setup**: Go to the **SQL Editor** in your Supabase project and run the following script. This will create the tables and functions required by the Telegram bot.

    ```sql
    -- 1. Enable the vector extension if not already enabled
    create extension if not exists vector;

    -- 2. Create the table for Telegram user profile data
    create table telegram_user_profile (
      user_id bigint not null, -- Telegram uses integer IDs
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
      user_id bigint not null, -- Telegram uses integer IDs
      memory_text text,
      memory_vector vector(3072), -- Use 3072 for 'gemini-embedding-2'
      created_at timestamptz default now()
    );

    -- 4. Create the RPC function for semantic search
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
      where tam.user_id = p_user_id and 1 - (tam.memory_vector <=> query_embedding) > match_threshold
      order by similarity desc
      limit match_count;
    $$;
    ```
2.  **Run the Bot**:
    ```bash
    python Bot_Telegram.py
    ```