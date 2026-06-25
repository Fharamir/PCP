# Personal Copilot Project

This project is a Python-based conversational AI agent that demonstrates long-term memory capabilities. It uses Google's Gemini models for its core intelligence and Supabase as a backend for data persistence, including a vector database for semantic memory retrieval (RAG).

## Features

*   **Conversational Loop**: An interactive command-line interface to chat with the agent.
*   **Long-Term Memory (RAG)**: The agent can remember past conversations by storing summaries in a vector database and retrieving relevant information semantically.
*   **User Profile Management**: The agent can learn and recall user preferences (e.g., name, favorite color) and stores them in a structured profile.
*   **Structured Data Extraction**: Uses Pydantic and Gemini's function calling to reliably extract information from user input and manage memory.
*   **API Resilience**: Automatically retries API calls to Gemini using an exponential backoff strategy to handle temporary service unavailability.
*   **Secure Credential Management**: Keeps API keys and credentials separate from the source code using a `.env` file.

## Tech Stack

*   **Language**: Python 3
*   **AI Model**: Google Gemini (`gemini-2.5-flash-lite`, `gemini-2.5-flash`, `gemini-embedding-2`)
*   **Database**: Supabase (PostgreSQL with `pgvector` extension)
*   **Key Python Libraries**:
    *   `google-generativeai`
    *   `supabase`
    *   `pydantic`
    *   `tenacity`
    *   `python-dotenv`

## Setup

Follow these steps to get the project running.

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd <repository-folder>
```

### 2. Install Dependencies

Install the required Python packages using pip:

```bash
pip install google-generativeai supabase pydantic tenacity python-dotenv
```

### 3. Set up Supabase

1.  Create a new project on Supabase.
2.  Go to the **SQL Editor** and run the following queries to set up the database:

    ```sql
    -- 1. Enable the vector extension
    create extension if not exists vector;

    -- 2. Create the table for user profile data
    create table user_profile (
      user_id uuid not null,
      key_name text not null,
      value_data text,
      context_desc text,
      is_sensitive boolean default true,
      created_at timestamptz default now(),
      primary key (user_id, key_name)
    );

    -- 3. Create the table for long-term vector memory
    create table agent_memories (
      id bigserial primary key,
      user_id uuid not null,
      memory_text text,
      memory_vector vector(768), -- Corresponds to gemini-embedding-2
      created_at timestamptz default now()
    );
    ```

### 4. Configure Environment Variables

1.  Rename the `accessdata.env.example` file (or create a new file) to `accessdata.env`.
2.  Fill it with your credentials. You will need your Supabase Project URL, Key, and your Google Gemini API Key.

    ```dotenv
    SUPABASE_URL="<your-supabase-project-url>"
    SUPABASE_KEY="<your-supabase-anon-key>"
    GEMINI_API_KEY="<your-gemini-api-key>"
    EMAIL_CLIENTE="<your-supabase-auth-email>"
    PASSWORD_SICURA="<your-supabase-auth-password>"
    ```

## Usage

Once the setup is complete, you can run the agent from your terminal:

```bash
python test_memoria.py
```

The script will start an interactive session where you can chat with the agent. Type `exit` to end the session.