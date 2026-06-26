"""
Test script for a personal AI assistant with long-term memory.

This script implements a conversational agent that uses Supabase for data persistence
(user profile and vector memory) and Google Gemini APIs for generative and embedding
capabilities. It includes caching mechanisms, API call retries, and an architecture
for structured memory management.
"""
import os
from typing import List, Literal
import asyncio
from google import genai
from google.genai import errors, types
from pydantic import BaseModel, Field
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from supabase import create_client, Client, ClientOptions
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
import concurrent.futures
from dotenv import load_dotenv

# --- Configuration and Constants ---
load_dotenv(dotenv_path='accessdata.env') # Load environment variables from the specified file
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
EMAIL_CLIENTE = os.getenv("EMAIL_CLIENTE")
PASSWORD_SICURA = os.getenv("PASSWORD_SICURA")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
MAX_STORAGE = 5

# --- Advanced Client Configuration for Resilience ---
# By specifying a regional endpoint in the Client constructor, we avoid global
google_client = genai.Client(api_key=GEMINI_API_KEY)

COMPLETE_SUPABASE_URL = f"https://{SUPABASE_URL}.supabase.co"

class AgentCache:
    """In-memory cache to temporarily store the user profile and reduce DB calls."""
    _storage = {}
    @classmethod
    def get_profile(cls, user_id): return cls._storage.get(user_id, None)
    @classmethod
    def set_profile(cls, user_id, data): cls._storage[user_id] = data
    @classmethod
    def invalidate(cls, user_id):
        if user_id in cls._storage: del cls._storage[user_id]

# --- Pydantic Data Models for Structured Extraction ---
class SinglePreference(BaseModel):
    key_name: str = Field(description="The specific category and entity in snake_case and strictly in English (e.g., max_budget_motorcycle).")
    value_data: str = Field(description="The extracted value translated to English. For DELETE actions, this can be empty.")
    context_desc: str = Field(description="The full context in English. For DELETE actions, describe what is being deleted.")
    action_type: Literal["upsert", "delete"] = Field(description="Use 'upsert' to save/update data. Use 'delete' ONLY if requested.")
    is_sensitive: bool = Field(description="True if the data involves personal, financial, private, or sensitive information (e.g., IBAN, health, tastes). False if it's a generic technical setting or app configuration.")
    memory_target: Literal["profile", "chat_memories", "all"] = Field(description="Only in case of deletion, use 'profile' to affect user_profile table. Use 'chat_memories' to delete recent semantic memories/chat logs. Use 'all' if the user wants to completely wipe out both profile data and chat memories.")

class ProfileUpdateBatch(BaseModel):
    preferences: List[SinglePreference] = Field(description="List of all profile preferences found.")

class GeminiDualOutput(BaseModel):
    """Model for a dual output containing both the user-facing response and the data operations."""
    assistant_response: str = Field(description="The final, user-facing response, written in the correct language.")
    preferences: List[SinglePreference] = Field(description="List of all profile preferences or data management actions found.")

def log_retry_attempt(retry_state):
    """Log the retry attempt number and the exception that caused it."""
    print(
        f"⚠️ Retrying API call... "
        f"Attempt #{retry_state.attempt_number}, "
        f"Exception: {retry_state.outcome.exception()}"
    )

@retry(
    retry=retry_if_exception_type((errors.APIError, errors.ServerError, errors.ClientError)),
    wait=wait_exponential(multiplier=2, min=2, max=20), 
    stop=stop_after_attempt(3),
    reraise=True,
    before_sleep=log_retry_attempt
)
def call_gemini_with_retry(model, prompt, system_prompt, schema_output=None, mime_type=None):
    """Makes a call to a Gemini model with automatic retry handling for API errors."""
    config_params = {
        "system_instruction": system_prompt,
        "temperature": 0.1,
        "response_schema": schema_output,
        "response_mime_type": mime_type
    }
    response = google_client.models.generate_content(model=model, contents=prompt, config=config_params)
    return response.text

@retry(
    retry=retry_if_exception_type((errors.APIError, errors.ServerError, errors.ClientError)),
    wait=wait_exponential(multiplier=2, min=2, max=20), 
    stop=stop_after_attempt(3),
    reraise=True,
    before_sleep=log_retry_attempt
)
def generate_embedding_with_retry(text_to_embed):
    """Generates text embedding with automatic retry handling, specific to embedding APIs."""
    response = google_client.models.embed_content(
        model="gemini-embedding-2",
        contents=text_to_embed
    )
    return response.embeddings[0].values

def get_authenticated_client(email, password):
    """Authenticates the user on Supabase and returns an authenticated client and the user ID."""
    complete_url = "https://" + SUPABASE_URL + ".supabase.co"
    client_base = create_client(complete_url, SUPABASE_KEY)
    session = client_base.auth.sign_in_with_password({"email": email, "password": password})
    client_options = ClientOptions(headers={"Authorization": f"Bearer {session.session.access_token}"})
    return create_client(complete_url, SUPABASE_KEY, options=client_options), session.user.id

def search_past_memories(client_sb, user_id, search_text):
    """Performs a semantic search (RAG) in the user's long-term memory on Supabase."""
    try:
        res_trad = call_gemini_with_retry(
            model='gemini-3.5-flash', # STRATEGY: Use the main, more stable model to avoid congestion on 'lite' versions.
            prompt=f"Translate this search query into a concise English keywords sentence: '{search_text}'",
            system_prompt="Translate the user's search intent into a concise English keyword-based sentence."
        )
        query_inglese = res_trad.strip()
		
        vettore = generate_embedding_with_retry(query_inglese)

        risposta_db = client_sb.rpc("match_memories", {
            "query_embedding": vettore,
            "match_threshold": 0.4,
            "match_count": 3,
            "p_user_id": user_id
        }).execute()
        return [riga['memory_text'] for riga in risposta_db.data]
    except Exception as e:
        print(f"⚠️ Vector search error: {e}")
        return []

def _get_chat_history(client_sb, user_id):
    """Fetches the recent chat history from Supabase."""
    try:
        response = client_sb.table("agent_memories").select("memory_text").eq("user_id", user_id).order("created_at", ascending=False).limit(MAX_STORAGE).execute()
        history = [row['memory_text'] for row in response.data]
        history.reverse()
        return history
    except Exception:
        return []

def _get_user_profile(client_sb, user_id):
    """Fetches the user profile, using cache if available."""
    profile = AgentCache.get_profile(user_id)
    if profile is None:
        try:
            response = client_sb.table("user_profile").select("key_name", "value_data", "context_desc", "is_sensitive").execute()
            profile = response.data
            AgentCache.set_profile(user_id, profile)
        except Exception:
            profile = []
    return profile

def _execute_background_tasks(client_sb, user_id, tasks):
    """Executes database write operations in a parallel background thread."""
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Task 1: Delete chat memories if requested
        if tasks.get("delete_chat_memories"):
            print("🔂 [BACKGROUND-DB] Submitting task: Delete all vector memories...")
            executor.submit(client_sb.table("agent_memories").delete().eq("user_id", user_id).execute)

        # Task 2: Delete profile keys if requested
        if tasks.get("profile_keys_to_delete"):
            keys = list(set(tasks["profile_keys_to_delete"]))
            print(f"🔂 [BACKGROUND-DB] Submitting task: Delete {len(keys)} profile keys...")
            executor.submit(client_sb.table("user_profile").delete().eq("user_id", user_id).in_("key_name", keys).execute)

        # Task 3: Upsert profile keys if requested
        if tasks.get("profile_records_to_upsert"):
            records = tasks["profile_records_to_upsert"]
            print(f"🔂 [BACKGROUND-DB] Submitting task: Upsert {len(records)} profile records...")
            executor.submit(client_sb.table("user_profile").upsert(records, on_conflict="user_id,key_name").execute)

        # Task 4: Save the current conversation to long-term memory
        if tasks.get("current_interaction_embedding"):
            embedding_data = tasks["current_interaction_embedding"]
            print("🔂 [BACKGROUND-DB] Submitting task: Save current interaction to memory...")
            executor.submit(client_sb.table("agent_memories").insert(embedding_data).execute)

    AgentCache.invalidate(user_id)

def process_input(client_sb, user_id, user_input):
    """Orchestrates the entire response process: retrieves context, updates memory, and generates a response."""
    action_list_summary = " no specific database operations were performed in this turn." # Safety initialization

    # --- Parallel Context Fetching ---
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_history = executor.submit(_get_chat_history, client_sb, user_id)
        future_profile = executor.submit(_get_user_profile, client_sb, user_id)
        future_memories = executor.submit(search_past_memories, client_sb, user_id, user_input)
        chat_history = future_history.result()
        current_profile = future_profile.result()
        relevant_memories = future_memories.result()

    try:
        # Lists to accumulate records for bulk operations.
        records_to_upsert = []
        keys_to_delete = []
        delete_chat_memory = False
        final_response = ""

        def process_preferences(preferences):
            nonlocal records_to_upsert, keys_to_delete, delete_chat_memory
            for preference_item in preferences:
                # Case 1: Handle deletion requests.
                if preference_item.action_type == "delete":
                    if preference_item.memory_target in ["chat_memories", "all"] and not delete_chat_memory:
                        print(f"🗑️ [GDPR PURGE] Deletion request detected for semantic chat memories.")
                        delete_chat_memory = True
                    if preference_item.memory_target in ["profile", "all"]:
                        if preference_item.memory_target == "all":
                            print("🗑️ [GDPR PURGE] Profile reset scheduled (system keys excluded).")
                            keys_to_delete.extend([r['key_name'] for r in current_profile if r.get('is_sensitive', False)])
                        else:
                            old_record = next((row for row in current_profile if row['key_name'] == preference_item.key_name), None)
                            if old_record and not old_record.get('is_sensitive', False):
                                print(f"🛡️ [SECURITY] Deletion denied for system key: '{preference_item.key_name}'")
                            else:
                                keys_to_delete.append(preference_item.key_name)
                # Case 2: Handle save/update requests.
                elif preference_item.action_type == "upsert":
                    old_record = next((row for row in current_profile if row['key_name'] == preference_item.key_name), None)
                    write_needed = True
                    if old_record and old_record['value_data'] == str(preference_item.value_data) and old_record['context_desc'] == preference_item.context_desc:
                        write_needed = False
                    if write_needed and preference_item.value_data:
                        records_to_upsert.append({
                            "user_id": user_id, 
                            "key_name": preference_item.key_name, 
                            "value_data": preference_item.value_data, 
                            "context_desc": preference_item.context_desc,
                            "is_sensitive": preference_item.is_sensitive
                        })

        language_setting = next((row['value_data'] for row in current_profile if row['key_name'] == 'user_language'), 'en')

        # 1. UNIFIED CALL: Generate response and extract data in a single API call.
        unified_prompt = f"""
        You are a professional personal AI assistant. Your task is to perform two actions in one go:
        1.  Generate a helpful, user-facing response.
        2.  Provide a JSON object detailing any data operations based on the user's prompt.

        **RESPONSE GENERATION RULES:**
        - You MUST reply strictly in the language with this ISO 639-1 code: {language_setting}.
        - Your response should be natural, conversational, and based on all the context provided.
        - You are working on behalf of the Personal Copilot Project.
        - If the data extraction operation finds someting to update on the user profile (upsert or delete), inform the user on what your're planning to do without technical words if possibile

        **DATA EXTRACTION RULES:**
        - After writing the user response, you MUST output a JSON object that conforms to the provided schema.
        - All output in the JSON must be in ENGLISH.
        - If the user expresses a preference, use `action_type: "upsert"`.
        - If the user asks to delete or forget data:
            - For a SPECIFIC item (e.g., "forget my favorite color"), use `action_type: "delete"` and specify the `key_name`.
            - For their PROFILE data (e.g., "delete my personal data"), use `action_type: "delete"` with `memory_target: "profile"`.
            - For CHAT HISTORY, use `action_type: "delete"` with `memory_target: "chat_memories"`.
            - For EVERYTHING, use `action_type: "delete"` with `memory_target: "all"`.
        - If no preferences or commands are found, the "preferences" list in the JSON must be empty.
        - Select operations only related to the user prompt, not to the provided context. Use the context only to determine if the new prompt contains a new preference to save or data to forget.

        **CONTEXT FOR YOUR RESPONSE AND ANALYSIS:**
        - USER PROFILE KEYS: {current_profile}
        - RECENT CHAT LOGS: {chat_history}
        - RETRIEVED LONG-TERM MEMORIES (RAG): {relevant_memories}
        """

        json_response = call_gemini_with_retry(
            model='gemini-3.5-flash',
            prompt=user_input,
            system_prompt=unified_prompt,
            schema_output=GeminiDualOutput,
            mime_type="application/json"
        )
        
        dual_output = GeminiDualOutput.model_validate_json(json_response)
        final_response = dual_output.assistant_response

        if dual_output.preferences:
            process_preferences(dual_output.preferences)
            
        azioni_eseguite = []
        if delete_chat_memory:
            azioni_eseguite.append("will delete all recent chat memories")
        if keys_to_delete:
            azioni_eseguite.append(f"will delete {len(set(keys_to_delete))} settings from profile")
        if records_to_upsert:
            azioni_eseguite.append(f"will save/update {len(records_to_upsert)} preferences in profile")
        
        if azioni_eseguite:
            action_list_summary = " " + ", and you ".join(azioni_eseguite) + "."
            print(f"✅ [DB-PREP] Actions prepared for background execution:{action_list_summary}")
        else:
            print("⏸️ [BACKGROUND SKIP] No preferences detected.")

        print(f"\n💬 User Input: '{user_input}'")
        print(f"🤖 Assistant Response ({language_setting}):\n{final_response}\n" + "-"*60)

        # Prepare text for long-term memory embedding
        interaction_text = f"User said: '{user_input}' | Assistant replied: '{final_response}'"

        # Background Task: Translate and summarize only if the conversation is not already in English.
        if language_setting != 'en':
            english_interaction = call_gemini_with_retry(
                model='gemini-3.5-flash', # STRATEGY: Use the main, more stable model to avoid congestion on 'lite' versions.
                prompt=f"Summarize this interaction into a concise English fact: {interaction_text}",
                system_prompt="Summarize interaction into clean facts."
            )
        else:
            english_interaction = interaction_text
        
        interaction_vector = generate_embedding_with_retry(english_interaction)

        # 2. Execute all database writes in the background
        background_tasks = {
            "delete_chat_memories": delete_chat_memory,
            "profile_keys_to_delete": keys_to_delete,
            "profile_records_to_upsert": records_to_upsert,
            "current_interaction_embedding": {
                "user_id": user_id, "memory_text": english_interaction, "memory_vector": interaction_vector
            }
        }
        _execute_background_tasks(client_sb, user_id, background_tasks)
        return final_response
    except Exception as e:
        print(f"❌ CRITICAL ERROR in processing/response generation: {e}")
        return "Sorry, error occurred."

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Sends a welcome message when the /start command is issued."""
    await update.message.reply_text("Hello! I am your personal memory assistant. Talk to me and I will remember our conversations.")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles incoming text messages and processes them with the agent logic."""
    user_input = update.message.text
    user_id = update.message.from_user.id  # Use Telegram user ID as the unique identifier

    # Run the synchronous processing function in a separate thread to avoid blocking the bot
    loop = asyncio.get_running_loop()
    response = await loop.run_in_executor(
        None, process_input, context.bot_data['supabase_client'], user_id, user_input
    )
    
    await update.message.reply_text(response)

if __name__ == "__main__":
    """Starts the Telegram bot."""
    print("--- STARTING TELEGRAM BOT ---")

    # Create a single, global Supabase client using the public API key
    supabase_client = create_client(COMPLETE_SUPABASE_URL, SUPABASE_KEY)

    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    application.bot_data['supabase_client'] = supabase_client

    # Register handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Run the bot
    application.run_polling()
