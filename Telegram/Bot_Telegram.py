"""
Telegram Bot for a personal AI assistant with long-term memory.
This script runs a Telegram bot that uses Supabase for data persistence (user
profile and vector memory) and Google Gemini APIs for its generative and
embedding capabilities. It features an asynchronous architecture, caching, API
call retries, and background processing for database operations.
"""
import os
import asyncio
import json
import logging
import time
from collections import defaultdict
from dotenv import load_dotenv
from typing import Any, List, Literal, Union
from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from pydantic import BaseModel, Field
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from supabase import create_client
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

# --- Configuration and Constants ---
_ENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "accessdata.env")
if not os.path.isfile(_ENV_PATH):
    raise RuntimeError(f"Environment file not found: {_ENV_PATH}")
load_dotenv(dotenv_path=_ENV_PATH)
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
MAX_STORAGE = 5
MAX_MEMORIES_PER_USER = int(os.getenv("MAX_MEMORIES_PER_USER", "100"))
EMBEDDING_DIMENSIONS = 3072
GEMINI_TIMEOUT_SECONDS = int(os.getenv("GEMINI_TIMEOUT_SECONDS", "120"))
RATE_LIMIT_SECONDS = float(os.getenv("RATE_LIMIT_SECONDS", "3"))
BOT_DEBUG = os.getenv("BOT_DEBUG", "").lower() in ("1", "true", "yes")
# Empty ALLOWED_USER_IDS = bot open to everyone; set comma-separated Telegram IDs to restrict access.
ALLOWED_USER_IDS = frozenset(
    uid.strip() for uid in os.getenv("ALLOWED_USER_IDS", "").split(",") if uid.strip()
)
REQUIRED_ENV_VARS = ("SUPABASE_URL", "SUPABASE_KEY", "GEMINI_API_KEY", "TELEGRAM_BOT_TOKEN")
_pending_background_tasks: set[asyncio.Task] = set()
_user_locks: dict[int, asyncio.Lock] = defaultdict(asyncio.Lock)
_user_last_message: dict[int, float] = {}
_RAG_RECALL_SIGNALS = (
    "remember", "recall", "forgot", "forget", "before", "last time", "earlier",
    "you said", "we discussed", "we talked", "prior", "previous", "my favorite",
    "what is my", "who is my", "ricord", "ricordi", "dimentic", "prima", "hai detto",
    "abbiamo parlato", "qual è il mio", "cosa ti ho", "predispos", "preferenz",
)
_WEB_SEARCH_SIGNALS = (
    "today", "current", "latest", "news", "weather", "now", "right now",
    "this week", "this year", "2024", "2025", "2026", "price of", "score",
    "risultat", "meteo", "oggi", "adesso", "attual", "notizie", "ultim",
)

def _validate_env():
    """Fail fast if required credentials are missing."""
    missing = [name for name in REQUIRED_ENV_VARS if not os.getenv(name)]
    if missing:
        raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")
_validate_env()
COMPLETE_SUPABASE_URL = f"https://{SUPABASE_URL}.supabase.co"

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("telegram").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("google_genai.models").setLevel(logging.WARNING)
logging.getLogger("google_genai._interactions").setLevel(logging.WARNING)

# --- Gemini API Configuration ---
google_client = genai.Client(api_key=GEMINI_API_KEY)

def _debug_log(message: str, *args):
    if BOT_DEBUG:
        logging.info(message, *args)

def _schedule_background_task(coro) -> asyncio.Task:
    task = asyncio.create_task(coro)
    _pending_background_tasks.add(task)
    task.add_done_callback(_pending_background_tasks.discard)
    return task

async def _cancel_background_tasks():
    if not _pending_background_tasks:
        return
    for task in list(_pending_background_tasks):
        task.cancel()
    await asyncio.gather(*_pending_background_tasks, return_exceptions=True)

def _validate_embedding(vector) -> bool:
    return bool(vector) and len(vector) == EMBEDDING_DIMENSIONS

def _needs_rag_search(text: str) -> bool:
    """Skip vector search on short or casual messages to save latency and API cost."""
    normalized = text.lower().strip()
    if len(normalized) < 12:
        return False
    if "?" in normalized:
        return True
    return any(signal in normalized for signal in _RAG_RECALL_SIGNALS)

def _needs_web_search(text: str) -> bool:
    """Keyword gate for a separate Gemini call with Google Search enabled."""
    normalized = text.lower().strip()
    return any(signal in normalized for signal in _WEB_SEARCH_SIGNALS)

def _is_user_allowed(user_id: int) -> bool:
    if not ALLOWED_USER_IDS:
        return True
    return str(user_id) in ALLOWED_USER_IDS

def _check_rate_limit(user_id: int) -> bool:
    now = time.monotonic()
    last = _user_last_message.get(user_id, 0.0)
    if now - last < RATE_LIMIT_SECONDS:
        return False
    _user_last_message[user_id] = now
    return True

async def _ensure_user_access(update: Update) -> bool:
    user_id = update.message.from_user.id
    if not _is_user_allowed(user_id):
        await update.message.reply_text("Sorry, this bot is private.")
        return False
    if not _check_rate_limit(user_id):
        await update.message.reply_text("Please wait a moment before sending another message.")
        return False
    return True

def _parse_structured_response(response, schema_output):
    if response.parsed is not None:
        if isinstance(response.parsed, schema_output):
            return response.parsed
        if isinstance(response.parsed, dict):
            return schema_output.model_validate(response.parsed)
    raw_text = (response.text or "").strip()
    if not raw_text:
        raise ValueError("Gemini returned an empty structured response")
    return schema_output.model_validate(json.loads(raw_text))

async def _send_text_reply(update: Update, text: str):
    if not text or not text.strip():
        text = "Sorry, an error occurred."
    if len(text) <= 4096:
        await update.message.reply_text(text)
        return
    for i in range(0, len(text), 4096):
        await update.message.reply_text(text[i:i + 4096])

class AgentCache:
    """In-memory cache to temporarily store the user profile and reduce DB calls."""
    _storage = {}
    @classmethod
    def get_profile(cls, user_id):
        return cls._storage.get(user_id, None)

    @classmethod
    def set_profile(cls, user_id, data):
        cls._storage[user_id] = data

    @classmethod
    def invalidate(cls, user_id):
        if user_id in cls._storage:
            del cls._storage[user_id]

class SinglePreference(BaseModel):
    key_name: str = Field(description="The specific category and entity in snake_case and strictly in English (e.g., max_budget_motorcycle).")
    value_data: str = Field(description="The extracted value in the user's language. For DELETE actions, this can be empty.")
    context_desc: str = Field(description="The full context in the user's language. For DELETE actions, describe what is being deleted.")
    action_type: Literal["upsert", "delete"] = Field(description="Use 'upsert' to save/update data. Use 'delete' ONLY if requested.")
    is_sensitive: bool = Field(description="True if the data involves personal, financial, private, or sensitive information (e.g., IBAN, health, tastes). False if it's a generic technical setting or app configuration.")
    memory_target: Literal["profile", "chat_memories", "all"] = Field(description="Only in case of deletion, use 'profile' to affect user_profile table. Use 'chat_memories' to delete recent semantic memories/chat logs. Use 'all' if the user wants to completely wipe out both profile data and chat memories.")

class GeminiDualOutput(BaseModel):
    """Model for a dual output containing both the user-facing response and the data operations."""
    assistant_response: str = Field(description="The final, user-facing response, written in the correct language.")
    preferences: List[SinglePreference] = Field(description="List of all profile preferences or data management actions found.")

@retry(
    retry=retry_if_exception_type((genai_errors.ServerError, genai_errors.APIError)),
    wait=wait_exponential(multiplier=2, min=2, max=20),
    stop=stop_after_attempt(3),
    reraise=True,
)
async def call_gemini_with_retry(
    model: str,
    contents: Union[str, List[Any]],
    system_prompt: str,
    schema_output=None,
    mime_type=None,
    temperature: float = 0.1,
):
    """Calls Gemini with retries. Structured JSON output cannot use Google Search in the same request."""
    generation_config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        temperature=temperature,
    )
    if schema_output:
        generation_config.response_schema = schema_output
        generation_config.response_mime_type = "application/json"
    elif mime_type:
        generation_config.response_mime_type = mime_type
    response = await asyncio.wait_for(
        google_client.aio.models.generate_content(
            model=model,
            contents=contents,
            config=generation_config,
        ),
        timeout=GEMINI_TIMEOUT_SECONDS,
    )
    if schema_output:
        return _parse_structured_response(response, schema_output)
    return response.text

@retry(
    retry=retry_if_exception_type((genai_errors.ServerError, genai_errors.APIError)),
    wait=wait_exponential(multiplier=2, min=2, max=20),
    stop=stop_after_attempt(3),
    reraise=True,
)
async def _call_gemini_with_web_search(
    model: str,
    contents: Union[str, List[Any]],
    system_prompt: str,
    temperature: float = 0.1,
):
    """On-demand web search: separate call because Google Search and JSON schema cannot be combined."""
    generation_config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        temperature=temperature,
        tools=[types.Tool(google_search=types.GoogleSearch())],
    )
    response = await asyncio.wait_for(
        google_client.aio.models.generate_content(
            model=model,
            contents=contents,
            config=generation_config,
        ),
        timeout=GEMINI_TIMEOUT_SECONDS,
    )
    return response.text

@retry(
    retry=retry_if_exception_type((genai_errors.ServerError, genai_errors.APIError)),
    wait=wait_exponential(multiplier=2, min=2, max=20),
    stop=stop_after_attempt(3),
    reraise=True,
)
async def generate_embedding_with_retry(text_to_embed: str, task_type="RETRIEVAL_DOCUMENT"):
    """Generates native 3072-dimension vectors for Supabase."""
    response = await asyncio.wait_for(
        google_client.aio.models.embed_content(
            model="gemini-embedding-2",
            contents=text_to_embed,
            config=types.EmbedContentConfig(task_type=task_type),
        ),
        timeout=GEMINI_TIMEOUT_SECONDS,
    )
    return response.embeddings[0].values if response.embeddings else []

async def search_past_memories(client_sb, user_id, search_text):
    """Performs semantic search (RAG) using embeddings in the original query language."""
    try:
        vector = await generate_embedding_with_retry(search_text.strip(), task_type="RETRIEVAL_QUERY")
        if not _validate_embedding(vector):
            logging.error("Invalid query embedding for user %d", user_id)
            return []
        loop = asyncio.get_running_loop()
        response_db = await loop.run_in_executor(
            None,
            lambda: client_sb.rpc("match_telegram_memories", {
                "query_embedding": vector,
                "match_threshold": 0.4,
                "match_count": 3,
                "p_user_id": user_id,
            }).execute(),
        )
        return [row["memory_text"] for row in response_db.data]
    except Exception as e:
        logging.error("Vector search error for user %d: %s", user_id, e, exc_info=BOT_DEBUG)
        return []

def _get_chat_history(client_sb, user_id):
    """Short-term context: the last MAX_STORAGE summaries, chronological."""
    try:
        response = client_sb.rpc("bot_get_recent_memories", {
            "p_user_id": user_id,
            "p_limit": MAX_STORAGE,
        }).execute()
        history = [row["memory_text"] for row in response.data]
        history.reverse()
        return history
    except Exception as e:
        logging.error("Failed to fetch chat history for user %d: %s", user_id, e)
        return []

def _get_user_profile(client_sb, user_id):
    """Fetches the user profile, using cache if available."""
    profile = AgentCache.get_profile(user_id)
    if profile is None:
        try:
            response = client_sb.rpc("bot_get_profile", {"p_user_id": user_id}).execute()
            profile = response.data
            AgentCache.set_profile(user_id, profile)
        except Exception as e:
            logging.error("Failed to fetch profile for user %d: %s", user_id, e)
            profile = []
    return profile

def _execute_background_tasks(client_sb, user_id, tasks):
    """Sequential writes: deletes must finish before inserts (GDPR-safe ordering)."""
    try:
        if tasks.get("delete_chat_memories"):
            client_sb.rpc("bot_delete_all_memories", {"p_user_id": user_id}).execute()
        if tasks.get("profile_keys_to_delete"):
            keys = list(set(tasks["profile_keys_to_delete"]))
            client_sb.rpc("bot_delete_profile_keys", {
                "p_user_id": user_id,
                "p_key_names": keys,
            }).execute()
        if tasks.get("profile_records_to_upsert"):
            client_sb.rpc("bot_upsert_profile_records", {
                "p_records": tasks["profile_records_to_upsert"],
            }).execute()
        if tasks.get("current_interaction_embedding"):
            embedding_data = tasks["current_interaction_embedding"]
            client_sb.rpc("bot_insert_memory", {
                "p_user_id": user_id,
                "p_memory_text": embedding_data["memory_text"],
                "p_memory_vector": embedding_data["memory_vector"],
            }).execute()
            client_sb.rpc("bot_prune_memories", {
                "p_user_id": user_id,
                "p_max_count": MAX_MEMORIES_PER_USER,
            }).execute()  # cap long-term memory growth per user
    except Exception as e:
        logging.error("Background DB write failed for user %d: %s", user_id, e)
        raise
    finally:
        AgentCache.invalidate(user_id)

async def _fetch_web_search_context(user_id: int, search_text: str) -> str:
    """Optional pre-call: fetch live web facts when keywords match, inject into main prompt."""
    if not _needs_web_search(search_text):
        return ""
    try:
        summary = await _call_gemini_with_web_search(
            model="gemini-3.1-flash-lite",
            contents=search_text,
            system_prompt="Provide a brief factual summary using current web information. Be concise.",
            temperature=0.1,
        )
        return f"\n        - WEB SEARCH SNAPSHOT: {summary}\n"
    except Exception as e:
        logging.error("Web search failed for user %d: %s", user_id, e, exc_info=BOT_DEBUG)
        return ""

async def process_input(client_sb, user_id, user_input: Union[str, List[Any]], rag_query_text: str | None = None):
    """Orchestrates context retrieval, response generation, and profile operations."""
    search_text = rag_query_text if rag_query_text is not None else (
        user_input if isinstance(user_input, str) else "image or media content"
    )
    history_task = asyncio.to_thread(_get_chat_history, client_sb, user_id)
    profile_task = asyncio.to_thread(_get_user_profile, client_sb, user_id)
    _debug_log("Pipeline user=%d stage=context_fetch", user_id)
    chat_history, current_profile = await asyncio.gather(history_task, profile_task)
    # Long-term RAG only when the message likely references past context; skip duplicates already in short-term history.
    if _needs_rag_search(search_text):
        rag_results = await search_past_memories(client_sb, user_id, search_text)
        recent_set = set(chat_history)
        relevant_memories = [memory for memory in rag_results if memory not in recent_set]
    else:
        relevant_memories = []
    _debug_log(
        "Pipeline user=%d stage=context_ready history=%d profile=%d rag=%d",
        user_id, len(chat_history), len(current_profile), len(relevant_memories),
    )
    try:
        records_to_upsert = []
        keys_to_delete = []
        delete_chat_memory = False
        def process_preferences(preferences):
            nonlocal records_to_upsert, keys_to_delete, delete_chat_memory
            for preference_item in preferences:
                if preference_item.action_type == "delete":
                    if preference_item.memory_target in ["chat_memories", "all"] and not delete_chat_memory:
                        delete_chat_memory = True
                    if preference_item.memory_target in ["profile", "all"]:
                        if preference_item.memory_target == "all":
                            keys_to_delete.extend([
                                row["key_name"] for row in current_profile if row.get("is_sensitive", False)
                            ])
                        else:
                            old_record = next(
                                (row for row in current_profile if row["key_name"] == preference_item.key_name),
                                None,
                            )
                            if not (old_record and not old_record.get("is_sensitive", False)):
                                keys_to_delete.append(preference_item.key_name)
                elif preference_item.action_type == "upsert":
                    old_record = next(
                        (row for row in current_profile if row["key_name"] == preference_item.key_name),
                        None,
                    )
                    write_needed = True
                    if old_record and old_record["value_data"] == str(preference_item.value_data) and old_record["context_desc"] == preference_item.context_desc:
                        write_needed = False
                    if write_needed and preference_item.value_data:
                        records_to_upsert.append({
                            "user_id": user_id,
                            "key_name": preference_item.key_name,
                            "value_data": preference_item.value_data,
                            "context_desc": preference_item.context_desc,
                            "is_sensitive": preference_item.is_sensitive,
                        })
        language_setting = next(
            (row["value_data"] for row in current_profile if row["key_name"] == "user_language"),
            "en",
        )
        web_search_context = await _fetch_web_search_context(user_id, search_text) if isinstance(user_input, str) else ""
        unified_prompt = f"""
        You are a professional personal AI assistant. Your task is to perform two actions in one go:
        1.  Generate a helpful, user-facing response.
        2.  Provide a JSON object detailing any data operations based on the user's prompt.
        **RESPONSE GENERATION RULES:**
        - You MUST reply strictly in the language with this ISO 639-1 code: {language_setting}.
        - Your response should be natural, conversational, and based on all the context provided.
        - If the data extraction finds something to update on the user profile (an upsert or delete), inform the user about what you are planning to do, using simple, non-technical language.
        **DATA EXTRACTION RULES:**
        - After writing the user response, you MUST output a JSON object that conforms to the provided schema.
        - `key_name` must always be in English snake_case.
        - `value_data` and `context_desc` must be in the user's language (ISO 639-1 code: {language_setting}).
        - If the user expresses a preference, use `action_type: "upsert"`.
        - If the user asks to delete or forget data:
            - For a SPECIFIC item (e.g., "forget my favorite color"), use `action_type: "delete"` and specify the `key_name`.
            - For their PROFILE data (e.g., "delete my personal data"), use `action_type: "delete"` with `memory_target: "profile"`.
            - For CHAT HISTORY (e.g., "forget our conversation"), use `action_type: "delete"` with `memory_target: "chat_memories"`.
            - For EVERYTHING (e.g., "forget everything about me"), use `action_type: "delete"` with `memory_target: "all"`.
        - If no preferences or commands are found, the "preferences" list in the JSON must be empty.
        - Base your data extraction *only* on the user's latest message. Use the provided context (profile, history, memories) solely to understand if the user's new message introduces a change or a request to forget something.
        **CONTEXT FOR YOUR RESPONSE AND ANALYSIS:**
        - USER PROFILE KEYS: {current_profile}
        - RECENT CHAT LOGS (short-term): {chat_history}
        - RETRIEVED LONG-TERM MEMORIES (RAG, excluding recent logs): {relevant_memories}
        {web_search_context}
        **ADDITIONAL INFO:**
        - You are working on behalf of Personal Copilot Project (PCP).
        - PCP saves conversation summaries and user preferences to build its memory.
        - Users can ask you to forget specific facts, their entire profile, or their chat history at any time.
        - You can handle Text, Audio and Images as input.
           - Media like audio and images are not stored. They are converted to text (transcriptions or descriptions) for you to access, so you cannot 're-watch' an image or 're-listen' to audio.
        """
        _debug_log("Pipeline user=%d stage=gemini_request", user_id)
        # Single structured call: reply + profile operations. Web search (if any) was already fetched above.
        dual_output = await call_gemini_with_retry(
            model="gemini-3.5-flash",
            contents=user_input,
            system_prompt=unified_prompt,
            schema_output=GeminiDualOutput,
            mime_type="application/json",
            temperature=0.4,
        )
        _debug_log("Pipeline user=%d stage=gemini_response", user_id)
        final_response = dual_output.assistant_response
        if not final_response or not str(final_response).strip():
            raise ValueError("Gemini returned an empty assistant_response")
        if dual_output.preferences:
            process_preferences(dual_output.preferences)
        return final_response, delete_chat_memory, keys_to_delete, records_to_upsert
    except asyncio.TimeoutError:
        logging.error("Processing timed out for user %d after %ds", user_id, GEMINI_TIMEOUT_SECONDS)
        return "Sorry, the request timed out. Please try again.", None, None, None
    except Exception as e:
        logging.error("Processing failed for user %d: %s", user_id, e, exc_info=True)
        return "Sorry, an error occurred.", None, None, None

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles the /start command and creates a default profile for new users."""
    user = update.message.from_user
    user_id = user.id
    supabase_client = context.bot_data["supabase_client"]
    exists = await asyncio.to_thread(
        lambda: supabase_client.rpc("bot_user_profile_exists", {"p_user_id": user_id}).execute()
    )
    if not exists.data:
        logging.info("New user registered: telegram_id=%d", user_id)
        default_language = user.language_code if user.language_code else "en"
        default_records = [
            {"user_id": user_id, "key_name": "user_name", "value_data": user.full_name, "context_desc": "The user's full name.", "is_sensitive": False},
            {"user_id": user_id, "key_name": "user_language", "value_data": default_language, "context_desc": "The user's preferred language code (ISO 639-1).", "is_sensitive": False},
            {"user_id": user_id, "key_name": "user_locale", "value_data": default_language, "context_desc": "The user's locale setting.", "is_sensitive": False},
            {"user_id": user_id, "key_name": "user_timezone", "value_data": "UTC", "context_desc": "The user's timezone.", "is_sensitive": False},
        ]
        try:
            await asyncio.to_thread(
                lambda: supabase_client.rpc("bot_upsert_profile_records", {"p_records": default_records}).execute()
            )
            AgentCache.invalidate(user_id)
        except Exception as e:
            logging.error("Failed to create default profile for user %d: %s", user_id, e)
    await update.message.reply_text(
        f"Hello {user.first_name}! I am your personal memory assistant. Talk to me and I will remember our conversations."
    )

async def run_background_processing(
    supabase_client, user_id, memory_user_text, final_response,
    delete_chat_memory, keys_to_delete, records_to_upsert,
):
    """Handles embedding generation and database writes after the user reply."""
    try:
        background_tasks = {
            "delete_chat_memories": delete_chat_memory,
            "profile_keys_to_delete": keys_to_delete,
            "profile_records_to_upsert": records_to_upsert,
        }
        if not delete_chat_memory:
            # Do not store the interaction being deleted (e.g. "forget our conversation").
            memory_text = f"User said: '{memory_user_text}' | Assistant replied: '{final_response}'"
            memory_vector = await generate_embedding_with_retry(memory_text, task_type="RETRIEVAL_DOCUMENT")
            if not _validate_embedding(memory_vector):
                logging.error("Invalid memory embedding for user %d", user_id)
                return
            background_tasks["current_interaction_embedding"] = {
                "user_id": user_id,
                "memory_text": memory_text,
                "memory_vector": memory_vector,
            }
        await asyncio.to_thread(_execute_background_tasks, supabase_client, user_id, background_tasks)
    except asyncio.TimeoutError:
        logging.error("Background processing timed out for user %d after %ds", user_id, GEMINI_TIMEOUT_SECONDS)
    except Exception as e:
        logging.error("Error in background processing for user %d: %s", user_id, e, exc_info=BOT_DEBUG)

async def process_and_reply(
    user_input: Union[str, List[Any]],
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    memory_user_text: str | None = None,
    rag_query_text: str | None = None,
    access_already_checked: bool = False,
):
    """Core logic to process user input, send a reply, and run background tasks."""
    if not access_already_checked and not await _ensure_user_access(update):
        return
    user_id = update.message.from_user.id
    supabase_client = context.bot_data["supabase_client"]
    # One in-flight request per user avoids profile/memory races on rapid messages.
    async with _user_locks[user_id]:
        try:
            await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)
            _debug_log("Pipeline user=%d stage=process_input", user_id)
            final_response, delete_chat, keys_delete, records_upsert = await process_input(
                supabase_client, user_id, user_input, rag_query_text=rag_query_text
            )
            _debug_log("Pipeline user=%d stage=reply", user_id)
            await _send_text_reply(update, final_response)
            if (
                final_response != "Sorry, an error occurred."
                and not final_response.startswith("Sorry, the request timed out")
            ):
                text_for_memory = memory_user_text if memory_user_text is not None else (
                    user_input if isinstance(user_input, str) else "[multimodal input]"
                )
                _schedule_background_task(run_background_processing(
                    supabase_client, user_id, text_for_memory, final_response,
                    delete_chat, keys_delete, records_upsert,
                ))
        except Exception as e:
            logging.error("Reply pipeline failed for user %d: %s", user_id, e, exc_info=True)
            await _send_text_reply(update, "Sorry, an error occurred.")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles incoming text messages and processes them with the agent logic."""
    if not update.message or not update.message.text:
        return
    await process_and_reply(update.message.text, update, context)

async def on_error(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logging.error("Unhandled handler error: %s", context.error, exc_info=True)
    if isinstance(update, Update) and update.effective_message:
        try:
            await update.effective_message.reply_text("Sorry, an error occurred.")
        except Exception as send_error:
            logging.error("Failed to send error reply: %s", send_error, exc_info=True)

async def on_shutdown(_application: Application) -> None:
    logging.info("Telegram bot shutting down")
    await _cancel_background_tasks()

async def handle_voice_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles voice messages by transcribing them and then processing the text."""
    if not await _ensure_user_access(update):
        return
    await update.message.reply_text("Transcribing your voice message...")
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)
    voice_file = await update.message.voice.get_file()
    voice_bytes = await voice_file.download_as_bytearray()
    voice_mime_type = update.message.voice.mime_type
    try:
        audio_part = types.Part.from_bytes(data=bytes(voice_bytes), mime_type=voice_mime_type)
        transcribed_text = await call_gemini_with_retry(
            model="gemini-3.1-flash-lite",
            contents=["Transcribe this audio message in its original spoken language.", audio_part],
            system_prompt="You are a transcription assistant. Transcribe the provided audio faithfully in the language it was spoken.",
        )
        await process_and_reply(transcribed_text, update, context, access_already_checked=True)
    except Exception as e:
        logging.error("Error processing voice message for user %d: %s", update.message.from_user.id, e, exc_info=True)
        await update.message.reply_text("Sorry, I couldn't process the voice message.")

async def handle_photo_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles photo messages through the main agent pipeline (profile, RAG, memory)."""
    if not await _ensure_user_access(update):
        return
    await update.message.reply_text("Analyzing the image...")
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)
    try:
        photo_file = await update.message.photo[-1].get_file()
        photo_bytes = await photo_file.download_as_bytearray()
        image_part = types.Part.from_bytes(data=bytes(photo_bytes), mime_type="image/jpeg")
        user_caption = update.message.caption or "The user sent an image without a caption. Describe and analyze it."
        prompt_contents = [user_caption, image_part]
        memory_text = f"[Image] {user_caption}" if update.message.caption else "[Image] (no caption)"
        await process_and_reply(
            prompt_contents, update, context,
            memory_user_text=memory_text,
            rag_query_text=user_caption,
            access_already_checked=True,
        )
    except Exception as e:
        logging.error("Error processing photo message for user %d: %s", update.message.from_user.id, e, exc_info=True)
        await update.message.reply_text("Sorry, I couldn't analyze the image.")

if __name__ == "__main__":
    logging.info(
        "Telegram bot starting (gemini_timeout=%ds, max_memories=%d, rate_limit=%ss)",
        GEMINI_TIMEOUT_SECONDS, MAX_MEMORIES_PER_USER, RATE_LIMIT_SECONDS,
    )
    if BOT_DEBUG:
        logging.info("BOT_DEBUG enabled — pipeline stage logs are active")
    if ALLOWED_USER_IDS:
        logging.info("Allowlist enabled for %d user(s)", len(ALLOWED_USER_IDS))
    supabase_client = create_client(COMPLETE_SUPABASE_URL, SUPABASE_KEY)  # anon key + RPC layer (see supabase_rls.sql)
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).post_shutdown(on_shutdown).build()
    application.bot_data["supabase_client"] = supabase_client
    application.add_error_handler(on_error)
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    application.add_handler(MessageHandler(filters.VOICE, handle_voice_message))
    application.add_handler(MessageHandler(filters.PHOTO, handle_photo_message))
    application.run_polling()
