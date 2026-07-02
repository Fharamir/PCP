"""
Telegram Bot for a personal AI assistant with long-term memory.

This script runs a Telegram bot that uses Supabase for data persistence (user
profile and vector memory) and Google Gemini APIs for its generative and
embedding capabilities. It features an asynchronous architecture, caching, API
call retries, and background processing for database operations.
"""
import os
import asyncio
import logging
import concurrent.futures
from dotenv import load_dotenv
from typing import List, Literal

from google import genai
from google.genai import errors as genai_errors
from google.genai import types

from pydantic import BaseModel, Field
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from supabase import create_client, Client, ClientOptions
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type


# --- Configuration and Constants ---
load_dotenv(dotenv_path='accessdata.env') # Load environment variables from the specified file
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY") # Use the service role key for admin access
MAX_STORAGE = 5

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# Set the logging level for httpx to WARNING to reduce polling noise
logging.getLogger("httpx").setLevel(logging.WARNING)

# --- Gemini API Configuration ---
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
    logging.warning(
        "Retrying API call... Attempt #%d, Exception: %s",
        retry_state.attempt_number, retry_state.outcome.exception()
    )

from typing import Union, List, Any

@retry(
    retry=retry_if_exception_type((genai_errors.ServerError, genai_errors.APIError)),
    wait=wait_exponential(multiplier=2, min=2, max=20), 
    stop=stop_after_attempt(3),
    reraise=True,
    before_sleep=log_retry_attempt
)
async def call_gemini_with_retry(
    model: str, 
    contents: Union[str, List[Any]], 
    system_prompt: str, 
    schema_output=None, 
    mime_type=None, 
    websearch: bool = False, 
    temperature: float = 0.1 # UPDATE: Dynamic parameter with a default value of 0.1
):
    """Asynchronously calls a Gemini model supporting text/multimodal inputs, dynamic temperature, and retries."""
    
    active_tools = []
    if websearch:
        active_tools.append(types.Tool(google_search=types.GoogleSearch()))

    # Configuration with dynamic temperature passed to the call
    generation_config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        temperature=temperature, # Apply the temperature passed as an argument
        tools=active_tools if active_tools else None
    )
    
    if schema_output:
        generation_config.response_schema = schema_output
        generation_config.response_mime_type = "application/json"
    elif mime_type:
        generation_config.response_mime_type = mime_type


    response = await google_client.aio.models.generate_content(
        model=model,
        contents=contents, 
        config=generation_config
    )
    
    if schema_output and mime_type == "application/json":
        return response.parsed
        
    return response.text


@retry(
    retry=retry_if_exception_type((genai_errors.ServerError, genai_errors.APIError)),
    wait=wait_exponential(multiplier=2, min=2, max=20), 
    stop=stop_after_attempt(3),
    reraise=True,
    before_sleep=log_retry_attempt
)
async def generate_embedding_with_retry(text_to_embed: str, task_type="RETRIEVAL_DOCUMENT"):
    """Generates native 3072-dimension vectors for Supabase."""
    
    response = await google_client.aio.models.embed_content(
        model="gemini-embedding-2",  # This model automatically generates 3072-dimension embeddings
        contents=text_to_embed, 
        config=types.EmbedContentConfig(
            task_type=task_type
        )
    )
    return response.embeddings[0].values if response.embeddings else []


async def search_past_memories(client_sb, user_id, search_text):
    """Asynchronously performs a semantic search (RAG) in the user's long-term memory."""
    try:
        # UPDATE: Using 'contents' parameter and implicit temperature of 0.1
        res_trad = await call_gemini_with_retry(
            model='gemini-3.1-flash-lite', 
            contents=f"Rephrase this search query for a vector database: '{search_text}'",
            system_prompt="Your task is to rephrase the user's search query into a concise, keyword-focused English sentence, optimized for semantic vector search."
        )
        query_inglese = res_trad.strip()
		
        vettore = await generate_embedding_with_retry(query_inglese, task_type="RETRIEVAL_QUERY")

        # Run the synchronous DB call in a separate thread
        loop = asyncio.get_running_loop()
        risposta_db = await loop.run_in_executor(
            None,
            lambda: client_sb.rpc("match_telegram_memories", {
                "query_embedding": vettore,
                "match_threshold": 0.4,
                "match_count": 3,
                "p_user_id": user_id
            }).execute()
        )

        return [riga['memory_text'] for riga in risposta_db.data]
    except Exception as e:
        logging.error("Vector search error: %s", e)
        return []

def _get_chat_history(client_sb, user_id):
    """Fetches the recent chat history from Supabase."""
    try:
        response = client_sb.table("telegram_agent_memories").select("memory_text").eq("user_id", user_id).order("created_at", ascending=False).limit(MAX_STORAGE).execute()
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
            response = client_sb.table("telegram_user_profile").select("key_name", "value_data", "context_desc", "is_sensitive").execute()
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
            logging.info("Submitting background task: Delete all vector memories...")
            executor.submit(client_sb.table("telegram_agent_memories").delete().eq("user_id", user_id).execute)

        # Task 2: Delete profile keys if requested
        if tasks.get("profile_keys_to_delete"):
            keys = list(set(tasks["profile_keys_to_delete"]))
            logging.info("Submitting background task: Delete %d profile keys...", len(keys))
            executor.submit(client_sb.table("telegram_user_profile").delete().eq("user_id", user_id).in_("key_name", keys).execute)

        # Task 3: Upsert profile keys if requested
        if tasks.get("profile_records_to_upsert"):
            records = tasks["profile_records_to_upsert"]
            logging.info("Submitting background task: Upsert %d profile records...", len(records))
            executor.submit(client_sb.table("telegram_user_profile").upsert(records, on_conflict="user_id,key_name").execute)

        # Task 4: Save the current conversation to long-term memory
        if tasks.get("current_interaction_embedding"):
            embedding_data = tasks["current_interaction_embedding"]
            logging.info("Submitting background task: Save current interaction to memory...")
            executor.submit(client_sb.table("telegram_agent_memories").insert(embedding_data).execute)

    AgentCache.invalidate(user_id) # Invalidate cache after writes are done

async def process_input(client_sb, user_id, user_input):
    """
    Asynchronously orchestrates the entire response process: retrieves context,
    updates memory, and generates a response.
    """
    # --- Parallel Asynchronous Context Fetching ---
    # Run synchronous DB/file operations in threads to avoid blocking
    history_task = asyncio.to_thread(_get_chat_history, client_sb, user_id)
    profile_task = asyncio.to_thread(_get_user_profile, client_sb, user_id)
    # Run async RAG search directly
    memories_task = search_past_memories(client_sb, user_id, user_input)

    chat_history, current_profile, relevant_memories = await asyncio.gather(
        history_task, profile_task, memories_task
    )

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
                        logging.info("GDPR PURGE: Deletion request detected for semantic chat memories.")
                        delete_chat_memory = True
                    if preference_item.memory_target in ["profile", "all"]:
                        if preference_item.memory_target == "all":
                            logging.info("GDPR PURGE: Profile reset scheduled (system keys excluded).")
                            keys_to_delete.extend([r['key_name'] for r in current_profile if r.get('is_sensitive', False)])
                        else:
                            old_record = next((row for row in current_profile if row['key_name'] == preference_item.key_name), None)
                            if old_record and not old_record.get('is_sensitive', False):
                                logging.warning("SECURITY: Deletion denied for system key: '%s'", preference_item.key_name)
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
        - If the data extraction finds something to update on the user profile (an upsert or delete), inform the user about what you are planning to do, using simple, non-technical language.

        **DATA EXTRACTION RULES:**
        - After writing the user response, you MUST output a JSON object that conforms to the provided schema.
        - All output in the JSON must be in ENGLISH.
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
        - RECENT CHAT LOGS: {chat_history}
        - RETRIEVED LONG-TERM MEMORIES (RAG): {relevant_memories}

        **ADDITIONAL INFO:**
        - You are working on behalf of Personal Copilot Project (PCP).
        - PCP saves conversation summaries and user preferences to build its memory.
        - Users can ask you to forget specific facts, their entire profile, or their chat history at any time.
        - You can handle Text, Audio and Images as input.
           - Media like audio and images are not stored. They are converted to text (transcriptions or descriptions) for you to access, so you cannot 're-watch' an image or 're-listen' to audio.
        """

        dual_output = await call_gemini_with_retry(
            model='gemini-3.5-flash', 
            contents=user_input, 
            system_prompt=unified_prompt,
            schema_output=GeminiDualOutput,
            mime_type="application/json", 
            websearch=True, 
            temperature=0.4 # Use a higher temperature for more natural-sounding conversation
        )
        
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
            logging.info("DB-PREP: Actions prepared for background execution:%s", action_list_summary)
        else:
            logging.info("DB-PREP: No database actions detected.")

        logging.info("User Input: '%s'", user_input)
        logging.info("Assistant Response (%s):\n%s", language_setting, final_response)

        # Return the response and the pending DB tasks to be executed in the background
        return final_response, delete_chat_memory, keys_to_delete, records_to_upsert, language_setting
    except Exception as e:
        logging.critical("CRITICAL ERROR in processing/response generation: %s", e)
        return "Sorry, an error occurred.", None, None, None, 'en'

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handles the /start command.

    Sends a welcome message and creates a default profile for new users.
    """
    user = update.message.from_user
    user_id = user.id
    supabase_client = context.bot_data['supabase_client']

    # Check if user profile already exists
    existing_profile = supabase_client.table("telegram_user_profile").select("user_id").eq("user_id", user_id).limit(1).execute()

    if not existing_profile.data:
        logging.info("New user detected (ID: %d). Creating default profile.", user_id)
        
        # Prepare default settings from Telegram user data
        default_language = user.language_code if user.language_code else 'en'
        default_records = [
            {"user_id": user_id, "key_name": "user_name", "value_data": user.full_name, "context_desc": "The user's full name.", "is_sensitive": False},
            {"user_id": user_id, "key_name": "user_language", "value_data": default_language, "context_desc": "The user's preferred language code (ISO 639-1).", "is_sensitive": False},
            {"user_id": user_id, "key_name": "user_locale", "value_data": default_language, "context_desc": "The user's locale setting.", "is_sensitive": False},
            {"user_id": user_id, "key_name": "user_timezone", "value_data": "UTC", "context_desc": "The user's timezone.", "is_sensitive": False},
        ]

        try:
            # Run the synchronous DB operation in a thread to avoid blocking
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None, 
                lambda: supabase_client.table("telegram_user_profile").upsert(default_records).execute()
            )
            logging.info("Default profile for user %d created successfully.", user_id)
        except Exception as e:
            logging.error("Failed to create default profile for user %d: %s", user_id, e)

    await update.message.reply_text(f"Hello {user.first_name}! I am your personal memory assistant. Talk to me and I will remember our conversations.")

async def run_background_processing(
    supabase_client, user_id, user_input, final_response, 
    delete_chat_memory, keys_to_delete, records_to_upsert, language_setting
):
    """
    Asynchronously handles all post-response tasks: embedding generation and database writes.
    This coroutine is designed to be run as a background task.
    """
    try:
        # Prepare text for long-term memory embedding
        interaction_text = f"User said: '{user_input}' | Assistant replied: '{final_response}'"

        # Translate and summarize if the conversation is not already in English.
        if language_setting != 'en':
            # UPDATE: Changed parameter to 'contents' (implicit temperature of 0.1)
            english_interaction = await call_gemini_with_retry(
                model='gemini-3.1-flash-lite', 
                contents=f"Summarize the following interaction into a single, self-contained English sentence that captures the core fact or outcome: {interaction_text}",
                system_prompt="You are an expert summarizer. Your goal is to create concise, factual summaries of interactions for a memory system."
            ) 
        else:
            english_interaction = interaction_text
        
        interaction_vector = await generate_embedding_with_retry(english_interaction, task_type="RETRIEVAL_DOCUMENT")

        # Execute all database writes in the background
        background_tasks = {
            "delete_chat_memories": delete_chat_memory,
            "profile_keys_to_delete": keys_to_delete,
            "profile_records_to_upsert": records_to_upsert,
            "current_interaction_embedding": {
                "user_id": user_id, "memory_text": english_interaction, "memory_vector": interaction_vector
            }
        }
        # Run the synchronous DB operations in a thread.
        await asyncio.to_thread(_execute_background_tasks, supabase_client, user_id, background_tasks)
    except Exception as e:
        logging.error("Error in background processing for user %d: %s", user_id, e)

async def process_and_reply(user_input: str, update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Core logic to process a text input, send a reply, and run background tasks.
    """
    user_id = update.message.from_user.id
    supabase_client = context.bot_data['supabase_client']

    # Let the user know the bot is thinking
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)

    # 1. Get the user-facing response as quickly as possible
    final_response, delete_chat, keys_delete, records_upsert, lang_setting = await process_input(
        supabase_client, user_id, user_input
    )

    # 2. Send the response to the user immediately
    await update.message.reply_text(final_response)

    # 3. Start all post-processing tasks in a background thread
    if final_response != "Sorry, an error occurred.":
        asyncio.create_task(run_background_processing(
                supabase_client, user_id, user_input, final_response,
                delete_chat, keys_delete, records_upsert, lang_setting
            )
        )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles incoming text messages and processes them with the agent logic."""
    await process_and_reply(update.message.text, update, context)

async def handle_voice_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles voice messages by transcribing them and then processing the text."""
    await update.message.reply_text("Trascrivo il tuo messaggio vocale e preparo una risposta...")
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)

    voice_file = await update.message.voice.get_file()
    voice_bytes = await voice_file.download_as_bytearray()
    voice_mime_type = update.message.voice.mime_type

    try:
        # SDK UPDATE: Correct generation of multimedia content
        audio_part = types.Part.from_bytes(data=bytes(voice_bytes), mime_type=voice_mime_type)

        # UPDATE: Now using call_gemini_with_retry protected by Tenacity
        transcribed_text = await call_gemini_with_retry(
            model='gemini-3.1-flash-lite', 
            contents=["Transcribe this audio message to English.", audio_part], # Multimodal list with text prompt and audio
            system_prompt="You are a transcription assistant. Transcribe the provided audio into English text."
            # Default temperature of 0.1 is used to avoid hallucinations in the transcription
        )

        logging.info("Transcription result: %s", transcribed_text)
        
        # Pass the transcribed text to the main bot logic
        await process_and_reply(transcribed_text, update, context)

    except Exception as e:
        logging.error("Error processing voice message: %s", e)
        await update.message.reply_text("Spiacente, non sono riuscito a elaborare il messaggio vocale.")


async def handle_photo_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles photo messages by analyzing them and sending a direct response."""
    await update.message.reply_text("Sto analizzando l'immagine...")
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)

    photo_file = await update.message.photo[-1].get_file()
    photo_bytes = await photo_file.download_as_bytearray()

    try:
        # SDK UPDATE: Correct generation of the image part
        image_part = types.Part.from_bytes(data=bytes(photo_bytes), mime_type="image/jpeg")

        user_id = update.message.from_user.id
        supabase_client = context.bot_data['supabase_client']
        current_profile = await asyncio.to_thread(_get_user_profile, supabase_client, user_id) 
        language_setting = next((row['value_data'] for row in current_profile if row['key_name'] == 'user_language'), 'en')

        # Build the content list (User caption + Image object)
        user_caption = update.message.caption or "Describe this image in detail."
        prompt_contents = [user_caption, image_part]

        # UPDATE: Now using call_gemini_with_retry with temperature at 0.4
        response_text = await call_gemini_with_retry(
            model='gemini-3.5-flash', # A top model, great with multimodal vision
            contents=prompt_contents, 
            system_prompt=f"You are a helpful assistant analyzing an image. Your response MUST be in the language with this ISO 639-1 code: {language_setting}.",
            temperature=0.4 # A higher temperature allows for more fluid and expressive text
        )

        await update.message.reply_text(response_text)

    except Exception as e:
        logging.error("Error processing photo message: %s", e)
        await update.message.reply_text("Spiacente, non sono riuscito ad analizzare l'immagine.")


if __name__ == "__main__":
    """Initializes and runs the Telegram bot."""
    logging.info("--- STARTING TELEGRAM BOT ---")

    # Create a single, global Supabase client using the powerful service_role key.
    # This client bypasses RLS, so security is handled within the Python code.
    supabase_client = create_client(COMPLETE_SUPABASE_URL, SUPABASE_SERVICE_KEY)

    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    application.bot_data['supabase_client'] = supabase_client

    # Register handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    application.add_handler(MessageHandler(filters.VOICE, handle_voice_message))
    application.add_handler(MessageHandler(filters.PHOTO, handle_photo_message))

    # Run the bot
    application.run_polling()
