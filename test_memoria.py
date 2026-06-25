"""
Test script for a personal AI assistant with long-term memory.

This script implements a conversational agent that uses Supabase for data persistence
(user profile and vector memory) and Google Gemini APIs for generative and embedding
capabilities. It includes caching mechanisms, API call retries, and an architecture
for structured memory management.
"""
import os
import time
from typing import List, Literal
from google import genai
from google.genai import errors, types
from pydantic import BaseModel, Field
from supabase import create_client, Client, ClientOptions
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from dotenv import load_dotenv

# --- Configuration and Constants ---
load_dotenv(dotenv_path='accessdata.env') # Load environment variables from the specified file
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
EMAIL_CLIENTE = os.getenv("EMAIL_CLIENTE")
PASSWORD_SICURA = os.getenv("PASSWORD_SICURA")
MAX_STORAGE = 5

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

@retry(
    retry=retry_if_exception_type((errors.APIError, errors.ServerError, errors.ClientError)),
    wait=wait_exponential(multiplier=1, min=1, max=10), 
    stop=stop_after_attempt(3),
    reraise=True 
)
def chiama_gemini_con_retry(modello, prompt, prompt_sistema, schema_output=None, mime_type=None):
    """Makes a call to a Gemini model with automatic retry handling for API errors."""
    config_params = {
        "system_instruction": prompt_sistema,
        "temperature": 0.1,
        "response_schema": schema_output,
        "response_mime_type": mime_type
    }
    response = google_client.models.generate_content(model=modello, contents=prompt, config=config_params)
    return response.text

@retry(
    retry=retry_if_exception_type((errors.APIError, errors.ServerError, errors.ClientError)),
    wait=wait_exponential(multiplier=1, min=1, max=10), 
    stop=stop_after_attempt(3),
    reraise=True 
)
def genera_embedding_con_retry(testo_da_convertire):
    """Generates text embedding with automatic retry handling, specific to embedding APIs."""
    response = google_client.models.embed_content(
        model="gemini-embedding-2", 
        contents=testo_da_convertire
    )
    return response.embeddings[0].values

def ottieni_client_autenticato(email, password):
    """Authenticates the user on Supabase and returns an authenticated client and the user ID."""
    complete_url = "https://" + SUPABASE_URL + ".supabase.co"
    client_base = create_client(complete_url, SUPABASE_KEY)
    sessione = client_base.auth.sign_in_with_password({"email": email, "password": password})
    opzioni_client = ClientOptions(headers={"Authorization": f"Bearer {sessione.session.access_token}"})
    return create_client(complete_url, SUPABASE_KEY, options=opzioni_client), sessione.user.id

def cerca_ricordi_passati(client_sb, user_id, testo_ricerca):
    """Performs a semantic search (RAG) in the user's long-term memory on Supabase."""
    try:
        res_trad = chiama_gemini_con_retry(
            modello='gemini-2.5-flash-lite',
            prompt=f"Translate this search query into a concise English keywords sentence: '{testo_ricerca}'",
            prompt_sistema="Translate the search intent into crisp English keywords."
        )
        query_inglese = res_trad.strip()
		
        try:
            vettore = genera_embedding_con_retry(query_inglese)
        # If embedding generation fails, return an empty list to avoid blocking the response.
        except Exception:
            return []
                
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

def elabora_input(client_sb, user_id, frase_utente):
    """Orchestrates the entire response process: retrieves context, updates memory, and generates a response."""
    lista_azioni = " no specific database operations were performed in this turn." # Safety initialization
											 
    try:
        risposta_cronologia = client_sb.table("agent_memories").select("memory_text").eq("user_id", user_id).order("created_at", ascending=False).limit(MAX_STORAGE).execute()
        chat_history = [riga['memory_text'] for riga in risposta_cronologia.data]
        chat_history.reverse()
    except Exception:
        chat_history = []

    ricordi_pertinenti = cerca_ricordi_passati(client_sb, user_id, frase_utente)

    profilo_attuale = AgentCache.get_profile(user_id)
    if profilo_attuale is None:
        try:
            risposta_db = client_sb.table("user_profile").select("key_name", "value_data", "context_desc", "is_sensitive").execute()
            profilo_attuale = risposta_db.data
            AgentCache.set_profile(user_id, profilo_attuale)
        except Exception:
            profilo_attuale = []

    avvenuto_aggiornamento = False

    # 1. Structured Extraction: Analyzes the input for data management commands or preferences.
    prompt_sistema_unificato = f"""
    You are a data extraction assistant. Your task is to find preferences or data management commands in the user's text and translate them into a structured JSON format.
    Rules:
    1. All output must be in ENGLISH JSON.
    2. If the user expresses a preference, use `action_type: "upsert"`.
    3. If the user asks to delete or forget data:
        - If the user asks to delete/forget a SPECIFIC item (e.g., "forget my favorite color"), use `action_type: "delete"` and specify the `key_name`.
        - If the user asks to delete/forget their PROFILE data (e.g., "delete my personal data"), use `action_type: "delete"` with `memory_target: "profile"`.
        - If the user asks to delete/forget CHAT HISTORY, use `action_type: "delete"` with `memory_target: "chat_memories"`.
        - If the user asks to delete/forget EVERYTHING, use `action_type: "delete"` with `memory_target: "all"`.
    4. If no preferences ("upsert") or forget commands ("delete") are found in the user prompt, return an empty list `[]`.
    - Existing profile for context: {profilo_attuale}
    - Recent chat for context: {chat_history}
    """
    
    try:
        json_risposta = chiama_gemini_con_retry('gemini-2.5-flash-lite', frase_utente, prompt_sistema_unificato, ProfileUpdateBatch, "application/json")
        batch_dati = ProfileUpdateBatch.model_validate_json(json_risposta)
        
        # Lists to accumulate records for bulk operations.
        records_da_scrivere = []
        chiavi_da_cancellare = []
        cancella_memoria_chat = False

        if batch_dati.preferences:
            for dati in batch_dati.preferences:
                # Case 1: Handle deletion requests.
                if dati.action_type == "delete":
                    
                    # Sub-case A: Deletion of chat memory or everything (GDPR).
                    if dati.memory_target in ["chat_memories", "all"] and not cancella_memoria_chat:
                        print(f"🗑️ [GDPR PURGE] Deletion request detected for semantic chat memories.")
                        cancella_memoria_chat = True
                    
                    # Sub-case B: Deletion of profile data.
                    if dati.memory_target in ["profile", "all"]:
                        # If the user requests a total wipe ('all'), we skip the single key check.
                        if dati.memory_target == "all":
                            print("🗑️ [GDPR PURGE] Profile reset scheduled (system keys excluded).")
                            # Accumulate all current keys that are deletable (is_sensitive=True).
                            chiavi_da_cancellare.extend([r['key_name'] for r in profilo_attuale if r.get('is_sensitive', False)])
                        else:
                            # Surgical deletion of a single profile key.
                            record_esistente = next((riga for riga in profilo_attuale if riga['key_name'] == dati.key_name), None)
                            # A key is protected if it is NOT marked as sensitive in the DB.
                            if record_esistente and not record_esistente.get('is_sensitive', False):
                                print(f"🛡️ [SECURITY] Deletion denied for system key: '{dati.key_name}'")
                            else: # If the key doesn't exist or is sensitive, it can be deleted.
                                chiavi_da_cancellare.append(dati.key_name)
                
                # Case 2: Handle save/update requests.
                elif dati.action_type == "upsert":
                    record_vecchio = next((riga for riga in profilo_attuale if riga['key_name'] == dati.key_name), None)
                    scrittura_necessaria = True
                    if record_vecchio and record_vecchio['value_data'] == str(dati.value_data) and record_vecchio['context_desc'] == dati.context_desc:
                        scrittura_necessaria = False
                    
                    if scrittura_necessaria and dati.value_data:
                        # Accumulate records to be written for a bulk operation.
                        records_da_scrivere.append({
                            "user_id": user_id, 
                            "key_name": dati.key_name, 
                            "value_data": dati.value_data, 
                            "context_desc": dati.context_desc,
                            "is_sensitive": dati.is_sensitive
                        })

            azioni_eseguite = []
            
            # Execute DML operations in bulk for efficiency.
            if cancella_memoria_chat:
                print("🗑️ [BULK DML] Deleting all vector memories from agent_memories...")
                client_sb.table("agent_memories").delete().eq("user_id", user_id).execute()
                avvenuto_aggiornamento = True
                azioni_eseguite.append("deleted all recent chat memories")

            if chiavi_da_cancellare:
                print(f"🗑️ [BULK DML] Deleting {len(chiavi_da_cancellare)} personal keys from the profile...")
                # Remove any duplicates before the query.
                client_sb.table("user_profile").delete().eq("user_id", user_id).in_("key_name", list(set(chiavi_da_cancellare))).execute()
                avvenuto_aggiornamento = True
                azioni_eseguite.append(f"deleted {len(chiavi_da_cancellare)} settings from profile")

            if records_da_scrivere:
                client_sb.table("user_profile").upsert(records_da_scrivere, on_conflict="user_id,key_name").execute()
                avvenuto_aggiornamento = True
                azioni_eseguite.append(f"saved/updated {len(records_da_scrivere)} preferences in profile")
            
            if azioni_eseguite:
                lista_azioni = " " + ", and you ".join(azioni_eseguite) + "."
            else:
                lista_azioni = " did no database operations in this specific turn."
            
            print(f"✅ [BACKGROUND] Actions performed:{lista_azioni}")
        else:
            print("⏸️ [BACKGROUND SKIP] No preferences detected.")
    except Exception as e:
        # If input analysis fails, stop the operation and inform the user.
        print(f"❌ CRITICAL ERROR in input analysis: {e}")
        messaggio_errore = "I'm sorry, but I'm having trouble processing your request due to a temporary issue. Could you please try again in a moment?"
        return messaggio_errore

    if avvenuto_aggiornamento:
        # Invalidate the local cache if the profile has been modified.
        AgentCache.invalidate(user_id)
        risposta_db = client_sb.table("user_profile").select("key_name", "value_data", "context_desc", "is_sensitive").execute()
        profilo_attuale = risposta_db.data
        AgentCache.set_profile(user_id, profilo_attuale)

    impostazione_lingua = next((riga['value_data'] for riga in profilo_attuale if riga['key_name'] == 'user_language'), 'en')

    # 2. Response Generation: Build the final prompt with all the collected context.
    prompt_risposta = f"""
    You are a professional personal AI assistant. You MUST reply strictly in the language with this ISO 639-1 code: {impostazione_lingua}.
    Context available to help you answer:
    - USER PROFILE KEYS: {profilo_attuale}
    - RECENT CHAT LOGS: {chat_history}
    - RETRIEVED LONG-TERM MEMORIES (RAG): {ricordi_pertinenti}
    - OTHER DATA to use for the answer if necessary:
      - You can use the user_name information (if provided) to enrich the response
      - Your're working on behalf of Personal Copilot Project (by Fharamir)
      - You have memory of user profile data and chat history provided by Personal Copilot Project
      - You're able to read, write and delete memory (both profile info and chat history) as requested by the user
      - You cannot delete user's system settings unless the user asks for profile deletion (different from memory delete)
      - In this confersation you {lista_azioni}
    """
    
    try:
        risposta_finale = chiama_gemini_con_retry('gemini-2.5-flash', frase_utente, prompt_risposta)
        print(f"\n💬 User Input: '{frase_utente}'")
        print(f"🤖 Assistant Response ({impostazione_lingua}):\n{risposta_finale}\n" + "-"*60)
        
        # 3. Memory Storage: Save the current interaction in the long-term vector memory.
        testo_scambio = f"User said: '{frase_utente}' | Assistant replied: '{risposta_finale}'"
        
        # Optimization: Translate and summarize only if the conversation is not already in English.
        if impostazione_lingua != 'en':
            scambio_inglese = chiama_gemini_con_retry(
                modello='gemini-2.5-flash-lite',
                prompt=f"Summarize this interaction into a concise English fact: {testo_scambio}",
                prompt_sistema="Summarize interaction into clean facts."
            )
        else:
            scambio_inglese = testo_scambio
        
        vettore_scambio = genera_embedding_con_retry(scambio_inglese)
        
        client_sb.table("agent_memories").insert({
            "user_id": user_id, "memory_text": scambio_inglese, "memory_vector": vettore_scambio
        }).execute()
        
        return risposta_finale
    except Exception as e:
        print(f"❌ Response error: {e}")
        return "Sorry, error occurred."

					   
if __name__ == "__main__":
    # Main execution block to test the agent's functionality.
    supabase_auth, id_cliente = ottieni_client_autenticato(EMAIL_CLIENTE, PASSWORD_SICURA)
    if id_cliente:
        print("--- CONVERSATIONAL AGENT STARTED (type 'exit' to end) ---\n")
        while True:
            frase_utente = input("You: ")
            if frase_utente.lower() in ["esci", "exit", "quit"]:
                print("--- SESSION TERMINATED ---")
                break
            elabora_input(supabase_auth, id_cliente, frase_utente)
