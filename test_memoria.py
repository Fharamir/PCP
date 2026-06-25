import os
import time
from typing import List, Literal
from google import genai
from google.genai import errors, types
from pydantic import BaseModel, Field
from supabase import create_client, Client, ClientOptions
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from dotenv import load_dotenv

# --- CONFIGURAZIONE CREDENZIALI (CARICATE IN MODO SICURO) ---
load_dotenv(dotenv_path='accessdata.env') # Carica le variabili dal file specificato
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
EMAIL_CLIENTE = os.getenv("EMAIL_CLIENTE")
PASSWORD_SICURA = os.getenv("PASSWORD_SICURA")
MAX_STORAGE = 5

google_client = genai.Client(api_key=GEMINI_API_KEY)

																 
COMPLETE_SUPABASE_URL = f"https://{SUPABASE_URL}.supabase.co"

# --- CACHE LOCALE IN RAM ---
class AgentCache:
    _storage = {}
    @classmethod
    def get_profile(cls, user_id): return cls._storage.get(user_id, None)
    @classmethod
    def set_profile(cls, user_id, data): cls._storage[user_id] = data
    @classmethod
    def invalidate(cls, user_id):
        if user_id in cls._storage: del cls._storage[user_id]

class SinglePreference(BaseModel):
    key_name: str = Field(description="The specific category and entity in snake_case and strictly in English (e.g., max_budget_motorcycle).")
    value_data: str = Field(description="The extracted value translated to English. For DELETE actions, this can be empty.")
    context_desc: str = Field(description="The full context in English. For DELETE actions, describe what is being deleted.")
    action_type: Literal["upsert", "delete"] = Field(description="Use 'upsert' to save/update data. Use 'delete' ONLY if requested.")
    is_sensitive: bool = Field(description="True if the data involves personal, financial, private, or sensitive information (e.g., IBAN, health, tastes). False if it's a generic technical setting or app configuration.")
    memory_target: Literal["profile", "chat_memories", "all"] = Field(description="Only in case of deletion, use 'profile' to affect user_profile table. Use 'chat_memories' to delete recent semantic memories/chat logs. Use 'all' if the user wants to completely wipe out both profile data and chat memories.")

class ProfileUpdateBatch(BaseModel):
    preferences: List[SinglePreference] = Field(description="List of all profile preferences found.")

# --- PARACADUTE 1: RETRY PER GENERAZIONE TESTI ---
@retry(
    retry=retry_if_exception_type((errors.APIError, errors.ServerError, errors.ClientError)),
    wait=wait_exponential(multiplier=1, min=1, max=10), 
    stop=stop_after_attempt(3),
    reraise=True 
)
def chiama_gemini_con_retry(modello, prompt, prompt_sistema, schema_output=None, mime_type=None):
    config_params = types.GenerateContentConfig(system_instruction=prompt_sistema, temperature=0.1)
    if schema_output: config_params.response_schema = schema_output
    if mime_type: config_params.response_mime_type = mime_type
    response = google_client.models.generate_content(model=modello, contents=prompt, config=config_params)
    return response.text

# --- 🔥 NUOVO PARACADUTE 2: RETRY SPECIFICO PER GLI EMBEDDING (ANTI-503) ---
@retry(
    retry=retry_if_exception_type((errors.APIError, errors.ServerError, errors.ClientError)),
    wait=wait_exponential(multiplier=1, min=1, max=10), 
    stop=stop_after_attempt(3),
    reraise=True 
)
def genera_embedding_con_retry(testo_da_convertire):
    """Genera il vettore numerico proteggendo l'app dai sovraccarichi 503 di Google"""
    print(f"🤖 Invocazione modello gemini-embedding-2 (Tentativo)...")
    response = google_client.models.embed_content(
        model="gemini-embedding-2", 
        contents=testo_da_convertire
    )
    return response.embeddings[0].values

def ottieni_client_autenticato(email, password):
    complete_url = "https://" + SUPABASE_URL + ".supabase.co"
    client_base = create_client(complete_url, SUPABASE_KEY)
    sessione = client_base.auth.sign_in_with_password({"email": email, "password": password})
    opzioni_client = ClientOptions(headers={"Authorization": f"Bearer {sessione.session.access_token}"})
    return create_client(complete_url, SUPABASE_KEY, options=opzioni_client), sessione.user.id

def cerca_ricordi_passati(client_sb, user_id, testo_ricerca):
    """Genera l'embedding ed esegue il RAG interrogando Supabase"""
    try:
        res_trad = chiama_gemini_con_retry(
            modello='gemini-2.5-flash-lite',
            prompt=f"Translate this search query into a concise English keywords sentence: '{testo_ricerca}'",
            prompt_sistema="Translate the search intent into crisp English keywords."
        )
        query_inglese = res_trad.strip()
		
																										 
        
        # CORREZIONE: Ora la generazione sfrutta il meccanismo di auto-retry protetto
        vettore = genera_embedding_con_retry(query_inglese)
        
        risposta_db = client_sb.rpc("match_memories", {
            "query_embedding": vettore,
            "match_threshold": 0.4,
            "match_count": 3,
            "p_user_id": user_id
        }).execute()
        return [riga['memory_text'] for riga in risposta_db.data]
    except Exception as e:
        print(f"⚠️ Errore ricerca vettoriale: {e}")
        return []

def elabora_input(client_sb, user_id, frase_utente):
    """Metodo unico che gestisce la cronologia e la memoria a lungo termine sul Cloud"""
											 
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
            risposta_db = client_sb.table("user_profile").select("key_name", "value_data", "context_desc").execute()
            profilo_attuale = risposta_db.data
            AgentCache.set_profile(user_id, profilo_attuale)
        except Exception:
            profilo_attuale = []

    avvenuto_aggiornamento = False

									
    print("🔍 [BACKGROUND] Analisi testo ed estrazione preferenze...")
    prompt_sistema_unificato = f"""
    You are an advanced data extraction assistant. Process the user text to find profile preferences, constraints, or any settings.
    Rules:
    1. Translate into ENGLISH JSON.
    2. ANTI-DUPLICATION: Existing profile: {profilo_attuale}. Reuse exact key names if they match the concept.
    3. If no preferences are found, return empty list [].
    4. GDPR COMPLIANCE: If the user asks to forget, delete, or clear personal data, identify the key and set action_type to 'delete'. Set 'is_sensitive' to True for any personal, financial, or private data.
    5. PROTECTED KEYS: Technical application settings like 'user_language' and 'user_timezone' CANNOT be deleted by the user. If the user asks to clear the entire memory, list all other existing keys as 'delete' but EXCLUDE these technical settings.
    5. Use the recent chat history for context if needed: {chat_history}
    """
    
    try:
        json_risposta = chiama_gemini_con_retry('gemini-2.5-flash-lite', frase_utente, prompt_sistema_unificato, ProfileUpdateBatch, "application/json")
        batch_dati = ProfileUpdateBatch.model_validate_json(json_risposta)
        
        # Lista per accumulare i record che hanno davvero bisogno di essere scritti
        records_da_scrivere = []
        chiavi_da_cancellare = []
        cancella_memoria_chat = False

        if batch_dati.preferences:
            for dati in batch_dati.preferences:
                # 🗑️ CASO 1: RICHIESTA DI CANCELLAZIONE (DELETE GDPR TOTAL)
                if dati.action_type == "delete":
                    
                    # Sotto-caso A: Cancellazione della sola memoria della chat o di tutto (GDPR)
                    if dati.memory_target in ["chat_memories", "all"]:
                        print(f"🗑️ [GDPR PURGE] Rilevata richiesta di cancellazione per la memoria semantica dei ricordi chat.")
                        cancella_memoria_chat = True
                    
                    # Sotto-caso B: Cancellazione dei dati del profilo
                    if dati.memory_target in ["profile", "all"]:
                        # Se l'utente chiede un wipe totale ('all'), saltiamo il controllo della singola chiave
                        if dati.memory_target == "all":
                            print("🗑️ [GDPR PURGE] Pianificato reset di tutte le chiavi del profilo (escluse quelle di sistema).")
                            # Accumuliamo tutte le chiavi attuali che sono cancellabili (is_sensitive=True)
                            chiavi_da_cancellare.extend([r['key_name'] for r in profilo_attuale if r.get('is_sensitive', False)])
                        else:
                            # Cancellazione chirurgica di una singola chiave del profilo
                            record_esistente = next((riga for riga in profilo_attuale if riga['key_name'] == dati.key_name), None)
                            # La chiave è protetta se NON è marcata come sensibile nel DB.
                            if record_esistente and not record_esistente.get('is_sensitive', False):
                                print(f"🛡️ [SECURITY] Cancellazione negata per chiave di sistema: '{dati.key_name}'")
                            else: # Se la chiave non esiste o è sensibile, può essere cancellata.
                                chiavi_da_cancellare.append(dati.key_name)
                
                # 💾 CASO 2: ACCUMULO SCRITTURE (UPSERT)
                elif dati.action_type == "upsert":
                    record_vecchio = next((riga for riga in profilo_attuale if riga['key_name'] == dati.key_name), None)
                    scrittura_necessaria = True
                    if record_vecchio and record_vecchio['value_data'] == str(dati.value_data) and record_vecchio['context_desc'] == dati.context_desc:
                        scrittura_necessaria = False
                    
                    if scrittura_necessaria:
                        # Accumuliamo il record nel dizionario invece di fare .execute() subito
                        records_da_scrivere.append({
                            "user_id": user_id, 
                            "key_name": dati.key_name, 
                            "value_data": dati.value_data, 
                            "context_desc": dati.context_desc,
                            "is_sensitive": dati.is_sensitive
                        })

            azioni_eseguite = []
            
            # 🔥 1. ESECUZIONE BULK DELETE MEMORIA SEMANTICA (Cancellazione ricordi chat)
            if cancella_memoria_chat:
                print("🗑️ [BULK DML] Eliminazione in corso di tutti i ricordi vettoriali da agent_memories...")
                client_sb.table("agent_memories").delete().eq("user_id", user_id).execute()
                avvenuto_aggiornamento = True
                azioni_eseguite.append("deleted all recent chat memories")

            # 🔥 2. ESECUZIONE BULK DELETE PROFILO
            if chiavi_da_cancellare:
                print(f"🗑️ [BULK DML] Eliminazione in corso di {len(chiavi_da_cancellare)} chiavi personali dal profilo...")
                # Usiamo set() per rimuovere eventuali duplicati prima della query
                client_sb.table("user_profile").delete().eq("user_id", user_id).in_("key_name", list(set(chiavi_da_cancellare))).execute()
                avvenuto_aggiornamento = True
                azioni_eseguite.append(f"deleted {len(chiavi_da_cancellare)} settings from profile")

            # 🔥 3. ESECUZIONE BULK UPSERT PROFILO
            if records_da_scrivere:
                # (Esegue l'upsert bulk visto prima)
                client_sb.table("user_profile").upsert(records_da_scrivere, on_conflict="user_id,key_name").execute()
                avvenuto_aggiornamento = True
                azioni_eseguite.append(f"saved/updated {len(records_da_scrivere)} preferences in profile")
            
            if azioni_eseguite:
                lista_azioni = " " + ", and you ".join(azioni_eseguite) + "."
            else:
                lista_azioni = " did no database operations in this specific turn."
                
        else:
            print("⏸️ [BACKGROUND SKIP] Nessuna preferenza rilevata.")
    
    except Exception as e:
        print(f"⚠️ Errore analisi: {e}")

    if avvenuto_aggiornamento:
        AgentCache.invalidate(user_id)
        risposta_db = client_sb.table("user_profile").select("key_name", "value_data", "context_desc").execute()
        profilo_attuale = risposta_db.data
        AgentCache.set_profile(user_id, profilo_attuale)
    
    impostazione_lingua = next((riga['value_data'] for riga in profilo_attuale if riga['key_name'] == 'user_language'), 'en')
    lingues_mappate = {"it": "ITALIAN", "en": "ENGLISH", "ja": "JAPANESE"}
    lingua_estesa = lingues_mappate.get(impostazione_lingua, "ENGLISH")

    prompt_risposta = f"""
    You are a professional personal AI assistant. You MUST reply strictly in {lingua_estesa}.
    Context available to help you answer:
    - USER PROFILE KEYS: {profilo_attuale}
    - RECENT CHAT LOGS: {chat_history}
    - RETRIEVED LONG-TERM MEMORIES (RAG): {ricordi_pertinenti}
    - OTHER DATA to use for the answer if necessary:
      - You can use the user_name information (if provided) to enrich the response
      - Your're working on behalf of Personal Copilot Project (by Fharamir)
      - You have memory of user profile data and chat history
      - You're able to read, write and delete memory (both profile info and chat history) as requested by the user
      - You cannot delete user's system settings unless the user asks for profile deletion (different from memory delete)
      - In this confersation you {lista_azioni}
    """
    
    try:
        risposta_finale = chiama_gemini_con_retry('gemini-2.5-flash', frase_utente, prompt_risposta)
        print(f"\n💬 Input Utente: '{frase_utente}'")
        print(f"🤖 Risposta Assistente ({lingua_estesa}):\n{risposta_finale}\n------------------------------------------------")
        
        # 4. SALVATAGGIO INTERAZIONE VETTORIALE CON PROTEZIONE RETRY
        testo_scambio = f"User said: '{frase_utente}' | Assistant replied: '{risposta_finale}'"
        
        scambio_inglese = chiama_gemini_con_retry(
            modello='gemini-2.5-flash-lite',
            prompt=f"Summarize this interaction into a concise English fact: {testo_scambio}",
            prompt_sistema="Summarize interaction into clean facts."
        )
											  
        
        # CORREZIONE: Usiamo la nuova funzione protetta anche per salvare la memoria finale
        vettore_scambio = genera_embedding_con_retry(scambio_inglese)
															 
														   
        
        client_sb.table("agent_memories").insert({
            "user_id": user_id, "memory_text": scambio_inglese, "memory_vector": vettore_scambio
        }).execute()
        
        return risposta_finale
    except Exception as e:
        print(f"❌ Errore risposta: {e}")
        return "Sorry, error occurred."

					   
if __name__ == "__main__":
    supabase_auth, id_cliente = ottieni_client_autenticato(EMAIL_CLIENTE, PASSWORD_SICURA)
    if id_cliente:
        print("--- AVVIO AGENTE AD ARCHITETTURA COMPLETAMENTE BLINDATA ---\n")
        elabora_input(supabase_auth, id_cliente, "Cancella i miei dati personali")
