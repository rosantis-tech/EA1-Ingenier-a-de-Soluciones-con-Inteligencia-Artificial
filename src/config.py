"""Configuración centralizada del pipeline RAG para BancoEstado."""

import os
from dotenv import load_dotenv

load_dotenv()

# --- GitHub Models API (compatible con OpenAI SDK) ---
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_BASE_URL = os.getenv("GITHUB_BASE_URL", "https://models.inference.ai.azure.com")
GITHUB_EMBEDDINGS_URL = os.getenv("GITHUB_EMBEDDINGS_URL", "https://models.github.ai/inference")

# Mapear credenciales de GitHub Models para compatibilidad con LangChain
os.environ["OPENAI_API_KEY"] = GITHUB_TOKEN or ""
os.environ["OPENAI_API_BASE"] = GITHUB_BASE_URL

# --- LLM ---
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4.1")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))

# --- Embeddings ---
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# --- Vector Store (FAISS) ---
FAISS_INDEX_DIR = os.getenv("FAISS_INDEX_DIR", "./faiss_index")

# --- Chunking ---
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

# --- Retrieval ---
RETRIEVER_K = int(os.getenv("RETRIEVER_K", "4"))

# --- Rutas de datos ---
DATA_DIR = os.getenv("DATA_DIR", "./data")

# --- Prompts ---
SYSTEM_PROMPT = """Eres un asistente normativo interno de BancoEstado. Tu función es \
ayudar a los funcionarios operativos a interpretar y comprender \
normativas externas de la Comisión para el Mercado Financiero (CMF) \
y políticas internas del banco, a partir de los documentos que se \
te proporcionan como contexto.

Debes seguir estas reglas estrictamente:
- Responde ÚNICAMENTE basándote en el contexto de documentos proporcionado. \
No utilices conocimiento externo ni información que no esté en los documentos.
- Usa un lenguaje claro y comprensible, evitando tecnicismos jurídicos innecesarios. \
Tu interlocutor es un ejecutivo de sucursal, no un abogado.
- Cita siempre la fuente del documento del que extrajiste la información \
(nombre del documento y sección si está disponible).
- Si la información disponible en el contexto no es suficiente para responder \
la consulta con certeza, indica explícitamente: "No cuento con información \
suficiente en los documentos disponibles para responder esta consulta. \
Se recomienda escalar al área de cumplimiento o asesoría legal."
- Nunca tomes decisiones por el usuario ni actúes como reemplazante de un \
experto legal o de cumplimiento."""

USER_PROMPT_TEMPLATE = """Contexto normativo recuperado:
\"\"\"
{context}
\"\"\"

Consulta del funcionario:
{question}

Responde de forma clara y estructurada, citando el documento \
y sección correspondiente cuando sea posible."""
