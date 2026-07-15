from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

# --- Model router ---
MODEL_ROUTER = {
    "default": "qwen3:8b",
    "fast":    "qwen3.5:4b",
    "embed":   "all-minilm:l6-v2",
    "cloud":   None,              # seam: wire a cloud model here later
}

# --- Ollama ---
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# --- OpenRouter ---
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# --- Paths ---
BASE_DIR         = Path(__file__).resolve().parent.parent
DATA_DIR         = BASE_DIR / "data"
CHROMA_PERSIST   = str(DATA_DIR / "chroma")
SQLITE_PATH      = str(DATA_DIR / "2plus.db")
LOG_DIR          = str(BASE_DIR / "logs")

# --- ReAct loop ---
MAX_REACT_STEPS  = 5

# --- RAG ---
TOP_K_RETRIEVAL  = 4
CHUNK_SIZE       = 800
CHUNK_OVERLAP    = 100

# --- Context budget (chars injected per prompt) ---
CTX_BUDGET_CHARS  = 6000
FACTS_MAX_CHARS   = 800
SUMMARY_MAX_CHARS = 600
RAG_CHUNK_MAX     = 1200   # max chars per retrieved chunk

# --- Conversation window ---
CONVO_WINDOW     = 6       # verbatim turns before summarising older ones
