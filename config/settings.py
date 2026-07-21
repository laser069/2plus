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
# How long Ollama keeps a model resident in VRAM after a call.
# "30m" = keep 30 min; "-1" = never unload; "0" = unload immediately.
# Pinning the chat model avoids cold reloads on tight-VRAM GPUs.
OLLAMA_KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "30m")

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

# --- Router cache ---
ROUTER_CACHE_SIZE = 128    # LRU slots for classify() result cache

# --- Router LLM fallback ---
# When False (default), ambiguous multi-signal queries resolve to the UNION of
# matched heuristic tags instead of calling an LLM. This avoids loading a second
# model (qwen3.5:4b) on tight-VRAM GPUs and never touches a local model when the
# chat model is a cloud one. Set True to restore LLM-based disambiguation.
ROUTER_LLM_FALLBACK = os.getenv("ROUTER_LLM_FALLBACK", "0") == "1"
