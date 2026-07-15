import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from memory.convo import ConvoMemory
from memory.chat_history import load_session, save_message, delete_session
from orchestrator.agent import Agent
import memory.user_facts as uf
from rag.ingestion import ingest, list_docs
from config.settings import MODEL_ROUTER, OPENROUTER_API_KEY

st.set_page_config(page_title="2Plus", page_icon="⚡", layout="wide")

st.markdown("""
<style>
/* ── Base ── */
html, body, [data-testid="stApp"] {
    background-color: #0a0a0f;
    color: #d8d8e4;
    font-size: 14px;
    line-height: 1.6;
    font-family: -apple-system, BlinkMacSystemFont, 'Inter', system-ui, sans-serif;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: #0d0d14 !important;
    border-right: 1px solid rgba(255,255,255,0.06) !important;
}
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] .stCaption {
    color: #6666888 !important;
}
[data-testid="stSidebar"] label {
    color: #888899 !important;
    font-size: 0.72rem !important;
}
[data-testid="stSidebar"] h1 {
    color: #d8d8e4 !important;
    font-size: 1.1rem;
    font-weight: 700;
    letter-spacing: -0.3px;
}
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
    color: #44446a !important;
    font-size: 0.65rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
    margin-top: 1.4rem !important;
    margin-bottom: 0.4rem !important;
}

/* ── Mode Switcher (radio as segmented control) ── */
[data-testid="stSidebar"] [data-testid="stRadio"] > div:first-child {
    display: none !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] {
    display: flex !important;
    gap: 2px !important;
    background: rgba(0,0,0,0.35) !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 9px !important;
    padding: 3px !important;
    width: 100% !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label {
    flex: 1 !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    padding: 6px 4px !important;
    border-radius: 7px !important;
    font-size: 0.68rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.5px !important;
    color: #33334a !important;
    cursor: pointer !important;
    transition: background 0.1s, color 0.1s !important;
    border: 1px solid transparent !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked),
[data-testid="stSidebar"] [data-testid="stRadio"] label:has([aria-checked="true"]) {
    background: #18182a !important;
    color: #d0d0e4 !important;
    border-color: rgba(255,255,255,0.09) !important;
}
/* Hide BaseWeb radio indicator circles */
[data-testid="stSidebar"] [data-testid="stRadio"] [data-baseweb="radio"] {
    width: 0 !important;
    height: 0 !important;
    overflow: hidden !important;
    display: none !important;
}

/* ── Main header ── */
h1 {
    color: #e8e8f0 !important;
    font-size: 1.15rem !important;
    font-weight: 700 !important;
    letter-spacing: -0.2px !important;
}

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    background-color: #111118 !important;
    border-radius: 8px !important;
    margin-bottom: 6px !important;
    padding: 0.75rem 1rem !important;
    border: 1px solid rgba(255,255,255,0.05) !important;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    border-left: 2px solid #4f8ef7 !important;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
    border-left: 2px solid #6644cc !important;
}

/* ── Chat input ── */
[data-testid="stChatInput"] {
    background-color: #111118 !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 10px !important;
    color: #d8d8e4 !important;
    font-size: 14px !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: rgba(79,142,247,0.4) !important;
    box-shadow: 0 0 0 2px rgba(79,142,247,0.12) !important;
}

/* ── Buttons ── */
.stButton > button {
    background-color: #111118 !important;
    color: #888899 !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 7px !important;
    font-size: 0.78rem !important;
    transition: all 0.12s !important;
}
.stButton > button:hover {
    border-color: rgba(255,255,255,0.15) !important;
    color: #d0d0e4 !important;
}

/* ── Selectbox / text inputs ── */
[data-testid="stSelectbox"] > div > div,
[data-testid="stTextInput"] > div > div > input {
    background-color: #0e0e18 !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    color: #d8d8e4 !important;
    border-radius: 7px !important;
    font-size: 0.8rem !important;
}
[data-testid="stTextInput"] > div > div > input::placeholder {
    color: #3a3a5a !important;
}
[data-testid="stTextInput"] > div > div > input:focus {
    border-color: rgba(255,255,255,0.15) !important;
    box-shadow: none !important;
}

/* ── Expanders ── */
[data-testid="stExpander"] {
    background-color: #0e0e16 !important;
    border: 1px solid rgba(255,255,255,0.05) !important;
    border-radius: 7px !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background-color: #0e0e16 !important;
    border: 1px dashed rgba(255,255,255,0.08) !important;
    border-radius: 7px !important;
}

/* ── Toggle ── */
[data-testid="stToggle"] label { color: #666688 !important; font-size: 0.78rem !important; }

/* ── Success / warning / info ── */
.stSuccess { background-color: #0d1f10 !important; border-color: rgba(45,212,191,0.2) !important; }
.stWarning { background-color: #1e180a !important; border-color: rgba(245,158,11,0.25) !important; }
.stInfo    { background-color: #0e111e !important; border-color: rgba(79,142,247,0.2) !important; }

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, [data-testid="stToolbar"] { display: none !important; }

/* ── Divider ── */
hr { border-color: rgba(255,255,255,0.05) !important; }

/* ── Caption text ── */
.stCaption, [data-testid="stCaptionContainer"] { color: #44446a !important; font-size: 0.68rem !important; }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]
if "convo" not in st.session_state:
    st.session_state.convo = ConvoMemory()
if "agent" not in st.session_state:
    st.session_state.agent = Agent(st.session_state.convo)
if "messages" not in st.session_state:
    st.session_state.messages = load_session(st.session_state.session_id)
if "selected_model" not in st.session_state:
    st.session_state.selected_model = MODEL_ROUTER["default"]
if "think_mode" not in st.session_state:
    st.session_state.think_mode = False
if "or_model" not in st.session_state:
    st.session_state.or_model = ""
if "inference_mode" not in st.session_state:
    st.session_state.inference_mode = "local"


# ── Ollama model list (cached 60 s) ───────────────────────────────────────────
@st.cache_data(ttl=60)
def _get_models() -> list[str]:
    try:
        import ollama as _ollama
        return [m.model for m in _ollama.list().models]
    except Exception:
        return [MODEL_ROUTER["default"]]


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚡ 2Plus")
    st.caption(f"session `{st.session_state.session_id}`")

    # ── Inference mode switcher ──
    st.subheader("Inference")
    _mode_labels = {"local": "⬡  LOCAL", "auto": "⟳  AUTO", "cloud": "☁  CLOUD"}
    _mode_list = list(_mode_labels.keys())
    _mode_idx = _mode_list.index(st.session_state.inference_mode)

    _selected = st.radio(
        "Mode",
        options=_mode_list,
        format_func=lambda x: _mode_labels[x],
        index=_mode_idx,
        horizontal=True,
        label_visibility="collapsed",
    )
    if _selected != st.session_state.inference_mode:
        st.session_state.inference_mode = _selected
        st.rerun()

    _mode = st.session_state.inference_mode

    # ── Local model (Ollama) ──
    if _mode in ("local", "auto"):
        st.subheader("Local Model")
        available_models = _get_models()
        _cur_idx = (
            available_models.index(st.session_state.selected_model)
            if st.session_state.selected_model in available_models
            else 0
        )
        st.session_state.selected_model = st.selectbox(
            "Ollama model",
            available_models,
            index=_cur_idx,
            label_visibility="collapsed",
        )
        if _mode == "auto":
            st.caption("Primary — falls back to cloud if unavailable")

    # ── Cloud model (OpenRouter) ──
    if _mode in ("cloud", "auto"):
        st.subheader("Cloud Model" if _mode == "cloud" else "Cloud Fallback")
        st.session_state.or_model = st.text_input(
            "OpenRouter model",
            value=st.session_state.or_model,
            placeholder="anthropic/claude-3.5-sonnet",
            help="Any OpenRouter model ID in provider/model format",
            label_visibility="collapsed",
        )
        if st.session_state.or_model and not OPENROUTER_API_KEY:
            st.warning("OPENROUTER_API_KEY missing in .env")
        if not st.session_state.or_model:
            st.caption("e.g. openai/gpt-4o · deepseek/deepseek-r1")

    # ── Think mode ──
    st.subheader("Options")
    st.session_state.think_mode = st.toggle(
        "Think mode", value=st.session_state.think_mode,
        help="Enables chain-of-thought (Qwen3 only, slower)"
    )

    # ── Doc upload ──
    st.subheader("Documents")
    uploaded = st.file_uploader(
        "PDF / TXT / MD",
        type=["txt", "md", "pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    if uploaded:
        for f in uploaded:
            content = f.read()
            text = (
                _pdf_to_text(content)
                if f.name.endswith(".pdf")
                else content.decode("utf-8", errors="replace")
            )
            with st.spinner(f"Ingesting {f.name}…"):
                n = ingest(text, doc_id=f.name, metadata={"filename": f.name})
            st.success(f"{f.name}: {n} chunks")

    docs = list_docs()
    if docs:
        with st.expander(f"Docs ({len(docs)})", expanded=False):
            for d in docs:
                st.markdown(f"- `{d}`")

    # ── Memory viewer ──
    st.subheader("Memory")
    facts = {k: v for k, v in uf.get_all().items() if k != "convo_summary"}
    if facts:
        with st.expander(f"Facts ({len(facts)})", expanded=False):
            for k, v in facts.items():
                st.markdown(f"**{k}**: {v}")
    else:
        st.caption("No facts yet.")

    st.markdown("---")
    if st.button("🗑  Clear conversation", use_container_width=True):
        st.session_state.convo.clear()
        delete_session(st.session_state.session_id)
        st.session_state.messages = []
        st.rerun()


# ── Active model resolution ───────────────────────────────────────────────────
_mode = st.session_state.inference_mode
_or = st.session_state.or_model.strip()
_local = st.session_state.selected_model

if _mode == "local":
    _active_model = _local
    _fallback_model = None
elif _mode == "cloud":
    _active_model = _or or _local  # graceful fallback if OR field empty
    _fallback_model = None
else:  # auto
    _active_model = _local
    _fallback_model = _or or None

# ── Header ────────────────────────────────────────────────────────────────────
_MODE_META = {
    "local": ("⬡", "#2dd4bf", "LOCAL"),
    "auto":  ("⟳", "#f59e0b", "AUTO"),
    "cloud": ("☁", "#9d7af7", "CLOUD"),
}
_icon, _color, _label = _MODE_META[_mode]
_display_model = _active_model.split("/")[-1] if "/" in _active_model else _active_model

col_title, col_badge = st.columns([5, 2])
with col_title:
    st.header("2Plus", divider=False)
with col_badge:
    st.markdown(
        f"""<div style='text-align:right;padding-top:1.1rem'>
          <span style='
            display:inline-flex;align-items:center;gap:5px;
            background:rgba(255,255,255,0.04);
            border:1px solid rgba(255,255,255,0.07);
            border-radius:20px;padding:3px 10px 3px 8px;
            font-size:0.65rem;font-weight:700;letter-spacing:0.5px;
            color:{_color}'>
            {_icon} {_label} · <span style='color:#888899;font-weight:500'>{_display_model}</span>
          </span>
        </div>""",
        unsafe_allow_html=True,
    )

# ── Chat history ──────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("citations"):
            with st.expander("Sources", expanded=False):
                for c in msg["citations"]:
                    st.markdown(f"- `{c}`")
        if msg.get("meta"):
            with st.expander("Debug", expanded=False):
                st.json(msg["meta"])

# ── Chat input ────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask me anything…"):
    with st.chat_message("user"):
        st.markdown(prompt)

    st.session_state.messages.append({"role": "user", "content": prompt})
    save_message(st.session_state.session_id, "user", prompt)

    with st.chat_message("assistant"):
        answer_ph = st.empty()
        status_ph = st.empty()
        full_answer = ""
        citations: list[str] = []
        meta: dict = {}

        for event in st.session_state.agent.run_stream(
            prompt,
            model=_active_model,
            think=st.session_state.think_mode,
            fallback_model=_fallback_model,
        ):
            if isinstance(event, dict):
                if event["type"] == "routing":
                    status_ph.caption(f"routing · {', '.join(event['routes']) or 'direct'}")
                elif event["type"] == "tool":
                    status_ph.caption(f"tool · **{event['name']}** (step {event['step']})")
                elif event["type"] == "done":
                    citations = event["citations"]
                    meta = {"routes": event["routes"], "steps": event["steps"], "mode": _mode}
                    status_ph.empty()
            else:
                full_answer += event
                answer_ph.markdown(full_answer + "▌")

        answer_ph.markdown(full_answer)

        if citations:
            with st.expander("Sources", expanded=False):
                for c in citations:
                    st.markdown(f"- `{c}`")
        with st.expander("Debug", expanded=False):
            st.json(meta)

    st.session_state.messages.append({
        "role": "assistant",
        "content": full_answer,
        "citations": citations,
        "meta": meta,
    })
    save_message(
        st.session_state.session_id, "assistant", full_answer,
        citations=citations, meta=meta,
    )


def _pdf_to_text(content: bytes) -> str:
    try:
        import io
        import importlib
        pypdf = importlib.import_module("pypdf")
        reader = pypdf.PdfReader(io.BytesIO(content))
        return "\n".join(p.extract_text() or "" for p in reader.pages)
    except Exception:
        return content.decode("utf-8", errors="replace")
