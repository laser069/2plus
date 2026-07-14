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
from config.settings import MODEL_ROUTER

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="2Plus", page_icon="⚡", layout="wide")

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Base ── */
html, body, [data-testid="stApp"] {
    background-color: #0f0f13;
    color: #e2e2e8;
    font-size: 15px;
    line-height: 1.6;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: #13131a !important;
    border-right: 1px solid #2a2a3a;
}
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stCaption {
    color: #9999bb !important;
}
[data-testid="stSidebar"] h1 {
    color: #4f8ef7 !important;
    font-size: 1.4rem;
    font-weight: 700;
    letter-spacing: -0.3px;
}
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
    color: #c2c2dd !important;
    font-size: 0.8rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-top: 1.2rem;
}

/* ── Main header ── */
h1 {
    color: #ffffff !important;
    font-size: 1.5rem !important;
    font-weight: 700 !important;
}

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    background-color: #1a1a24 !important;
    border-radius: 10px;
    margin-bottom: 0.5rem;
    padding: 0.75rem 1rem !important;
    border: 1px solid #2a2a3a;
}
[data-testid="stChatMessage"][data-testid*="user"] {
    border-left: 3px solid #4f8ef7;
}
[data-testid="stChatMessage"][data-testid*="assistant"] {
    border-left: 3px solid #7c5af7;
}

/* ── Chat input ── */
[data-testid="stChatInput"] {
    background-color: #1a1a24 !important;
    border: 1px solid #2a2a3a !important;
    border-radius: 10px !important;
    color: #e2e2e8 !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: #4f8ef7 !important;
    box-shadow: 0 0 0 2px rgba(79,142,247,0.2) !important;
}

/* ── Buttons ── */
.stButton > button {
    background-color: #1e1e2e !important;
    color: #c2c2dd !important;
    border: 1px solid #2a2a3a !important;
    border-radius: 8px !important;
    transition: all 0.15s;
}
.stButton > button:hover {
    border-color: #4f8ef7 !important;
    color: #4f8ef7 !important;
}

/* ── Selectbox / inputs ── */
[data-testid="stSelectbox"] > div > div,
[data-testid="stTextInput"] > div > div > input {
    background-color: #1a1a24 !important;
    border-color: #2a2a3a !important;
    color: #e2e2e8 !important;
    border-radius: 8px !important;
}

/* ── Expanders ── */
[data-testid="stExpander"] {
    background-color: #16161f !important;
    border: 1px solid #2a2a3a !important;
    border-radius: 8px !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background-color: #1a1a24 !important;
    border: 1px dashed #2a2a3a !important;
    border-radius: 8px !important;
}

/* ── Status / spinner ── */
[data-testid="stStatusWidget"] { color: #4f8ef7 !important; }

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, [data-testid="stToolbar"] { display: none !important; }

/* ── Toggle ── */
[data-testid="stToggle"] label { color: #9999bb !important; }

/* ── Success / info ── */
.stSuccess { background-color: #1a2e1a !important; border-color: #2a6a2a !important; }
.stInfo    { background-color: #1a1e2e !important; border-color: #2a3a6a !important; }
</style>
""", unsafe_allow_html=True)

# ── Session state bootstrap ───────────────────────────────────────────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]
if "convo" not in st.session_state:
    st.session_state.convo = ConvoMemory()
if "agent" not in st.session_state:
    st.session_state.agent = Agent(st.session_state.convo)
# Load persisted messages for this session (survives page refresh)
if "messages" not in st.session_state:
    st.session_state.messages = load_session(st.session_state.session_id)
if "selected_model" not in st.session_state:
    st.session_state.selected_model = MODEL_ROUTER["default"]
if "think_mode" not in st.session_state:
    st.session_state.think_mode = False


# ── Model list (cached 60s) ───────────────────────────────────────────────────
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
    st.caption(f"Session `{st.session_state.session_id}`")

    # ── Model selector ──
    st.subheader("Model")
    available_models = _get_models()
    current_idx = (
        available_models.index(st.session_state.selected_model)
        if st.session_state.selected_model in available_models
        else 0
    )
    st.session_state.selected_model = st.selectbox(
        "Active model",
        available_models,
        index=current_idx,
        label_visibility="collapsed",
    )
    st.session_state.think_mode = st.toggle("Think mode (slower, deeper)", value=st.session_state.think_mode)

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
        with st.expander(f"Ingested docs ({len(docs)})", expanded=False):
            for d in docs:
                st.markdown(f"- `{d}`")

    # ── Memory viewer ──
    st.subheader("Memory")
    facts = {k: v for k, v in uf.get_all().items() if k != "convo_summary"}
    if facts:
        with st.expander(f"Stored facts ({len(facts)})", expanded=False):
            for k, v in facts.items():
                st.markdown(f"**{k}**: {v}")
    else:
        st.caption("No facts stored yet.")

    # ── Clear ──
    st.markdown("---")
    if st.button("🗑 Clear conversation", use_container_width=True):
        st.session_state.convo.clear()
        delete_session(st.session_state.session_id)
        st.session_state.messages = []
        st.rerun()


# ── Main chat area ────────────────────────────────────────────────────────────
col_title, col_badge = st.columns([5, 1])
with col_title:
    st.header("2Plus — Local AI Assistant", divider="gray")
with col_badge:
    st.markdown(
        f"<div style='text-align:right;padding-top:1.4rem;color:#4f8ef7;font-size:0.75rem;font-weight:600'>"
        f"⚡ {st.session_state.selected_model}</div>",
        unsafe_allow_html=True,
    )

# Render history
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

# Chat input
if prompt := st.chat_input("Ask me anything…"):
    # Show user message
    with st.chat_message("user"):
        st.markdown(prompt)

    user_msg = {"role": "user", "content": prompt}
    st.session_state.messages.append(user_msg)
    save_message(st.session_state.session_id, "user", prompt)

    # Stream assistant reply
    with st.chat_message("assistant"):
        answer_ph = st.empty()
        status_ph = st.empty()
        full_answer = ""
        citations: list[str] = []
        meta: dict = {}

        for event in st.session_state.agent.run_stream(
            prompt,
            model=st.session_state.selected_model,
            think=st.session_state.think_mode,
        ):
            if isinstance(event, dict):
                if event["type"] == "routing":
                    status_ph.caption(f"Routes: {', '.join(event['routes']) or 'direct'}")
                elif event["type"] == "tool":
                    status_ph.caption(f"Tool: **{event['name']}** (step {event['step']})")
                elif event["type"] == "done":
                    citations = event["citations"]
                    meta = {"routes": event["routes"], "steps": event["steps"]}
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

    assistant_msg = {
        "role": "assistant",
        "content": full_answer,
        "citations": citations,
        "meta": meta,
    }
    st.session_state.messages.append(assistant_msg)
    save_message(
        st.session_state.session_id,
        "assistant",
        full_answer,
        citations=citations,
        meta=meta,
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
