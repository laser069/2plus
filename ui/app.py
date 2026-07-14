import sys
import uuid
from pathlib import Path

# Make project root importable when running: streamlit run ui/app.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from memory.convo import ConvoMemory
from orchestrator.agent import Agent
import memory.user_facts as uf
from rag.ingestion import ingest, list_docs

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="2Plus", page_icon="⚡", layout="wide")

# ── Session state bootstrap ───────────────────────────────────────────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]
if "convo" not in st.session_state:
    st.session_state.convo = ConvoMemory()
if "agent" not in st.session_state:
    st.session_state.agent = Agent(st.session_state.convo)
if "messages" not in st.session_state:
    st.session_state.messages = []   # display history: [{role, content, citations}]

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚡ 2Plus")
    st.caption(f"Session `{st.session_state.session_id}`")
    st.divider()

    # Doc upload
    st.subheader("📄 Upload Documents")
    uploaded = st.file_uploader(
        "PDF / TXT / MD",
        type=["txt", "md", "pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    if uploaded:
        for f in uploaded:
            content = f.read()
            text = content.decode("utf-8", errors="replace") if not f.name.endswith(".pdf") else _pdf_to_text(content)
            with st.spinner(f"Ingesting {f.name}…"):
                n = ingest(text, doc_id=f.name, metadata={"filename": f.name})
            st.success(f"{f.name}: {n} chunks")

    # Ingested docs list
    docs = list_docs()
    if docs:
        st.divider()
        st.subheader("📚 Ingested Docs")
        for d in docs:
            st.markdown(f"- `{d}`")

    # User memory viewer
    st.divider()
    st.subheader("🧠 Memory")
    facts = uf.get_all()
    if facts:
        with st.expander("View stored facts", expanded=False):
            for k, v in facts.items():
                if k == "convo_summary":
                    continue
                st.markdown(f"**{k}**: {v}")
    else:
        st.caption("No facts stored yet.")

    # Clear session
    st.divider()
    if st.button("🗑 Clear conversation", use_container_width=True):
        st.session_state.convo.clear()
        st.session_state.messages = []
        st.rerun()

# ── Main chat area ────────────────────────────────────────────────────────────
st.header("2Plus — Your Local AI Assistant", divider="gray")

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
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Run agent
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            result = st.session_state.agent.run(prompt)
        st.markdown(result.answer)
        if result.citations:
            with st.expander("Sources", expanded=False):
                for c in result.citations:
                    st.markdown(f"- `{c}`")
        with st.expander("Debug", expanded=False):
            st.json({"routes": result.routes, "steps": result.steps})

    st.session_state.messages.append({
        "role": "assistant",
        "content": result.answer,
        "citations": result.citations,
        "meta": {"routes": result.routes, "steps": result.steps},
    })


def _pdf_to_text(content: bytes) -> str:
    """Best-effort PDF text extraction without extra dependencies."""
    try:
        import io
        # Try pypdf if available
        import importlib
        pypdf = importlib.import_module("pypdf")
        reader = pypdf.PdfReader(io.BytesIO(content))
        return "\n".join(p.extract_text() or "" for p in reader.pages)
    except Exception:
        return content.decode("utf-8", errors="replace")
