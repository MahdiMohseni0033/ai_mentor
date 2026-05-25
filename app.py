"""Streamlit demo for the Mini AI Mentor Engine.

    streamlit run app.py

Chat UI with conversation memory, the decision-engine label, and the retrieved
chunks + similarity scores behind every grounded answer.
"""

from __future__ import annotations

import os

from CONFIG import CONFIG

# Select the GPU before torch / sentence_transformers get imported anywhere.
if CONFIG.model.use_gpu:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", CONFIG.model.gpu_device)

import streamlit as st

from src.controller import MentorController
from src.generation.llm_client import LLMUnavailableError

EXAMPLE_QUERIES = [
    "I feel stuck in my career and don't know what to do next.",
    "How can I become a better leader?",
    "I'm afraid of failing in a new role.",
    "Give me a 7-day plan to improve my focus.",
    "Help",
    "How do I find my purpose?",
]

st.set_page_config(page_title=CONFIG.app.page_title, page_icon="🧭", layout="centered")


@st.cache_resource(show_spinner="Loading embedding model and vector store...")
def get_controller() -> MentorController:
    """Build the controller once per session (embedding model is heavy)."""
    from src.embeddings.embedding_model import EmbeddingModel
    from src.retrieval.retriever import Retriever
    from src.vectorstore.chroma_store import ChromaStore

    store = ChromaStore()
    if store.count() == 0:
        raise RuntimeError(
            "The vector index is empty. Build it first:\n\n"
            "    python -m src.ingestion.pipeline\n"
            "    python scripts/build_index.py"
        )
    return MentorController(Retriever(EmbeddingModel(), store))


def render_sources(retrieved) -> None:
    if not retrieved or not CONFIG.app.show_retrieved_chunks:
        return
    st.markdown("**Retrieved context**")
    for c in retrieved:
        with st.expander(f"🔎 {c.similarity:.3f} · {c.speaker} — {c.title}"):
            st.caption(
                f"chunk_id: `{c.chunk_id}` · topic: {c.topic} · "
                f"source: {c.source_type} · cosine similarity: {c.similarity:.3f}"
            )
            if c.source_url:
                st.caption(c.source_url)
            st.write(c.text)


def render_classification(cls) -> None:
    icons = {"emotional": "💙", "strategic": "🎯", "vague": "❓",
             "general": "💬", "off_topic": "🧭", "follow_up": "🔁",
             "appreciation": "✅"}
    st.caption(f"{icons.get(cls['label'], '•')} **{cls['label']}** — {cls['reason']}")


# --- Sidebar ---------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuration")
    st.write(f"**LLM:** `{CONFIG.model.llm_model}`")
    st.write(f"**Embedding:** `{CONFIG.model.embedding_model}`")
    st.write(f"**Embedding dim/context:** `{CONFIG.model.embedding_dim}` / `{CONFIG.model.embedding_max_tokens}` tokens")
    st.write(f"**Collection:** `{CONFIG.retrieval.collection_name}`")
    st.write(f"**top_k:** `{CONFIG.retrieval.top_k}`")
    st.write(f"**Memory:** last `{CONFIG.app.conversation_memory_size}` messages")

    st.divider()
    st.subheader("💡 Try an example")
    for q in EXAMPLE_QUERIES:
        if st.button(q, key=f"ex_{q}", use_container_width=True):
            st.session_state.pending = q

    st.divider()
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# --- Main ------------------------------------------------------------------
st.title("🧭 Mini AI Mentor Engine")
st.caption(
    "A grounded RAG mentor over curated TED talks on career, leadership, "
    "psychology, and productivity."
)

if "messages" not in st.session_state:
    st.session_state.messages = []

try:
    controller = get_controller()
except Exception as exc:  # noqa: BLE001 - show a clear setup message instead of a stack trace
    st.error(str(exc))
    st.stop()

# Replay history.
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg.get("classification"):
            render_classification(msg["classification"])
        st.markdown(msg["content"])
        render_sources(msg.get("retrieved"))

prompt = st.chat_input("Ask the mentor...") or st.session_state.pop("pending", None)

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    history = [{"role": m["role"], "content": m["content"]}
               for m in st.session_state.messages[:-1]]

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = controller.respond(prompt, history)
            except LLMUnavailableError as exc:
                st.error(str(exc))
                st.stop()

        render_classification(result.classification.to_dict())
        st.markdown(result.answer)
        render_sources(result.retrieved)

    st.session_state.messages.append({
        "role": "assistant",
        "content": result.answer,
        "classification": result.classification.to_dict(),
        "retrieved": result.retrieved,
    })
