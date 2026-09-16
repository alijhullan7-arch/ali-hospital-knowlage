"""
Hospital Knowledge Base Assistant
----------------------------------
Streamlit chat app that retrieves relevant policy chunks from a local
FAISS index and asks a Groq-hosted LLM to answer using only that context.
"""

import os
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from groq import Groq

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
INDEX_DIR = "faiss_index"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL = "openai/gpt-oss-120b"
TOP_K = 4

st.set_page_config(page_title="Hospital Knowledge Base Assistant", page_icon="🏥")

# ----------------------------------------------------------------------
# API key — read from Streamlit secrets, never shown in a text box
# ----------------------------------------------------------------------
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error(
        "No Groq API key found. Add it to `.streamlit/secrets.toml` as:\n\n"
        '```\nGROQ_API_KEY = "your_key_here"\n```'
    )
    st.stop()

client = Groq(api_key=GROQ_API_KEY)


# ----------------------------------------------------------------------
# Load embeddings + FAISS index (cached so it only loads once)
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading knowledge base...")
def load_vectorstore():
    if not os.path.exists(INDEX_DIR):
        st.error(f"Index folder '{INDEX_DIR}' not found. Place it next to app.py.")
        st.stop()

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )
    return FAISS.load_local(
        INDEX_DIR, embeddings, allow_dangerous_deserialization=True
    )


vectorstore = load_vectorstore()


# ----------------------------------------------------------------------
# Retrieval + generation
# ----------------------------------------------------------------------
def retrieve_chunks(question, k=TOP_K):
    return vectorstore.similarity_search(question, k=k)


def build_context(chunks):
    parts = []
    for i, c in enumerate(chunks, start=1):
        src = c.metadata.get("file_name", "unknown")
        page = c.metadata.get("page", "?")
        parts.append(f"[Source {i}: {src}, page {page}]\n{c.page_content}")
    return "\n\n".join(parts)


def ask_groq(question, context):
    system_prompt = (
        "You are a hospital policy assistant. Answer the user's question "
        "using ONLY the provided context from hospital policy documents. "
        "If the answer is not in the context, say you don't have that "
        "information in the available policies — do not guess or use "
        "outside knowledge. Be clear and concise."
    )
    user_prompt = f"Context:\n{context}\n\nQuestion: {question}"

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content


# ----------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------
st.title("🏥 Hospital Knowledge Base Assistant")
st.caption("Ask a question about hospital policy. Answers are grounded in your indexed documents.")

if "messages" not in st.session_state:
    st.session_state.messages = []

# render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("Sources"):
                for s in msg["sources"]:
                    st.markdown(f"- **{s['file']}** — page {s['page']} ({s['category']})")

question = st.chat_input("Ask about a hospital policy...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching policies..."):
            chunks = retrieve_chunks(question)
            context = build_context(chunks)

        with st.spinner("Generating answer..."):
            answer = ask_groq(question, context)

        st.markdown(answer)

        sources = [
            {
                "file": c.metadata.get("file_name", "unknown"),
                "page": c.metadata.get("page", "?"),
                "category": c.metadata.get("category", "root"),
            }
            for c in chunks
        ]
        # de-duplicate sources while keeping order
        seen = set()
        unique_sources = []
        for s in sources:
            key = (s["file"], s["page"])
            if key not in seen:
                seen.add(key)
                unique_sources.append(s)

        with st.expander("Sources"):
            for s in unique_sources:
                st.markdown(f"- **{s['file']}** — page {s['page']} ({s['category']})")

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": unique_sources}
    )
