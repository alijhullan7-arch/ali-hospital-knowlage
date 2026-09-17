"""
Hospital Knowledge Base Assistant
----------------------------------
Streamlit chat app that retrieves relevant policy chunks from a local
FAISS index and asks a Groq-hosted LLM to answer using only that context.
"""

import os
from pathlib import Path
import streamlit as st
from pypdf import PdfReader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from groq import Groq

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL = "openai/gpt-oss-120b"
TOP_K = 4

st.set_page_config(page_title="Hospital Knowledge Base Assistant", page_icon="🏥")

# ----------------------------------------------------------------------
# Path Configurations Matching Your Exact Repo Layout
# ----------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
DOCUMENTS_DIR = BASE_DIR / "Hospital_Knowledge_Base_PDFs" / "Hospital_Knowledge_Base_PDFs"
INDEX_DIR = BASE_DIR / "faiss_index"

# ----------------------------------------------------------------------
# API key — read from Streamlit secrets, never shown in a text box
# ----------------------------------------------------------------------
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error(
        "No Groq API key found. Add it to `.streamlit/secrets.toml` or Streamlit Cloud Secrets as:\n\n"
        '```\nGROQ_API_KEY = "your_key_here"\n```'
    )
    st.stop()

client = Groq(api_key=GROQ_API_KEY)


# ----------------------------------------------------------------------
# PDF Extraction & Indexing Functions
# ----------------------------------------------------------------------
def extract_pdf_documents():
    documents = []
    
    if not DOCUMENTS_DIR.exists():
        st.error(f"Directory `{DOCUMENTS_DIR.name}` not found. Please verify PDF location.")
        st.stop()

    pdf_files = list(DOCUMENTS_DIR.glob("*.pdf"))

    for pdf_file in pdf_files:
        reader = PdfReader(str(pdf_file))

        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""

            if text.strip():
                documents.append({
                    "text": text,
                    "metadata": {
                        "file_name": pdf_file.name,
                        "page": page_number,
                        "category": pdf_file.stem
                    }
                })

    return documents


def create_faiss_index(embeddings):
    raw_documents = extract_pdf_documents()

    if not raw_documents:
        st.error(f"No PDF files found inside `{DOCUMENTS_DIR.name}`.")
        st.stop()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )

    texts = []
    metadatas = []

    for item in raw_documents:
        chunks = splitter.split_text(item["text"])

        for chunk in chunks:
            texts.append(chunk)
            metadatas.append(item["metadata"])

    vectorstore = FAISS.from_texts(
        texts,
        embeddings,
        metadatas=metadatas
    )

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(INDEX_DIR))

    return vectorstore


@st.cache_resource(show_spinner="Loading hospital knowledge base...")
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        encode_kwargs={"normalize_embeddings": True}
    )

    # Load existing index if already created
    if (INDEX_DIR / "index.faiss").exists():
        return FAISS.load_local(
            str(INDEX_DIR),
            embeddings,
            allow_dangerous_deserialization=True
        )

    # Generate FAISS index from Hospital_Knowledge_Base_PDFs if missing
    st.info("FAISS index not found. Creating it from hospital PDFs...")
    return create_faiss_index(embeddings)


vectorstore = load_vectorstore()


# ----------------------------------------------------------------------
# Retrieval + Generation Functions
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
# Chat Interface
# ----------------------------------------------------------------------
st.title("🏥 Hospital Knowledge Base Assistant")
st.caption("Ask a question about hospital policy. Answers are grounded in your indexed documents.")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Render chat history
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
        
        # Deduplicate sources while keeping order
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
