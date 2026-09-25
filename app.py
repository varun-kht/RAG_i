"""
Modern Multi-Document Streamlit Web UI Application for CILI RAG System.
Supports Groq & Gemini LLM Providers. Compatible with Python 3.11+.
"""

import os
import time
from pathlib import Path
import streamlit as st

from config import RAGConfig, default_config, SAMPLE_DATA_DIR
from rag_engine import RAGEngine
from document_loader import Document

# Page setup
st.set_page_config(
    page_title="CILI RAG System Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
CUSTOM_CSS = """
<style>
    .stApp {
        background-color: #0f172a;
        color: #f8fafc;
        font-family: 'Inter', -apple-system, sans-serif;
    }
    
    .main-title {
        font-size: 2.25rem;
        font-weight: 800;
        background: linear-gradient(135deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    section[data-testid="stSidebar"] {
        background-color: #1e293b !important;
        border-right: 1px solid #334155;
    }

    .stChatMessage {
        background-color: #1e293b !important;
        border: 1px solid #334155 !important;
        border-radius: 12px !important;
    }

    div[data-testid="stMetricValue"] {
        font-size: 1.5rem !important;
        color: #38bdf8 !important;
        font-weight: 700 !important;
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Initialize Session State
if "engine" not in st.session_state:
    st.session_state.engine = RAGEngine(config=default_config)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

engine: RAGEngine = st.session_state.engine

# Sidebar Controls
st.sidebar.markdown("## 🤖 CILI RAG System")
st.sidebar.markdown("### Python 3.11+ RAG Dashboard")

provider_choice = st.sidebar.selectbox(
    "LLM Provider",
    options=["auto", "groq", "gemini", "local"],
    index=0
)
engine.config.llm_provider = provider_choice

# Groq Configuration
groq_key_input = st.sidebar.text_input(
    "Groq API Key",
    type="password",
    value=engine.config.groq_api_key or os.getenv("GROQ_API_KEY", ""),
    help="Enter Groq API key"
)

groq_model_choice = st.sidebar.selectbox(
    "Groq Model",
    options=["llama-3.3-70b-versatile", "llama3-8b-8192", "mixtral-8x7b-32768", "gemma2-9b-it"],
    index=0
)

if groq_key_input:
    engine.config.groq_api_key = groq_key_input
engine.config.groq_model = groq_model_choice

# Gemini Configuration
gemini_key_input = st.sidebar.text_input(
    "Gemini API Key",
    type="password",
    value=engine.config.gemini_api_key or os.getenv("GEMINI_API_KEY", ""),
    help="Enter Google Gemini API Key"
)
if gemini_key_input:
    engine.config.gemini_api_key = gemini_key_input

st.sidebar.markdown("---")
st.sidebar.subheader("🔍 Retrieval Settings")
top_k = st.sidebar.slider("Top-K Chunks", min_value=1, max_value=10, value=4)
use_hybrid = st.sidebar.checkbox("Enable Hybrid Search (Vector + Keyword)", value=True)

st.sidebar.markdown("---")
st.sidebar.subheader("📊 Index Status")
chunk_count = engine.vector_store.count()
st.sidebar.metric("Total Chunks", chunk_count)

if st.sidebar.button("Index Sample Dataset"):
    with st.spinner("Indexing sample dataset..."):
        count = engine.ingest_directory(SAMPLE_DATA_DIR)
        st.sidebar.success(f"Indexed {count} chunks!")
        st.rerun()

if st.sidebar.button("Clear Vector Index"):
    engine.vector_store.chunks.clear()
    engine.vector_store.embeddings.clear()
    engine.vector_store.doc_ids.clear()
    if default_config.vector_db_path.exists():
        default_config.vector_db_path.unlink()
    st.sidebar.warning("Vector index cleared.")
    st.rerun()


# Main UI Header
st.markdown("<div class='main-title'>📚 Retrieval-Augmented Generation Dashboard</div>", unsafe_allow_html=True)
st.markdown("Upload documents, manage vector indices, and ask questions powered by **Groq** & **Gemini**.")

# Document Upload Section
with st.expander("📁 Upload & Index New Documents", expanded=(chunk_count == 0)):
    uploaded_files = st.file_uploader(
        "Upload files (.txt, .md, .csv, .json, .pdf)",
        accept_multiple_files=True,
        type=["txt", "md", "csv", "json", "pdf"]
    )
    if uploaded_files and st.button("Ingest Uploaded Documents"):
        with st.spinner("Processing documents..."):
            docs = []
            for ufile in uploaded_files:
                content = ufile.read().decode("utf-8", errors="ignore")
                doc = Document(
                    content=content,
                    metadata={"source": ufile.name, "file_size": ufile.size},
                    doc_id=ufile.name
                )
                docs.append(doc)
            count = engine.ingest_documents(docs)
            st.success(f"Successfully indexed {count} new text chunks!")
            st.rerun()

# Chat Query Section
st.subheader("💬 Ask Questions")

# Display Chat History
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("🔍 View Retrieved Source Context"):
                for idx, src in enumerate(message["sources"], 1):
                    st.markdown(f"**[{idx}] {src['source']}** (Score: `{src['score']}`)")
                    st.text(src["text"])

# User Prompt Input
user_query = st.chat_input("Ask a question about your indexed documents...")
if user_query:
    st.session_state.chat_history.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.write(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving context and generating answer..."):
            result = engine.query(
                question=user_query,
                top_k=top_k,
                use_hybrid=use_hybrid
            )

        st.write(result.answer)
        st.caption(f"Provider: {result.llm_provider} | Confidence Score: {result.confidence_score}")

        sources_data = [
            {
                "source": src.metadata.get("source", "doc"),
                "score": src.similarity_score,
                "text": src.text
            }
            for src in result.sources
        ]

        if result.sources:
            with st.expander("🔍 View Retrieved Source Context"):
                for idx, src in enumerate(sources_data, 1):
                    st.markdown(f"**[{idx}] {src['source']}** (Similarity: `{src['score']}`)")
                    st.text(src["text"])

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": result.answer,
            "sources": sources_data
        })
