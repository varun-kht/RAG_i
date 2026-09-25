"""
Modern, Premium PDF Conversational Chat Web UI powered by Groq API & RAG Knowledge Tool.
Compatible with Python 3.11+.
"""

import os
import time
from pathlib import Path
import streamlit as st

from config import RAGConfig, default_config
from document_loader import DocumentLoader
from rag_engine import RAGEngine

# Page Configuration
st.set_page_config(
    page_title="PDF AI Conversational Assistant | Groq RAG",
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
        background: linear-gradient(135deg, #6366f1 0%, #a855f7 50%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.25rem;
    }
    
    .subtitle {
        color: #94a3b8;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }

    .css-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1rem;
    }
    
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
        background: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid rgba(52, 211, 153, 0.3);
    }

    .status-badge-groq {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
        background: rgba(99, 102, 241, 0.15);
        color: #818cf8;
        border: 1px solid rgba(129, 140, 248, 0.3);
    }

    section[data-testid="stSidebar"] {
        background-color: #1e293b !important;
        border-right: 1px solid #334155;
    }

    .stChatMessage {
        background-color: #1e293b !important;
        border: 1px solid #334155 !important;
        border-radius: 12px !important;
        margin-bottom: 0.75rem !important;
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

if "indexed_file_name" not in st.session_state:
    st.session_state.indexed_file_name = None

if "doc_preview" not in st.session_state:
    st.session_state.doc_preview = None

engine: RAGEngine = st.session_state.engine

# Sidebar Section
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.markdown("<span class='status-badge-groq'>⚡ Groq AI Conversational Engine</span>", unsafe_allow_html=True)
    st.write("")

    groq_key = st.text_input(
        "🔑 Groq API Key",
        type="password",
        value=engine.config.groq_api_key or os.getenv("GROQ_API_KEY", ""),
        help="Enter Groq API Key"
    )
    if groq_key:
        engine.config.groq_api_key = groq_key
        engine.config.llm_provider = "groq"

    # Active Groq Models List
    groq_model = st.selectbox(
        "🧠 Select Groq Model",
        options=[
            "openai/gpt-oss-120b",
            "openai/gpt-oss-20b",
            "qwen/qwen3.8-27b",
            "allam-2-7b"
        ],
        index=0
    )
    engine.config.groq_model = groq_model

    st.markdown("---")
    st.markdown("### 🎛️ RAG Tool Settings")
    top_k = st.slider("Context Chunks (Top-K)", min_value=1, max_value=8, value=4)
    use_hybrid = st.checkbox("Enable Hybrid Vector + Keyword Search", value=True)

    st.markdown("---")
    if st.button("🔄 Reset Chat & Document"):
        engine.vector_store.chunks.clear()
        engine.vector_store.embeddings.clear()
        engine.vector_store.doc_ids.clear()
        st.session_state.chat_history = []
        st.session_state.indexed_file_name = None
        st.session_state.doc_preview = None
        st.rerun()


# Main Application Content
st.markdown("<div class='main-title'>🤖 Groq Conversational AI + RAG Document Tool</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>A full conversational chat session powered by Groq LLM with live PDF RAG knowledge tool retrieval.</div>", unsafe_allow_html=True)

# Top Bar Indicators
col_a, col_b, col_c = st.columns([1, 1, 1])
with col_a:
    if groq_key or os.getenv("GROQ_API_KEY"):
        st.markdown("<span class='status-badge'>🟢 Groq Chat Connected</span>", unsafe_allow_html=True)
    else:
        st.markdown("<span class='status-badge-groq'>⚠️ Set GROQ_API_KEY for Groq Chat</span>", unsafe_allow_html=True)
with col_b:
    st.markdown(f"<span class='status-badge-groq'>⚡ Model: {groq_model}</span>", unsafe_allow_html=True)
with col_c:
    st.markdown(f"<span class='status-badge'>📚 Document Chunks: {engine.vector_store.count()}</span>", unsafe_allow_html=True)

st.write("")

# PDF File Uploader Widget
uploaded_pdf = st.file_uploader(
    "Upload your PDF document to empower Groq's RAG knowledge tool",
    type=["pdf"],
    help="Drag and drop a PDF file"
)

if uploaded_pdf is not None:
    if st.session_state.indexed_file_name != uploaded_pdf.name:
        with st.spinner(f"⚡ Indexing PDF document '{uploaded_pdf.name}' into RAG tool..."):
            start_time = time.time()
            pdf_bytes = uploaded_pdf.read()
            doc = DocumentLoader.load_pdf_stream(pdf_bytes, filename=uploaded_pdf.name)
            
            engine.vector_store.chunks.clear()
            engine.vector_store.embeddings.clear()
            engine.vector_store.doc_ids.clear()
            
            chunk_count = engine.ingest_documents([doc])
            elapsed = round(time.time() - start_time, 2)
            
            st.session_state.indexed_file_name = uploaded_pdf.name
            st.session_state.doc_preview = doc.content[:400] + "..." if len(doc.content) > 400 else doc.content
            st.session_state.chat_history = []
            st.success(f"Indexed **{uploaded_pdf.name}** ({chunk_count} text chunks ready for RAG tool in {elapsed}s)!")

# Active Document Summary Card
if st.session_state.indexed_file_name:
    st.markdown("<div class='css-card'>", unsafe_allow_html=True)
    m1, m2, m3 = st.columns(3)
    m1.metric("Active PDF", st.session_state.indexed_file_name)
    m2.metric("RAG Knowledge Chunks", engine.vector_store.count())
    m3.metric("Chat Engine", f"Groq ({groq_model})")
    st.markdown("</div>", unsafe_allow_html=True)

# Conversational Chat Interface
st.markdown("### 💬 Conversational Session")

# Display Conversation History
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander("🔍 RAG Knowledge Tool Context & Citations"):
                for idx, src in enumerate(msg["sources"], 1):
                    st.markdown(f"**[{idx}] Source:** `{src['source']}` | **Similarity Score:** `{src['score']}`")
                    st.text(src["text"])

# User Chat Input Box
user_input = st.chat_input("Chat with Groq AI assistant (Ask anything or discuss your uploaded PDF)...")

if user_input:
    # 1. Append User message to Chat History
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    # 2. Call Conversational Agent with full chat history + RAG tool
    with st.chat_message("assistant"):
        with st.spinner("🤖 Groq AI thinking & retrieving document knowledge..."):
            result = engine.chat(
                messages=st.session_state.chat_history,
                top_k=top_k,
                use_hybrid=use_hybrid
            )

        st.markdown(result.answer)
        st.caption(f"⚡ Engine: {result.llm_provider} | RAG Retrieval Relevance: {result.confidence_score}")

        sources_info = [
            {
                "source": src.metadata.get("source", st.session_state.indexed_file_name or "document"),
                "score": src.similarity_score,
                "text": src.text
            }
            for src in result.sources
        ]

        if result.sources:
            with st.expander("🔍 RAG Knowledge Tool Context & Citations"):
                for idx, src in enumerate(sources_info, 1):
                    st.markdown(f"**[{idx}] Source:** `{src['source']}` | **Similarity Score:** `{src['score']}`")
                    st.text(src["text"])

        # 3. Append Assistant response to Chat History
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": result.answer,
            "sources": sources_info
        })
