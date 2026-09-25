"""
Configuration settings for the CILI RAG System.
Automatically loads environment variables from .env file.
Compatible with Python 3.11+. Supports Vercel Serverless Read-Only Filesystem.
"""

import os
from pathlib import Path
from dataclasses import dataclass

# Base Directory
BASE_DIR = Path(__file__).resolve().parent

def load_dotenv(env_file_path: Path):
    """Zero-dependency .env file reader."""
    if not env_file_path.exists():
        return
    try:
        with open(env_file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, val = line.split("=", 1)
                key = key.strip()
                val = val.strip().strip("'\"")
                if key and val and key not in os.environ:
                    os.environ[key] = val
    except Exception:
        pass

# Auto-load .env file from project root
load_dotenv(BASE_DIR / ".env")

# Detect Vercel / AWS Lambda Serverless Environment (Read-Only Filesystem)
IS_VERCEL = bool(os.getenv("VERCEL") or os.getenv("AWS_LAMBDA_FUNCTION_NAME"))

if IS_VERCEL:
    DATA_DIR = Path("/tmp/data")
    STORAGE_DIR = Path("/tmp/storage")
    SAMPLE_DATA_DIR = BASE_DIR / "sample_data"
else:
    DATA_DIR = BASE_DIR / "data"
    STORAGE_DIR = BASE_DIR / "storage"
    SAMPLE_DATA_DIR = BASE_DIR / "sample_data"

# Ensure required directories exist (safely catch read-only errors)
try:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    SAMPLE_DATA_DIR.mkdir(parents=True, exist_ok=True)
except Exception:
    pass

@dataclass
class RAGConfig:
    # Text Chunker settings
    chunk_size: int = 500
    chunk_overlap: int = 100

    # Retrieval settings
    top_k: int = 4
    similarity_threshold: float = 0.2

    # Embedding settings
    embedding_model_name: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384

    # Storage settings
    vector_db_path: Path = STORAGE_DIR / "vector_store.json"

    # LLM Provider settings: 'auto', 'groq', 'gemini', 'local'
    llm_provider: str = os.getenv("LLM_PROVIDER", "auto")

    # Groq Settings (Updated to active models for API Key)
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    groq_model: str = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

    # Gemini Settings
    gemini_api_key: str = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY", "")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    # Generation Parameters
    temperature: float = 0.2
    max_tokens: int = 1024

# Default instance
default_config = RAGConfig()

RAG_PROMPT_TEMPLATE = """You are a helpful and precise AI assistant. Answer the user's question based strictly on the provided retrieved context.

Retrieved Context:
-------------------
{context}
-------------------

User Question: {question}

Instructions:
1. Provide a direct, factual, and concise answer based ONLY on the context provided above.
2. If the context does not contain enough information to answer the question, state clearly: "I cannot answer this based on the provided document context."
3. Include references to the source documents when mentioning specific facts (e.g., [Source: filename.txt, Chunk 2]).

Answer:
"""
