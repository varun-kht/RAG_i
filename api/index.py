"""
FastAPI Serverless Function Entrypoint for CILI RAG System on Vercel.
Exports top-level 'app' and 'handler' variables.
Compatible with Python 3.11+.
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config import default_config
from rag_engine import RAGEngine

# Initialize RAG Engine
engine = RAGEngine(config=default_config)

# Top-level FastAPI application export required by Vercel
app = FastAPI(
    title="CILI RAG System API",
    description="FastAPI Serverless RAG API for Vercel",
    version="1.0.0"
)

# Alias for Vercel handler scanner
handler = app

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = 4


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    top_k: Optional[int] = 4


@app.get("/")
@app.get("/api")
@app.get("/api/health")
def health_check():
    return {
        "status": "online",
        "system": "CILI RAG System",
        "model": default_config.groq_model,
        "indexed_chunks": engine.vector_store.count()
    }


@app.post("/api/query")
def query_rag(req: QueryRequest):
    try:
        res = engine.query(req.question, top_k=req.top_k)
        return {
            "question": res.question,
            "answer": res.answer,
            "llm_provider": res.llm_provider,
            "confidence_score": res.confidence_score,
            "sources": [
                {
                    "source": s.metadata.get("source", "doc"),
                    "score": s.similarity_score,
                    "text": s.text[:200]
                }
                for s in res.sources
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat")
def chat_rag(req: ChatRequest):
    try:
        history = [{"role": m.role, "content": m.content} for m in req.messages]
        res = engine.chat(messages=history, top_k=req.top_k)
        return {
            "question": res.question,
            "answer": res.answer,
            "llm_provider": res.llm_provider,
            "confidence_score": res.confidence_score,
            "sources": [
                {
                    "source": s.metadata.get("source", "doc"),
                    "score": s.similarity_score,
                    "text": s.text[:200]
                }
                for s in res.sources
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
