"""
RAG Engine Conversational Agent for CILI RAG System.
Integrates Groq API with multi-turn chat history & RAG document retrieval.
Compatible with Python 3.11+.
"""

import os
import re
import json
import urllib.request
import urllib.error
from typing import List, Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass, field

from config import RAGConfig, default_config
from document_loader import Document, DocumentLoader, RecursiveTextSplitter, TextChunk
from embeddings import EmbeddingManager, BaseEmbeddingProvider
from vector_store import VectorStore

@dataclass
class RetrievalSource:
    chunk_id: str
    doc_id: str
    text: str
    similarity_score: float
    metadata: Dict[str, Any]

@dataclass
class RAGResult:
    question: str
    answer: str
    sources: List[RetrievalSource]
    context_used: str
    llm_provider: str
    confidence_score: float

class RAGEngine:
    """Conversational RAG Engine using Groq as core LLM with Document Retrieval Tool."""

    def __init__(
        self,
        config: Optional[RAGConfig] = None,
        embedding_provider: Optional[BaseEmbeddingProvider] = None,
        vector_store: Optional[VectorStore] = None
    ):
        self.config = config or default_config
        self.splitter = RecursiveTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap
        )
        self.embedding_provider = embedding_provider or EmbeddingManager.get_provider(
            self.config.embedding_model_name
        )
        self.vector_store = vector_store or VectorStore(storage_path=self.config.vector_db_path)
        
        # Load existing index if present
        if self.config.vector_db_path and self.config.vector_db_path.exists():
            self.vector_store.load()

    def ingest_files(self, file_paths: List[str | Path]) -> int:
        docs = [DocumentLoader.load_file(p) for p in file_paths]
        return self.ingest_documents(docs)

    def ingest_directory(self, dir_path: str | Path) -> int:
        docs = DocumentLoader.load_directory(dir_path)
        return self.ingest_documents(docs)

    def ingest_documents(self, documents: List[Document]) -> int:
        all_chunks: List[TextChunk] = []
        for doc in documents:
            chunks = self.splitter.split_document(doc)
            all_chunks.extend(chunks)

        if not all_chunks:
            return 0

        texts = [c.text for c in all_chunks]
        if hasattr(self.embedding_provider, 'fit'):
            self.embedding_provider.fit(texts)

        embeddings = self.embedding_provider.embed_texts(texts)
        self.vector_store.add_chunks(all_chunks, embeddings)
        
        if self.config.vector_db_path:
            self.vector_store.save()

        return len(all_chunks)

    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
        use_hybrid: bool = True
    ) -> RAGResult:
        """Single-turn query wrapper."""
        messages = [{"role": "user", "content": question}]
        return self.chat(messages=messages, top_k=top_k, use_hybrid=use_hybrid)

    def chat(
        self,
        messages: List[Dict[str, str]],
        top_k: Optional[int] = None,
        use_hybrid: bool = True
    ) -> RAGResult:
        """Multi-turn conversational chat agent powered by Groq LLM + RAG Retrieval Tool."""
        if not messages:
            return RAGResult("", "Hello! How can I assist you?", [], "", "Chat Agent", 1.0)

        latest_user_message = messages[-1]["content"]
        
        # 1. Retrieve RAG Document Knowledge
        k = top_k or self.config.top_k
        sources: List[RetrievalSource] = []
        context_parts = []
        scores = []

        if self.vector_store.count() > 0:
            query_emb = self.embedding_provider.embed_query(latest_user_message)
            if use_hybrid:
                retrieved = self.vector_store.hybrid_search(
                    query_text=latest_user_message,
                    query_embedding=query_emb,
                    top_k=k
                )
            else:
                retrieved = self.vector_store.similarity_search(
                    query_embedding=query_emb,
                    top_k=k,
                    similarity_threshold=self.config.similarity_threshold
                )

            for chunk, score in retrieved:
                scores.append(score)
                source_info = f"[Source: {chunk.metadata.get('source', 'document')}, Chunk {chunk.chunk_index}]"
                context_parts.append(f"{source_info}\n{chunk.text}")
                sources.append(RetrievalSource(
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    text=chunk.text,
                    similarity_score=round(score, 4),
                    metadata=chunk.metadata
                ))

        context_used = "\n\n".join(context_parts) if context_parts else "No document chunks retrieved."
        avg_confidence = round(sum(scores) / len(scores), 4) if scores else 0.0

        # 2. Build Agent System Instruction with Document Tool Knowledge
        system_instruction = (
            "You are a friendly, intelligent, and articulate AI assistant engaged in a conversation.\n"
            "You have access to a RAG Document Knowledge Base extracted from the user's uploaded files.\n\n"
            f"=== RETRIEVED DOCUMENT KNOWLEDGE ===\n{context_used}\n====================================\n\n"
            "INSTRUCTIONS:\n"
            "1. Chat naturally and conversationally with the user.\n"
            "2. When answering questions about document facts, topics, or details, ground your answer in the RETRIEVED DOCUMENT KNOWLEDGE.\n"
            "3. For general chat, greetings, or follow-up discussion, maintain a warm, helpful conversational tone.\n"
            "4. If the retrieved context is relevant, reference specific facts from it clearly."
        )

        # 3. Generate Answer using Groq API
        groq_key = self.config.groq_api_key or os.getenv("GROQ_API_KEY", "")
        gemini_key = self.config.gemini_api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY", "")

        if groq_key:
            ans, prov, err = self._call_groq_chat(system_instruction, messages, groq_key)
            if ans:
                return RAGResult(
                    question=latest_user_message,
                    answer=ans,
                    sources=sources,
                    context_used=context_used,
                    llm_provider=prov,
                    confidence_score=avg_confidence
                )
            elif err:
                return RAGResult(
                    question=latest_user_message,
                    answer=f"⚠️ **Groq API Error:** {err}\n\nPlease check your `GROQ_API_KEY` in `.env` or sidebar.",
                    sources=sources,
                    context_used=context_used,
                    llm_provider="Groq API Error",
                    confidence_score=avg_confidence
                )

        if gemini_key:
            ans, prov = self._call_gemini_chat(system_instruction, latest_user_message, gemini_key)
            if ans:
                return RAGResult(
                    question=latest_user_message,
                    answer=ans,
                    sources=sources,
                    context_used=context_used,
                    llm_provider=prov,
                    confidence_score=avg_confidence
                )

        # Offline Local Synthesizer
        local_ans = self._offline_chat(latest_user_message, sources)
        return RAGResult(
            question=latest_user_message,
            answer=local_ans,
            sources=sources,
            context_used=context_used,
            llm_provider="Offline Chat Synthesizer (Set GROQ_API_KEY in .env)",
            confidence_score=avg_confidence
        )

    def _call_groq_chat(self, system_instruction: str, history: List[Dict[str, str]], api_key: str) -> tuple[Optional[str], str, str]:
        """Send multi-turn chat history + system instruction to Groq LLM."""
        model_name = self.config.groq_model

        # Build complete payload messages
        payload_messages = [{"role": "system", "content": system_instruction}]
        for m in history:
            role = "assistant" if m.get("role") in ["assistant", "bot"] else "user"
            payload_messages.append({"role": role, "content": m.get("content", "")})

        # Attempt 1: Groq Python SDK
        try:
            from groq import Groq
            client = Groq(api_key=api_key)
            completion = client.chat.completions.create(
                messages=payload_messages,
                model=model_name,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            if completion.choices and completion.choices[0].message.content:
                return completion.choices[0].message.content.strip(), f"Groq ({model_name})", ""
        except Exception as e:
            pass

        # Attempt 2: Direct HTTP POST with Browser User-Agent (Bypasses Cloudflare 403)
        try:
            url = "https://api.groq.com/openai/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            }
            payload = {
                "model": model_name,
                "messages": payload_messages,
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens
            }
            req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                content = data["choices"][0]["message"]["content"]
                return content.strip(), f"Groq ({model_name})", ""
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8")
            return None, "", f"HTTP {e.code}: {err_body}"
        except Exception as e:
            return None, "", str(e)

        return None, "", "Unknown error calling Groq API"

    def _call_gemini_chat(self, system_instruction: str, question: str, api_key: str) -> tuple[Optional[str], str]:
        """Gemini Chat fallback."""
        model_name = self.config.gemini_model
        prompt = f"{system_instruction}\n\nUser Question: {question}"
        try:
            from google import genai
            client = genai.Client(api_key=api_key)
            response = client.models.generate_content(model=model_name, contents=prompt)
            if response and response.text:
                return response.text.strip(), f"Gemini ({model_name})"
        except Exception:
            pass
        return None, ""

    def _offline_chat(self, question: str, sources: List[RetrievalSource]) -> str:
        """Local offline mode response."""
        q_lower = question.lower()
        if re.match(r"^(hi|hello|hey|greetings|good morning|good evening)\b", q_lower):
            return "Hello! I am your offline AI document assistant. Add your `GROQ_API_KEY` to `.env` for Groq LLM answers."

        if not sources:
            return "No document context found."

        top_text = sources[0].text[:400].replace("\n", " ")
        return f"**Retrieved Document Excerpt:**\n\n\"{top_text}...\"\n\n*(Add `GROQ_API_KEY` to `.env` to enable full Groq AI Chat).* "
