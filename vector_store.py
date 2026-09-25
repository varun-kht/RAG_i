"""
Vector Store module for CILI RAG System.
Provides high-performance vector indexing, similarity search, hybrid search, and persistence.
Supports NumPy or Pure Python. Compatible with Python 3.11+.
"""

import json
import math
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

from document_loader import TextChunk

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


def _dot_product(vec1, vec2) -> float:
    if HAS_NUMPY and isinstance(vec1, np.ndarray):
        return float(np.dot(vec1, vec2))
    return sum(a * b for a, b in zip(vec1, vec2))


def _l2_norm(vec) -> float:
    if HAS_NUMPY and isinstance(vec, np.ndarray):
        return float(np.linalg.norm(vec))
    return math.sqrt(sum(x * x for x in vec))


def _normalize(vec):
    norm = _l2_norm(vec)
    if norm == 0:
        return vec
    if HAS_NUMPY and isinstance(vec, np.ndarray):
        return vec / norm
    return [x / norm for x in vec]


class VectorStore:
    """In-Memory Vector Database with Persistence and Hybrid Search capabilities."""

    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = Path(storage_path) if storage_path else None
        self.chunks: Dict[str, TextChunk] = {}
        self.embeddings: Dict[str, Any] = {}
        self.doc_ids: List[str] = []

    def add_chunks(self, chunks: List[TextChunk], embeddings):
        """Add text chunks and corresponding embedding vectors to the vector store."""
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks and embeddings must match.")

        for chunk, embedding in zip(chunks, embeddings):
            self.chunks[chunk.chunk_id] = chunk
            norm_vec = _normalize(embedding)
            self.embeddings[chunk.chunk_id] = norm_vec
            if chunk.chunk_id not in self.doc_ids:
                self.doc_ids.append(chunk.chunk_id)

    def similarity_search(
        self,
        query_embedding,
        top_k: int = 4,
        similarity_threshold: float = 0.0,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[TextChunk, float]]:
        """Perform Vector Cosine Similarity Search."""
        if not self.doc_ids:
            return []

        q_vec = _normalize(query_embedding)

        results = []
        for chunk_id in self.doc_ids:
            chunk = self.chunks[chunk_id]

            # Metadata Filter Check
            if filter_metadata:
                match = all(chunk.metadata.get(k) == v for k, v in filter_metadata.items())
                if not match:
                    continue

            emb = self.embeddings[chunk_id]
            score = _dot_product(q_vec, emb)

            if score >= similarity_threshold:
                results.append((chunk, score))

        # Sort by similarity score descending
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def hybrid_search(
        self,
        query_text: str,
        query_embedding,
        top_k: int = 4,
        alpha: float = 0.7
    ) -> List[Tuple[TextChunk, float]]:
        """
        Hybrid Search combining Vector Cosine Similarity (weight alpha)
        and Keyword Match score (weight 1 - alpha).
        """
        if not self.doc_ids:
            return []

        vector_results = self.similarity_search(query_embedding, top_k=len(self.doc_ids))
        vec_scores = {chunk.chunk_id: score for chunk, score in vector_results}

        # Keyword BM25-like scoring
        query_words = set(query_text.lower().split())
        keyword_scores = {}
        for chunk_id, chunk in self.chunks.items():
            chunk_words = chunk.text.lower().split()
            if not chunk_words:
                keyword_scores[chunk_id] = 0.0
                continue
            matches = sum(1 for w in query_words if w in chunk_words)
            keyword_scores[chunk_id] = matches / (len(query_words) + 1)

        combined_results = []
        for chunk_id in self.doc_ids:
            chunk = self.chunks[chunk_id]
            v_score = vec_scores.get(chunk_id, 0.0)
            k_score = keyword_scores.get(chunk_id, 0.0)
            final_score = (alpha * v_score) + ((1.0 - alpha) * k_score)
            combined_results.append((chunk, float(final_score)))

        combined_results.sort(key=lambda x: x[1], reverse=True)
        return combined_results[:top_k]

    def save(self, path: Optional[Path] = None):
        """Save vector store to disk."""
        target_path = Path(path) if path else self.storage_path
        if not target_path:
            raise ValueError("No storage path provided.")

        target_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "chunks": [
                {
                    "chunk_id": c.chunk_id,
                    "doc_id": c.doc_id,
                    "text": c.text,
                    "metadata": c.metadata,
                    "chunk_index": c.chunk_index
                }
                for c in self.chunks.values()
            ],
            "embeddings": {
                chunk_id: (emb.tolist() if HAS_NUMPY and isinstance(emb, np.ndarray) else list(emb))
                for chunk_id, emb in self.embeddings.items()
            }
        }
        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load(self, path: Optional[Path] = None):
        """Load vector store from disk."""
        target_path = Path(path) if path else self.storage_path
        if not target_path or not target_path.exists():
            return

        with open(target_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.chunks.clear()
        self.embeddings.clear()
        self.doc_ids.clear()

        for cdata in data.get("chunks", []):
            chunk = TextChunk(
                chunk_id=cdata["chunk_id"],
                doc_id=cdata["doc_id"],
                text=cdata["text"],
                metadata=cdata["metadata"],
                chunk_index=cdata["chunk_index"]
            )
            self.chunks[chunk.chunk_id] = chunk
            self.doc_ids.append(chunk.chunk_id)

        for chunk_id, emb_list in data.get("embeddings", {}).items():
            self.embeddings[chunk_id] = np.array(emb_list, dtype=np.float32) if HAS_NUMPY else emb_list

    def count(self) -> int:
        return len(self.chunks)
