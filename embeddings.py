"""
Embeddings module for CILI RAG System.
Provides vector embedding extraction using SentenceTransformers or pure Python / NumPy TF-IDF.
Compatible with Python 3.11+.
"""

import math
import re
from typing import List, Union

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

class BaseEmbeddingProvider:
    """Abstract base class for embedding providers."""
    def embed_texts(self, texts: List[str]):
        raise NotImplementedError

    def embed_query(self, query: str):
        return self.embed_texts([query])[0]


class TFIDFEmbeddingProvider(BaseEmbeddingProvider):
    """
    Lightweight, zero-dependency TF-IDF Embedding Provider using pure Python or NumPy.
    Guarantees operation out of the box without external packages.
    """

    def __init__(self, vocab_size: int = 384):
        self.vocab_size = vocab_size
        self.vocabulary = {}
        self.idf = {}
        self.is_fitted = False

    def fit(self, texts: List[str]):
        """Build vocabulary and compute IDF scores from text corpus."""
        doc_count = len(texts)
        if doc_count == 0:
            return

        term_doc_freq = {}
        vocab_counts = {}

        for text in texts:
            words = self._tokenize(text)
            unique_words = set(words)
            for word in words:
                vocab_counts[word] = vocab_counts.get(word, 0) + 1
            for word in unique_words:
                term_doc_freq[word] = term_doc_freq.get(word, 0) + 1

        # Select top N most frequent words
        sorted_vocab = sorted(vocab_counts.items(), key=lambda x: x[1], reverse=True)[:self.vocab_size]
        self.vocabulary = {word: idx for idx, (word, _) in enumerate(sorted_vocab)}

        # Compute Inverse Document Frequency (IDF)
        for word, idx in self.vocabulary.items():
            df = term_doc_freq.get(word, 1)
            self.idf[word] = math.log((doc_count + 1) / (df + 1)) + 1.0

        self.is_fitted = True

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"\b\w+\b", text.lower())

    def embed_texts(self, texts: List[str]):
        if not self.is_fitted:
            self.fit(texts)

        embeddings = []
        for text in texts:
            vec = [0.0] * self.vocab_size
            words = self._tokenize(text)
            if not words:
                embeddings.append(np.array(vec, dtype=np.float32) if HAS_NUMPY else vec)
                continue

            tf = {}
            for w in words:
                tf[w] = tf.get(w, 0) + 1

            total_words = len(words)
            for word, count in tf.items():
                if word in self.vocabulary:
                    idx = self.vocabulary[word]
                    tf_val = count / total_words
                    idf_val = self.idf.get(word, 1.0)
                    vec[idx] = tf_val * idf_val

            # L2 Normalize
            norm = math.sqrt(sum(v * v for v in vec))
            if norm > 0:
                vec = [v / norm for v in vec]
            
            embeddings.append(np.array(vec, dtype=np.float32) if HAS_NUMPY else vec)

        return np.array(embeddings, dtype=np.float32) if HAS_NUMPY else embeddings


class SentenceTransformerEmbeddingProvider(BaseEmbeddingProvider):
    """Embedding provider using PyTorch + sentence-transformers."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts: List[str]):
        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return embeddings.astype(np.float32) if HAS_NUMPY else embeddings.tolist()


class EmbeddingManager:
    """Factory to initialize and manage embeddings."""

    @staticmethod
    def get_provider(preferred_model: str = "all-MiniLM-L6-v2") -> BaseEmbeddingProvider:
        try:
            return SentenceTransformerEmbeddingProvider(preferred_model)
        except Exception:
            return TFIDFEmbeddingProvider(vocab_size=384)
