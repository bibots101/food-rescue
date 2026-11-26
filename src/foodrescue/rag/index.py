"""Finding the passages that answer a question.

Two retrievers, same interface (`search(query, k)` -> best passages first):

* `EmbeddingIndex`: dense retrieval. Each passage and each query becomes a vector with `bge-m3`
  (multilingual embedding model served by Ollama), and passages are ranked by cosine similarity.
  Vectors are saved to disk (passages once, queries as they come), so after one run with Ollama the
  demo works offline.
* `BM25Index`: keyword retrieval (Okapi BM25), written here in a few lines. It needs no model, so it is
  the fallback when Ollama is not available.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np

from foodrescue.config import PROJECT_ROOT
from foodrescue.rag.corpus import Passage

CACHE = PROJECT_ROOT / "data" / "safety" / "index"
_WORD = re.compile(r"[a-zà-ÿ0-9°]+")
STOPWORDS = set("the a an and or of to in on for is are be with by as at from that this it its not no can "
                "should must may which their they there these those than then was were has have had".split())


def tokenize(text: str) -> list[str]:
    return [w for w in _WORD.findall(text.lower()) if w not in STOPWORDS and len(w) > 1]


class BM25Index:
    """Okapi BM25 (k1 = 1.5, b = 0.75): the classic keyword ranking."""

    def __init__(self, passages: list[Passage], k1: float = 1.5, b: float = 0.75) -> None:
        self.passages, self.k1, self.b = passages, k1, b
        self.docs = [Counter(tokenize(p.text)) for p in passages]
        self.lengths = [sum(d.values()) for d in self.docs]
        self.avg_length = sum(self.lengths) / len(self.lengths)
        frequency = Counter(word for doc in self.docs for word in doc)
        n = len(passages)
        self.idf = {w: math.log(1 + (n - df + 0.5) / (df + 0.5)) for w, df in frequency.items()}

    def scores(self, query: str) -> list[float]:
        words = tokenize(query)
        result = []
        for doc, length in zip(self.docs, self.lengths, strict=True):
            score = 0.0
            for word in words:
                tf = doc.get(word, 0)
                if tf:
                    norm = tf + self.k1 * (1 - self.b + self.b * length / self.avg_length)
                    score += self.idf[word] * tf * (self.k1 + 1) / norm
            result.append(score)
        return result

    def search(self, query: str, k: int = 3) -> list[tuple[Passage, float]]:
        scores = self.scores(query)
        best = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        return [(self.passages[i], round(scores[i], 3)) for i in best if scores[i] > 0]


class EmbeddingIndex:
    """Cosine similarity over `bge-m3` vectors, with an on-disk cache of every vector."""

    def __init__(self, passages: list[Passage], model: str = "bge-m3", host: str | None = None,
                 cache_dir: Path = CACHE, offline: bool = False) -> None:
        self.passages, self.model, self.cache_dir, self.offline = passages, model, cache_dir, offline
        self._client = None if offline else _ollama_client(host)
        self._queries = self._load_queries()
        self.matrix = self._passage_vectors()

    def search(self, query: str, k: int = 3) -> list[tuple[Passage, float]]:
        vector = self._query_vector(query)
        similarities = self.matrix @ vector
        best = np.argsort(-similarities)[:k]
        return [(self.passages[i], round(float(similarities[i]), 3)) for i in best]

    # --- vectors -------------------------------------------------------------------------

    def _embed(self, texts: list[str]) -> np.ndarray:
        if self._client is None:
            raise LookupError("offline and not in the cache")
        response = self._client.embed(model=self.model, input=texts)
        vectors = np.asarray(response.embeddings, dtype=np.float32)
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)  # unit length: dot = cosine

    def _passage_vectors(self) -> np.ndarray:
        digest = hashlib.sha256(json.dumps([self.model] + [p.text for p in self.passages]).encode()).hexdigest()
        path = self.cache_dir / f"passages-{self.model}-{digest[:16]}.npy"
        if path.exists():
            return np.load(path)
        vectors = np.vstack([self._embed([p.text for p in self.passages[i:i + 32]])
                             for i in range(0, len(self.passages), 32)])
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, vectors)
        return vectors

    def _query_vector(self, query: str) -> np.ndarray:
        key = hashlib.sha256(f"{self.model}|{query}".encode()).hexdigest()[:24]
        if key not in self._queries:
            self._queries[key] = self._embed([query])[0].round(6).tolist()
            self._save_queries()
        return np.asarray(self._queries[key], dtype=np.float32)

    def _queries_path(self) -> Path:
        return self.cache_dir / f"queries-{self.model}.json"

    def _load_queries(self) -> dict[str, list[float]]:
        path = self._queries_path()
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    def _save_queries(self) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._queries_path().write_text(json.dumps(self._queries), encoding="utf-8")


def _ollama_client(host: str | None):
    import ollama

    return ollama.Client(host=host) if host else ollama.Client()


class Retriever:
    """Embeddings when possible, BM25 otherwise: the RAG never stops the demo."""

    def __init__(self, passages: list[Passage], model: str = "bge-m3", host: str | None = None,
                 cache_dir: Path = CACHE, offline: bool = False) -> None:
        self.bm25 = BM25Index(passages)
        try:
            self.dense: EmbeddingIndex | None = EmbeddingIndex(passages, model, host, cache_dir, offline)
        except Exception:  # Ollama down or model missing, and no cached vectors
            self.dense = None
        self.last_method = ""

    def search(self, query: str, k: int = 3) -> list[tuple[Passage, float]]:
        if self.dense is not None:
            try:
                results = self.dense.search(query, k)
                self.last_method = "embeddings"
                return results
            except Exception:  # this query was never embedded and Ollama is not reachable
                pass
        self.last_method = "bm25"
        return self.bm25.search(query, k)
