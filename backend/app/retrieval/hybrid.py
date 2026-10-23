"""Hybrid retrieval: BM25 (exact terms) + vector search (meaning), fused with RRF.

Reciprocal Rank Fusion: score(d) = sum over rankings of 1 / (k + rank(d)).
It only uses ranks, so BM25 scores and cosine similarities (different scales)
can be combined without tuning weights.
"""

from __future__ import annotations

from app.models import Chunk, ScoredChunk
from app.retrieval.bm25 import BM25Index
from app.retrieval.embeddings import Embedder
from app.retrieval.store import VectorStore


class HybridRetriever:
    def __init__(self, embedder: Embedder, store: VectorStore, rrf_k: int = 60):
        self.embedder = embedder
        self.store = store
        self.rrf_k = rrf_k
        self.bm25 = BM25Index()
        self._chunks: dict[str, Chunk] = {}
        for chunk in store.all_chunks():  # rebuild the keyword index from the store
            self._index_keywords(chunk)

    def _index_keywords(self, chunk: Chunk) -> None:
        self._chunks[chunk.id] = chunk
        self.bm25.add(chunk.id, f"{chunk.doc_name}\n{chunk.text}")

    def add(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        vectors = self.embedder.embed([c.text for c in chunks])
        self.store.add(chunks, vectors)
        for chunk in chunks:
            self._index_keywords(chunk)

    def delete_document(self, doc_id: str) -> int:
        removed = set(self.store.delete_document(doc_id))
        self.bm25.remove(removed)
        for chunk_id in removed:
            self._chunks.pop(chunk_id, None)
        return len(removed)

    def chunks_of(self, doc_id: str) -> list[Chunk]:
        return sorted(
            (c for c in self._chunks.values() if c.doc_id == doc_id), key=lambda c: c.position
        )

    def search(self, query: str, k: int = 5, doc_ids: set[str] | None = None) -> list[ScoredChunk]:
        candidates = max(k * 4, 20)
        allowed = (
            {cid for cid, c in self._chunks.items() if c.doc_id in doc_ids}
            if doc_ids is not None
            else None
        )

        keyword_hits = self.bm25.search(query, candidates, allowed)
        query_vector = self.embedder.embed([query])[0]
        vector_hits = self.store.search(query_vector, candidates, doc_ids)

        fused: dict[str, ScoredChunk] = {}
        for rank, (chunk_id, score) in enumerate(keyword_hits, start=1):
            item = fused.setdefault(chunk_id, ScoredChunk(self._chunks[chunk_id], 0.0))
            item.score += 1 / (self.rrf_k + rank)
            item.bm25_rank, item.bm25_score = rank, score
        for rank, (chunk, similarity) in enumerate(vector_hits, start=1):
            item = fused.setdefault(chunk.id, ScoredChunk(chunk, 0.0))
            item.score += 1 / (self.rrf_k + rank)
            item.vector_rank, item.vector_score = rank, similarity

        ranked = sorted(fused.values(), key=lambda item: item.score, reverse=True)
        return ranked[:k]
