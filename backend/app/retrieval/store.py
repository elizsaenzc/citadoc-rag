"""Vector stores.

``VectorStore`` is the port; ``InMemoryVectorStore`` (persisted to disk) and
``PgVectorStore`` (PostgreSQL + pgvector) are adapters. The rest of the app
only depends on the port, so switching stores is a configuration change.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Protocol

import numpy as np

from app.models import Chunk


class VectorStore(Protocol):
    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None: ...

    def delete_document(self, doc_id: str) -> list[str]:
        """Remove a document; return the ids of the removed chunks."""
        ...

    def search(
        self, vector: np.ndarray, k: int, doc_ids: set[str] | None = None
    ) -> list[tuple[Chunk, float]]: ...

    def all_chunks(self) -> list[Chunk]: ...


class InMemoryVectorStore:
    """Exact cosine search with NumPy, saved as JSON + .npy so it survives restarts."""

    def __init__(self, path: Path | None = None):
        self.path = path
        self.chunks: list[Chunk] = []
        self.vectors: np.ndarray | None = None
        if path and (path / "chunks.json").exists():
            self._load()

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        if not chunks:
            return
        self.chunks.extend(chunks)
        self.vectors = vectors if self.vectors is None else np.vstack([self.vectors, vectors])
        self._save()

    def delete_document(self, doc_id: str) -> list[str]:
        keep = [i for i, c in enumerate(self.chunks) if c.doc_id != doc_id]
        removed = [c.id for c in self.chunks if c.doc_id == doc_id]
        self.chunks = [self.chunks[i] for i in keep]
        self.vectors = self.vectors[keep] if self.vectors is not None and keep else None
        self._save()
        return removed

    def search(
        self, vector: np.ndarray, k: int, doc_ids: set[str] | None = None
    ) -> list[tuple[Chunk, float]]:
        if self.vectors is None or not self.chunks:
            return []
        similarities = self.vectors @ vector  # rows are normalized: dot = cosine
        order = np.argsort(-similarities)
        results: list[tuple[Chunk, float]] = []
        for index in order:
            chunk = self.chunks[index]
            if doc_ids is not None and chunk.doc_id not in doc_ids:
                continue
            results.append((chunk, float(similarities[index])))
            if len(results) == k:
                break
        return results

    def all_chunks(self) -> list[Chunk]:
        return list(self.chunks)

    def _save(self) -> None:
        if not self.path:
            return
        self.path.mkdir(parents=True, exist_ok=True)
        (self.path / "chunks.json").write_text(
            json.dumps([asdict(c) for c in self.chunks], ensure_ascii=False), encoding="utf-8"
        )
        if self.vectors is not None:
            np.save(self.path / "vectors.npy", self.vectors)
        elif (self.path / "vectors.npy").exists():
            (self.path / "vectors.npy").unlink()

    def _load(self) -> None:
        data = json.loads((self.path / "chunks.json").read_text(encoding="utf-8"))
        self.chunks = [Chunk(**item) for item in data]
        vectors_file = self.path / "vectors.npy"
        self.vectors = np.load(vectors_file) if vectors_file.exists() else None


class PgVectorStore:
    """PostgreSQL + pgvector adapter (cosine distance with an HNSW index).

    The table name includes the vector size (``chunks_1536``...), so changing the
    embedding model never mixes vectors of different dimensions.
    """

    def __init__(self, database_url: str, dim: int):
        import psycopg  # optional dependency: pip install "psycopg[binary]"

        self.dim = dim
        self.table = f"chunks_{int(dim)}"
        self.conn = psycopg.connect(database_url, autocommit=True)
        with self.conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.table} (
                    id        TEXT PRIMARY KEY,
                    doc_id    TEXT NOT NULL,
                    doc_name  TEXT NOT NULL,
                    text      TEXT NOT NULL,
                    page      INT,
                    position  INT NOT NULL,
                    metadata  JSONB NOT NULL DEFAULT '{{}}',
                    embedding vector({dim}) NOT NULL
                )"""
            )
            cur.execute(f"CREATE INDEX IF NOT EXISTS {self.table}_doc_idx ON {self.table} (doc_id)")
            cur.execute(
                f"CREATE INDEX IF NOT EXISTS {self.table}_embedding_idx "
                f"ON {self.table} USING hnsw (embedding vector_cosine_ops)"
            )

    @staticmethod
    def _vec(vector: np.ndarray) -> str:
        return "[" + ",".join(f"{x:.6f}" for x in vector) + "]"

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        with self.conn.cursor() as cur:
            cur.executemany(
                f"INSERT INTO {self.table} (id, doc_id, doc_name, text, page, position, metadata, embedding)"
                " VALUES (%s, %s, %s, %s, %s, %s, %s, %s::vector) ON CONFLICT (id) DO NOTHING",
                [
                    (c.id, c.doc_id, c.doc_name, c.text, c.page, c.position,
                     json.dumps(c.metadata), self._vec(v))
                    for c, v in zip(chunks, vectors, strict=True)
                ],
            )

    def delete_document(self, doc_id: str) -> list[str]:
        with self.conn.cursor() as cur:
            cur.execute(f"DELETE FROM {self.table} WHERE doc_id = %s RETURNING id", (doc_id,))
            return [row[0] for row in cur.fetchall()]

    def search(
        self, vector: np.ndarray, k: int, doc_ids: set[str] | None = None
    ) -> list[tuple[Chunk, float]]:
        sql = (
            "SELECT id, doc_id, doc_name, text, page, position, metadata, "
            f"1 - (embedding <=> %s::vector) AS similarity FROM {self.table}"
        )
        params: list = [self._vec(vector)]
        if doc_ids is not None:
            sql += " WHERE doc_id = ANY(%s)"
            params.append(list(doc_ids))
        sql += " ORDER BY embedding <=> %s::vector LIMIT %s"
        params += [self._vec(vector), k]
        with self.conn.cursor() as cur:
            cur.execute(sql, params)
            return [(Chunk(*row[:7]), float(row[7])) for row in cur.fetchall()]

    def all_chunks(self) -> list[Chunk]:
        with self.conn.cursor() as cur:
            cur.execute(
                f"SELECT id, doc_id, doc_name, text, page, position, metadata FROM {self.table} "
                "ORDER BY doc_id, position"
            )
            return [Chunk(*row) for row in cur.fetchall()]
