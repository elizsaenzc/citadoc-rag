"""Runs only when a PostgreSQL + pgvector database is available, e.g.
CITADOC_TEST_DATABASE_URL=postgresql://user:pass@localhost:5432/db pytest tests/test_pgvector.py
(CI starts one with the pgvector/pgvector Docker image)."""

import os

import pytest

from app.config import Settings
from app.container import build_container

DATABASE_URL = os.environ.get("CITADOC_TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(not DATABASE_URL, reason="no test database configured")


def _settings(tmp_path, name: str) -> Settings:
    return Settings(
        data_dir=tmp_path / name, vector_store="pgvector", database_url=DATABASE_URL, _env_file=None
    )


def test_pgvector_end_to_end_and_survives_a_fresh_container(tmp_path):
    import psycopg

    with psycopg.connect(DATABASE_URL, autocommit=True) as conn:
        conn.execute("DROP TABLE IF EXISTS chunks_1024")

    first = build_container(_settings(tmp_path, "a"))
    assert len(first.rag.documents) == 3
    answer = first.rag.ask("¿Cuántas horas dura el curso de prevención de riesgos laborales?")
    assert "20 horas" in answer.answer

    # new container with an empty data dir: documents are recovered, not duplicated
    second = build_container(_settings(tmp_path, "b"))
    assert len(second.rag.documents) == 3
    assert len(second.rag.retriever.store.all_chunks()) == len(first.rag.retriever.store.all_chunks())
