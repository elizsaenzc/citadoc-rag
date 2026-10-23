"""Wires the adapters together (the only place that knows the concrete classes)."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from app.agent.agent import Agent
from app.config import Settings
from app.conversations import ConversationStore
from app.llm.providers import build_llm
from app.rag.service import RagService
from app.retrieval.embeddings import AzureOpenAIEmbedder, Embedder, HashingEmbedder, OpenAIEmbedder
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.store import InMemoryVectorStore, PgVectorStore, VectorStore

log = logging.getLogger("citadoc")


@dataclass
class Container:
    settings: Settings
    rag: RagService
    agent: Agent
    conversations: ConversationStore


def build_embedder(settings: Settings) -> Embedder:
    if settings.embedding_provider == "openai":
        return OpenAIEmbedder(settings.openai_api_key, settings.embedding_model, settings.openai_base_url)
    if settings.embedding_provider == "azure":
        return AzureOpenAIEmbedder(
            settings.azure_openai_endpoint,
            settings.azure_openai_api_key,
            settings.azure_openai_embedding_deployment,
            settings.azure_openai_api_version,
        )
    return HashingEmbedder(settings.embedding_dim)


def build_store(settings: Settings, dim: int) -> VectorStore:
    if settings.vector_store == "pgvector":
        return PgVectorStore(settings.database_url, dim)
    return InMemoryVectorStore(settings.data_dir / "index")


def build_container(settings: Settings) -> Container:
    embedder = build_embedder(settings)
    store = build_store(settings, embedder.dim)
    retriever = HybridRetriever(embedder, store, settings.rrf_k)
    rag = RagService(settings, retriever, build_llm(settings))
    if settings.load_sample_docs:
        load_samples(rag)
    log.info("Citadoc ready: mode=%s, documents=%d", rag.mode, len(rag.documents))
    return Container(
        settings=settings,
        rag=rag,
        agent=Agent(rag),
        conversations=ConversationStore(settings.data_dir / "conversations.db"),
    )


def load_samples(rag: RagService) -> None:
    folder = rag.settings.sample_docs_dir
    if not folder.exists():
        return
    loaded = {d.name for d in rag.documents.values()}
    for path in sorted(folder.iterdir()):
        if path.is_file() and path.name not in loaded:
            rag.ingest(path.name, path.read_bytes(), sample=True)
