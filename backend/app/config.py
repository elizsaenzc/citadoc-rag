"""Application settings, read from environment variables (or a .env file).

Without any API key the app runs in *demo mode*: local hashing embeddings and an
extractive answer generator. Set a provider key to switch to real LLMs.
"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="CITADOC_", extra="ignore")

    # Where uploaded documents and the local index are stored
    data_dir: Path = BASE_DIR / "data"
    # Documents loaded on startup so the demo is never empty
    sample_docs_dir: Path = BASE_DIR / "sample_docs"
    load_sample_docs: bool = True

    # Chunking
    chunk_size: int = 800  # characters
    chunk_overlap: int = 120

    # Retrieval
    top_k: int = 5
    rrf_k: int = 60  # Reciprocal Rank Fusion constant
    # Grounding guard: the question is answered only if the best chunks cover
    # enough of the question's key terms OR are semantically close enough.
    # Otherwise the answer is "not found in the documents" (no hallucination).
    min_term_coverage: float = 0.4
    min_vector_similarity: float = 0.35

    # LLM provider: "demo", "azure" (Azure OpenAI), "anthropic" or "openai"
    # (any OpenAI-compatible API: OpenAI, OpenRouter, Ollama...)
    llm_provider: str = "demo"
    llm_model: str = "claude-sonnet-4-5"
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"

    # Azure OpenAI (used when llm_provider or embedding_provider is "azure")
    azure_openai_endpoint: str = ""  # https://<resource>.openai.azure.com
    azure_openai_api_key: str = ""
    azure_openai_api_version: str = "2024-10-21"
    azure_openai_chat_deployment: str = ""  # e.g. gpt-4o-mini
    azure_openai_embedding_deployment: str = ""  # e.g. text-embedding-3-small

    # Embeddings: "hashing" (local, no key), "azure" or "openai" (OpenAI-compatible endpoint)
    embedding_provider: str = "hashing"
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1024  # used by the hashing embedder

    # Vector store: "memory" (persisted to data_dir) or "pgvector"
    vector_store: str = "memory"
    database_url: str = ""

    # Max upload size in MB
    max_upload_mb: int = 10

    @property
    def demo_mode(self) -> bool:
        return self.llm_provider == "demo"


@lru_cache
def get_settings() -> Settings:
    return Settings()
