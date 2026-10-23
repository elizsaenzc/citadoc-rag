"""Domain objects and API schemas."""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import BaseModel, Field


@dataclass
class Chunk:
    """A piece of a document that is indexed and retrieved."""

    id: str
    doc_id: str
    doc_name: str
    text: str
    page: int | None = None  # 1-based page for PDFs, None otherwise
    position: int = 0  # order of the chunk inside its document
    metadata: dict = field(default_factory=dict)


@dataclass
class ScoredChunk:
    chunk: Chunk
    score: float  # fused score (Reciprocal Rank Fusion)
    bm25_rank: int | None = None
    vector_rank: int | None = None
    bm25_score: float = 0.0
    vector_score: float = 0.0  # cosine similarity


# ---------- API schemas ----------


class DocumentInfo(BaseModel):
    id: str
    name: str
    chunks: int
    pages: int | None = None
    sample: bool = False


class Source(BaseModel):
    ref: int = Field(description="Number used in the answer, e.g. [1]")
    doc_id: str
    doc_name: str
    page: int | None
    chunk_id: str
    snippet: str
    score: float


class AskRequest(BaseModel):
    question: str = Field(min_length=2, max_length=2000)
    doc_ids: list[str] | None = Field(default=None, description="Restrict search to these documents")
    top_k: int | None = Field(default=None, ge=1, le=20)
    conversation_id: str | None = Field(
        default=None, description="Continue this conversation; a new one is created if empty"
    )


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]
    grounded: bool = Field(description="False when nothing relevant was found")
    mode: str
    search_query: str | None = Field(
        default=None, description="Standalone question used for retrieval (follow-up questions)"
    )
    conversation_id: str | None = None


class AgentStep(BaseModel):
    tool: str
    input: dict
    output_preview: str


class AgentResponse(BaseModel):
    answer: str
    steps: list[AgentStep]
    sources: list[Source]
    mode: str
    conversation_id: str | None = None


class StoredMessage(BaseModel):
    role: str
    content: str
    created_at: str
    sources: list[Source] = []
    steps: list[AgentStep] = []
    grounded: bool | None = None
    search_query: str | None = None


class ConversationSummary(BaseModel):
    id: str
    title: str
    created_at: str
    updated_at: str
    messages: int


class ConversationDetail(BaseModel):
    id: str
    title: str
    created_at: str
    updated_at: str
    messages: list[StoredMessage]
