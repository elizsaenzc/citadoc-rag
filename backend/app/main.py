"""HTTP API (FastAPI). Interactive docs at /docs (Swagger UI) and /redoc.

Also serves the built React frontend when it is present.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.config import BASE_DIR, get_settings
from app.container import Container, build_container
from app.conversations import ConversationNotFoundError
from app.ingestion.loaders import UnsupportedFileError
from app.llm.providers import LLMError
from app.models import (
    AgentResponse,
    AskRequest,
    AskResponse,
    ConversationDetail,
    ConversationSummary,
    DocumentInfo,
)
from app.rag.service import DocumentNotFoundError

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

STATIC_DIR = BASE_DIR / "static"  # filled by the Docker build with the React app

TAGS = [
    {"name": "Documentos", "description": "Subir, listar y eliminar documentos del índice."},
    {"name": "Preguntas", "description": "Preguntas con respuesta citada y agente con herramientas."},
    {"name": "Conversaciones", "description": "Historial persistente de preguntas y respuestas."},
    {"name": "Sistema", "description": "Estado y configuración activa."},
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.container = build_container(get_settings())
    yield


app = FastAPI(
    title="Citadoc RAG API",
    version="2.0.0",
    description=(
        "Pregunta a tus documentos con RAG: búsqueda híbrida (BM25 + vectores), respuestas "
        "con citas a la fuente, control de alucinaciones, agente con herramientas y "
        "conversaciones persistentes."
    ),
    openapi_tags=TAGS,
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def container(request: Request) -> Container:
    return request.app.state.container


# ---------- system ----------


@app.get("/api/health", tags=["Sistema"], summary="Estado y modo de funcionamiento")
def health(request: Request) -> dict:
    c = container(request)
    return {
        "status": "ok",
        "mode": c.rag.mode,
        "embeddings": c.settings.embedding_provider,
        "vector_store": c.settings.vector_store,
        "documents": len(c.rag.documents),
    }


# ---------- documents ----------


@app.get(
    "/api/documents", response_model=list[DocumentInfo], tags=["Documentos"], summary="Listar documentos"
)
def list_documents(request: Request) -> list[DocumentInfo]:
    return list(container(request).rag.documents.values())


@app.post(
    "/api/documents",
    response_model=DocumentInfo,
    status_code=201,
    tags=["Documentos"],
    summary="Subir un documento",
    description="Extrae el texto (PDF, DOCX, TXT o MD), lo divide en fragmentos, "
    "genera los embeddings y los indexa.",
)
async def upload_document(request: Request, file: UploadFile = File(...)) -> DocumentInfo:
    c = container(request)
    content = await file.read()
    if len(content) > c.settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(413, f"El archivo supera {c.settings.max_upload_mb} MB")
    try:
        return c.rag.ingest(Path(file.filename or "documento.txt").name, content)
    except UnsupportedFileError as exc:
        raise HTTPException(415, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@app.delete("/api/documents/{doc_id}", status_code=204, tags=["Documentos"], summary="Eliminar un documento")
def delete_document(request: Request, doc_id: str) -> None:
    try:
        container(request).rag.delete(doc_id)
    except DocumentNotFoundError as exc:
        raise HTTPException(404, "Documento no encontrado") from exc


# ---------- questions ----------


def _conversation_for(c: Container, body: AskRequest) -> str:
    if body.conversation_id:
        if not c.conversations.exists(body.conversation_id):
            raise HTTPException(404, "Conversación no encontrada")
        return body.conversation_id
    return c.conversations.create(body.question)


@app.post(
    "/api/ask",
    response_model=AskResponse,
    tags=["Preguntas"],
    summary="Preguntar con respuesta citada",
    description="Recupera los fragmentos más relevantes y responde citándolos con [n]. "
    "Si no hay base suficiente en los documentos, lo dice. Guarda la pregunta y la respuesta "
    "en la conversación indicada (o en una nueva).",
)
def ask(request: Request, body: AskRequest) -> AskResponse:
    c = container(request)
    conversation_id = _conversation_for(c, body)
    history = c.conversations.history(conversation_id)
    try:
        result = c.rag.ask(body.question, body.doc_ids, body.top_k, history=history)
    except LLMError as exc:
        raise HTTPException(502, str(exc)) from exc
    c.conversations.add_message(conversation_id, "user", body.question)
    c.conversations.add_message(
        conversation_id,
        "assistant",
        result.answer,
        {
            "sources": [s.model_dump() for s in result.sources],
            "grounded": result.grounded,
            "search_query": result.search_query,
        },
    )
    result.conversation_id = conversation_id
    return result


@app.post(
    "/api/agent",
    response_model=AgentResponse,
    tags=["Preguntas"],
    summary="Petición al agente con herramientas",
    description="El agente elige entre listar, buscar, resumir o comparar documentos y "
    "devuelve cada paso que ha dado.",
)
def agent(request: Request, body: AskRequest) -> AgentResponse:
    c = container(request)
    conversation_id = _conversation_for(c, body)
    try:
        result = c.agent.run(body.question)
    except LLMError as exc:
        raise HTTPException(502, str(exc)) from exc
    c.conversations.add_message(conversation_id, "user", body.question)
    c.conversations.add_message(
        conversation_id,
        "assistant",
        result.answer,
        {
            "sources": [s.model_dump() for s in result.sources],
            "steps": [s.model_dump() for s in result.steps],
        },
    )
    result.conversation_id = conversation_id
    return result


# ---------- conversations ----------


@app.get(
    "/api/conversations",
    response_model=list[ConversationSummary],
    tags=["Conversaciones"],
    summary="Listar conversaciones",
)
def list_conversations(request: Request) -> list[ConversationSummary]:
    return container(request).conversations.list()


@app.get(
    "/api/conversations/{conversation_id}",
    response_model=ConversationDetail,
    tags=["Conversaciones"],
    summary="Ver una conversación con sus mensajes",
)
def get_conversation(request: Request, conversation_id: str) -> ConversationDetail:
    try:
        return container(request).conversations.get(conversation_id)
    except ConversationNotFoundError as exc:
        raise HTTPException(404, "Conversación no encontrada") from exc


@app.delete(
    "/api/conversations/{conversation_id}",
    status_code=204,
    tags=["Conversaciones"],
    summary="Eliminar una conversación",
)
def delete_conversation(request: Request, conversation_id: str) -> None:
    try:
        container(request).conversations.delete(conversation_id)
    except ConversationNotFoundError as exc:
        raise HTTPException(404, "Conversación no encontrada") from exc


# ---------- frontend ----------

if STATIC_DIR.exists():
    app.mount("/assets", StaticFiles(directory=STATIC_DIR / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str) -> FileResponse:
        file = STATIC_DIR / path
        if path and file.is_file() and file.resolve().is_relative_to(STATIC_DIR.resolve()):
            return FileResponse(file)
        return FileResponse(STATIC_DIR / "index.html")
