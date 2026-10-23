"""MCP server: exposes the document base as tools for any MCP client
(Claude Desktop, Claude Code, Cursor, VS Code...).

Run it with:
    python -m app.mcp_server            # stdio transport

Example Claude Desktop config (claude_desktop_config.json):
    {
      "mcpServers": {
        "citadoc": {
          "command": "python",
          "args": ["-m", "app.mcp_server"],
          "cwd": "/ruta/a/citadoc/backend"
        }
      }
    }
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from app.config import get_settings
from app.container import build_container

mcp = FastMCP("citadoc")
_container = None


def _rag():
    global _container
    if _container is None:
        _container = build_container(get_settings())
    return _container.rag


@mcp.tool()
def list_documents() -> list[dict]:
    """List the documents in the knowledge base."""
    return [d.model_dump() for d in _rag().documents.values()]


@mcp.tool()
def search_documents(query: str, top_k: int = 5) -> dict:
    """Search the documents and return the most relevant passages with their source
    (document and page). Use it to answer questions grounded in the documents."""
    hits, grounded = _rag().retrieve(query, top_k)
    return {
        "grounded": grounded,
        "passages": [
            {
                "ref": i,
                "document": h.chunk.doc_name,
                "page": h.chunk.page,
                "text": h.chunk.text,
                "score": round(h.score, 4),
            }
            for i, h in enumerate(hits, start=1)
        ],
    }


@mcp.tool()
def ask_documents(question: str) -> dict:
    """Answer a question using only the documents, with numbered citations."""
    return _rag().ask(question).model_dump()


@mcp.tool()
def summarize_document(document: str) -> str:
    """Summarize one document, given its name (or part of it) or id."""
    doc, summary = _rag().summarize(document)
    return f"{doc.name}\n{summary}"


if __name__ == "__main__":
    mcp.run()
