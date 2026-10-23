"""A small tool-using agent on top of the RAG service.

With an LLM configured, the model chooses the tools (JSON protocol, works with
any provider). In demo mode a deterministic router picks the tool from the
wording of the request, so the flow can still be shown without API keys.
Every step is returned to the client for transparency.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable

from app.models import AgentResponse, AgentStep, Source
from app.rag import prompts
from app.rag.service import DocumentNotFoundError, RagService
from app.retrieval.text import strip_accents

MAX_STEPS = 3


class Agent:
    def __init__(self, rag: RagService):
        self.rag = rag
        self.tools: dict[str, Callable[..., tuple[str, list[Source]]]] = {
            "list_documents": self._list_documents,
            "search_documents": self._search_documents,
            "summarize_document": self._summarize_document,
            "compare_documents": self._compare_documents,
        }

    # ---------- tools ----------

    def _list_documents(self) -> tuple[str, list[Source]]:
        if not self.rag.documents:
            return "No hay documentos cargados.", []
        lines = [f"- {d.name} ({d.chunks} fragmentos)" for d in self.rag.documents.values()]
        return "Documentos disponibles:\n" + "\n".join(lines), []

    def _search_documents(self, query: str) -> tuple[str, list[Source]]:
        result = self.rag.ask(query)
        return result.answer, result.sources

    def _summarize_document(self, document: str) -> tuple[str, list[Source]]:
        doc, summary = self.rag.summarize(document)
        return f"Resumen de {doc.name}:\n{summary}", []

    def _compare_documents(self, a: str, b: str) -> tuple[str, list[Source]]:
        doc_a, doc_b, comparison = self.rag.compare(a, b)
        return f"Comparación entre {doc_a.name} y {doc_b.name}:\n{comparison}", []

    def call(self, tool: str, tool_input: dict) -> tuple[str, list[Source]]:
        if tool not in self.tools:
            return f"Herramienta desconocida: {tool}", []
        try:
            return self.tools[tool](**tool_input)
        except DocumentNotFoundError as exc:
            return f"No encuentro el documento '{exc.args[0]}'.", []
        except TypeError as exc:
            return f"Parámetros no válidos para {tool}: {exc}", []

    # ---------- planning ----------

    def run(self, request: str) -> AgentResponse:
        if self.rag.llm:
            return self._run_with_llm(request)
        return self._run_demo(request)

    def _run_with_llm(self, request: str) -> AgentResponse:
        steps: list[AgentStep] = []
        sources: list[Source] = []
        transcript = f"Petición del usuario: {request}"
        for _ in range(MAX_STEPS):
            raw = self.rag.llm.complete(prompts.AGENT_SYSTEM, transcript, max_tokens=700)
            decision = _parse_json(raw)
            if decision is None or "final" in decision:
                answer = decision.get("final") if decision else raw
                return AgentResponse(answer=answer, steps=steps, sources=sources, mode=self.rag.mode)
            tool, tool_input = decision.get("tool", ""), decision.get("input", {}) or {}
            output, found = self.call(tool, tool_input)
            sources = _merge_sources(sources, found)
            steps.append(AgentStep(tool=tool, input=tool_input, output_preview=output[:300]))
            transcript += f"\n\nResultado de {tool}({json.dumps(tool_input, ensure_ascii=False)}):\n{output}"
        final = self.rag.llm.complete(
            prompts.AGENT_SYSTEM, transcript + '\n\nResponde ya con {"final": ...}.', max_tokens=700
        )
        decision = _parse_json(final) or {}
        return AgentResponse(
            answer=decision.get("final", final), steps=steps, sources=sources, mode=self.rag.mode
        )

    def _run_demo(self, request: str) -> AgentResponse:
        text = strip_accents(request.lower())
        names = self._mentioned_documents(text)
        if re.search(r"\b(compar|diferenc|vs\b)", text) and len(names) >= 2:
            tool, tool_input = "compare_documents", {"a": names[0], "b": names[1]}
        elif re.search(r"\b(resum|sintesis|de que trata)", text) and names:
            tool, tool_input = "summarize_document", {"document": names[0]}
        elif re.search(r"\b(lista|que documentos|cuales documentos|documentos hay)", text):
            tool, tool_input = "list_documents", {}
        else:
            tool, tool_input = "search_documents", {"query": request}
        output, sources = self.call(tool, tool_input)
        step = AgentStep(tool=tool, input=tool_input, output_preview=output[:300])
        return AgentResponse(answer=output, steps=[step], sources=sources, mode=self.rag.mode)

    def _mentioned_documents(self, text: str) -> list[str]:
        """Documents whose name words appear in the request, in order of appearance."""
        found: list[tuple[int, str]] = []
        for doc in self.rag.documents.values():
            stem = strip_accents(doc.name.rsplit(".", 1)[0].lower()).replace("_", " ").replace("-", " ")
            words = [w for w in stem.split() if len(w) > 3]
            positions = [text.find(w) for w in words if w in text]
            if positions:
                found.append((min(positions), doc.id))
        return [doc_id for _, doc_id in sorted(found)]


def _parse_json(raw: str) -> dict | None:
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return None
    try:
        value = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _merge_sources(existing: list[Source], new: list[Source]) -> list[Source]:
    seen = {s.chunk_id for s in existing}
    return existing + [s for s in new if s.chunk_id not in seen]
