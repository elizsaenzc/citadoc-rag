"""RAG application service: ingest documents and answer questions with citations.

Flow for a question:
    retrieve (hybrid BM25 + vectors) -> grounding check -> generate -> validate citations
"""

from __future__ import annotations

import json
import re
import uuid
from collections import Counter
from pathlib import Path

from app.config import Settings
from app.ingestion.chunker import chunk_text
from app.ingestion.loaders import load_document
from app.llm.providers import LLM
from app.models import AskResponse, Chunk, DocumentInfo, ScoredChunk, Source
from app.rag import prompts
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.text import tokenize, tokenize_with_surface

NOT_FOUND = "No encuentro esa información en los documentos."
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
_CITATION = re.compile(r"\[(\d+)\]")
_MD_HEADING = re.compile(r"^#{1,6}\s*", re.MULTILINE)


class DocumentNotFoundError(LookupError):
    pass


def _sentences(text: str) -> list[str]:
    """Split into sentences, dropping headings, very short lines and fragments that
    start mid-sentence (left by the chunk overlap)."""
    out = []
    for raw in _SENTENCE_SPLIT.split(text):
        if raw.lstrip().startswith("#"):
            continue  # markdown heading
        sentence = raw.strip(" -*\t")
        if len(sentence) < 25 or not sentence[0].isupper() and not sentence[0].isdigit():
            continue
        if not sentence.endswith((".", "!", "?", ":")) or len(sentence.split()) < 8:
            continue  # headings, titles or cut text
        out.append(sentence)
    return out


def _dedupe(sentences: list[str]) -> list[str]:
    kept: list[str] = []
    for sentence in sentences:
        if not any(sentence in other or other in sentence for other in kept):
            kept.append(sentence)
    return kept


class RagService:
    def __init__(self, settings: Settings, retriever: HybridRetriever, llm: LLM | None):
        self.settings = settings
        self.retriever = retriever
        self.llm = llm
        self._registry_path: Path = settings.data_dir / "documents.json"
        self.documents: dict[str, DocumentInfo] = self._load_registry()
        if not self.documents:
            self._rebuild_registry_from_index()

    @property
    def mode(self) -> str:
        return f"llm:{self.llm.name}" if self.llm else "demo"

    # ---------- documents ----------

    def ingest(self, filename: str, content: bytes, sample: bool = False) -> DocumentInfo:
        pages = load_document(filename, content)
        doc_id = uuid.uuid4().hex[:12]
        chunks: list[Chunk] = []
        for page in pages:
            for text in chunk_text(page.text, self.settings.chunk_size, self.settings.chunk_overlap):
                chunks.append(
                    Chunk(
                        id=f"{doc_id}-{len(chunks)}",
                        doc_id=doc_id,
                        doc_name=filename,
                        text=text,
                        page=page.number,
                        position=len(chunks),
                    )
                )
        if not chunks:
            raise ValueError("No se ha podido extraer texto del documento (¿es un PDF escaneado?).")
        self.retriever.add(chunks)
        pdf_pages = [p.number for p in pages if p.number is not None]
        info = DocumentInfo(
            id=doc_id,
            name=filename,
            chunks=len(chunks),
            pages=max(pdf_pages) if pdf_pages else None,
            sample=sample,
        )
        self.documents[doc_id] = info
        self._save_registry()
        return info

    def delete(self, doc_id: str) -> None:
        if doc_id not in self.documents:
            raise DocumentNotFoundError(doc_id)
        self.retriever.delete_document(doc_id)
        del self.documents[doc_id]
        self._save_registry()

    def find_document(self, name_or_id: str) -> DocumentInfo:
        if name_or_id in self.documents:
            return self.documents[name_or_id]
        wanted = set(tokenize(name_or_id))
        best, best_overlap = None, 0
        for doc in self.documents.values():
            overlap = len(wanted & set(tokenize(Path(doc.name).stem.replace("_", " "))))
            if overlap > best_overlap:
                best, best_overlap = doc, overlap
        if best is None:
            raise DocumentNotFoundError(name_or_id)
        return best

    # ---------- question answering ----------

    def retrieve(self, question: str, top_k: int | None = None, doc_ids: list[str] | None = None):
        k = top_k or self.settings.top_k
        hits = self.retriever.search(question, k, set(doc_ids) if doc_ids else None)
        return hits, self._is_grounded(question, hits)

    def standalone_question(self, question: str, history: list[tuple[str, str]] | None) -> str:
        """Turn a follow-up ("¿y para jornada parcial?") into a self-contained search query.

        With an LLM, the model rewrites it using the conversation. In demo mode, a short
        question (few key terms) is combined with the previous user question.
        """
        previous = [content for role, content in (history or []) if role == "user"]
        if not previous:
            return question
        if self.llm:
            transcript = "\n".join(f"{role}: {content}" for role, content in (history or [])[-6:])
            rewritten = self.llm.complete(
                prompts.CONDENSE_SYSTEM,
                prompts.condense_user_prompt(transcript, question),
                max_tokens=150,
            ).strip()
            return rewritten or question
        if len(tokenize(question)) <= 3:
            return f"{previous[-1]} {question}"
        return question

    def ask(
        self,
        question: str,
        doc_ids: list[str] | None = None,
        top_k: int | None = None,
        history: list[tuple[str, str]] | None = None,
    ) -> AskResponse:
        query = self.standalone_question(question, history)
        search_query = query if query != question else None
        hits, grounded = self.retrieve(query, top_k, doc_ids)
        if not grounded:
            return AskResponse(
                answer=NOT_FOUND, sources=[], grounded=False, mode=self.mode, search_query=search_query
            )

        sources = self._to_sources(hits)
        if self.llm:
            context = prompts.build_context(
                [(s.ref, s.doc_name, h.chunk.text, s.page) for s, h in zip(sources, hits, strict=True)]
            )
            answer = self.llm.complete(prompts.ANSWER_SYSTEM, prompts.answer_user_prompt(query, context))
            answer = self._drop_invalid_citations(answer, len(sources))
        else:
            answer = self._extractive_answer(query, hits)

        if answer.strip().startswith(NOT_FOUND):
            return AskResponse(
                answer=NOT_FOUND, sources=[], grounded=False, mode=self.mode, search_query=search_query
            )
        cited = {int(n) for n in _CITATION.findall(answer)}
        used = [s for s in sources if s.ref in cited] or sources
        return AskResponse(
            answer=answer, sources=used, grounded=True, mode=self.mode, search_query=search_query
        )

    def summarize(self, name_or_id: str) -> tuple[DocumentInfo, str]:
        doc = self.find_document(name_or_id)
        text = "\n\n".join(c.text for c in self.retriever.chunks_of(doc.id))
        if self.llm:
            return doc, self.llm.complete(prompts.SUMMARY_SYSTEM, text[:24000], max_tokens=600)
        return doc, self._extractive_summary(text)

    def compare(self, a: str, b: str) -> tuple[DocumentInfo, DocumentInfo, str]:
        doc_a, doc_b = self.find_document(a), self.find_document(b)
        text_a = "\n\n".join(c.text for c in self.retriever.chunks_of(doc_a.id))
        text_b = "\n\n".join(c.text for c in self.retriever.chunks_of(doc_b.id))
        if self.llm:
            user = (
                f"Documento A: {doc_a.name}\n{text_a[:12000]}\n\n"
                f"Documento B: {doc_b.name}\n{text_b[:12000]}"
            )
            return doc_a, doc_b, self.llm.complete(prompts.COMPARE_SYSTEM, user, max_tokens=700)
        return doc_a, doc_b, self._extractive_compare(doc_a.name, text_a, doc_b.name, text_b)

    # ---------- helpers ----------

    def _is_grounded(self, question: str, hits: list[ScoredChunk]) -> bool:
        """Answer only if the retrieved text really relates to the question."""
        if not hits:
            return False
        if max(h.vector_score for h in hits) >= self.settings.min_vector_similarity:
            return True
        terms = set(tokenize(question))
        if not terms:
            return False
        best_coverage = max(
            len(terms & set(tokenize(f"{h.chunk.doc_name} {h.chunk.text}"))) / len(terms)
            for h in hits[:3]
        )
        return best_coverage >= self.settings.min_term_coverage

    @staticmethod
    def _to_sources(hits: list[ScoredChunk]) -> list[Source]:
        sources = []
        for ref, hit in enumerate(hits, start=1):
            text = _MD_HEADING.sub("", hit.chunk.text)  # cleaner to read in the UI
            if text[:1].islower():  # chunk starts mid-sentence (overlap): start at the next one
                cut = min((i for i in (text.find(". "), text.find("\n")) if i != -1), default=-1)
                if 0 <= cut < 200:
                    text = text[cut + 1 :].lstrip()
            snippet = text if len(text) <= 600 else text[:600] + "…"
            sources.append(
                Source(
                    ref=ref,
                    doc_id=hit.chunk.doc_id,
                    doc_name=hit.chunk.doc_name,
                    page=hit.chunk.page,
                    chunk_id=hit.chunk.id,
                    snippet=snippet,
                    score=round(hit.score, 4),
                )
            )
        return sources

    @staticmethod
    def _drop_invalid_citations(answer: str, n_sources: int) -> str:
        """Remove citations to sources that do not exist (a common LLM failure)."""
        return _CITATION.sub(lambda m: m.group(0) if 1 <= int(m.group(1)) <= n_sources else "", answer)

    def _extractive_answer(self, question: str, hits: list[ScoredChunk]) -> str:
        """Demo mode: pick the sentences that best match the question, with citations.

        Each sentence is scored by shared key terms plus its embedding similarity
        to the question (so 'teletrabajar' still matches 'teletrabajo'), with a
        bonus for chunks that ranked higher.
        """
        terms = set(tokenize(question))
        candidates: list[tuple[int, int, str]] = []  # (ref, position, sentence)
        for ref, hit in enumerate(hits[:3], start=1):
            for position, sentence in enumerate(_sentences(hit.chunk.text)):
                candidates.append((ref, position, sentence))
        if not candidates:
            return f"{hits[0].chunk.text[:300]} [1]"

        embedder = self.retriever.embedder
        query_vector = embedder.embed([question])[0]
        similarities = embedder.embed([s for _, _, s in candidates]) @ query_vector
        scored = []
        for (ref, position, sentence), similarity in zip(candidates, similarities, strict=True):
            overlap = len(terms & set(tokenize(sentence)))
            score = overlap + 3 * float(similarity) + 1.0 / ref - 0.01 * position
            scored.append((score, ref, sentence))
        scored.sort(key=lambda item: item[0], reverse=True)

        chosen: list[tuple[int, str]] = []
        for _, ref, sentence in scored:
            if any(sentence in other or other in sentence for _, other in chosen):
                continue  # same sentence repeated by the chunk overlap
            chosen.append((ref, sentence))
            if len(chosen) == 3:
                break
        return "Según los documentos:\n" + "\n".join(f"- {s} [{ref}]" for ref, s in chosen)

    @staticmethod
    def _extractive_summary(text: str, n: int = 5) -> str:
        sentences = _dedupe([s for s in _sentences(text) if len(s) > 40])
        if not sentences:
            return text[:500]
        freq = Counter(tokenize(text))
        scored = []
        for index, sentence in enumerate(sentences):
            tokens = tokenize(sentence)
            if tokens:
                scored.append((sum(freq[t] for t in tokens) / len(tokens) ** 0.5, index, sentence))
        top = sorted(sorted(scored, reverse=True)[:n], key=lambda item: item[1])
        return "\n".join(f"- {sentence}" for _, _, sentence in top)

    def _extractive_compare(self, name_a: str, text_a: str, name_b: str, text_b: str) -> str:
        surface: dict[str, Counter] = {}

        def key_terms(text: str) -> list[str]:
            pairs = [(s, w) for s, w in tokenize_with_surface(text) if s.isalpha() and len(s) > 3]
            for stem, word in pairs:
                surface.setdefault(stem, Counter())[word] += 1
            return [t for t, _ in Counter(s for s, _ in pairs).most_common(60)]

        def show(stems: list[str]) -> str:
            return ", ".join(surface[s].most_common(1)[0][0] for s in stems)

        # ignore words present in every document (company name, boilerplate...)
        everywhere = set.intersection(
            *(
                {t for t in tokenize(" ".join(c.text for c in self.retriever.chunks_of(d.id)))}
                for d in self.documents.values()
            )
        ) if len(self.documents) > 2 else set()
        terms_a = [t for t in key_terms(text_a) if t not in everywhere]
        terms_b = [t for t in key_terms(text_b) if t not in everywhere]
        shared = [t for t in terms_a if t in terms_b][:8]
        only_a = [t for t in terms_a if t not in terms_b][:8]
        only_b = [t for t in terms_b if t not in terms_a][:8]
        return (
            f"- Temas en común: {show(shared) or 'ninguno destacado'}\n"
            f"- Propios de {name_a}: {show(only_a)}\n"
            f"- Propios de {name_b}: {show(only_b)}"
        )

    def _rebuild_registry_from_index(self) -> None:
        """If the index outlives the registry file (e.g. pgvector + a fresh container),
        recover the document list from the stored chunks instead of re-ingesting."""
        samples = (
            {p.name for p in self.settings.sample_docs_dir.iterdir()}
            if self.settings.sample_docs_dir.exists()
            else set()
        )
        by_doc: dict[str, list[Chunk]] = {}
        for chunk in self.retriever.store.all_chunks():
            by_doc.setdefault(chunk.doc_id, []).append(chunk)
        for doc_id, chunks in by_doc.items():
            pages = [c.page for c in chunks if c.page]
            self.documents[doc_id] = DocumentInfo(
                id=doc_id,
                name=chunks[0].doc_name,
                chunks=len(chunks),
                pages=max(pages) if pages else None,
                sample=chunks[0].doc_name in samples,
            )
        if by_doc:
            self._save_registry()

    def _load_registry(self) -> dict[str, DocumentInfo]:
        if self._registry_path.exists():
            data = json.loads(self._registry_path.read_text(encoding="utf-8"))
            return {d["id"]: DocumentInfo(**d) for d in data}
        return {}

    def _save_registry(self) -> None:
        self._registry_path.parent.mkdir(parents=True, exist_ok=True)
        self._registry_path.write_text(
            json.dumps([d.model_dump() for d in self.documents.values()], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
