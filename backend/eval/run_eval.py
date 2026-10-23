"""Offline evaluation of the RAG pipeline on the sample documents.

Measures, for each retrieval strategy (BM25 only, vectors only, hybrid):
  * hit@1 / hit@3: the right document is in the top 1 / top 3 chunks
  * MRR: mean reciprocal rank of the first chunk from the right document
and, for the full pipeline:
  * answer accuracy: the expected fact appears in the answer
  * refusal rate on out-of-scope questions (should be 100%)

Run from backend/:  python -m eval.run_eval
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from app.config import Settings
from app.container import build_container
from app.rag.service import NOT_FOUND

DATASET = Path(__file__).with_name("dataset.json")


def rank_of(doc_names: list[str], expected: str) -> int | None:
    for rank, name in enumerate(doc_names, start=1):
        if name == expected:
            return rank
    return None


def main() -> None:
    data = json.loads(DATASET.read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory() as tmp:
        container = build_container(Settings(data_dir=Path(tmp), _env_file=None))
        rag, retriever = container.rag, container.rag.retriever

        def bm25(question: str) -> list[str]:
            return [retriever._chunks[cid].doc_name for cid, _ in retriever.bm25.search(question, 5)]

        def vectors(question: str) -> list[str]:
            vector = retriever.embedder.embed([question])[0]
            return [chunk.doc_name for chunk, _ in retriever.store.search(vector, 5)]

        def hybrid(question: str) -> list[str]:
            return [hit.chunk.doc_name for hit in retriever.search(question, 5)]

        print(f"{'Estrategia':<12}{'hit@1':>8}{'hit@3':>8}{'MRR':>8}")
        for name, strategy in [("BM25", bm25), ("Vectores", vectors), ("Híbrida", hybrid)]:
            ranks = [rank_of(strategy(item["question"]), item["doc"]) for item in data["in_scope"]]
            n = len(ranks)
            hit1 = sum(1 for r in ranks if r == 1) / n
            hit3 = sum(1 for r in ranks if r and r <= 3) / n
            mrr = sum(1 / r for r in ranks if r) / n
            print(f"{name:<12}{hit1:>8.2f}{hit3:>8.2f}{mrr:>8.2f}")

        correct = 0
        for item in data["in_scope"]:
            answer = rag.ask(item["question"]).answer
            ok = item["expected"].lower() in answer.lower()
            correct += ok
            if not ok:
                print(f"  ✗ {item['question']}\n    → {answer[:160]!r}")
        refused = sum(rag.ask(q).answer == NOT_FOUND for q in data["out_of_scope"])

        print(f"\nRespuestas correctas: {correct}/{len(data['in_scope'])} (modo {rag.mode})")
        print(f"Preguntas fuera de tema rechazadas: {refused}/{len(data['out_of_scope'])}")


if __name__ == "__main__":
    main()
