import numpy as np

from app.models import Chunk
from app.retrieval.bm25 import BM25Index
from app.retrieval.embeddings import HashingEmbedder
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.store import InMemoryVectorStore
from app.retrieval.text import tokenize


def test_tokenize_removes_accents_stopwords_and_plurals():
    assert tokenize("Las facturas de los Proveedores") == ["factura", "proveedor"]


def test_hashing_embeddings_are_normalized_and_similar_for_related_text():
    embedder = HashingEmbedder(dim=512)
    vectors = embedder.embed(["casco de seguridad en obra", "cascos de seguridad", "receta de paella"])
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)
    assert vectors[0] @ vectors[1] > vectors[0] @ vectors[2]


def test_bm25_ranks_the_matching_document_first_and_supports_removal():
    index = BM25Index()
    index.add("a", "vacaciones del empleado y días libres")
    index.add("b", "pedidos a proveedores y facturas")
    assert index.search("facturas de proveedores", k=2)[0][0] == "b"
    index.remove({"b"})
    assert index.search("facturas", k=2) == []


def _chunk(cid: str, doc: str, text: str) -> Chunk:
    return Chunk(id=cid, doc_id=doc, doc_name=f"{doc}.md", text=text)


def test_hybrid_search_filters_by_document_and_persists(tmp_path):
    store = InMemoryVectorStore(tmp_path / "index")
    retriever = HybridRetriever(HashingEmbedder(256), store)
    retriever.add(
        [
            _chunk("1", "rrhh", "Los empleados tienen 23 días de vacaciones al año."),
            _chunk("2", "compras", "Los pedidos de más de 30.000 euros necesitan tres presupuestos."),
        ]
    )
    assert retriever.search("vacaciones", k=1)[0].chunk.id == "1"
    assert all(h.chunk.doc_id == "compras" for h in retriever.search("vacaciones", 5, {"compras"}))

    reloaded = HybridRetriever(HashingEmbedder(256), InMemoryVectorStore(tmp_path / "index"))
    assert reloaded.search("presupuestos", k=1)[0].chunk.id == "2"

    assert reloaded.delete_document("rrhh") == 1
    assert all(h.chunk.doc_id != "rrhh" for h in reloaded.search("vacaciones", 5))
