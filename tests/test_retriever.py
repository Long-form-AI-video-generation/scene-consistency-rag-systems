import numpy as np
import faiss

from src.rag.retriever import Retriever


def test_hybrid_retrieval():
    docs = [
        {"chunk_text": "red apple on tree"},
        {"chunk_text": "green banana in jungle"},
    ]
    embs = np.random.rand(2, 512).astype(np.float32)
    embs = embs / np.linalg.norm(embs, axis=1, keepdims=True)

    retriever = Retriever(docs, embs, set())
    index = faiss.IndexFlatIP(512)
    index.add(embs)
    retriever.set_faiss(index)

    query_emb = embs[0]
    candidates = retriever.hybrid_with_query_emb("red apple", query_emb, top_k=1)
    assert len(candidates) == 1
    assert "red apple" in candidates[0]["chunk_text"]