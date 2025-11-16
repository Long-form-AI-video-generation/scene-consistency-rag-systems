
from src.rag.reranker import Reranker

def test_reranker():
    reranker = Reranker()
    query = "red apple"
    candidates = [
        {"chunk_text": "red apple on tree"},
        {"chunk_text": "green banana"},
    ]
    ranked = reranker.rerank(query, candidates, top_k=1)
    assert ranked[0]["chunk_text"] == "red apple on tree"