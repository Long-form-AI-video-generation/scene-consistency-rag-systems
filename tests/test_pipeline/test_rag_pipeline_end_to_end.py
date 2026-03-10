#!/usr/bin/env python3
"""RAGPipeline smoke test.

Validates that RAGPipeline can:
- Load a small subset of entities
- Build indices (BM25 + Chroma) via build_indices()
- Execute a query() end-to-end using HybridRetriever
- Integrate a (dummy) reranker without errors

NOTE: Some tests in this file use deprecated PromptAssembler and need refactoring.
"""

import sys
from typing import List, Dict
from pathlib import Path

import pytest

sys.path.insert(0, "src")  # ensure src is on sys.path

from pipeline import RAGPipeline  # noqa: E402
from retriever.chroma_client import chroma_client  # noqa: E402

try:  # Support running file directly with `python tests/...`
    from conftest import register_checklist_item
except ModuleNotFoundError:  # pragma: no cover - fallback for direct execution
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))
    from tests.conftest import register_checklist_item


@pytest.fixture(autouse=True)
def mock_image_embedding(monkeypatch):
    """Mock image embedding to avoid CLIP model and image file dependencies in CI/CD."""
    import numpy as np
    from embedder import ClipEmbedder
    
    def dummy_embed_image(self, path: str) -> np.ndarray:
        # Return a dummy embedding vector (512-dim for ViT-B/32)
        return np.random.rand(512).astype(np.float32)
    
    monkeypatch.setattr(ClipEmbedder, "embed_image", dummy_embed_image)


@pytest.fixture(scope="module")
def rag_pipeline() -> RAGPipeline:
    return RAGPipeline()


def _create_test_entities() -> Dict[str, List[Dict]]:
    """Create minimal test entities without image file dependencies.
    
    This allows tests to run in CI/CD environments without requiring
    actual image files to be present in the repository.
    """
    chars = [{
        "character_id": "char_test_001",
        "name": "TestCharacter",
        "appearance": "A test character wearing casual clothes in an office setting with a desk and computer",
        "tags": ["test", "office"]
    }]
    locs = [{
        "location_id": "loc_test_001",
        "name": "TestOffice",
        "description": "A modern office with desks, computers, and large windows overlooking the city",
        "tags": ["test", "office"]
    }]
    return {"characters": chars, "locations": locs}


class _DummyReranker:
    """Lightweight reranker for tests.

    Avoids downloading the real CrossEncoder model while still exercising
    the RAGPipeline.rerank integration path.
    """

    def rerank(self, query: str, results: List[Dict], top_k: int = 5) -> List[Dict]:
        if not results:
            return []
        # Use hybrid_score as a stand-in for rerank_score
        for r in results:
            r["rerank_score"] = float(r.get("hybrid_score", 0.0))
        return sorted(results, key=lambda x: x["rerank_score"], reverse=True)[:top_k]


def test_rag_pipeline_build_and_query(monkeypatch, rag_pipeline: RAGPipeline):
    data = _create_test_entities()
    chars = data["characters"]
    locs = data["locations"]

    assert chars, "No character data loaded for RAGPipeline test"
    assert locs, "No location data loaded for RAGPipeline test"

    rag_pipeline.build_indices(characters=chars, locations=locs, rebuild=True)

    # Collections should have some documents
    assert chroma_client.get_collection_count("characters") > 0
    assert chroma_client.get_collection_count("locations") > 0

    # Monkeypatch the reranker to avoid loading a heavy CrossEncoder model
    dummy = _DummyReranker()

    def _get_dummy_reranker():
        return dummy

    monkeypatch.setattr(rag_pipeline, "_get_reranker", _get_dummy_reranker)

    # Run a simple query end-to-end
    results = rag_pipeline.query(
        query_text="office",
        collection="characters",
        top_k_retrieval=5,
        top_k_rerank=3,
    )

    assert isinstance(results, list)
    assert len(results) <= 3
    # If there are results, they should contain basic fields
    for r in results:
        assert "entity_id" in r
        assert "hybrid_score" in r
        assert "rerank_score" in r


# ============================================================================
# Additional RAGPipeline Tests (from test_stage7_readiness.py)
# ============================================================================

from rich.console import Console

console = Console()


def test_retrieve_query(rag_pipeline: RAGPipeline):
    """Test: retrieve(query) works"""
    data = _create_test_entities()
    chars = data["characters"]
    locs = data["locations"]
    rag_pipeline.build_indices(characters=chars, locations=locs, rebuild=True)

    # Test retrieval
    retriever = rag_pipeline._get_retriever()
    results = retriever.hybrid_search("office", collection_name="characters", top_k=5)

    assert results, "retrieve(query) returned no results"
    assert len(results) > 0
    for r in results:
        assert "entity_id" in r or "id" in r

    register_checklist_item("stage3_rag", "retrieve", True)


def test_retrieve_and_rerank(rag_pipeline: RAGPipeline, monkeypatch):
    """Test: retrieve_and_rerank(query) works"""
    data = _create_test_entities()
    chars = data["characters"]
    locs = data["locations"]
    rag_pipeline.build_indices(characters=chars, locations=locs, rebuild=True)

    # Monkeypatch reranker
    dummy = _DummyReranker()
    monkeypatch.setattr(rag_pipeline, "_get_reranker", lambda: dummy)

    # Test query (which includes reranking)
    results = rag_pipeline.query(
        query_text="office", collection="characters", top_k_retrieval=5, top_k_rerank=3
    )

    assert results, "retrieve_and_rerank(query) returned no results"
    for r in results:
        assert "rerank_score" in r

    register_checklist_item("stage3_rag", "retrieve_rerank", True)


def test_malformed_query_handling(rag_pipeline: RAGPipeline, monkeypatch):
    """Test: RAGPipeline handles malformed queries gracefully"""
    data = _create_test_entities()
    chars = data["characters"]
    locs = data["locations"]
    rag_pipeline.build_indices(characters=chars, locations=locs, rebuild=True)

    # Monkeypatch reranker
    dummy = _DummyReranker()
    monkeypatch.setattr(rag_pipeline, "_get_reranker", lambda: dummy)

    test_cases = [
        ("", "empty query"),
        ("   ", "whitespace-only query"),
        ("\x00\x01\x02", "control characters"),
        ("x" * 5000, "extremely long query"),
    ]

    all_passed = True
    for query, description in test_cases:
        try:
            results = rag_pipeline.query(
                query_text=query,
                collection="characters",
                top_k_retrieval=5,
                top_k_rerank=3,
            )
            # Empty query might return empty results, which is acceptable
            assert isinstance(results, list), f"{description} should return a list"
        except Exception as e:
            # Some queries may raise exceptions, which is acceptable for malformed input
            # But we want to ensure it doesn't crash the system
            if description == "empty query":
                # Empty query might raise ValueError, which is acceptable
                pass
            else:
                all_passed = False

    assert all_passed, "RAGPipeline failed to handle some malformed queries gracefully"

    register_checklist_item("stage3_rag", "malformed", all_passed)
