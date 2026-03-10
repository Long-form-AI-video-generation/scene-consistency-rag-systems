"""
ContextRetriever: Queries RAG pipeline with entity IDs to retrieve relevant context chunks.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pipeline import RAGPipeline
from .entity_extractor import EntityExtractionResult

@dataclass
class RetrievedContext:
    character_context: Dict[str, List[str]]
    location_context: Dict[str, List[str]]
    character_scores: Optional[Dict[str, Dict[str, Any]]] = None
    location_scores: Optional[Dict[str, Dict[str, Any]]] = None

class ContextRetriever:
    def __init__(self, rag_pipeline: RAGPipeline):
        self.rag = rag_pipeline

    def retrieve(self, prompt: str, entities: EntityExtractionResult) -> RetrievedContext:
        """Query RAG system for each entity ID using the full prompt as context."""
        char_data: Dict[str, List[str]] = {}
        loc_data: Dict[str, List[str]] = {}
        char_scores: Dict[str, Dict[str, Any]] = {}
        loc_scores: Dict[str, Dict[str, Any]] = {}

        for char_id in entities.characters:
            results = self.rag.query(
                query_text=prompt,
                collection="characters",
                top_k_retrieval=3,
                top_k_rerank=1,
                where={"entity_id": char_id}
            )

            if not results:
                continue

            chunks: List[str] = []
            for r in results:
                text_value = r.get("text") or r.get("document")
                if not text_value:
                    continue
                chunks.append(text_value)

            if not chunks:
                continue

            char_data[char_id] = chunks

            best = results[0]
            metadata = best.get("metadata") or {}
            chunk_index = best.get("chunk_index")
            if chunk_index is None and isinstance(metadata, dict):
                chunk_index = metadata.get("chunk_index")

            char_scores[char_id] = {
                "bm25_score": best.get("bm25_score"),
                "dense_score": best.get("dense_score"),
                "hybrid_score": best.get("hybrid_score"),
                "rerank_score": best.get("rerank_score"),
                "rerank_score_norm": best.get("rerank_score_norm"),
                "collection": "characters",
                "chunk_index": chunk_index,
                "chunk_id": best.get("chunk_id"),
                "doc_id": best.get("doc_id"),
            }

        for loc_id in entities.locations:
            results = self.rag.query(
                query_text=prompt,
                collection="locations",
                top_k_retrieval=3,
                top_k_rerank=1,
                where={"entity_id": loc_id}
            )

            if not results:
                continue

            chunks: List[str] = []
            for r in results:
                text_value = r.get("text") or r.get("document")
                if not text_value:
                    continue
                chunks.append(text_value)

            if not chunks:
                continue

            loc_data[loc_id] = chunks

            best = results[0]
            metadata = best.get("metadata") or {}
            chunk_index = best.get("chunk_index")
            if chunk_index is None and isinstance(metadata, dict):
                chunk_index = metadata.get("chunk_index")

            loc_scores[loc_id] = {
                "bm25_score": best.get("bm25_score"),
                "dense_score": best.get("dense_score"),
                "hybrid_score": best.get("hybrid_score"),
                "rerank_score": best.get("rerank_score"),
                "rerank_score_norm": best.get("rerank_score_norm"),
                "collection": "locations",
                "chunk_index": chunk_index,
                "chunk_id": best.get("chunk_id"),
                "doc_id": best.get("doc_id"),
            }

        return RetrievedContext(
            character_context=char_data,
            location_context=loc_data,
            character_scores=char_scores or None,
            location_scores=loc_scores or None,
        )


def retrieve_context(
    prompt: str,
    entities: EntityExtractionResult,
    pipeline_instance: RAGPipeline
) -> RetrievedContext:
    retriever = ContextRetriever(pipeline_instance)
    return retriever.retrieve(prompt, entities)