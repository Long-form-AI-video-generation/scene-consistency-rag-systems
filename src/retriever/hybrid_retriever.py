"""Hybrid Retrieval System: Combines BM25 (sparse) and CLIP (dense) retrieval."""

import numpy as np
import logging
from typing import List, Dict, Any, Optional
from rich.console import Console
from config import cfg
from dataset import DatasetPreparer
from retriever.chroma_client import chroma_client
from embedder import ClipEmbedder

console = Console()
logger = logging.getLogger(__name__)


class HybridRetriever:
    """Hybrid retrieval combining BM25 and dense vector search."""

    def __init__(self):
        self.dataset_prep = DatasetPreparer()
        self.text_embedder = ClipEmbedder()
        self.chroma = chroma_client
        self.bm25_weight = cfg.get("bm25_weight", 0.3)
        self.dense_weight = 1.0 - self.bm25_weight
        console.print(
            f"[cyan]Hybrid Weights:[/cyan] bm25_w={self.bm25_weight:.2f}, dense_w={self.dense_weight:.2f}"
        )
        self.default_top_k = cfg.get("top_k_retrieval", 20)
        self.bm25, self.bm25_documents = self.dataset_prep.load_bm25_index()
        if self.bm25 is None:
            console.print(
                "[yellow]⚠[/yellow] BM25 index not loaded - run build_indices first"
            )

    def bm25_search(
        self, query: str, top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Perform BM25 sparse retrieval."""
        if self.bm25 is None:
            logger.warning(
                "BM25 search requested but index not loaded. "
                "Run build_indices first. Returning empty results."
            )
            console.print("[yellow]⚠[/yellow] BM25 index not loaded")
            return []

        top_k = top_k or self.default_top_k

        try:
            tokenized_query = self.dataset_prep.tokenize(query)
            scores = self.bm25.get_scores(tokenized_query)
        except Exception as e:
            logger.error(f"BM25 scoring failed for query '{query}': {e}", exc_info=True)
            console.print(f"[red]✗[/red] BM25 scoring failed: {e}")
            return []

        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append(
                    {**self.bm25_documents[idx], "bm25_score": float(scores[idx])}
                )

        if not results:
            logger.info(f"BM25 search found no results for query: '{query}'")

        return results

    def dense_search(
        self,
        query: str,
        collection_name: str,
        top_k: Optional[int] = None,
        where: Optional[Dict] = None,
    ) -> List[Dict[str, Any]]:
        """Perform dense vector retrieval using CLIP."""
        top_k = top_k or self.default_top_k

        try:
            query_embedding = self.text_embedder.embed_text(query)
        except Exception as exc:
            logger.error(
                f"Dense embedding failed for query '{query}': {exc}",
                exc_info=True
            )
            console.print(f"[red]✗[/red] Dense embedding failed: {exc}")
            return []

        try:
            results = self.chroma.query(
                collection_name=collection_name,
                query_embedding=query_embedding.tolist(),
                n_results=top_k,
                where=where,
            )
        except Exception as exc:
            logger.error(
                f"ChromaDB query failed for collection '{collection_name}' "
                f"with filter {where}: {exc}",
                exc_info=True
            )
            console.print(f"[red]✗[/red] ChromaDB query failed: {exc}")
            return []

        # Format results
        formatted_results = []
        try:
            for i in range(len(results["ids"][0])):
                md = results["metadatas"][0][i] if results.get("metadatas") else {}
                entity_id = (
                    md.get("entity_id") if isinstance(md, dict) else None
                ) or results["ids"][0][i]
                text = results["documents"][0][i]
                dist_list = results.get("distances")
                dense_score = (1.0 - dist_list[0][i]) if dist_list else 0.0
                formatted_results.append(
                    {
                        "entity_id": entity_id,
                        "text": text,
                        "metadata": md,
                        "dense_score": dense_score,  # distance -> similarity
                    }
                )
        except (KeyError, IndexError, TypeError) as e:
            logger.error(
                f"Failed to format ChromaDB results - malformed response structure: {e}",
                exc_info=True
            )
            console.print(f"[red]✗[/red] Failed to format search results: {e}")
            return []

        if not formatted_results:
            logger.info(
                f"Dense search found no results in '{collection_name}' "
                f"for query: '{query}' with filter {where}"
            )

        return formatted_results

    @staticmethod
    def _safe_normalize(values: List[float]) -> List[float]:
        if not values:
            return []
        vmin = min(values)
        vmax = max(values)
        denom = vmax - vmin
        if denom <= 1e-12:
            return [1.0 for _ in values]
        return [(v - vmin) / denom for v in values]

    def hybrid_search(
        self,
        query: str,
        collection_name: str = "characters",
        top_k: Optional[int] = None,
        where: Optional[Dict] = None,
        bm25_k: Optional[int] = None,
        dense_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid retrieval combining BM25 and dense search.

        Fusion formula: final_score = (1-bm25_weight)*dense_score + bm25_weight*bm25_score
        """
        # Get BM25 results
        bm25_results = self.bm25_search(query, top_k=bm25_k or top_k)
        
        # Apply entity_id filter to BM25 results if where clause is provided
        if where and "entity_id" in where:
            target_entity_id = where["entity_id"]
            bm25_results = [
                r for r in bm25_results 
                if r.get("entity_id") == target_entity_id
            ]
        dense_results = self.dense_search(
            query,
            collection_name,
            top_k=dense_k or top_k,
            where=where,
        )
        return self._hybrid_fuse(
            bm25_results,
            dense_results,
            top_k,
        )

    def _hybrid_fuse(
        self,
        bm25_results: List[Dict[str, Any]],
        dense_results: List[Dict[str, Any]],
        top_k: Optional[int],
    ) -> List[Dict[str, Any]]:
        top_k = top_k or self.default_top_k

        merged = {}

        for result in bm25_results:
            entity_id = result["entity_id"]
            merged[entity_id] = {
                **result,
                "bm25_score": result.get("bm25_score", 0.0),
                "dense_score": 0.0,
            }

        for result in dense_results:
            entity_id = result["entity_id"]
            if entity_id in merged:
                merged[entity_id]["dense_score"] = result.get("dense_score", 0.0)
            else:
                merged[entity_id] = {
                    **result,
                    "bm25_score": 0.0,
                    "dense_score": result.get("dense_score", 0.0),
                }

        items = list(merged.values())
        bm25_scores = [item["bm25_score"] for item in items]
        dense_scores = [item["dense_score"] for item in items]

        norm_bm25 = self._safe_normalize(bm25_scores)
        norm_dense = self._safe_normalize(dense_scores)

        for item, nb, nd in zip(items, norm_bm25, norm_dense):
            item["hybrid_score"] = self.dense_weight * nd + self.bm25_weight * nb

        results = sorted(items, key=lambda x: x["hybrid_score"], reverse=True)
        return results[:top_k]
