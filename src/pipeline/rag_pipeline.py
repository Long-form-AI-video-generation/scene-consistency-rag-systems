"""RAG Pipeline: Orchestrates retrieval-augmented generation workflow."""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from config import cfg
from dataset import DatasetPreparer
from embedder import ClipEmbedder
from retriever.chroma_client import chroma_client
from retriever import HybridRetriever
from reranker import Reranker

console = Console()


class RAGPipeline:
    """Main RAG pipeline orchestrator."""

    def __init__(self):
        self.dataset_prep = DatasetPreparer()
        self.embedder = ClipEmbedder()
        self._retriever: Optional[HybridRetriever] = None
        self._reranker: Optional[Reranker] = None
        console.print(Panel("[green]RAG Pipeline Initialized[/green]", expand=False))

    def _get_retriever(self) -> HybridRetriever:
        if self._retriever is None:
            self._retriever = HybridRetriever()
        return self._retriever

    def _get_reranker(self) -> Reranker:
        if self._reranker is None:
            self._reranker = Reranker()
        return self._reranker

    def load_json_data(self, file_path: str) -> List[Dict[str, Any]]:
        """Load entity data from JSON file."""
        path = Path(file_path)
        if not path.exists():
            console.print(f"[red]✗[/red] File not found: {file_path}")
            return []

        with open(path, "r") as f:
            data = json.load(f)

        # Handle both single objects and arrays
        if isinstance(data, dict):
            data = [data]

        console.print(f"[green]✓[/green] Loaded {len(data)} entities from {file_path}")
        return data

    @staticmethod
    def _entity_lookup(
        entities: List[Dict[str, Any]], id_key: str
    ) -> Dict[str, Dict[str, Any]]:
        return {entity[id_key]: entity for entity in entities if entity.get(id_key)}

    def _index_documents(
        self,
        collection_name: str,
        docs: List[Dict[str, Any]],
        entity_lookup: Dict[str, Dict[str, Any]],
        entity_type: str,
        include_image: bool = True,
    ) -> int:
        if not docs:
            console.print(f"[yellow]⚠[/yellow] No documents to index for {collection_name}")
            return 0

        indexed_count = 0
        failed_count = 0

        try:
            texts = [doc.get("text", "") for doc in docs]
            if not all(texts):
                console.print(f"[yellow]⚠[/yellow] Some documents missing 'text' field in {collection_name}")
            text_embs = self.embedder.embed_text_batch(texts)
        except Exception as e:
            console.print(f"[red]✗[/red] Text embedding failed for {collection_name}: {e}")
            return 0

        image_cache: Dict[str, Optional[Any]] = {}
        ids, embeddings, documents, metadatas = [], [], [], []

        for idx, doc in enumerate(docs):
            try:
                if not doc.get("chunk_id"):
                    console.print(f"[yellow]⚠[/yellow] Document {idx} missing 'chunk_id', skipping")
                    failed_count += 1
                    continue

                entity_id = doc.get("entity_id")
                if not entity_id:
                    console.print(f"[yellow]⚠[/yellow] Document {idx} missing 'entity_id', skipping")
                    failed_count += 1
                    continue

                entity = entity_lookup.get(entity_id, {})

                image_emb = None
                if include_image and entity:
                    if entity_id not in image_cache:
                        path = entity.get("canonical_image_path")
                        if path:
                            try:
                                image_cache[entity_id] = self.embedder.embed_image(path)
                            except Exception as e:
                                console.print(
                                    f"[yellow]⚠[/yellow] Image embedding failed for {entity_id} "
                                    f"(path: {path}): {e}"
                                )
                                image_cache[entity_id] = None
                        else:
                            image_cache[entity_id] = None
                    image_emb = image_cache[entity_id]

                fused = self.embedder.fuse(text_embs[idx], image_emb)

                tags_value = doc.get("tags", [])
                if isinstance(tags_value, list):
                    tags_value = ",".join(str(t) for t in tags_value)
                else:
                    tags_value = str(tags_value or "")

                ids.append(doc["chunk_id"])
                embeddings.append(fused.astype(float).tolist())
                documents.append(doc.get("text", ""))
                metadatas.append({
                    "entity_id": entity_id,
                    "entity_type": entity_type,
                    "chunk_index": doc.get("chunk_index", 0),
                    "tags": tags_value,
                })
                indexed_count += 1

            except Exception as e:
                console.print(
                    f"[yellow]⚠[/yellow] Failed to process document {idx} "
                    f"(chunk_id: {doc.get('chunk_id', 'unknown')}): {e}"
                )
                failed_count += 1
                continue

        if ids:
            try:
                chroma_client.add_documents(
                    collection_name, ids, embeddings, documents, metadatas
                )
                console.print(
                    f"[green]✓[/green] Indexed {indexed_count}/{len(docs)} documents to {collection_name}"
                )
            except Exception as e:
                console.print(
                    f"[red]✗[/red] ChromaDB insertion failed for {collection_name}: {e}"
                )
                return 0
        else:
            console.print(f"[yellow]⚠[/yellow] No valid documents to index for {collection_name}")

        if failed_count > 0:
            console.print(
                f"[yellow]⚠[/yellow] {failed_count} document(s) failed to process in {collection_name}"
            )

        return indexed_count

    def build_indices(
        self,
        characters: List[Dict] = None,
        locations: List[Dict] = None,
        relationships: List[Dict] = None,
        rebuild: bool = False,
    ):
        """
        Build BM25 and ChromaDB indices from entity data.

        Args:
            characters: List of character dictionaries
            locations: List of location dictionaries
            relationships: List of relationship dictionaries
            rebuild: If True, rebuild indices from scratch
        """
        console.print("\n[bold cyan]═══ Building Indices ═══[/bold cyan]\n")

        if rebuild:
            console.print("[yellow]⟳[/yellow] Rebuilding indices...")
            chroma_client.reset_collection("characters")
            chroma_client.reset_collection("locations")
            chroma_client.reset_collection("relationships")

        all_documents = []

        characters = characters or []
        locations = locations or []
        relationships = relationships or []

        char_docs = (
            self.dataset_prep.prepare_documents(characters, "character")
            if characters
            else []
        )
        loc_docs = (
            self.dataset_prep.prepare_documents(locations, "location")
            if locations
            else []
        )
        rel_docs = (
            self.dataset_prep.prepare_documents(relationships, "relationship")
            if relationships
            else []
        )

        all_documents.extend(char_docs)
        all_documents.extend(loc_docs)
        all_documents.extend(rel_docs)

        char_lookup = (
            self._entity_lookup(characters, "character_id") if characters else {}
        )
        loc_lookup = self._entity_lookup(locations, "location_id") if locations else {}
        rel_lookup = (
            self._entity_lookup(relationships, "relationship_id")
            if relationships
            else {}
        )

        self._index_documents(
            "characters", char_docs, char_lookup, "character", include_image=True
        )
        self._index_documents(
            "locations", loc_docs, loc_lookup, "location", include_image=True
        )
        self._index_documents(
            "relationships", rel_docs, rel_lookup, "relationship", include_image=False
        )

        bm25 = self.dataset_prep.build_bm25_index(all_documents)
        self.dataset_prep.save_bm25_index(bm25, all_documents)

        table = Table(title="Index Summary")
        table.add_column("Collection", style="cyan")
        table.add_column("Count", style="green")

        table.add_row(
            "Characters", str(chroma_client.get_collection_count("characters"))
        )
        table.add_row("Locations", str(chroma_client.get_collection_count("locations")))
        table.add_row(
            "Relationships", str(chroma_client.get_collection_count("relationships"))
        )
        table.add_row("BM25 Documents", str(len(all_documents)))

        console.print(table)
        console.print("\n[green]✓ Indices built successfully[/green]\n")

    def query(
        self,
        query_text: str,
        collection: str = "characters",
        top_k_retrieval: int = None,
        top_k_rerank: int = None,
        where: Optional[Dict] = None,
    ) -> List[Dict[str, Any]]:
        """
        Query the RAG system.

        Args:
            query_text: Query string
            collection: Collection to search ("characters", "locations", "relationships")
            top_k_retrieval: Number of results to retrieve (before reranking)
            top_k_rerank: Number of final results (after reranking)
            where: Metadata filter

        Returns:
            List of ranked results with scores
        """
        if not query_text or not query_text.strip():
            console.print(
                "[yellow]⚠[/yellow] Empty query provided, returning no results"
            )
            return []

        top_k_retrieval = top_k_retrieval or cfg.get("top_k_retrieval", 20)
        top_k_rerank = top_k_rerank or cfg.get("top_k_rerank", 5)

        console.print(f"\n[bold cyan]═══ Query: '{query_text}' ═══[/bold cyan]\n")

        retriever = self._get_retriever()
        reranker = self._get_reranker()

        try:
            results = retriever.hybrid_search(
                query=query_text,
                collection_name=collection,
                top_k=top_k_retrieval,
                where=where,
            )
        except Exception as e:
            console.print(f"[yellow]⚠[/yellow] Query processing failed: {e}")
            return []

        if not results:
            console.print("[yellow]⚠[/yellow] No results found")
            return []

        reranked_results = reranker.rerank(query_text, results, top_k=top_k_rerank)
        self._display_results(reranked_results)

        return reranked_results

    def _display_results(self, results: List[Dict[str, Any]]):
        if not results:
            return

        table = Table(title="Top Results")
        table.add_column("#", style="dim", width=3)
        table.add_column("Entity ID", style="cyan")
        table.add_column("Text Preview", style="white", no_wrap=False)
        table.add_column("BM25", style="magenta", justify="right")
        table.add_column("Dense", style="blue", justify="right")
        table.add_column("Hybrid", style="yellow", justify="right")
        table.add_column("RerankNorm", style="green", justify="right")

        for i, result in enumerate(results[:10], 1):
            text = result.get("text") or result.get("document", "")
            text_preview = (text[:80] + "...") if len(text) > 80 else text

            bm25_raw = result.get("bm25_score")
            dense_raw = result.get("dense_score")
            hybrid_raw = result.get("hybrid_score")
            rerank_norm_raw = result.get("rerank_score_norm")

            bm25_score = f"{bm25_raw:.3f}" if isinstance(bm25_raw, (int, float)) else "-"
            dense_score = f"{dense_raw:.3f}" if isinstance(dense_raw, (int, float)) else "-"
            hybrid_score = f"{hybrid_raw:.3f}" if isinstance(hybrid_raw, (int, float)) else "-"
            rerank_score = (
                f"{rerank_norm_raw:.3f}"
                if isinstance(rerank_norm_raw, (int, float))
                else "-"
            )

            table.add_row(
                str(i),
                result.get("entity_id", ""),
                text_preview,
                bm25_score,
                dense_score,
                hybrid_score,
                rerank_score,
            )

        console.print(table)
