"""Dataset Preparation: Text normalization, chunking, and indexing."""

import json
import pickle
import re
from pathlib import Path
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi
from rich.console import Console
from rich.progress import track
from config import cfg

console = Console()

# Stopwords for BM25 
STOPWORDS = set(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "has",
        "he",
        "in",
        "it",
        "its",
        "of",
        "on",
        "that",
        "the",
        "to",
        "was",
        "will",
    ]
)


class DatasetPreparer:
    """Prepares dataset for RAG retrieval: chunking, tokenization, BM25 indexing."""

    def __init__(self):
        self.chunk_size = cfg.get("chunk_size", 500)
        self.chunk_overlap = cfg.get("chunk_overlap", 50)
        console.print("[green]✓[/green] DatasetPreparer initialized")

    @staticmethod
    def bm25_preprocess(text: str) -> List[str]:
        text = text.lower()
        text = re.sub(r"[^\w\s]", " ", text)
        tokens = [t for t in text.split() if t and t not in STOPWORDS and len(t) >= 2]
        return tokens

    # Backward compatibility alias
    tokenize = bm25_preprocess

    def chunk_text(self, text: str, chunk_id_prefix: str) -> List[Dict[str, Any]]:
        if len(text) <= self.chunk_size:
            doc_id = f"{chunk_id_prefix}_{0:02d}"
            return [
                {
                    "doc_id": doc_id,
                    "chunk_id": doc_id,
                    "text": text,
                    "chunk_index": 0,
                }
            ]

        chunks = []
        start = 0
        chunk_idx = 0

        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end]

            doc_id = f"{chunk_id_prefix}_{chunk_idx:02d}"
            chunks.append(
                {
                    "doc_id": doc_id,
                    "chunk_id": doc_id,
                    "text": chunk_text,
                    "chunk_index": chunk_idx,
                }
            )

            chunk_idx += 1
            start = end - self.chunk_overlap

        return chunks

    def prepare_documents(
        self, entities: List[Dict[str, Any]], entity_type: str
    ) -> List[Dict[str, Any]]:
        documents = []

        for entity in track(entities, description=f"Preparing {entity_type}s"):
            if entity_type == "character":
                text = entity.get("appearance", "")
                entity_id = entity.get("character_id")
            elif entity_type == "location":
                text = entity.get("description", "")
                entity_id = entity.get("location_id")
            elif entity_type == "relationship":
                text = f"{entity.get('relationship_type', '')} between {entity.get('source_entity', '')} and {entity.get('target_entity', '')}"
                entity_id = entity.get("relationship_id")
            else:
                continue

            if not text:
                continue

            chunks = self.chunk_text(text, entity_id)

            for chunk in chunks:
                tokens = self.bm25_preprocess(chunk["text"])
                doc = {
                    "doc_id": chunk["doc_id"],
                    "chunk_id": chunk["chunk_id"],
                    "entity_id": entity_id,
                    "entity_type": entity_type,
                    "text": chunk["text"],
                    "bm25_tokens": tokens,
                    "tokens": tokens,
                    "chunk_index": chunk["chunk_index"],
                    "metadata": entity.get("metadata", {}),
                    "tags": entity.get("tags", []),
                }
                documents.append(doc)

        console.print(
            f"[green]✓[/green] Prepared {len(documents)} document chunks from {len(entities)} {entity_type}s"
        )
        return documents

    def build_bm25_index(self, documents: List[Dict[str, Any]]) -> BM25Okapi:
        """Build BM25 index from documents."""
        tokenized_docs = [
            doc.get("bm25_tokens") or self.bm25_preprocess(doc["text"])
            for doc in documents
        ]
        bm25 = BM25Okapi(tokenized_docs)
        try:
            setattr(bm25, "corpus", tokenized_docs)
        except Exception:
            pass

        console.print(
            f"[green]✓[/green] BM25 index built with {len(documents)} documents"
        )
        return bm25

    def save_bm25_index(self, bm25: BM25Okapi, documents: List[Dict[str, Any]]):
        """Save BM25 index and documents to disk."""
        index_path = Path(cfg.get("bm25_index_path", "data/bm25_index.pkl"))
        index_path.parent.mkdir(parents=True, exist_ok=True)

        with open(index_path, "wb") as f:
            pickle.dump({"bm25": bm25, "documents": documents}, f)

        console.print(f"[green]✓[/green] BM25 index saved to {index_path}")

    def load_bm25_index(self) -> tuple:
        """Load BM25 index and documents from disk."""
        index_path = Path(cfg.get("bm25_index_path", "data/bm25_index.pkl"))

        if not index_path.exists():
            console.print(f"[yellow]⚠[/yellow] BM25 index not found at {index_path}")
            return None, None

        with open(index_path, "rb") as f:
            data = pickle.load(f)

        console.print(f"[green]✓[/green] BM25 index loaded from {index_path}")
        return data["bm25"], data["documents"]

    def load_entities(self, dir_path: str, entity_type: str) -> List[Dict[str, Any]]:
        base = Path(dir_path)
        entities: List[Dict[str, Any]] = []
        for p in sorted(base.glob("*.json")):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    entities.extend(data)
                else:
                    entities.append(data)
            except Exception as e:
                console.print(f"[red]✗[/red] Failed to load {p}: {e}")
                raise
        console.print(
            f"[green]✓[/green] Loaded {len(entities)} {entity_type}(s) from {dir_path}"
        )
        return entities

    def chunk_entity(
        self, entity: Dict[str, Any], entity_type: str
    ) -> List[Dict[str, Any]]:
        if entity_type == "character":
            text = entity.get("appearance", "")
            eid = entity.get("character_id")
        elif entity_type == "location":
            text = entity.get("description", "")
            eid = entity.get("location_id")
        elif entity_type == "relationship":
            text = f"{entity.get('relationship_type', '')} between {entity.get('source_entity', '')} and {entity.get('target_entity', '')}"
            eid = entity.get("relationship_id")
        else:
            text, eid = "", None
        return self.chunk_text(text, eid) if (text and eid) else []

    def prepare(
        self,
        characters_dir: str = "data/characters",
        locations_dir: str = "data/locations",
        relationships_dir: str = None,
    ) -> Dict[str, Any]:
        characters = (
            self.load_entities(characters_dir, "character") if characters_dir else []
        )
        locations = (
            self.load_entities(locations_dir, "location") if locations_dir else []
        )
        relationships = (
            self.load_entities(relationships_dir, "relationship")
            if relationships_dir
            else []
        )

        char_docs = (
            self.prepare_documents(characters, "character") if characters else []
        )
        loc_docs = self.prepare_documents(locations, "location") if locations else []
        rel_docs = (
            self.prepare_documents(relationships, "relationship")
            if relationships
            else []
        )

        all_docs = char_docs + loc_docs + rel_docs
        bm25 = self.build_bm25_index(all_docs) if all_docs else None
        if bm25 is not None:
            self.save_bm25_index(bm25, all_docs)

        return {
            "characters": char_docs,
            "locations": loc_docs,
            "bm25": bm25,
        }
