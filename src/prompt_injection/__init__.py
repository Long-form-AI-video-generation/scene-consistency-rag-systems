"""
Scene Consistency Engine
-------------------------
Orchestrates entity extraction, context retrieval, and shot enrichment.
"""

from typing import Dict, Any, Optional
from .entity_extractor import EntityExtractor, EntityExtractionResult
from .context_retriever import ContextRetriever, RetrievedContext
from .shot_enricher import ShotEnricher, EnrichedShot

# Import the RAGPipeline type for type hints
from pipeline import RAGPipeline


class SceneConsistencyEngine:
    """
    The main orchestrator for the prompt injection pipeline.

    Usage:
        engine = SceneConsistencyEngine(rag_pipeline)
        enriched_shot = engine.process_shot(shot_dict)
    """

    def __init__(
        self,
        rag_pipeline: RAGPipeline,
        characters_dir: str = "data/characters",
        locations_dir: str = "data/locations"
    ):
        """
        Args:
            rag_pipeline: An initialized RAGPipeline instance (share models)
            characters_dir: Path to character JSON directory
            locations_dir: Path to location JSON directory
        """
        self.rag = rag_pipeline

        # Initialize the three components
        self.entity_extractor = EntityExtractor(
            characters_dir=characters_dir,
            locations_dir=locations_dir
        )
        self.context_retriever = ContextRetriever(rag_pipeline=rag_pipeline)
        self.shot_enricher = ShotEnricher(
            characters_dir=characters_dir,
            locations_dir=locations_dir
        )

    def process_shot(
        self,
        shot: Dict[str, Any],
        entities: Optional[EntityExtractionResult] = None,
        context: Optional[RetrievedContext] = None,
    ) -> EnrichedShot:
        """
        The main pipeline: Extract → Retrieve → Enrich

        Args:
            shot: A shot dictionary with at minimum:
                  {"description": "Isaac monitors display lines of code..."}

        Returns:
            EnrichedShot: Fully structured shot with RAG-enriched context
        """
        # Step 1: Extract entities from the shot description
        description = shot.get("description", "")
        if entities is None:
            entities = self.entity_extractor.extract(description)

        # Ensure the enriched shot carries the resolved entity IDs when
        # the original shot did not already specify them.
        shot_for_enrichment = dict(shot)
        if "characters" not in shot_for_enrichment:
            shot_for_enrichment["characters"] = list(entities.characters)
        if "locations" not in shot_for_enrichment:
            shot_for_enrichment["locations"] = list(entities.locations)

        # Step 2: Retrieve context using the FULL description
        if context is None:
            context = self.context_retriever.retrieve(
                prompt=description,
                entities=entities,
            )

        # Step 3: Enrich the shot with RAG context
        enriched = self.shot_enricher.enrich(shot_for_enrichment, context)

        return enriched

    def extract_entities(self, text: str) -> EntityExtractionResult:
        """
        Convenience method: Extract entities from any text.
        """
        return self.entity_extractor.extract(text)

    def retrieve_context(
        self,
        prompt: str,
        entities: EntityExtractionResult
    ) -> RetrievedContext:
        """
        Convenience method: Retrieve context for extracted entities.
        """
        return self.context_retriever.retrieve(prompt, entities)


# Public API
__all__ = [
    "SceneConsistencyEngine",
    "EntityExtractor",
    "ContextRetriever",
    "ShotEnricher",
    "EntityExtractionResult",
    "RetrievedContext",
    "EnrichedShot",
]
