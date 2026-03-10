"""ShotEnricher: Merges RAG-retrieved descriptions into shot objects for prompt generation."""

from dataclasses import dataclass
from typing import Dict, Any
from pathlib import Path
import json

from .context_retriever import RetrievedContext

@dataclass
class EnrichedShot:
    shot_id: str
    scene_id: str
    raw_description: str
    characters: list
    locations: list
    character_names: Dict[str, str] 
    location_names: Dict[str, str] 
    rag_characters: Dict[str, str]
    rag_locations: Dict[str, str]
    rag_scores: Dict[str, Dict[str, Any]]
    actions: Any
    camera: Any
    metadata: Dict[str, Any]


class ShotEnricher:

    def __init__(self, characters_dir: str = "data/characters", locations_dir: str = "data/locations"):
        self.characters_dir = Path(characters_dir)
        self.locations_dir = Path(locations_dir)

        self.char_names = self._build_name_map(self.characters_dir, "character_id")
        self.loc_names = self._build_name_map(self.locations_dir, "location_id")

    def _build_name_map(self, directory: Path, id_field: str) -> Dict[str, str]:
        name_map = {}
        if not directory.exists():
            return name_map

        for file in directory.glob("*.json"):
            try:
                with open(file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                entity_id = data.get(id_field) or data.get("entity_id")
                name = data.get("name")

                if entity_id and name:
                    name_map[entity_id] = name
            except Exception:
                pass

        return name_map

    def enrich(self, shot: Dict[str, Any], context: RetrievedContext) -> EnrichedShot:
        """
        Combines the original shot metadata with RAG descriptions.

        Args:
            shot: Original shot dict ("id", "scene", "description", "characters", ...)
            context: RetrievedContext from the RAG pipeline

        Returns:
            EnrichedShot: fully structured shot ready for prompt generation
        """

        # Flatten the text chunks into single strings per entity
        rag_chars = {
            k: " ".join(v).strip()
            for k, v in context.character_context.items()
        }

        rag_locs = {
            k: " ".join(v).strip()
            for k, v in context.location_context.items()
        }

        rag_scores: Dict[str, Dict[str, Any]] = {}
        if getattr(context, "character_scores", None):
            for entity_id, scores in context.character_scores.items():
                rag_scores[entity_id] = dict(scores)
        if getattr(context, "location_scores", None):
            for entity_id, scores in context.location_scores.items():
                rag_scores[entity_id] = dict(scores)

        char_names = {
            char_id: self.char_names.get(char_id, char_id)
            for char_id in rag_chars.keys()
        }

        loc_names = {
            loc_id: self.loc_names.get(loc_id, loc_id)
            for loc_id in rag_locs.keys()
        }

        return EnrichedShot(
            shot_id=shot.get("shot_id"),
            scene_id=shot.get("scene_id"),
            raw_description=shot.get("description", ""),
            characters=shot.get("characters", []),
            locations=shot.get("locations", []),
            character_names=char_names,
            location_names=loc_names,
            rag_characters=rag_chars,
            rag_locations=rag_locs,
            rag_scores=rag_scores,
            actions=shot.get("actions"),
            camera=shot.get("camera"),
            metadata=shot.get("metadata", {})
        )
