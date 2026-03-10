#!/usr/bin/env python3
"""
RAG Consistency Engine Demo
----------------------------
Demonstrates the prompt injection pipeline's core responsibility:
1. Entity Extraction
2. RAG Context Retrieval  
3. Shot Enrichment

OUTPUT: EnrichedShot JSON (ready for downstream prompt generator)
"""

import sys
import json
from dataclasses import asdict

sys.path.insert(0, "src")

from pipeline import RAGPipeline
from prompt_injection import SceneConsistencyEngine
from rich.console import Console
from rich.panel import Panel
from rich.json import JSON

console = Console()


def main():
    console.print("\n[bold cyan]═══ RAG Consistency Engine Demo ═══[/bold cyan]\n")

    # SETUP: Initialize RAG Pipeline (prerequisite)
    console.print("[bold yellow]Setup: Initializing RAG Pipeline[/bold yellow]")
    rag_pipeline = RAGPipeline()
    
    console.print("  Loading entity data from directories...")
    
    # Load ALL characters from directory
    from pathlib import Path
    characters = []
    characters_dir = Path("data/characters")
    for char_file in characters_dir.glob("*.json"):
        char_data = rag_pipeline.load_json_data(str(char_file))
        if char_data:
            characters.extend(char_data if isinstance(char_data, list) else [char_data])
    
    # Load ALL locations from directory
    locations = []
    locations_dir = Path("data/locations")
    for loc_file in locations_dir.glob("*.json"):
        loc_data = rag_pipeline.load_json_data(str(loc_file))
        if loc_data:
            locations.extend(loc_data if isinstance(loc_data, list) else [loc_data])
    
    console.print(f"  Loaded {len(characters)} characters and {len(locations)} locations")
    
    if characters and locations:
        console.print("  Building indices (this may take a moment)...")
        rag_pipeline.build_indices(
            characters=characters,
            locations=locations,
            rebuild=False  # Use existing indices if available
        )
    console.print("[green]✓[/green] RAG Pipeline ready\n")

    # Initialize the Scene Consistency Engine
    engine = SceneConsistencyEngine(
        rag_pipeline=rag_pipeline,
        characters_dir="data/characters",
        locations_dir="data/locations"
    )
    console.print("[green]✓[/green] Scene Consistency Engine ready\n")

    # Build simple lookup maps for pretty-printing surface forms
    char_name_map = {c.get("character_id"): c.get("name") for c in (characters or [])}
    loc_name_map = {l.get("location_id"): l.get("name") for l in (locations or [])}

    # STEP 1: Input Shot Description
    console.print("[bold] Step 1: Input Shot Description[/bold]\n")
    
    sample_shot = {
        "shot_id": "shot_001",
        "scene_id": "scene_001",
        "description": "PANEL 001\nSHOT_TYPE: ESTABLISHING SHOT\nSUBJECT: Electrical Power Facility\nACTION_DESCRIPTION: A wide, ominous shot of the power facility at night. Grimy metal, pulsing pipes, intermittent sparks. The facility dominates the frame, looking like a dark, brooding organism, baolin baolin.\nDIALOGUE: (Sound of Techno-heavy METAL MUSIC)",
        "actions": {"type": "entering", "intensity": "casual"},
        "camera": {"angle": "medium_shot", "movement": "static"},
        "metadata": {"duration": 3.5}
    }
    
    console.print(Panel(
        sample_shot["description"],
        title="[cyan]Input: Narrative Shot Description[/cyan]",
        border_style="cyan"
    ))
    console.print()

    # STEP 2: Entity Extraction
    console.print("[bold] Step 2: Entity Extraction[/bold]\n")
    
    entities = engine.extract_entities(sample_shot["description"])
    
    console.print("  [green]✓[/green] Extracted Entities:")

    if entities.characters:
        console.print("    Characters:")
        for cid in entities.characters:
            name = char_name_map.get(cid, cid)
            console.print(f"      • {cid} (\"{name}\")")
    else:
        console.print("    [dim]No characters found[/dim]")

    if entities.locations:
        console.print("    Locations:")
        for lid in entities.locations:
            name = loc_name_map.get(lid, lid)
            console.print(f"      • {lid} (\"{name}\")")
    else:
        console.print("    [dim]No locations found[/dim]")

    console.print()

    # STEP 3: Context Retrieval (RAG Retrieval Step)
    console.print("[bold] Step 3: RAG Context Retrieval[/bold]\n")
    
    context = engine.retrieve_context(sample_shot["description"], entities)
    
    # Quick RAG summary before detailed printout
    num_chars = len(entities.characters)
    num_locs = len(entities.locations)
    total_char_chunks = sum(len(chunks) for chunks in context.character_context.values())
    total_loc_chunks = sum(len(chunks) for chunks in context.location_context.values())

    all_norms = []
    char_scores_map = getattr(context, "character_scores", None) or {}
    for s in char_scores_map.values():
        v = s.get("rerank_score_norm")
        if isinstance(v, (int, float)):
            all_norms.append(v)
    loc_scores_map = getattr(context, "location_scores", None) or {}
    for s in loc_scores_map.values():
        v = s.get("rerank_score_norm")
        if isinstance(v, (int, float)):
            all_norms.append(v)
    best_conf = max(all_norms) if all_norms else None

    console.print("[bold]RAG Summary[/bold]")
    console.print(
        f"  Entities: {num_chars} character(s), {num_locs} location(s)"
    )
    console.print(
        f"  Chunks retrieved: {total_char_chunks} character + {total_loc_chunks} location"
    )
    if best_conf is not None:
        console.print(
            f"  Best match confidence (rerank_norm): {best_conf:.3f}"
        )
    console.print()

    console.print("  [green]✓[/green] Retrieved Canonical Descriptions:\n")
    
    # Display character context and scores
    if context.character_context:
        for char_id, chunks in context.character_context.items():
            name = char_name_map.get(char_id, char_id)
            console.print(f"    [cyan]• {char_id} (\"{name}\"):[/cyan] {len(chunks)} chunk(s)")
            for i, chunk in enumerate(chunks, 1):
                console.print(f"      [{i}] {chunk[:100]}{'...' if len(chunk) > 100 else ''}")

            scores_map = getattr(context, "character_scores", None) or {}
            scores = scores_map.get(char_id)
            if isinstance(scores, dict):
                bm25 = scores.get("bm25_score")
                dense = scores.get("dense_score")
                hybrid = scores.get("hybrid_score")
                rerank_norm = scores.get("rerank_score_norm")

                def _fmt(v):
                    return "-" if not isinstance(v, (int, float)) else f"{v:.3f}"

                console.print(
                    f"      [dim]Scores:[/dim] "
                    f"BM25={_fmt(bm25)} | "
                    f"Dense={_fmt(dense)} | "
                    f"Hybrid={_fmt(hybrid)} | "
                    f"RerankConf={_fmt(rerank_norm)}"
                )

                chunk_index = scores.get("chunk_index")
                chunk_id = scores.get("chunk_id")
                doc_id = scores.get("doc_id")
                if chunk_index is not None or chunk_id or doc_id:
                    console.print(
                        f"      [dim]Source:[/dim] "
                        f"doc_id={doc_id or '-'} | "
                        f"chunk_id={chunk_id or '-'} | "
                        f"index={chunk_index if chunk_index is not None else '-'}"
                    )
    else:
        console.print("    [dim]No character context retrieved[/dim]")
    
    # Display location context and scores
    if context.location_context:
        for loc_id, chunks in context.location_context.items():
            name = loc_name_map.get(loc_id, loc_id)
            console.print(f"    [cyan]• {loc_id} (\"{name}\"):[/cyan] {len(chunks)} chunk(s)")
            for i, chunk in enumerate(chunks, 1):
                console.print(f"      [{i}] {chunk[:100]}{'...' if len(chunk) > 100 else ''}")

            scores_map = getattr(context, "location_scores", None) or {}
            scores = scores_map.get(loc_id)
            if isinstance(scores, dict):
                bm25 = scores.get("bm25_score")
                dense = scores.get("dense_score")
                hybrid = scores.get("hybrid_score")
                rerank_norm = scores.get("rerank_score_norm")

                def _fmt(v):
                    return "-" if not isinstance(v, (int, float)) else f"{v:.3f}"

                console.print(
                    f"      [dim]Scores:[/dim] "
                    f"BM25={_fmt(bm25)} | "
                    f"Dense={_fmt(dense)} | "
                    f"Hybrid={_fmt(hybrid)} | "
                    f"RerankConf={_fmt(rerank_norm)}"
                )

                chunk_index = scores.get("chunk_index")
                chunk_id = scores.get("chunk_id")
                doc_id = scores.get("doc_id")
                if chunk_index is not None or chunk_id or doc_id:
                    console.print(
                        f"      [dim]Source:[/dim] "
                        f"doc_id={doc_id or '-'} | "
                        f"chunk_id={chunk_id or '-'} | "
                        f"index={chunk_index if chunk_index is not None else '-'}"
                    )
    else:
        console.print("    [dim]No location context retrieved[/dim]")
    
    console.print()

    # STEP 4: Shot Enrichment
    console.print("[bold] Step 4: Shot Enrichment[/bold]\n")
    
    # Reuse extracted entities and retrieved context to avoid duplicate queries
    enriched_shot = engine.process_shot(sample_shot, entities=entities, context=context)
    
    console.print("  [green]✓[/green] EnrichedShot created")
    console.print(f"    • Original description preserved")
    console.print(f"    • {len(enriched_shot.rag_characters)} character(s) enriched")
    console.print(f"    • {len(enriched_shot.rag_locations)} location(s) enriched")

    # Enrichment confidence indicator based on normalized rerank scores
    char_conf_values = []
    for cid in enriched_shot.characters:
        scores = enriched_shot.rag_scores.get(cid)
        if isinstance(scores, dict):
            v = scores.get("rerank_score_norm")
            if isinstance(v, (int, float)):
                char_conf_values.append(v)

    loc_conf_values = []
    for lid in enriched_shot.locations:
        scores = enriched_shot.rag_scores.get(lid)
        if isinstance(scores, dict):
            v = scores.get("rerank_score_norm")
            if isinstance(v, (int, float)):
                loc_conf_values.append(v)

    if char_conf_values or loc_conf_values:
        console.print("    • Enrichment confidence:")
        if char_conf_values:
            console.print(f"        Characters: {max(char_conf_values):.3f}")
        if loc_conf_values:
            console.print(f"        Locations: {max(loc_conf_values):.3f}")

    console.print()

    # Show a compact enriched narrative view
    merged_lines = [f"Shot: {enriched_shot.raw_description}", ""]
    if enriched_shot.rag_characters:
        merged_lines.append("Canonical Characters:")
        for cid, text in enriched_shot.rag_characters.items():
            name = enriched_shot.character_names.get(cid, cid)
            preview = (text[:120] + "...") if len(text) > 120 else text
            merged_lines.append(f"  - {cid} (\"{name}\"): {preview}")
    if enriched_shot.rag_locations:
        merged_lines.append("")
        merged_lines.append("Canonical Locations:")
        for lid, text in enriched_shot.rag_locations.items():
            name = enriched_shot.location_names.get(lid, lid)
            preview = (text[:120] + "...") if len(text) > 120 else text
            merged_lines.append(f"  - {lid} (\"{name}\"): {preview}")

    console.print(Panel("\n".join(merged_lines), title="[magenta]Enriched Narrative View[/magenta]", border_style="magenta"))
    console.print()

    # STEP 5: Display Enriched Shot JSON
    console.print("[bold] Step 5: Final Output (EnrichedShot JSON)[/bold]\n")
    
    # Convert EnrichedShot to JSON-serializable dict
    enriched_dict = {
        "shot_id": enriched_shot.shot_id,
        "scene_id": enriched_shot.scene_id,
        "raw_description": enriched_shot.raw_description,
        "characters": enriched_shot.characters,
        "locations": enriched_shot.locations,
        "character_names": enriched_shot.character_names, 
        "location_names": enriched_shot.location_names, 
        "rag_characters": enriched_shot.rag_characters,
        "rag_locations": enriched_shot.rag_locations,
        "rag_scores": enriched_shot.rag_scores,
        "actions": enriched_shot.actions,
        "camera": enriched_shot.camera,
        "metadata": enriched_shot.metadata
    }
    
    # Display as formatted JSON
    console.print(Panel(
        JSON(json.dumps(enriched_dict, indent=2)),
        title="[green bold]EnrichedShot JSON Output[/green bold]",
        subtitle="[dim]Ready for downstream 'Final Prompt Generator'[/dim]",
        border_style="green"
    ))

    
    console.print("[bold]What This Module Demonstrates:[/bold]")
    console.print("  [green]✓[/green] Entity extraction from narrative")
    console.print("  [green]✓[/green] RAG retrieval of canonical descriptions")
    console.print("  [green]✓[/green] Shot enrichment (merge + assemble)")
    console.print("  [green]✓[/green] Deterministic JSON structure for downstream pipeline\n")

    # Final summary recap
    num_chars_extracted = len(entities.characters)
    num_locs_extracted = len(entities.locations)
    num_chars_ctx = len(context.character_context)
    num_locs_ctx = len(context.location_context)

    console.print("[bold cyan]========= Summary =========[/bold cyan]")
    console.print(f"[green]✓[/green] Extracted: {num_chars_extracted} character(s), {num_locs_extracted} location(s)")
    console.print(f"[green]✓[/green] Retrieved: {num_chars_ctx} character context(s), {num_locs_ctx} location context(s)")
    console.print("[green]✓[/green] EnrichedShot ready")
    console.print("[cyan]→[/cyan] Pass to Final Prompt Generator\n")
    
if __name__ == "__main__":
    main()
