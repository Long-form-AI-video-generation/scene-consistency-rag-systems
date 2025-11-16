from pathlib import Path


from src.rag.pipeline import RAGPipeline

def test_full_rag_flow(tmp_data_dir, monkeypatch, config, tmp_path):  
    out_dir = tmp_path / "output"
    out_dir.mkdir()

    # CRITICAL: Set config attributes BEFORE pipeline
    config.DATASET_PATH = tmp_data_dir
    config.OUTPUT_DIR = out_dir
    config.CLIP_INDEX_PATH = out_dir / "faiss_clip.index"
    config.METADATA_PATH = out_dir / "dataset_metadata.json"
    config.CLIP_EMBEDDINGS_PATH = out_dir / "clip_embeddings.npy"

    monkeypatch.setattr("src.rag.config.DATASET_PATH", tmp_data_dir)
    monkeypatch.setattr("src.rag.config.OUTPUT_DIR", out_dir)

    pipeline = RAGPipeline(config)
    pipeline.build_all_indices(overwrite=True)

    result = pipeline.enrich_panel("mage with staff")
    assert "glowing staff" in result or "blue robes" in result
    assert len(result.split("REF")) > 1