
from src.rag.pipeline import RAGPipeline
from src.rag.config import *


def test_pipeline_build_and_enrich(tmp_data_dir, monkeypatch, config, tmp_path):
    temp_out = tmp_path / "output"
    temp_out.mkdir()

    # CRITICAL: Set config attributes BEFORE pipeline
    config.DATASET_PATH = tmp_data_dir
    config.OUTPUT_DIR = temp_out
    config.CLIP_INDEX_PATH = temp_out / "faiss_clip.index"
    config.METADATA_PATH = temp_out / "dataset_metadata.json"
    config.CLIP_EMBEDDINGS_PATH = temp_out / "clip_embeddings.npy"

    # Optional: monkeypatch module (defensive)
    monkeypatch.setattr("src.rag.config.DATASET_PATH", tmp_data_dir)
    monkeypatch.setattr("src.rag.config.OUTPUT_DIR", temp_out)

    pipeline = RAGPipeline(config)
    pipeline.build_all_indices(overwrite=True)

    assert (temp_out / "faiss_clip.index").exists()
    assert (temp_out / "clip_embeddings.npy").exists()

    result = pipeline.enrich_panel("warrior with armor")
    assert "CHARACTER CONSISTENCY ANCHOR" in result
    assert "silver armor" in result or "red cape" in result