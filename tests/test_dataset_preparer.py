from pathlib import Path
from src.rag.dataset_preparer import DatasetPreparer

def test_load_dataset(tmp_data_dir): 
    preparer = DatasetPreparer(tmp_data_dir)
    docs = preparer.load()
    assert len(docs) == 2
    assert any("warrior" in d["chunk_text"].lower() for d in docs)
    assert any("mage" in d["chunk_text"].lower() for d in docs)