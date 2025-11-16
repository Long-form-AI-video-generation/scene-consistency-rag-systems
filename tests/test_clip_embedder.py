from PIL import Image
from pathlib import Path
from PIL import Image

from src.rag.clip_embedder import ClipEmbedder



def test_clip_embedder(tmp_path):  
    img_path = tmp_path / "fake.png"
    img = Image.new("RGB", (64, 64), color="red")
    img.save(img_path)

    embedder = ClipEmbedder()
    docs = [
        {"image_path": str(img_path), "chunk_text": "test"},
    ]
    embeddings = embedder.encode(docs)
    assert embeddings.shape == (1, 512)