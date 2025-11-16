# tests/conftest.py
import pytest
from pathlib import Path
from types import SimpleNamespace
from src.rag.config import TEMPLATES
from PIL import Image
import io


def _create_fake_image(path: Path, color: str, format: str = "PNG"):
    """Generate a real image file (PNG/JPEG) using PIL."""
    img = Image.new("RGB", (64, 64), color=color)
    if format == "JPEG":
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=95)
        path.write_bytes(buffer.getvalue())
    else:
        img.save(path, format=format)


@pytest.fixture
def tmp_data_dir(tmp_path: Path):
    """Create dataset with 2 characters"""
    data_dir = tmp_path / "characters"
    data_dir.mkdir()

    # Character 1: Warrior
    char1 = data_dir / "Warrior"
    char1.mkdir()
    _create_fake_image(char1 / "img1.png", color="red", format="PNG")  
    (char1 / "img1.txt").write_text(
        "Tall warrior with silver armor and red cape. Strong pose.", encoding="utf-8"
    )

    # Character 2: Mage
    char2 = data_dir / "Mage"
    char2.mkdir()
    _create_fake_image(char2 / "img2.jpg", color="blue", format="JPEG")  
    (char2 / "img2.txt").write_text(
        "Old mage in blue robes holding a glowing staff. Wise expression.", encoding="utf-8"
    )

    return data_dir


@pytest.fixture
def config():
    return SimpleNamespace(
        DATASET_PATH=None,
        OUTPUT_DIR=None,
        DEVICE="cpu",
        BATCH_SIZE=8,
        CLIP_INDEX_PATH=Path("/tmp/faiss_clip.index"),
        METADATA_PATH=Path("/tmp/dataset_metadata.json"),
        CLIP_EMBEDDINGS_PATH=Path("/tmp/clip_embeddings.npy"),
        TEMPLATES=TEMPLATES,
    )