from pathlib import Path
from typing import List, Dict
from .logger import get_logger
from .utils import sanitize_text, clean_file, chunk_text


logger = get_logger(__name__)

class DatasetPreparer:
    def __init__(self, dataset_path: Path):
        self.dataset_path = Path(dataset_path)

    def load(self) -> List[Dict]:
        docs = []
        next_id = 0
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"{self.dataset_path} missing")
        for character_dir in sorted(self.dataset_path.iterdir()):
            if not character_dir.is_dir(): continue
            for item in sorted(character_dir.iterdir()):
                if item.suffix.lower() in [".png", ".jpg", ".jpeg"]:
                    image_file = item
                    text_file = item.with_suffix(".txt")
                    if not text_file.exists(): continue
                    raw = text_file.read_text(encoding="utf-8")
                    cleaned = sanitize_text(clean_file(raw))
                    chunks = chunk_text(cleaned)
                    for chunk in chunks:
                        docs.append({
                            "id": f"chunk_{next_id:07d}",   # stable id per chunk
                            "character": character_dir.name,
                            "image_path": str(image_file),
                            "chunk_text": chunk
                        })
                        next_id += 1
        logger.info("Loaded %d chunks", len(docs))
        return docs
