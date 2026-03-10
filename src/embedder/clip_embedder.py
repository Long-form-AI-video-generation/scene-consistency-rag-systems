"""ClipEmbedder: Batched text/image encoding, fusion, and caching."""

from pathlib import Path
from typing import List, Dict, Optional, Tuple
from collections import OrderedDict
import numpy as np
import torch
from PIL import Image
import logging

from .backend import load_clip_model
from .utils import l2_normalize, l2_normalize_batch

try:
    from joblib import Memory

    _JOBLIB_AVAILABLE = True
except Exception:
    _JOBLIB_AVAILABLE = False

logger = logging.getLogger(__name__)


class ClipEmbedder:
    def __init__(self, config: Optional[Dict] = None):
        config = config or {}
        self.model_name = config.get("clip_model", "ViT-B/32")
        self.device = config.get("device") or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.alpha = float(config.get("fusion_alpha", 0.5))
        self.dtype = np.float32
        self._model, self._preprocess, self._tokenize = load_clip_model(
            self.model_name, device=self.device
        )
        try:
            self.dim = int(
                getattr(getattr(self._model, "visual", self._model), "output_dim", 512)
            )
        except Exception:
            self.dim = int(config.get("embedding_dim", 512))
        cache_dir = Path(config.get("embed_cache_dir", "data/embed_cache"))
        cache_dir.mkdir(parents=True, exist_ok=True)
        self._mem = (
            Memory(str(cache_dir), verbose=0)
            if (_JOBLIB_AVAILABLE and config.get("use_disk_cache", True))
            else None
        )

        # LRU cache with configurable max size to prevent unbounded memory growth
        self.max_cache_size = int(config.get("max_cache_size", 1000))
        self._text_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._image_cache: OrderedDict[str, np.ndarray] = OrderedDict()

        logger.info(
            f"ClipEmbedder initialized: model={self.model_name} device={self.device} "
            f"dim={self.dim} max_cache_size={self.max_cache_size}"
        )

    def _add_to_cache(self, cache: OrderedDict, key: str, value: np.ndarray) -> None:
        if key in cache:
            cache.move_to_end(key)
            cache[key] = value
        else:
            cache[key] = value
            if len(cache) > self.max_cache_size:
                evicted_key = cache.popitem(last=False)[0]
                logger.debug(f"LRU cache evicted: {evicted_key[:50]}...")

    def embed_text_batch(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=self.dtype)
        results: List[Optional[np.ndarray]] = [None] * len(texts)
        to_compute, order = [], []
        for i, t in enumerate(texts):
            if t in self._text_cache:
                self._text_cache.move_to_end(t)
                results[i] = self._text_cache[t]
            else:
                to_compute.append(t)
                order.append(i)
        if to_compute:
            with torch.no_grad():
                feats = self._model.encode_text(self._tokenize(to_compute))
            arrs = l2_normalize_batch(feats.cpu().numpy().astype(self.dtype))
            for j, t in enumerate(to_compute):
                vec = arrs[j].copy()
                self._add_to_cache(self._text_cache, t, vec)
                results[order[j]] = vec
        if any(v is None for v in results):
            raise RuntimeError("Failed to produce text embedding")
        return np.stack(results, axis=0)

    def embed_text(self, text: str) -> np.ndarray:
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        if text in self._text_cache:
            self._text_cache.move_to_end(text)
            return self._text_cache[text]
        self.embed_text_batch([text])
        return self._text_cache[text]

    def _load_images(self, paths: List[str]) -> Tuple[List, List[int]]:
        imgs, valid_idx = [], []
        for i, p in enumerate(paths):
            try:
                imgs.append(self._preprocess(Image.open(p).convert("RGB")))
                valid_idx.append(i)
            except Exception as e:
                logger.warning(f"Failed to load image {p}: {e}")
                valid_idx.append(-1)
        return imgs, valid_idx

    def embed_image_batch(self, paths: List[str]) -> np.ndarray:
        if not paths:
            return np.zeros((0, self.dim), dtype=self.dtype)
        results: List[Optional[np.ndarray]] = [None] * len(paths)
        to_compute, order = [], []
        for i, p in enumerate(paths):
            if p in self._image_cache:
                self._image_cache.move_to_end(p)
                results[i] = self._image_cache[p]
            else:
                to_compute.append(p)
            order.append(i)
        if to_compute:
            imgs, valid_idx = self._load_images(to_compute)
            if imgs:
                with torch.no_grad():
                    feats = self._model.encode_image(
                        torch.stack(imgs, dim=0).to(self.device)
                    )
                arrs = l2_normalize_batch(feats.cpu().numpy().astype(self.dtype))
            else:
                arrs = np.zeros((0, self.dim), dtype=self.dtype)
            vi = 0
            for j, p in enumerate(to_compute):
                if valid_idx[j] >= 0:
                    vec = arrs[vi].copy()
                    self._add_to_cache(self._image_cache, p, vec)
                    results[order[j]] = vec
                    vi += 1
                else:
                    raise RuntimeError(f"Failed to load image: {p}")
        if any(v is None for v in results):
            raise RuntimeError("Missing image embedding")
        return np.stack(results, axis=0)

    def embed_image(self, path: str) -> np.ndarray:
        if not path or not path.strip():
            raise ValueError("Image path cannot be empty")
        if path in self._image_cache:
            self._image_cache.move_to_end(path)
            return self._image_cache[path]
        self.embed_image_batch([path])
        return self._image_cache[path]

    def _fuse(
        self, text_emb: np.ndarray, image_emb: Optional[np.ndarray]
    ) -> np.ndarray:
        if image_emb is None:
            return l2_normalize(text_emb)
        if text_emb.shape != image_emb.shape:
            raise ValueError("Embedding dim mismatch")
        return l2_normalize(
            self.alpha * text_emb.astype(self.dtype)
            + (1.0 - self.alpha) * image_emb.astype(self.dtype)
        )

    def embed_entity(self, entity: Dict) -> Dict:
        eid = (
            entity.get("character_id") or entity.get("location_id") or entity.get("id")
        )
        if not eid:
            raise ValueError(
                "Entity must have an id field (character_id/location_id/id)"
            )
        text = (
            entity.get("appearance")
            or entity.get("description")
            or entity.get("name")
            or ""
        ).strip()
        text_emb = self.embed_text(text)
        image_emb = None
        if entity.get("canonical_image_path"):
            try:
                image_emb = self.embed_image(entity["canonical_image_path"])
            except Exception:
                logger.warning(
                    f"Image embedding failed for {entity.get('canonical_image_path')}; falling back to text only."
                )
        return {"entity_id": eid, "fused_embedding": self._fuse(text_emb, image_emb)}

    def embed_entities(self, entities: List[Dict]) -> List[Dict]:
        texts = [
            (e.get("appearance") or e.get("description") or e.get("name") or "")
            for e in entities
        ]
        text_embs = self.embed_text_batch(texts)
        image_paths = [e.get("canonical_image_path") for e in entities]
        imgs_to_compute = [p for p in image_paths if p]
        img_order = [i for i, p in enumerate(image_paths) if p]
        image_embs_map = {}
        if imgs_to_compute:
            img_embs = self.embed_image_batch(imgs_to_compute)
            for k, idx in enumerate(img_order):
                image_embs_map[idx] = img_embs[k]
        return [
            {
                "entity_id": e.get("character_id")
                or e.get("location_id")
                or e.get("id"),
                "fused_embedding": self._fuse(text_embs[i], image_embs_map.get(i)),
            }
            for i, e in enumerate(entities)
        ]

    # Backward Compatibility 
    def fuse(
        self,
        text_emb: np.ndarray,
        image_emb: Optional[np.ndarray] = None,
        alpha: Optional[float] = None,
    ) -> np.ndarray:
        """Public fusion method for backward compatibility. Uses alpha parameter if provided, otherwise uses self.alpha."""
        if alpha is not None:
            old_alpha = self.alpha
            self.alpha = float(alpha)
            result = self._fuse(text_emb, image_emb)
            self.alpha = old_alpha
            return result
        return self._fuse(text_emb, image_emb)

    def clear_cache(self):
        """Clear in-memory caches."""
        self._text_cache.clear()
        self._image_cache.clear()
        if self._mem:
            self._mem.clear()

    def format_output(
        self, chunk_id: str, embedding: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Format embedding output for backward compatibility."""
        return {"chunk_id": chunk_id, "embedding": embedding.astype("float32")}

    @property
    def embed_dim(self) -> int:
        """Backward compatibility: embed_dim property."""
        return self.dim

    @property
    def model(self):
        """Backward compatibility: model property."""
        return self._model

    @property
    def preprocess(self):
        """Backward compatibility: preprocess property."""
        return self._preprocess
