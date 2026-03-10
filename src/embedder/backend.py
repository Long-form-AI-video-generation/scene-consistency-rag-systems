"""
Backend CLIP loader with deterministic, cached loader that supports
OpenAI CLIP and OpenCLIP fallback. Exposes: load_clip_model()
"""

from functools import lru_cache
from typing import Tuple, Callable
import torch
import logging

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def load_clip_model(
    model_name: str = "ViT-B/32", device: str = None
) -> Tuple[object, object, Callable]:
    """Load a CLIP model and preprocessing function once and cache it."""
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Loading CLIP model {model_name} on device: {dev}")
    try:
        import clip as openai_clip 

        model, preprocess = openai_clip.load(model_name, device=dev)

        def _tokenize(texts):
            return openai_clip.tokenize(texts, truncate=True).to(dev)

        model.eval()
        return model, preprocess, _tokenize
    except Exception:
        logger.debug("OpenAI CLIP load failed", exc_info=True)
        try:
            import open_clip 

            model_name_mapped = model_name.replace("/", "-")
            logger.info(f"Loading OpenCLIP model {model_name_mapped} on {dev}")
            model, _, preprocess = open_clip.create_model_and_transforms(
                model_name_mapped, pretrained="openai"
            )
            model = model.to(dev)

            def _tokenize(texts):
                return open_clip.tokenize(texts).to(dev)

            model.eval()
            return model, preprocess, _tokenize
        except Exception:
            logger.exception("Both OpenAI CLIP and OpenCLIP failed to load")
            raise RuntimeError(
                "Could not load CLIP (tried openai-clip and open_clip)."
                " Install one of them to use embeddings."
            )
