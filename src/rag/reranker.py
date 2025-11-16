from sentence_transformers import CrossEncoder
from typing import List, Dict
from .config import CROSS_ENCODER_MODEL_NAME, DEVICE
from .logger import get_logger
import numpy as np

logger = get_logger(__name__)

class Reranker:
    def __init__(self, model_name=CROSS_ENCODER_MODEL_NAME, device=DEVICE):
        self.model = CrossEncoder(model_name, device=device)

    def rerank(self, query: str, candidates: List[Dict], top_k=5):
        pairs = [[query, c["chunk_text"]] for c in candidates]
        scores = self.model.predict(pairs)
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return [c for c, s in ranked[:top_k]]
