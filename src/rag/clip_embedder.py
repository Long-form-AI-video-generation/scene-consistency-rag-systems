
from typing import List, Dict
from .config import EMBEDDING_MODEL_NAME, DEVICE, BATCH_SIZE
from .logger import get_logger

import numpy as np
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
from sklearn.preprocessing import normalize


logger = get_logger(__name__)

class ClipEmbedder:
    def __init__(self, model_name=EMBEDDING_MODEL_NAME, device=DEVICE):
        self.device = device
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()

    def _encode_batch(self, images, texts):
        # tokenize
        tokenizer = self.processor.tokenizer
        tokenized = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
        image_inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        inputs = {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized.get("attention_mask"),
            "pixel_values": image_inputs["pixel_values"]
        }
        with torch.no_grad():
            outputs = self.model(**{k:v for k,v in inputs.items() if v is not None})
            image_embs = outputs.image_embeds.cpu().numpy()
            text_embs = outputs.text_embeds.cpu().numpy()
            # average then L2 renormalize
            fused = (image_embs + text_embs) / 2.0
            fused = normalize(fused, axis=1)
            return fused

    def encode(self, docs: List[Dict], batch_size: int = BATCH_SIZE) -> np.ndarray:
        embs = []
        for i in range(0, len(docs), batch_size):
            batch = docs[i:i+batch_size]
            images = [Image.open(d["image_path"]).convert("RGB") for d in batch]
            texts = [d["chunk_text"] for d in batch]
            embs.append(self._encode_batch(images, texts))
        return np.vstack(embs)
