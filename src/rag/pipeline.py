import json
from pathlib import Path

import faiss
import numpy as np
import torch
import nltk
from nltk.corpus import stopwords as _stopwords


from .config import (CLIP_INDEX_PATH, METADATA_PATH, CLIP_EMBEDDINGS_PATH, DEVICE, BATCH_SIZE)
from .dataset_preparer import DatasetPreparer
from .clip_embedder import ClipEmbedder
from .retriever import Retriever
from .reranker import Reranker
from .logger import get_logger

logger = get_logger(__name__)

# ensure nltk stopwords present
try:
    STOPWORDS = set(_stopwords.words("english"))
except Exception:
    import nltk
    nltk.download("stopwords", quiet=True)
    STOPWORDS = set(_stopwords.words("english"))

class RAGPipeline:
    def __init__(self, config):
        self.config = config
        self.preparer = DatasetPreparer(config.DATASET_PATH)
        self.embedder = ClipEmbedder(device=config.DEVICE)
        self.reranker = Reranker()
        self._docs = None
        self._embeddings = None
        self._faiss = None
        self._retriever = None

    def build_all_indices(self, overwrite=False, use_hnsw=True):
        """
        Loads dataset, encodes, builds faiss index and BM25 structures, and saves artifacts.
        """
        logger.info("Building indices...")
        docs = self.preparer.load()
        if not docs:
            raise RuntimeError("No docs found in dataset")
        self._docs = docs

        # encode embeddings with batching
        embeddings = self.embedder.encode(docs, batch_size=self.config.BATCH_SIZE)
        # ensure L2 normalised
        faiss.normalize_L2(embeddings)

        # build FAISS index
        dim = embeddings.shape[1]
        if use_hnsw:
            index = faiss.IndexHNSWFlat(dim, 32)
        else:
            index = faiss.IndexFlatIP(dim)
        index.add(embeddings)

        # save artifacts
        faiss.write_index(index, str(self.config.CLIP_INDEX_PATH))
        np.save(str(self.config.CLIP_EMBEDDINGS_PATH), embeddings)
        with open(self.config.METADATA_PATH, "w", encoding="utf-8") as f:
            json.dump(docs, f, ensure_ascii=False, indent=2)

        # store locally
        self._embeddings = embeddings
        self._faiss = index
        self._retriever = Retriever(docs, embeddings, STOPWORDS)
        self._retriever.set_faiss(index)

        logger.info("Built indices: %d chunks", len(docs))
        return docs

    def load_indices(self):
        # loader used at runtime to avoid rebuild
        if self._docs is not None:
            return
        if not Path(self.config.METADATA_PATH).exists():
            raise FileNotFoundError("metadata missing; run build_all_indices")
        with open(self.config.METADATA_PATH, "r", encoding="utf-8") as f:
            self._docs = json.load(f)
        self._embeddings = np.load(str(self.config.CLIP_EMBEDDINGS_PATH))
        self._faiss = faiss.read_index(str(self.config.CLIP_INDEX_PATH))
        self._retriever = Retriever(self._docs, self._embeddings, STOPWORDS)
        self._retriever.set_faiss(self._faiss)


    def enrich_panel(self, panel_text: str, top_k=5):
        self.load_indices()
        # produce query embedding (decoupled from retriever)
        # using embedder text encoding only
        processor = self.embedder.processor
        tokenizer = processor.tokenizer
        inputs = tokenizer([panel_text], padding=True, truncation=True, return_tensors="pt").to(self.embedder.device)
        with torch.no_grad():
            q_emb = self.embedder.model.get_text_features(**inputs).cpu().numpy()[0]
        q_emb = q_emb / np.linalg.norm(q_emb)

        candidates = self._retriever.hybrid_with_query_emb(panel_text, q_emb, top_k=50)
        reranked = self.reranker.rerank(panel_text, candidates, top_k=top_k)

        # assemble anchor template
        entries = []
        for i, c in enumerate(reranked, start=1):
            appearance = c["chunk_text"]
            style = "Copy lighting, clothing, hair, pose, expression exactly."
            entries.append(self.config.TEMPLATES["entry_format"].format(i=i, appearance=appearance, style=style))
        anchor = self.config.TEMPLATES["character_anchor"].format(entries="\n".join(entries))
        return panel_text + "\n" + anchor
