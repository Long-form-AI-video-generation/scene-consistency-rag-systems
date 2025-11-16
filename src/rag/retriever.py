
from .utils import tokenize_for_bm25
from .config import BM25_WEIGHT, FAISS_WEIGHT, BM25_CANDIDATES, ANN_CANDIDATES
from typing import List, Dict, Tuple
from .logger import get_logger

from rank_bm25 import BM25Okapi
import numpy as np
import nltk
from sklearn.preprocessing import normalize

logger = get_logger(__name__)

class Retriever:
    def __init__(self, docs: List[Dict], clip_embeddings: np.ndarray, stopwords: set):
        self.docs = docs
        self.clip_embeddings = clip_embeddings
        self.stopwords = stopwords
        tokenized = [tokenize_for_bm25(d["chunk_text"], stopwords) for d in docs]
        self.bm25 = BM25Okapi(tokenized)
        # Expect faiss index provided separately at runtime (faiss_index) to decouple
        self.faiss_index = None

    def set_faiss(self, faiss_index):
        self.faiss_index = faiss_index

    def bm25_topk(self, query: str, k=BM25_CANDIDATES):
        tokens = tokenize_for_bm25(query, self.stopwords)
        scores = self.bm25.get_scores(tokens)
        k = min(len(scores), k)
        indices = np.argsort(scores)[-k:][::-1]
        return indices, scores[indices]

    def faiss_topk(self, query_embedding: np.ndarray, k=ANN_CANDIDATES):
        # query_embedding should be normalized
        D, I = self.faiss_index.search(np.asarray([query_embedding]), k)
        return I[0], D[0]

    def hybrid(self, query: str, top_k=50, bm25_k=None, ann_k=None):
        # Weighted fusion of BM25 score and FAISS similarity
        if bm25_k is None: bm25_k = BM25_CANDIDATES
        if ann_k is None: ann_k = ANN_CANDIDATES

        # BM25 candidates
        bm25_idx, bm25_scores = self.bm25_topk(query, k=bm25_k)

        # FAISS: encode query with external clip model -> caller must provide embedding
        # Here the person need to pass query_emb.
        raise NotImplementedError("Use hybrid_with_query_emb(query, query_emb) to fuse using query embedding")

    def hybrid_with_query_emb(self, query: str, query_emb: np.ndarray, top_k=50,
                              bm25_k=None, ann_k=None, bm25_weight=BM25_WEIGHT, faiss_weight=FAISS_WEIGHT):
        bm25_k = bm25_k or BM25_CANDIDATES
        ann_k = ann_k or ANN_CANDIDATES

        bm25_idx, bm25_scores = self.bm25_topk(query, k=bm25_k)

        # subset embeddings for bm25 candidates
        subset_embs = self.clip_embeddings[bm25_idx]
        # ensure query is normalized
        q = query_emb / np.linalg.norm(query_emb)
        sims = subset_embs @ q  # inner product because embeddings L2-normalised
        # normalize both score streams to 0..1
        if len(bm25_scores) == 0:
            bm25_norm = np.zeros_like(sims)
        else:
            bm25_norm = (bm25_scores - bm25_scores.min()) / (np.ptp(bm25_scores) + 1e-8)
        if len(sims) == 0:
            sims_norm = np.zeros_like(bm25_norm)
        else:
            sims_norm = (sims - sims.min()) / (np.ptp(sims) + 1e-8)

        fused = bm25_weight * bm25_norm + faiss_weight * sims_norm
        topk = min(top_k, len(fused))
        idx_in_subset = np.argsort(-fused)[:topk]
        chosen_indices = bm25_idx[idx_in_subset]
        candidates = [self.docs[i] for i in chosen_indices]
        return candidates
