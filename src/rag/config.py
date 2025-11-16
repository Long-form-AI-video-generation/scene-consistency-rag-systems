from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parents[2]
DATASET_PATH = Path(os.getenv("RAG_DATASET_PATH", BASE_DIR / "data" / "characters"))
OUTPUT_DIR = Path(os.getenv("RAG_OUTPUT_DIR", BASE_DIR / "data" / "rag_out"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "openai/clip-vit-base-patch32")
CROSS_ENCODER_MODEL_NAME = os.getenv("CROSS_ENCODER_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L-6-v2")
DEVICE = "cuda" if os.getenv("RAG_FORCE_CPU", "0") != "1" and __import__("torch").cuda.is_available() else "cpu"

CLIP_INDEX_PATH = OUTPUT_DIR / "faiss_clip.index"
METADATA_PATH = OUTPUT_DIR / "dataset_metadata.json"
CLIP_EMBEDDINGS_PATH = OUTPUT_DIR / "clip_embeddings.npy"

# Hyerparams
BM25_CANDIDATES = int(os.getenv("BM25_CANDIDATES", "100"))
ANN_CANDIDATES = int(os.getenv("ANN_CANDIDATES", "50"))
RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "5"))

# Hybrid weights (0..1)
BM25_WEIGHT = float(os.getenv("BM25_WEIGHT", "0.4"))
FAISS_WEIGHT = float(os.getenv("FAISS_WEIGHT", "0.6"))

BATCH_SIZE = int(os.getenv("RAG_BATCH_SIZE", "8"))
CACHE_EXPIRES_SEC = int(os.getenv("RAG_CACHE_EXPIRES_SEC", "3600"))
 
CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "2000"))
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "200"))

# Text templates
TEMPLATES = {
    "character_anchor": "=== CHARACTER CONSISTENCY ANCHOR (OBEY EXACTLY) ===\n{entries}\n=== END ANCHOR ===\n",
    "entry_format": "REF {i}: APPEARANCE: {appearance}\nSTYLE: {style}\n"
}
