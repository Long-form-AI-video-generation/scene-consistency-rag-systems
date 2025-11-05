"""
RAG Pipeline 
 - ingests character image/text datasets (folder-per-character)
 - computes CLIP (image+text fused) embeddings 
 - builds FAISS indices (CLIP index with normalization
 - provides a simple BM25 keyword filter to build a hybrid retriever
 - reranks candidates with a Cross-Encoder
 - assembles a structured prompt 

Notes / Requirements:
 - Change DATASET_PATH to your local dataset location.

Dataset structure expected:
 datasets/
   character_name/
     img1.png
     img1.txt
     img2.png
     img2.txt

"""



import os
from pathlib import Path
from typing import List, Dict, Tuple
from PIL import Image

import numpy as np
import faiss
import torch
import nltk
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import CLIPProcessor, CLIPModel
from rank_bm25 import BM25Okapi
from sklearn.preprocessing import normalize
import prepare


# Ensure punkt is available
try:
    nltk.data.find('tokenizers/punkt_tab/english')
except LookupError:
    import nltk
nltk.download('punkt_tab', quiet=True)

# configurations
DATASET_PATH = "datasets/"
EMBEDDING_MODEL_NAME = "openai/clip-vit-base-patch32"
CROSS_ENCODER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# Index filenames
CLIP_INDEX_PATH = "faiss_clip.index"
METADATA_PATH = "dataset_metadata.json"
CLIP_EMBEDDINGS_PATH = "clip_embeddings.npy"


# Hyperparameters
BM25_CANDIDATES = 100
ANN_CANDIDATES = 50
RERANK_TOP_K = 5

# Load models
clip_model = CLIPModel.from_pretrained(EMBEDDING_MODEL_NAME).to(DEVICE)
clip_processor = CLIPProcessor.from_pretrained(EMBEDDING_MODEL_NAME)
cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL_NAME, device=DEVICE)


def load_and_prepare_dataset(dataset_path: str) -> List[Dict]:
    """
    Loads the dataset from the specified path and prepares a list of documents.
    Each document is a dictionary with 'image_path' and 'text' keys.

    Args:
        dataset_path (str): Path to the dataset directory.

    Returns:
        List[Dict]: A list of documents with image paths and associated text.
    """
    base = Path(dataset_path)
    docs = []
    next_id = 0
    for character_dir in base.iterdir():
        if not character_dir.is_dir():
            continue
        for item in character_dir.iterdir():
            if item.suffix.lower() in [".png", ".jpg", ".jpeg"]:
                image_file = item
                text_file = item.with_suffix('.txt')
                if not text_file.exists():
                    continue
                raw = text_file.read_text(encoding='utf-8')
                cleaned = prepare.clean_file(raw)
                chunks = prepare.chunk_text(cleaned)
                for chunk in chunks:
                    docs.append({
                        "id": next_id,
                        "character": character_dir.name,
                        "image_path": str(image_file),
                        "chunk_text": chunk,
                        
                    })
                next_id += 1
    return docs


def encode_clip_fused_batch(docs: List[Dict], batch_size: int = 8) -> np.ndarray:
    """
        Encode image and text(captions) using CLIP.
    """
    all_embs = []
    clip_model.eval()

    # determine the max token length the CLIP text encoder supports so we can
    # truncate long chunks before passing them to the model. Some chunk sizes
    # (e.g. 1000 chars) will tokenise to more tokens than the model allows
    # and raise ValueError: sequence length must be less than max_position_embeddings.
    try:
        max_text_len = clip_model.config.text_config.max_position_embeddings
    except Exception:
        # fallback to tokenizer's model max length
        max_text_len = getattr(clip_processor.tokenizer, "model_max_length", None)

    with torch.no_grad():
        for i in range(0, len(docs), batch_size):
            batch = docs[i:i+batch_size]
            images = [Image.open(d['image_path']).convert('RGB') for d in batch]
            texts = [d['chunk_text'] for d in batch]

            # Tokenize texts explicitly with truncation so we guarantee sequences
            # are at or below the model's max length. Then separately process
            # images to pixel values and combine tensors for the model call.
            tokenizer = clip_processor.tokenizer
            tokenized = tokenizer(texts, padding=True, truncation=True, max_length=max_text_len, return_tensors='pt')

            image_inputs = clip_processor(images=images, return_tensors='pt')

            # Combine into the input dict expected by CLIPModel and move to device
            inputs = {
                'input_ids': tokenized['input_ids'].to(DEVICE),
                'attention_mask': tokenized.get('attention_mask').to(DEVICE) if tokenized.get('attention_mask') is not None else None,
                'pixel_values': image_inputs['pixel_values'].to(DEVICE),
            }

            # Remove None entries to avoid passing unexpected kwargs
            inputs = {k: v for k, v in inputs.items() if v is not None}

            outputs = clip_model(**inputs)

            image_embs = outputs.image_embeds.cpu().numpy()
            text_embs = outputs.text_embeds.cpu().numpy()
            fused = image_embs + text_embs
            fused = normalize(fused)
            all_embs.append(fused)
   

    return np.vstack(all_embs)


def build_bm25_index(docs: List[Dict]) -> Tuple[BM25Okapi, List[List[str]]]:
    """
     Build BM25 index from the text chunks in the documents.
    Args:
        docs (List[Dict]): List of documents with 'chunk_text' key.
    Returns:
        Tuple[BM25Okapi, List[List[str]]]: The BM25 index and tokenized texts.
    """

    tokenized_texts = [nltk.word_tokenize(d['chunk_text'].lower()) for d in docs]
    bm25 = BM25Okapi(tokenized_texts)

    return bm25, tokenized_texts



def build_faiss_index(embeddings: np.ndarray, use_hnsw: bool = True) -> faiss.Index:
    """
     Set up two types of faiss database form:
      1. Heirarchical Navigable Small World(HNSW): graph based Approximate Nearest Neihbor algorith(ANN)
         It is not guarante to search closest words or token but it is efficeint in term of time complexity.
      2. Flat Inner Product: Calculate similarity between all words or token. It is precise and accurate
         but less efficeint interm of time complexity.
    """

    dimensions = embeddings.shape[1]
    if use_hnsw:
        idx = faiss.IndexHNSWFlat(dimensions, 32)
        faiss.normalize_L2(embeddings)
        idx.add(embeddings)
    else:
        idx = faiss.IndexFlatIP(dimensions)
        faiss.normalize_L2(embeddings)
        idx.add(embeddings)
    return idx

def hybrid_retrieve(query: str,
                    docs: List[Dict],
                    bm25: BM25Okapi,
                    faiss_index: faiss.Index,
                    clip_embeddings: np.ndarray,
                    top_k_bm25: int = BM25_CANDIDATES,
                    top_k_faiss: int = ANN_CANDIDATES) -> List[Dict]:
    """
    Hybrid retriever combining BM25 and FAISS ANN-HNSW search.

    Args:
        query (str): The input query string.
        docs (List[Dict]): The list of documents.
        bm25 (BM25Okapi): The BM25 index.
        faiss_index (faiss.index): The FAISS index.
        clip_embeddings (np.ndarray): The CLIP embeddings for FAISS.
        top_k_bm25 (int): Number of top candidates to retrieve from BM25.
        top_k_faiss (int): Number of top candidates to retrieve from FAISS.

    Returns:
        List[Dict]: A list of tuples containing document IDs and their scores.
    """

    # Step 1: BM25 retrieval
    tokenized_query = nltk.word_tokenize(query.lower())
    bm25_scores = bm25.get_scores(tokenized_query)
    # select the indices of the top `top_k_bm25` BM25 scores (descending)
    if top_k_bm25 <= 0:
        bm25_top_indices = np.array([], dtype=int)
    else:
        k = min(len(bm25_scores), top_k_bm25)
        # argsort gives ascending order; take the last `k` entries for top scores
        bm25_top_indices = np.argsort(bm25_scores)[-k:][::-1]

    # Step 2: FAISS ANN retrieval
    inputs = clip_processor.tokenizer([query], 
                                      return_tensors='pt', 
                                      padding=True).to(DEVICE)
    with torch.no_grad():
        query_clip = clip_model.get_text_features(**inputs).cpu().numpy()[0]
    query_clip = query_clip / np.linalg.norm(query_clip)  
    bm25_subset_embeddings = clip_embeddings[bm25_top_indices]

    # embeddings were normalized during encoding, here inner product is used 
    sims = bm25_subset_embeddings @ query_clip

    topk = min(top_k_faiss, sims.shape[0])
    if topk <= 0:
        return []

    # get indices of the best `topk` items within the subset (descending)
    top_subset_idx = np.argsort(-sims)[:topk]

    # map back to original document indices
    candidates_indices = bm25_top_indices[top_subset_idx]

    return [docs[idx] for idx in candidates_indices]



def rerank_with_cross_encoder(query: str, candidates: List[Dict], top_k: int = RERANK_TOP_K) -> List[Dict]:
    """
    Rerank candidates using a Cross-Encoder model.

    Args:
        query (str): The input query string.
        candidates (List[Dict]): The list of candidate documents.
        top_k (int): Number of top candidates to return after reranking.

    Returns:
        List[Dict]: The reranked list of candidate documents.
    """

    pairs = [[query, cand['chunk_text']] for cand in candidates]
    scores = cross_encoder.predict(pairs)

    ranked_candidates = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    top_candidates = [cand for cand, score in ranked_candidates[:top_k]]

    return top_candidates


def assemble_prompt(scene: str, context_docs: List[Dict]) -> str:
    """
    Assemble the final prompt using the scene description and context documents.

    Args:
        scene (str): The scene description.
        context_docs (List[Dict]): The list of context documents.

    Returns:
        str: The assembled prompt.
    """

    blocks = []
    for doc in context_docs:
        block = f"Character: {doc['character']} \n Source image: {doc['image_path']} \n Important facts: {doc['chunk_text']}\n"
        blocks.append(block)
    
    prompt = (
        f"Scene Description: \n {scene}\n\n"
        "Character Contexts (use these to maintain visual and behavioral fidelity) : \n"
         + "\n---\n".join(blocks)
    )
 
    return prompt


def build_all_indices(dataset_path: str = DATASET_PATH):
    """
    Build all necessary indices (BM25 and FAISS) from the dataset.

    Args:
        dataset_path (str): Path to the dataset directory.
    """

    print("Loading and preparing dataset...")
    docs = load_and_prepare_dataset(dataset_path)

    print(f"Prepared {len(docs)} document chunks.")
    if len(docs) == 0:
        raise RuntimeError("No documents found in the dataset. Please check the dataset path and structure.")
    
    print("Building BM25 index...")
    bm25, tokenized = build_bm25_index(docs)

    print("Encoding documents with CLIP fused embeddings...")
    clip_embeddings = encode_clip_fused_batch(docs)

    print("Building FAISS index...")
    faiss_index = build_faiss_index(clip_embeddings, use_hnsw=False)

    print("Saving indices and metadata...")
    faiss.write_index(faiss_index, CLIP_INDEX_PATH)
    # Save embeddings to disk so they can be re-used for hybrid retrieval
    try:
        np.save(CLIP_EMBEDDINGS_PATH, clip_embeddings)
    except Exception:
        pass
    prepare.save_metadata(docs, METADATA_PATH)

    print("All indices and metadata saved successfully.")

    return docs, bm25, tokenized, faiss_index, clip_embeddings

def sample_query(scene_query: str):

    print("Loading metadata...")
    docs = prepare.load_metadata(METADATA_PATH)
    print("Loading FAISS index...")
    faiss_index = faiss.read_index(CLIP_INDEX_PATH)
    print("Building BM25 index...")
    bm25, tokenized = build_bm25_index(docs)

    # Load or compute CLIP embeddings needed for hybrid retrieval
    if os.path.exists(CLIP_EMBEDDINGS_PATH):
        try:
            clip_embeddings = np.load(CLIP_EMBEDDINGS_PATH)
        except Exception:
            clip_embeddings = encode_clip_fused_batch(docs)
    else:
        print("Encoding documents with CLIP fused embeddings (for retrieval)...")
        clip_embeddings = encode_clip_fused_batch(docs)

    print("Retrieving hybrid candidates...")
    candidates = hybrid_retrieve(scene_query, docs, bm25, faiss_index, clip_embeddings)


    reranked = rerank_with_cross_encoder(scene_query, candidates, top_k=RERANK_TOP_K)
    print("Top reranked documents:")
    for r in reranked:
        print(f"- [{r['character']}] {r['chunk_text'][:200]}...")

    final_prompt = assemble_prompt(scene_query, reranked)
    print("\nFinal assembled prompt:\n")
    print(final_prompt)
    return final_prompt

if __name__ == "__main__":
    need_build = not (os.path.exists(METADATA_PATH)  and os.path.exists(CLIP_INDEX_PATH))
    if need_build:
        print("Indices or metadata missing — building from dataset...")
        docs, bm25, tokenized, faiss_index, clip_embeddings = build_all_indices(DATASET_PATH)
    else:
        print("Indices and metadata found — skipping build.")

    example_query = "Baolin runs through the forest in the evening rain, looking frightened but determined."
    sample_query(example_query)