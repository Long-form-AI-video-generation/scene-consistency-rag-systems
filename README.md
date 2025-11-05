
## Highlights

- Hybrid retrieval combining BM25 (keyword) and FAISS (CLIP embeddings) for robust candidate selection
- CLIP fused embeddings (image + text) for multimodal retrieval
- Cross-encoder reranking for high-quality top-k selection
- Simple utilities for cleaning/chunking textual annotations and saving metadata

## Quick summary

- Entry point: `main.py` — builds indices from `datasets/` and contains a `sample_query()` helper to demonstrate retrieval and prompt assembly.
- Utilities: `prepare.py` — text cleaning, chunking, metadata save/load.
- Outputs: `faiss_clip.index`, `clip_embeddings.npy`, `dataset_metadata.json` (metadata and indices used for retrieval).

## Requirements

- Python >= 3.12
- See `pyproject.toml` for the pinned runtime dependencies used by the project (faiss-cpu, torch, transformers, sentence-transformers, rank-bm25, nltk, numpy, scikit-learn).

Recommended: a machine with CUDA if you plan to use large models and GPU acceleration; the code falls back to CPU when CUDA isn't available.

## Installation

1. Create and activate a virtual environment (example using python builtin venv):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
```

2. Install runtime dependencies (adjust for GPU builds of PyTorch if needed):

```bash
pip install -r requirements.txt  # optional: create from pyproject or install the listed deps manually
```

Note: This repo includes a `pyproject.toml` listing dependencies. If you prefer, install those packages directly with pip. If you plan to use CUDA builds of PyTorch, follow the official PyTorch instructions to install a matching `torch` wheel.

## Dataset format

Place character image/text pairs under the `datasets/` folder. The expected layout is:

```
datasets/
	character_name/
		img001.png
		img001.txt   # caption/annotation for img001.png
		img002.jpg
		img002.txt
```

Each `.txt` file should contain the textual description for the paired image. `main.py` will iterate character subfolders, pair images with same-stem `.txt` files, clean and chunk the text, and construct per-chunk metadata entries.

## How it works (high-level)

1. `main.build_all_indices()` loads the dataset using `load_and_prepare_dataset()`.
2. Text is cleaned (`prepare.clean_file`) and chunked (`prepare.chunk_text`) to produce manageable chunks for encoding.
3. CLIP (image + text) embeddings are computed, fused (image_emb + text_emb), normalized, and optionally saved to `clip_embeddings.npy`.
4. FAISS index is created (`faiss_clip.index`) and stored to disk.
5. BM25 index (text-only) is built from chunked texts to provide a keyword-based candidate set.
6. Hybrid retrieval (BM25 -> FAISS) returns candidates which are reranked with a Cross-Encoder. The top-k are assembled into a structured prompt.

## Usage

- Build indices and metadata (default dataset path `datasets/`):

```bash
python main.py
```

By default, `main.py` checks whether `dataset_metadata.json` and `faiss_clip.index` already exist. If either is missing it will build indices from the dataset.

- Run the sample query routine (see `main.sample_query` for the example prompt flow):

```bash
python -c "import main; main.sample_query('A short example scene description here')"
```

Or modify `example_query` inside `main.py` and run the script to see retrieval and the assembled prompt printed to stdout.

## Configuration knobs

- `DATASET_PATH` (in `main.py`) — path to dataset directory (defaults to `datasets/`).
- `EMBEDDING_MODEL_NAME` — CLIP model path/identifier (defaults to `openai/clip-vit-base-patch32`).
- `CROSS_ENCODER_MODEL_NAME` — cross-encoder used for reranking (defaults to `cross-encoder/ms-marco-MiniLM-L-6-v2`).
- `BM25_CANDIDATES`, `ANN_CANDIDATES`, `RERANK_TOP_K` — retrieval and re-ranking hyperparameters.

Adjust these values in `main.py` for your environment or experiments.

## Files and purpose

- `main.py` — Full RAG pipeline: data ingestion, CLIP encoding (image+text fusion), FAISS index creation, BM25 indexing, hybrid retrieval, cross-encoder reranking, and prompt assembly. Also includes `sample_query()` and a small CLI-style behavior in `__main__`.
- `prepare.py` — small utilities: `clean_file()`, `chunk_text()`, `save_metadata()` and `load_metadata()`.
- `pyproject.toml` — project metadata and dependency list used for development and reproducibility.
- `dataset_metadata.json` — generated metadata listing all document chunks (id, character, image_path, chunk_text). This file is produced by `prepare.save_metadata()` during the indexing step.
- `clip_embeddings.npy` — (optional) saved fused CLIP embeddings for faster repeated retrieval.
- `faiss_clip.index` — FAISS index persisted to disk for ANN retrieval.

## Troubleshooting and tips

- If you see tokenizer length errors when encoding very large chunks, reduce `CHUNK_SIZE` in `prepare.py` or increase truncation handling in `main.encode_clip_fused_batch()`.
- NLTK tokenizers: the script attempts to download required resources (punkt). Ensure the environment can reach the internet or install `nltk` datasets manually (e.g. `python -m nltk.downloader punkt`).
- If you want GPU acceleration: install `torch` with CUDA support and run on a machine with an NVIDIA GPU. `main.py` auto-detects CUDA via `torch.cuda.is_available()`.
- FAISS: the repo lists `faiss-cpu` in `pyproject.toml`; if you have GPU-enabled FAISS builds, you can change the package and rebuild indices accordingly.
