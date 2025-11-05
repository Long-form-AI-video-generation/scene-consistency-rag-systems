"""
 This module contains utility functions for cleaning and chunking.
  The first function clean extra or whitespace.
  The second one chunks text into smaller pieces based on a specified size and overlap.
"""

import re
import json
from typing import List, Dict


CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
METADATA_PATH = "dataset_metadata.json"
DATASET_PATH = "datasets/"

def clean_file(text: str) -> str:
    """
    Cleans the input text by removing extra whitespace and special characters.

    Args:
        text (str): The input text to be cleaned.

    Returns:
        str: The cleaned text.
    """

    text = text.replace('\n', ' ').strip()
    text = re.sub(r'\s+', ' ', text)
    return text



def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Chunks the input text into smaller pieces based on the specified chunk size and overlap.

    Args:
        text (str): The input text to be chunked.
        chunk_size (int): The size of each chunk.
        overlap (int): The number of overlapping characters between chunks.

    Returns:
        List[str]: A list of text chunks.
    """

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        chunk = text[start: start + chunk_size]
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


def save_metadata(docs: List[Dict], path: str = METADATA_PATH):
    """
    Saves the metadata of documents to a JSON file.

    Args:
        docs (List[Dict]): The list of document metadata.
        path (str): The path to save the metadata JSON file.
    """
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(docs, f, ensure_ascii=False, indent=4)


def load_metadata(path: str = METADATA_PATH) -> List[Dict]:
    """
    Loads the metadata of documents from a JSON file.

    Args:
        path (str): The path to the metadata JSON file.
    Returns:
        List[Dict]: The list of document metadata.
    """
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)
