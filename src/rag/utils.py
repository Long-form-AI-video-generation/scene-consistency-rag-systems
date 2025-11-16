import re
import html
from typing import List
from .config import CHUNK_SIZE, CHUNK_OVERLAP

def sanitize_text(s: str) -> str:
    # Basic sanitisation: unescape & remove control chars
    s = html.unescape(s)
    s = re.sub(r'[\x00-\x09\x0b-\x1f\x7f]+', ' ', s)
    return s.strip()

def tokenize_for_bm25(text: str, stopwords: set):
    # lower, remove punctuation (simple), split
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    tokens = text.split()
    return [t for t in tokens if t and t not in stopwords]


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


