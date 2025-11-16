
from src.rag.utils import sanitize_text, tokenize_for_bm25, clean_file



def test_sanitize_text():
    assert sanitize_text("Hello &amp; World") == "Hello & World"
    assert sanitize_text("Bad\x00Char") == "Bad Char"

def test_tokenize_for_bm25():
    stopwords = {"the", "and"}
    tokens = tokenize_for_bm25("The quick, brown fox!", stopwords)
    assert tokens == ["quick", "brown", "fox"]

def test_clean_file():
    text = "Line1\nLine2   extra   spaces"
    assert clean_file(text) == "Line1 Line2 extra spaces"
