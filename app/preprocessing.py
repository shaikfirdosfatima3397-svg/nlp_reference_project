"""Normalize model input while keeping the original ticket for evidence spans."""
import re
import unicodedata


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    text = text.replace("’", "'")
    text = re.sub(r"\b[\w.+-]+@[\w.-]+\.[a-z]{2,}\b", " emailtoken ", text)
    text = re.sub(r"\b(?:order\s*(?:id|number|no\.?|#)?\s*[:#-]?\s*)[a-z]*\d[\w-]*", " ordertoken ", text)
    return re.sub(r"\s+", " ", text).strip()
