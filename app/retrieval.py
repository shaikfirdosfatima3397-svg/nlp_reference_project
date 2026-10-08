"""Sparse retrieval by default; optional dense retrieval explicitly reports its mode."""
import json
import os
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from .preprocessing import normalize

ROOT = Path(__file__).resolve().parents[1]


class PolicyRetriever:
    def __init__(self):
        self.policies = json.loads((ROOT / "data/policies.json").read_text(encoding="utf-8"))
        self.texts = [p["title"] + " " + p["text"] + " " + p["keywords"] for p in self.policies]
        self.vectorizer = TfidfVectorizer(preprocessor=normalize, ngram_range=(1, 2), sublinear_tf=True)
        self.matrix = self.vectorizer.fit_transform(self.texts)
        self.mode, self.warning, self.model, self.embeddings = "tfidf", None, None, None
        if os.getenv("SUPPORT_RETRIEVAL", "tfidf") == "embeddings":
            try:
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(os.getenv("SUPPORT_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))
                self.embeddings = self.model.encode(self.texts, normalize_embeddings=True)
                self.mode = "embeddings"
            except Exception as error:
                self.warning = f"Embedding model unavailable ({type(error).__name__}); using TF-IDF."

    def search(self, query: str, top_k: int = 3) -> dict:
        if self.mode == "embeddings":
            vector = self.model.encode([query], normalize_embeddings=True)
            scores = (vector @ self.embeddings.T)[0]
            threshold = 0.35
        else:
            scores = cosine_similarity(self.vectorizer.transform([query]), self.matrix)[0]
            threshold = 0.08
        matches = []
        for index in sorted(range(len(scores)), key=lambda i: -scores[i])[:top_k]:
            if float(scores[index]) >= threshold:
                matches.append({**self.policies[index], "score": round(float(scores[index]), 4)})
        return {"matches": matches, "mode": self.mode, "warning": self.warning,
                "threshold": threshold, "method": "Sentence embeddings + cosine similarity" if self.mode == "embeddings" else "TF-IDF + cosine similarity"}
