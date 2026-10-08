"""A learned TF-IDF + multinomial logistic regression baseline, with explanations."""
import csv
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from .preprocessing import normalize

ROOT = Path(__file__).resolve().parents[1]


class TicketClassifier:
    def __init__(self):
        with (ROOT / "data/train.csv").open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(preprocessor=normalize, ngram_range=(1, 2), sublinear_tf=True)),
            ("classifier", LogisticRegression(C=8.0, max_iter=1000, random_state=42)),
        ])
        self.pipeline.fit([r["text"] for r in rows], [r["label"] for r in rows])
        self.training_count = len(rows)

    def predict(self, text: str) -> dict:
        vectorizer = self.pipeline.named_steps["tfidf"]
        model = self.pipeline.named_steps["classifier"]
        vector = vectorizer.transform([text])
        probabilities = model.predict_proba(vector)[0]
        best = int(np.argmax(probabilities))
        label = str(model.classes_[best])
        confidence = float(probabilities[best])
        # These are deliberately transparent heuristics, not calibrated guarantees.
        words = set(vectorizer.build_analyzer()(text))
        unigrams = {w for w in words if " " not in w}
        known = {w for w in unigrams if w in vectorizer.vocabulary_}
        coverage = len(known) / max(1, len(unigrams))
        review = confidence < 0.50 or coverage < 0.25 or vector.nnz == 0
        terms = vectorizer.get_feature_names_out()
        contributions = vector.toarray()[0] * model.coef_[best]
        indices = np.argsort(contributions)[::-1]
        evidence = [{"term": str(terms[i]), "contribution": round(float(contributions[i]), 4)}
                    for i in indices if contributions[i] > 0][:6]
        return {
            "category": "needs_review" if review else label,
            "suggested_category": label,
            "confidence": round(confidence, 4),
            "coverage": round(coverage, 4),
            "needs_review": review,
            "scores": [{"category": str(c), "score": round(float(p), 4)}
                       for c, p in sorted(zip(model.classes_, probabilities), key=lambda x: -x[1])],
            "evidence": evidence,
            "method": "TF-IDF + logistic regression",
        }
