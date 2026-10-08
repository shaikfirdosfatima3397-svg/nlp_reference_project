"""Honest evaluation on checked-in synthetic holdout data, with per-case results."""
import csv
import json
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from .extraction import extract

ROOT = Path(__file__).resolve().parents[1]


def evaluate(pipeline):
    with (ROOT / "data/test.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    labels = list(pipeline.classifier.pipeline.named_steps["classifier"].classes_)
    actual, predicted, accepted = [], [], []
    cases = []
    for row in rows:
        result = pipeline.classifier.predict(row["text"])
        actual.append(row["label"])
        predicted.append(result["suggested_category"])
        if not result["needs_review"]:
            accepted.append(row["label"] == result["category"])
        cases.append({"text": row["text"], "actual": row["label"], "predicted": result["category"],
                      "suggested": result["suggested_category"], "confidence": result["confidence"]})
    report = classification_report(actual, predicted, labels=labels, output_dict=True, zero_division=0)
    extraction_cases = json.loads((ROOT / "data/extraction_test.json").read_text(encoding="utf-8"))
    tp = fp = fn = 0
    for case in extraction_cases:
        expected = {(s["type"], s["value"]) for s in case["entities"]}
        result = extract(case["text"])
        found = {(kind, value) for kind, values in result["fields"].items() for value in values}
        tp += len(expected & found)
        fp += len(found - expected)
        fn += len(expected - found)
    retrieval_cases = json.loads((ROOT / "data/retrieval_test.json").read_text(encoding="utf-8"))
    retrieval_results = []
    for case in retrieval_cases:
        matches = pipeline.retriever.search(case["query"])["matches"]
        ids = [p["id"] for p in matches]
        retrieval_results.append({**case, "retrieved": ids, "hit": case["policy_id"] in ids})
    return {
        "dataset": "Small synthetic teaching holdout; not a real-world benchmark",
        "train_count": pipeline.classifier.training_count, "test_count": len(rows),
        "accuracy": round(float(accuracy_score(actual, predicted)), 4),
        "macro_f1": round(report["macro avg"]["f1-score"], 4),
        "coverage": round(len(accepted) / len(rows), 4),
        "accepted_accuracy": round(sum(accepted) / len(accepted), 4) if accepted else None,
        "per_class": {label: report[label] for label in labels},
        "confusion_matrix": confusion_matrix(actual, predicted, labels=labels).tolist(), "labels": labels,
        "classification_cases": cases,
        "extraction": {"precision": tp / max(1, tp + fp), "recall": tp / max(1, tp + fn),
                       "f1": 2 * tp / max(1, 2 * tp + fp + fn), "cases": len(extraction_cases), "tp": tp, "fp": fp, "fn": fn},
        "retrieval": {"recall_at_3": sum(c["hit"] for c in retrieval_results) / len(retrieval_results),
                      "cases": retrieval_results, "mode": pipeline.retriever.mode},
        "drafting": "Human grounding checklist in docs; no automated factuality score is claimed.",
    }
