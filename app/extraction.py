"""Rule-based extraction. All spans use offsets in the untouched original text."""
import re

PATTERNS = {
    "order_id": r"\border\s*(?:id|number|no\.?)?\s*[:#-]?\s*([A-Za-z0-9]+(?:-[A-Za-z0-9]+)*)\b",
    "email": r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b",
    "amount": r"(?:[$₹£€]\s*\d[\d,]*(?:\.\d{1,2})?|\b(?:USD|INR|GBP|EUR|Rs\.?)\s*\d[\d,]*(?:\.\d{1,2})?)",
    "date": r"\b(?:\d{4}-\d{2}-\d{2}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{1,2}\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)(?:\s+\d{4})?)\b",
}
ACTION_PATTERNS = {
    "Replacement": r"\b(?:replace(?:ment)?|exchange)\b",
    "Refund": r"\b(?:refund|money back|reimburse(?:ment)?)\b",
    "Cancellation": r"\b(?:cancel|cancellation)\b",
    "Delivery update": r"\b(?:track(?:ing)?|where is|delivery update|when\b[^.!?\n]{0,60}\barriv(?:e|ing)|(?:has|have) not arrived)\b",
    "Account access": r"\b(?:reset|unlock|log ?in|sign ?in|password)\b",
}


def is_negated(text: str, start: int) -> bool:
    """A small local window; deliberately not a full syntactic negation model."""
    before = re.findall(r"\b[\w']+\b", text[:start].lower().replace("’", "'"))[-4:]
    return any(w in {"not", "no", "never", "don't", "dont", "without"} for w in before)


def extract(text: str) -> dict:
    spans = []
    fields = {key: [] for key in PATTERNS}
    for key, pattern in PATTERNS.items():
        for match in re.finditer(pattern, text, flags=re.I):
            group = 1 if key == "order_id" else 0
            value = match.group(group)
            if key == "order_id" and not any(c.isdigit() for c in value):
                continue  # Never mistake "order arrived" for an order identifier.
            fields[key].append(value)
            spans.append({"type": key, "value": value, "start": match.start(group), "end": match.end(group)})
    actions, action_spans = [], []
    for action, pattern in ACTION_PATTERNS.items():
        for match in re.finditer(pattern, text, flags=re.I):
            if not is_negated(text, match.start()):
                if action not in actions:
                    actions.append(action)
                action_spans.append({"type": "action", "value": match.group(), "start": match.start(), "end": match.end()})
    return {"fields": fields, "spans": sorted(spans + action_spans, key=lambda s: (s["start"], s["end"])),
            "actions": actions, "method": "Regular expressions + local negation rules"}
