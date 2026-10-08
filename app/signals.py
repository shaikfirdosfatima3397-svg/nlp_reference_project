"""Readable attention and sentiment heuristics; these are not learned predictions."""
import re
from .extraction import is_negated


def detect_signals(text: str) -> dict:
    flags = []
    patterns = [
        ("Repeated contact", r"\b(?:(?:emailed|contacted|called|messaged|written)\s+(?:you\s+)?(?:twice|three times|again)|no response|already (?:emailed|contacted)|second (?:email|message))\b"),
        ("Urgent wording", r"\b(?:urgent|asap|immediately|today|emergency)\b"),
        ("Possible safety concern", r"\b(?:smoke|sparks|fire|injur\w*|burning|electric shock)\b"),
    ]
    for label, pattern in patterns:
        match = next((m for m in re.finditer(pattern, text, re.I) if not is_negated(text, m.start())), None)
        if match:
            flags.append({"label": label, "quote": match.group(), "start": match.start(), "end": match.end()})
    positive, negative = [], []
    for match in re.finditer(r"\b(?:happy|great|excellent|love|helpful|disappointed|angry|terrible|frustrated|awful|broken|damaged|late|failed)\b", text, re.I):
        word = match.group().lower()
        is_positive = word in {"happy", "great", "excellent", "love", "helpful"}
        if is_negated(text, match.start()):
            is_positive = not is_positive
        (positive if is_positive else negative).append(match.group())
    sentiment = "Negative cues" if len(negative) > len(positive) else "Positive cues" if len(positive) > len(negative) else "Mixed / no clear cues"
    priority = "High" if flags else "Normal"
    return {"flags": flags, "priority": priority, "sentiment": sentiment,
            "positive_cues": positive, "negative_cues": negative,
            "method": "Lexicon + attention rules"}
