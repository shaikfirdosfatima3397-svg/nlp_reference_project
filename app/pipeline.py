"""Orchestrates independent NLP modules without conflating their outputs."""
from .classifier import TicketClassifier
from .extraction import extract
from .signals import detect_signals
from .retrieval import PolicyRetriever
from .drafting import template_draft, llm_draft


class SupportPipeline:
    def __init__(self):
        self.classifier = TicketClassifier()
        self.retriever = PolicyRetriever()

    def analyze(self, text: str, use_llm: bool = False) -> dict:
        classification = self.classifier.predict(text)
        extraction = extract(text)
        signals = detect_signals(text)
        retrieval = self.retriever.search(text)
        draft = (llm_draft(text, classification, extraction, retrieval) if use_llm
                 else template_draft(classification, extraction, retrieval))
        return {"classification": classification, "extraction": extraction, "signals": signals,
                "retrieval": retrieval, "draft": draft}
