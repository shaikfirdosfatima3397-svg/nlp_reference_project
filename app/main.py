"""Local FastAPI application. Start using: python run.py."""
import json
import os
from functools import lru_cache
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import Literal
from .pipeline import SupportPipeline
from .storage import TicketStore
from .evaluation import evaluate
from .drafting import llm_configured

ROOT = Path(__file__).resolve().parents[1]


@asynccontextmanager
async def lifespan(application):
    services()  # Warm once before concurrent browser requests can seed demo data.
    yield


app = FastAPI(title="Support Studio", version="1.0.0", lifespan=lifespan)


@lru_cache
def services():
    pipeline = SupportPipeline()
    store = TicketStore(os.getenv("SUPPORT_DB", str(ROOT / "runtime/support.db")))
    if not store.list():
        examples = json.loads((ROOT / "data/demo_tickets.json").read_text(encoding="utf-8"))
        for ticket in reversed(examples):
            store.create(**ticket, analysis=pipeline.analyze(ticket["message"]))
    return pipeline, store


class AnalyzeInput(BaseModel):
    text: str = Field(min_length=3, max_length=12000)
    use_llm: bool = False


class TicketInput(BaseModel):
    customer: str = Field(min_length=1, max_length=80)
    subject: str = Field(min_length=1, max_length=120)
    message: str = Field(min_length=3, max_length=12000)


class TicketUpdate(BaseModel):
    status: Literal["open", "in_review", "resolved"]
    draft: str = Field(max_length=15000)


def meaningful(value):
    if not value.strip():
        raise HTTPException(422, "Text cannot be blank")
    return value.strip()


@app.get("/api/health")
def health():
    pipeline, _ = services()
    return {"status": "ok", "retrieval_mode": pipeline.retriever.mode, "retrieval_warning": pipeline.retriever.warning,
            "llm_configured": llm_configured(), "training_count": pipeline.classifier.training_count}


@app.post("/api/analyze")
def analyze(body: AnalyzeInput):
    try:
        return services()[0].analyze(meaningful(body.text), body.use_llm)
    except ValueError as error:
        raise HTTPException(400, str(error)) from error


@app.get("/api/tickets")
def tickets():
    return services()[1].list()


@app.post("/api/tickets", status_code=201)
def create_ticket(body: TicketInput):
    customer, subject, message = map(meaningful, (body.customer, body.subject, body.message))
    analysis = services()[0].analyze(message)
    return services()[1].create(customer, subject, message, analysis)


@app.patch("/api/tickets/{ticket_id}")
def update_ticket(ticket_id: int, body: TicketUpdate):
    ticket = services()[1].update(ticket_id, body.status, body.draft)
    if not ticket:
        raise HTTPException(404, "Ticket not found")
    return ticket


@app.post("/api/tickets/{ticket_id}/analyze")
def analyze_ticket(ticket_id: int):
    pipeline, store = services()
    ticket = store.get(ticket_id)
    if not ticket:
        raise HTTPException(404, "Ticket not found")
    return store.save_analysis(ticket_id, pipeline.analyze(ticket["message"]))


@app.get("/api/policies")
def policies():
    return services()[0].retriever.policies


@app.get("/api/evaluation")
def evaluation():
    return evaluate(services()[0])


@app.get("/")
def index():
    return FileResponse(ROOT / "web/index.html")


app.mount("/assets", StaticFiles(directory=ROOT / "web"), name="assets")
app.mount("/guide", StaticFiles(directory=ROOT / "docs", html=True), name="guide")
