# Support Studio

A complete local NLP teaching project: a polished customer-support workspace, a learned ticket classifier, span extraction, explainable attention rules, policy retrieval, editable source-linked drafts, SQLite persistence, evaluation, tests, and a student handbook.

## Start here

Requires Python 3.11 or 3.12 and internet access once to install packages. No Node build, GPU, paid API, or model download is needed for the baseline.

```powershell
cd support-studio
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe run.py
```

macOS or Linux:

```sh
cd support-studio
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python run.py
```

Open http://127.0.0.1:8000. To use a different port: `python run.py --port 8001` (use your virtual environment's Python). Stop with Ctrl+C.

Use **Student handbook** in the sidebar for the illustrated, searchable module guide. A printable PDF and Markdown version are in `docs/`. API documentation is at `/docs` while the server is running.

## What works

- Inbox with six illustrative conversations; search and status filters.
- Create and analyze a ticket; edit and persist drafts and ticket status.
- Workbench for arbitrary English messages, with evidence and highlighted spans.
- Five learned categories plus a human-review outcome.
- Order IDs, emails, amounts, literal date strings and possible action cues.
- Rule-based repeated-contact, urgent-wording and safety flags; lexicon sentiment cues.
- Seven fictional policy passages, TF-IDF cosine search, optional dense search.
- Offline reply templates with policy IDs and clarification on uncertainty.
- Optional, explicitly selected external LLM reply adapter with source-ID checks.
- Evaluation dashboard with classification, coverage, entity and retrieval metrics.
- Responsive keyboard-accessible UI; no frontend build tools or external font dependencies.

## Architecture

```text
web/index.html + styles.css + app.js
              | HTTP JSON
         app/main.py (FastAPI)
              | orchestrates
         app/pipeline.py
         /     |      |       \
classifier  extraction  signals  retrieval -> drafting
              |                         |
           storage.py (SQLite)      policies.json
```

The baseline uses TF-IDF + logistic regression for classification. Retrieval is lexical TF-IDF, not transformer semantic search. Extraction and attention signals are rules, not trained NER or sentiment models. The offline draft is a template, not LLM generation. The UI and guide label these distinctions.

## Verify and evaluate

Run inside the virtual environment:

```sh
python -m unittest discover -s tests -v
python evaluate.py
```

`docs/evaluation-baseline.json` is the recorded baseline run. Training: 120 synthetic tickets. Holdout: 40 separately written tickets. Candidate accuracy: 90%; macro-F1: 0.8971; auto-routing coverage: 65%. Accepted accuracy was 100% on just 26 examples. Extraction scored 100% on eight narrow fixtures and retrieval Recall@3 was 100% on twelve queries. These small synthetic results are not claims of production performance.

## Optional semantic search

```powershell
.venv\Scripts\python.exe -m pip install -r requirements-advanced.txt
$env:SUPPORT_RETRIEVAL = 'embeddings'
$env:SUPPORT_EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
.venv\Scripts\python.exe run.py
```

On macOS/Linux, prefix the run command with `SUPPORT_RETRIEVAL=embeddings`. The first load downloads a pretrained model; subsequent cached loads can run offline. A failed model load reports a warning and uses the lexical baseline. Re-run evaluation because similarity thresholds differ by mode. The optional model path is implemented but was not downloaded or exercised in the delivered baseline verification.

## Optional LLM drafting

Set `SUPPORT_LLM_URL`, `SUPPORT_LLM_MODEL`, and `SUPPORT_LLM_KEY` in the server environment, then restart. `.env.example` documents the variables; Python does **not** load `.env` automatically. The adapter expects an OpenAI-compatible Chat Completions endpoint that accepts JSON response format. Provider compatibility must be tested with your chosen model. Never put keys in frontend files or commit them.

The workbench checkbox sends that message, extracted fields, and retrieved policies to the configured external provider. It is off by default. Inbox analysis uses offline drafts. Live-provider behavior is unverified without credentials; adapter rejection paths have mocked tests. Source validation is not hallucination proof. Drafts require human review.

## Data and persistence

All included messages, customers, emails and policies are fictional. See `data/README.md`. The train and test split is fixed and checked in; feature fitting only uses training data. Policies include transparent manually curated keyword aliases.

State lives in `runtime/support.db` by default. Override `SUPPORT_DB` for a different database. A new empty database seeds the six demo tickets. Changes survive server restarts. To start fresh, stop the server and choose a new database path. No customer mail, SMS or payment service is connected.

## Student learning path

1. Run the demo and follow Maya's damaged-screen ticket.
2. Read preprocessing and TF-IDF modules in the handbook.
3. Change a message in the workbench and inspect scores and spans.
4. Run evaluation and explain the four candidate errors.
5. Add independently labeled examples; preserve a final untouched test set.
6. Compare lexical retrieval with embeddings as an advanced experiment.

## Practical limits

Local single-user English teaching app; no authentication, production deployment, order database lookup, real email sending, or automatic eligibility decision. Single-label classification cannot fully represent multi-issue tickets. Negation rules are intentionally limited; action matches are possible intents, not verified requests. Dates are extracted without validity or locale interpretation. Resolve tickets only after reviewing evidence. Do not expose this server to the public internet.

## Files

`app/`: documented NLP and API modules. `web/`: UI. `data/`: train/test and policies. `tests/`: regression suite. `docs/`: handbook, exercises and recorded evaluation. `run.py`: server launcher. `evaluate.py`: reproducible metrics. `requirements-advanced.txt`: optional transformer dependency.

This project is licensed under MIT; see `LICENSE`. Third-party dependencies retain their own licenses.
