# Verification record

Baseline verified on 7 October 2026 using Python 3.12 on Windows. This record describes the delivered local project, not a production deployment.

## Automated regression checks

16 unittest tests passed. They cover learned classification, unknown-topic review, extraction offsets and false identifiers, simple action and sentiment negation, avoiding duplicate-payment/repeated-contact confusion, category-matched citations, clarification without evidence, offline injection behavior, mocked LLM configuration and citation rejection, train/test separation, SQLite persistence and SQL parameters, API validation, and static routes.

The suite can be reproduced with `python -m unittest discover -s tests -v` after installing requirements.txt.

## Browser verification

Headless Chrome checks passed at 1440px desktop and 390px mobile widths. The checked workflow included exactly one set of six seed tickets; normal and uncertain workbench analysis; policy filtering; evaluation rendering; ticket creation; HTML-like input rendered as plain text; editing, saving and recovering a draft after reload; status and search filters; no document-level mobile overflow; all 19 handbook lessons; handbook title search; and no browser JavaScript errors.

The browser run used an isolated preview database, separate from the delivered project's default database. Browser automation helpers and scratch screenshots are not required by the student project. Manual reproduction steps are in lesson 17.

## Held-out baseline metrics

- 120 synthetic training tickets and 40 synthetic test tickets.
- Candidate accuracy 0.9000; macro-F1 0.8971.
- Automatic-routing coverage 0.6500; accepted accuracy 1.0000 on 26 accepted tickets.
- Entity precision/recall/F1 1.0000 on eight fixtures containing 13 distinct type/value matches.
- Policy Recall@3 1.0000 on twelve single-relevant-policy queries, in TF-IDF mode.

See evaluation-baseline.json for per-class results, every test prediction and retrieval case. The small synthetic datasets do not establish performance on real customer messages. Action, priority and sentiment rules have narrow regression cases, not a representative labeled benchmark. No automated factuality score is claimed for drafts.

## Documentation review

The 21-page PDF was rendered to images and every page was visually inspected. Browser and Markdown versions come from the same student-handbook.md source. The guide includes module concepts, inputs/outputs, code examples, limitations, exercises, a project rubric and official further reading.

## Optional extensions

The embedding adapter is implemented, but no Sentence Transformers model was downloaded or benchmarked in baseline verification. The LLM adapter's rejection logic has mocked tests; live provider compatibility, factuality and latency remain unverified without configured credentials. The offline baseline works without either extension.
