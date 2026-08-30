# GOVERNANCE_INDEX.md — Vantara Governance and Reference Map

## Purpose

This pack constrains Codex so the implementation remains faithful to the supplied Vantara internship PRD, the audited architecture decisions, the supplied Online Retail II workbook, and the owner's gated/atomic delivery workflow.

Detailed governance exists to reduce agent improvisation, not to make the runtime architecture unnecessarily complex.

## Mutable vs immutable

### Mutable state — Codex may edit

- `/STATUS.md`
- `/NEXT_STEPS.md`

### Immutable — Codex must never edit

- `/AGENTS.md`
- `/governance/REFERENCE_LOCK.json`
- `/governance/source/*`
- `/governance/tools/*`
- every file under `/governance/reference/`

## Mandatory reading order for Step 00

1. `/AGENTS.md`
2. this file
3. `PROJECT.md`
4. `PRD_BASELINE.md`
5. `REQUIREMENTS.md`
6. `DECISIONS_LOCK.md`
7. `ARCHITECTURE.md`
8. `SOURCE_MANIFEST.md`
9. `DATASET_MANIFEST.md`
10. `DATA_CONTRACT.md`
11. `FEATURE_CONTRACT.md`
12. `MODELING_SPEC.md`
13. `DATABASE_SPEC.md`
14. `BACKEND_SPEC.md`
15. `FRONTEND_SPEC.md`
16. `CODE_STANDARDS.md`
17. `FILE_HANDLING.md`
18. `SECURITY_REPRODUCIBILITY.md`
19. `TESTING_VALIDATION.md`
20. `ACCEPTANCE_MATRIX.md`
21. `PRD_TRACEABILITY.md`
22. `GIT_WORKFLOW.md`
23. `REPORTING_PROTOCOL.md`
24. `ROADMAP.md`
25. `STEP_REPORT_TEMPLATE.md`
26. `/STATUS.md`
27. `/NEXT_STEPS.md`
28. the source PRD itself at `/governance/source/Vantara_requirements.docx`

Then execute `python governance/tools/verify_reference_lock.py`.

## File map

| File | Purpose |
|---|---|
| `PROJECT.md` | Product identity, goals, non-goals, success philosophy |
| `PRD_BASELINE.md` | Structured faithful baseline of the supplied 17-page PRD |
| `REQUIREMENTS.md` | Actionable functional/non-functional requirement contract |
| `DECISIONS_LOCK.md` | Audited choices where the PRD allowed alternatives or left gaps |
| `ARCHITECTURE.md` | End-to-end technical ownership and data/service boundaries |
| `SOURCE_MANIFEST.md` | Exact supplied PRD identity and source-authority rules |
| `DATASET_MANIFEST.md` | Exact supplied workbook identity and expected structure |
| `DATA_CONTRACT.md` | Raw/interim/processed semantics, cleaning, validation, cutoffs |
| `FEATURE_CONTRACT.md` | Required engineered features and leakage-safe definitions |
| `MODELING_SPEC.md` | Classical ML, DL, clustering, recommender, XAI, evaluation rules |
| `DATABASE_SPEC.md` | PostgreSQL schema responsibilities and persistence rules |
| `BACKEND_SPEC.md` | FastAPI endpoints, validation, startup, scoring boundaries |
| `FRONTEND_SPEC.md` | Streamlit pages, required views, API-only access, UX/performance |
| `CODE_STANDARDS.md` | Python/module/config/logging/error-handling rules |
| `FILE_HANDLING.md` | Source preservation, Git inclusion/ignore rules, artifact handling |
| `SECURITY_REPRODUCIBILITY.md` | Secrets, deterministic runs, environment/config rules |
| `TESTING_VALIDATION.md` | Test layers, leakage tests, lint/format/coverage/benchmark rules |
| `ACCEPTANCE_MATRIX.md` | Milestone gates and final PRD acceptance conditions |
| `PRD_TRACEABILITY.md` | PRD section → implementation → test/evidence mapping |
| `GIT_WORKFLOW.md` | Step commits/pushes, main-branch safety, remote reporting |
| `REPORTING_PROTOCOL.md` | Mandatory end-of-step report and approval behavior |
| `ROADMAP.md` | Locked M0–M3 / Step 00–09 build sequence |
| `STEP_REPORT_TEMPLATE.md` | Chat report format Codex must use after every step |

## Key principle

Codex should never need to invent a project architecture after Step 00. When a genuine unknown remains, it must stop and ask rather than create a new durable decision.
