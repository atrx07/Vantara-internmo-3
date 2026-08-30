# ACCEPTANCE_MATRIX.md — Vantara Milestone and Final Acceptance

## Status values

Use only:

- `NOT_STARTED`
- `IN_PROGRESS`
- `PASS`
- `FAIL`
- `BLOCKED`
- `N/A_PRD_ALLOWED`

## Milestone M1 — Data Ready

Before Step 03 is accepted:

- [ ] supplied workbook validated against manifest;
- [ ] both sheets loaded and chronological combined table produced;
- [ ] cleaning rules implemented/tested;
- [ ] required customer features implemented;
- [ ] 90-day churn labels implemented;
- [ ] 180-day CLV target implemented;
- [ ] product taxonomy/affinities implemented;
- [ ] one immutable customer split produced;
- [ ] leakage tests pass;
- [ ] EDA notebook contains required analyses/hypotheses;
- [ ] VIF/correlation evidence exists;
- [ ] `python -m src.pipeline` succeeds without notebook intervention;
- [ ] code-quality tests for M1 pass.

## Milestone M2 — Intelligence Ready

Before Step 06 is accepted:

- [ ] six required classical churn models trained/logged;
- [ ] ANN trained/logged on same final churn feature schema;
- [ ] Ridge + XGBRegressor CLV evaluation complete;
- [ ] K-Means + GMM complete with business profiles;
- [ ] next-category model/baseline complete;
- [ ] recommender + offline metrics complete;
- [ ] LSTM trained/evaluated with grouped snapshots;
- [ ] autoencoder trained/evaluated honestly;
- [ ] model comparison generated from logs;
- [ ] production churn model + threshold frozen on validation evidence;
- [ ] required SHAP/LIME/PDP/plain-language outputs generated;
- [ ] final held-out test executed once after freeze;
- [ ] final metrics artifact preserved;
- [ ] no leakage violation remains.

## Milestone M3 — Product Ready

Before Step 09 is accepted:

- [ ] PostgreSQL schema/migrations work from fresh DB;
- [ ] required FastAPI endpoints work and validate errors;
- [ ] scored predictions/segments persist;
- [ ] required Streamlit views work through API;
- [ ] batch CSV scoring works;
- [ ] downloadable CSV/PDF works;
- [ ] API p95 benchmark recorded (<400 ms target or documented miss);
- [ ] segment-view benchmark recorded (<3 s target or documented miss);
- [ ] Docker Compose brings DB/API/dashboard up cleanly;
- [ ] `src/` coverage >=70%; all leakage tests pass;
- [ ] Ruff/Black/tests pass;
- [ ] architecture, ER and workflow diagrams exist;
- [ ] README clean-clone/setup/run instructions verified;
- [ ] final report contains model comparison, limitations, future enhancements and 2–4 page math appendix tied to actual implementation;
- [ ] required serialized artifacts inventoried and distributable;
- [ ] recorded walkthrough checklist prepared.

## PRD success metrics

| Metric | Target | Final handling |
|---|---:|---|
| Churn held-out ROC-AUC | >=0.80 | must report actual |
| Churn recall | >=0.70 | must report actual |
| CLV test R² | >=0.60 | must report actual |
| Single prediction API p95 | <400 ms | benchmark actual |
| Segment dashboard load | <3 s | benchmark actual |
| Production pipeline scripted | 100% | no notebook-only production step |
| Source package coverage | >=70% | actual coverage report |

The PRD permits a success metric to be missed if the final report explicitly discusses the reason. Therefore a genuine metric miss does not authorize label leakage or evaluation manipulation.

## Final Definition of Done

Project status may become `COMPLETE` only after Step 09 and owner approval, with every hard acceptance item satisfied and any permitted metric miss documented with real evidence.
