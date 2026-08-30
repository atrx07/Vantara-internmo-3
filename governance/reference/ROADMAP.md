# ROADMAP.md — Locked Fast Vantara Build Sequence

## Roadmap contract

This is the full implementation sequence. It intentionally compresses the PRD's 20-day plan into **3 build milestones / 9 implementation steps** plus Step 00 governance scan.

Do not combine steps to save a chat turn. The stop/approval boundary is part of the build safety model.

---

# M0 — Governance Ready

## STEP 00 — PRD + Governance + Repository Scan

### Purpose

Understand the complete project before any code is written and bind the owner-provided remote safely.

### Required work

- read all governance/reference files and source PRD;
- verify reference lock;
- inspect dataset manifest and supplied workbook read-only;
- inspect repository tree/current code if any;
- inspect Git status/history/branch/ignore state;
- receive remote URL in chat;
- add/verify/fetch `origin` safely;
- confirm `main` strategy or report conflict;
- compare PRD baseline/traceability against source;
- identify blockers only; do not redesign.

### Allowed file changes

Only `STATUS.md` and `NEXT_STEPS.md`.

### Git

No push.

### Exit

Report full scan and proposed Step 01, then STOP for approval.

---

# M1 — Data Ready

## STEP 01 — Bootstrap + Raw Ingestion

### Build

- scaffold PRD-required repository structure;
- create Python 3.11 venv workflow and dependency manifests;
- resolve/install compatible approved dependencies and pin exact versions;
- configure Ruff, Black, Pytest, logging, YAML config, `.gitignore`, `.env.example`;
- place/copy canonical workbook to `data/raw/` without mutation if not already present;
- create `data/raw/README.md` with source/hash instructions;
- implement both-sheet loader, canonical names/types, chronological merge;
- implement initial Pandera/schema/hash/date/null validation;
- create unit tests for ingestion/validation;
- ensure governance pack remains unchanged.

### Validate

- governance lock;
- dataset hash/sheets/headers;
- imports/environment;
- loader/validation tests;
- Ruff + Black check on created code;
- clean Git diff without raw dataset/secrets.

### Push

Green atomic commit(s) to `main`, then report all commits/heads and STOP.

---

## STEP 02 — Cleaning + Snapshot + Feature Pipeline

### Build

- cleaning/quality flags: missing IDs, exact duplicates, returns/cancellations, price issues, admin StockCodes, description canonicalization, outliers;
- interim Parquet output;
- CustomerSnapshot abstraction;
- derived observation end and canonical cutoffs;
- 90-day churn label;
- 180-day CLV target;
- all required feature families;
- training-only product-frequency encoding;
- markdown affinity proxy;
- product taxonomy + category affinity;
- one persisted customer split 70/15/15 seed 42;
- model preprocessing contracts/scaler separation where appropriate;
- comprehensive leakage tests.

### Validate

- all data/feature/leakage tests;
- deterministic rerun on sample/full data as practical;
- no future data affects historical features;
- split disjointness;
- Ruff/Black/Pytest.

### Push

Green atomic commit(s), report, STOP.

---

## STEP 03 — EDA + Data Freeze

### Build

- required `01_eda.ipynb`, `02_feature_engineering.ipynb` consumers of `src/` logic;
- spend/frequency/recency summaries;
- target/class analysis;
- correlation + VIF;
- country/seasonality analysis;
- outlier review;
- written hypotheses;
- final churn feature schema decision based on evidence without changing locked required feature-table content;
- `python -m src.pipeline` end-to-end single-command path;
- final M1 processed/evidence outputs.

### Validate

- end-to-end raw -> processed command;
- notebooks execute/smoke without owning unique production logic;
- all M1 tests + leakage tests;
- M1 acceptance checklist.

### Push

Green atomic commit(s), report M1 complete/waiting, STOP.

---

# M2 — Intelligence Ready

## STEP 04 — Classical ML + CLV + Segmentation + Product Intelligence

### Build

- Logistic Regression;
- Decision Tree;
- Random Forest;
- XGBoost;
- LightGBM;
- SVM;
- Ridge + XGBRegressor CLV;
- K-Means + GMM segmentation/profile/PCA data;
- next-purchase-category LightGBM + baseline;
- item-to-item recommender + evaluation;
- training-only CV/tuning;
- MLflow local logging;
- `03_model_experiments.ipynb` as consumer/analysis notebook;
- generated comparison/evaluation evidence.

### Important

Do not access final held-out test for model selection.

### Validate

- model training smoke/full runs as authorized by compute/time;
- CV/logging evidence;
- required metrics;
- artifact reload smoke;
- tests/lint/format.

### Push

Green atomic commit(s), report, STOP.

---

## STEP 05 — Deep Learning

### Build

- PyTorch ANN on exact final churn feature schema;
- rolling-snapshot LSTM for 30-day purchase probability;
- autoencoder anomaly model;
- early stopping/loss curves;
- grouped LSTM CV;
- serialized DL artifacts + metadata;
- metric logging.

### Validate

- training/reload inference smoke;
- ANN required churn metrics;
- LSTM required binary metrics;
- autoencoder reconstruction evidence;
- customer-group isolation;
- tests/lint/format.

### Push

Green atomic commit(s), report, STOP.

---

## STEP 06 — Model Freeze + XAI + Final Evaluation

### Build

- generate consolidated validation comparison across classical churn + ANN;
- choose production churn model using locked selection rule;
- freeze threshold/model/hyperparameters/feature schema;
- generate SHAP global + low/borderline/high local outputs;
- generate one LIME comparison;
- generate PDPs;
- plain-language explanation layer;
- other model explanation adapters;
- with explicit Step 06 approval and freeze recorded, run held-out final test exactly once;
- generate immutable-style final metrics evidence/report outputs;
- create serving artifact inventory.

### Validate

- XAI artifacts load/render;
- comparison generated from logs;
- final metrics complete;
- no test-set tuning loop;
- M2 acceptance checklist;
- all relevant tests/lint/format.

### Push

Green atomic commit(s), report M2 complete/waiting, STOP.

---

# M3 — Product Ready

## STEP 07 — PostgreSQL + FastAPI

### Build

- PostgreSQL models/schema;
- Alembic migrations;
- deterministic serving data load/init path;
- FastAPI app/lifespan artifact loading;
- health, metadata, single prediction and batch CSV endpoints;
- approved supporting customer/segment/recommendation/analytics endpoints needed by dashboard;
- Pydantic validation;
- persistence of predictions/segments;
- API tests.

### Validate

- fresh DB migration;
- API test suite;
- real artifact loading/scoring smoke;
- DB persistence smoke;
- no secrets/model reload per request;
- tests/lint/format.

### Push

Green atomic commit(s), report, STOP.

---

## STEP 08 — Streamlit Dashboard + Reports

### Build

- executive overview;
- segment filters/view;
- churn priority leaderboard;
- customer explorer + XAI;
- revenue trend + Holt-Winters forecast;
- batch CSV upload;
- CSV/PDF downloads;
- model insights;
- API-only frontend data access;
- caching/error states appropriate to performance.

### Validate

- every PRD dashboard view/filter;
- customer lookup and XAI;
- batch/report flows;
- frontend does not load models directly;
- Streamlit smoke/integration tests where practical;
- tests/lint/format.

### Push

Green atomic commit(s), report, STOP.

---

## STEP 09 — Docker + Benchmarks + Docs + Final Acceptance

### Build/finalize

- Dockerfile + `docker-compose.yml` for postgres/api/dashboard;
- environment-based configuration;
- API p95 benchmark;
- dashboard segment-view benchmark;
- coverage >=70% and complete test suite;
- architecture, ER and workflow diagrams;
- README clean-clone/setup/run/API/dashboard instructions;
- final report PDF with actual results, production choice justification, limitations, future enhancements;
- 2–4 page math appendix tied to actual implementation;
- coverage report;
- required model artifact size/distribution audit;
- recorded walkthrough checklist;
- final PRD traceability/acceptance review through evidence, without editing immutable governance.

### Validate

Run full final gates including:

```bash
ruff check .
black --check .
pytest --cov=src --cov=api --cov-fail-under=70
```

and clean Docker Compose startup. Exercise required API/dashboard paths.

### Push

Only after green final validation. Report all commits/pushed heads, final HEAD, metrics, acceptance status and any PRD-permitted documented metric misses. STOP for owner acceptance.

Project is not `COMPLETE` until owner approves final result.
