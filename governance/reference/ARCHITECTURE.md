# ARCHITECTURE.md — Vantara Technical Architecture

## 1. Architectural style

Vantara is a local, batch-oriented ML analytics platform with four PRD-aligned layers:

1. **Data layer** — immutable raw workbook + PostgreSQL persisted serving data.
2. **Processing layer** — Pandas/NumPy loading, cleaning, snapshot creation, feature engineering and validation.
3. **Modeling layer** — classical ML, PyTorch DL, clustering, recommender, explainability and serialized artifacts.
4. **Serving layer** — FastAPI + PostgreSQL + Streamlit.

No streaming/event bus, auth service, cloud dependency or unnecessary microservice is part of this build.

## 2. End-to-end flow

```text
Online Retail II XLSX (read-only)
        |
        v
raw loader + schema validation
        |
        v
chronological standardized transactions
        |
        v
cleaning + quality flags + product metadata
        |
        +----------------------+
        |                      |
        v                      v
product taxonomy        CustomerSnapshot engine
        |                      |
        v                      v
product affinity       customer feature tables
        |                      |
        +----------+-----------+
                   |
       +-----------+-----------------------------+
       |           |            |                |
       v           v            v                v
 churn/ANN        CLV        segmentation     recommender
       |                         |
       v                         v
    SHAP/LIME                profiles/PCA

rolling purchase snapshots -> LSTM -> next-purchase probability
behavior features          -> autoencoder -> anomaly score
customer features          -> LightGBM -> next-purchase category

model/preprocessor/taxonomy artifacts
                   |
                   v
             FastAPI service
                   |
          +--------+--------+
          |                 |
          v                 v
      PostgreSQL       scoring artifacts
          |
          v
      Streamlit dashboard (API consumer)
```

## 3. Repository layout

Preserve the PRD-required paths and add only the approved support folders:

```text
customer-behavior-prediction/
├── AGENTS.md
├── STATUS.md
├── NEXT_STEPS.md
├── governance/
│   ├── REFERENCE_LOCK.json
│   ├── reference/
│   ├── source/
│   └── tools/
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_experiments.ipynb
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── segmentation/
│   ├── explainability/
│   ├── recommendation/
│   └── utils/
├── api/
│   ├── main.py
│   ├── routers/
│   └── schemas/
├── frontend/
│   └── dashboard.py
├── models_artifacts/
├── config/
│   └── config.yaml
├── tests/
├── docs/
│   ├── architecture_diagram.png
│   ├── er_diagram.png
│   ├── workflow_diagram.png
│   └── final_report.pdf
├── reports/
├── scripts/
├── migrations/
├── docker-compose.yml
├── Dockerfile
├── requirements.in
├── requirements.txt
└── README.md
```

Do not create parallel app roots such as `backend2/`, `new_src/`, `final_app/`, `v2/`, or duplicated model directories.

## 4. Module ownership

### `src/data/`

Owns:

- workbook loading;
- canonical column/type normalization;
- validation;
- cleaning and quality flags;
- product description canonicalization;
- source/interim persistence.

Must not own model training.

### `src/features/`

Owns:

- CustomerSnapshot abstraction;
- target-window-safe feature builders;
- churn/CLV label builders;
- RFM and all required business features;
- fitted feature-preprocessing contracts;
- split generation and persistence.

### `src/models/`

Owns:

- classical churn training/evaluation;
- CLV regression;
- next-category model;
- PyTorch ANN/LSTM/autoencoder;
- experiment logging;
- production model selection;
- final held-out test command.

### `src/segmentation/`

Owns K-Means/GMM training, profiling, labels, metrics and PCA visualization data.

### `src/recommendation/`

Owns item interaction matrix, similarity, recommendation generation, fallback logic and offline recommender evaluation.

### `src/explainability/`

Owns SHAP/LIME/PDP generation, deterministic human-readable explanations and non-churn explanation adapters.

### `src/utils/`

Owns configuration loading, seed setup, paths, artifact metadata, structured logging and truly shared utilities.

No business feature implementation should migrate here merely for convenience.

### `api/`

Owns HTTP boundary, request/response schemas, orchestration, artifact loading and persistence service integration. It calls reusable `src/` logic.

### `frontend/`

Owns presentation and user interaction only. It calls FastAPI. It does not import model artifacts or implement feature engineering.

## 5. Data ownership

- raw workbook: immutable local source;
- interim Parquet: cleaned transaction-level truth;
- processed Parquet: final customer/snapshot/model-ready data;
- PostgreSQL: serving/persisted prediction state, not training source-of-truth replacement;
- MLflow: development experiment evidence;
- `models_artifacts/`: serialized evaluation/serving artifacts and metadata;
- `reports/`: generated tables/metrics/plots intended as evidence;
- `docs/`: final human documentation and diagrams.

## 6. Artifact contract

Each serialized model family must have model + preprocessor/metadata required for deterministic inference. Production artifacts must include at least:

```text
models_artifacts/
├── churn/
├── clv/
├── next_purchase/
├── next_category/
├── autoencoder/
├── segmentation/
├── recommendation/
└── product_taxonomy/
```

Every production model metadata record includes:

- model version;
- dataset SHA-256;
- cutoff/horizon;
- feature schema/version;
- random seed;
- training/validation metrics;
- selected threshold where relevant;
- library/package versions where practical.

## 7. Serving boundary

FastAPI loads serving artifacts once at startup where practical. Single-customer and batch paths must reuse the same preprocessing/feature code as training. Streamlit does not duplicate those transformations.

## 8. Failure behavior

Validation failures fail loudly with actionable error messages. The platform must not silently drop required columns, use fallback models without disclosure, regenerate taxonomy unexpectedly, or substitute missing artifacts.

## 9. Architecture change policy

Architecture changes are governance changes. Codex cannot change this file. If implementation requires a material boundary change, stop and request owner action.
