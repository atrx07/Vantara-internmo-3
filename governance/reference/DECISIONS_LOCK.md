# DECISIONS_LOCK.md — Audited Vantara Architecture Lock v2.0

These choices were audited against the supplied PRD and are intentionally frozen to remove agent improvisation. Codex must not change them.

## Technology

| Concern | Locked choice |
|---|---|
| Runtime | Python 3.11 |
| Environment | `venv` + `pip` |
| Data | Pandas + NumPy + PyArrow/Parquet |
| Validation | Pandera + explicit business-rule checks |
| Classical ML | scikit-learn |
| Boosting | XGBoost + LightGBM |
| Distance baseline | SVM, not KNN |
| Deep learning | PyTorch consistently for ANN/LSTM/Autoencoder |
| Segmentation | K-Means + Gaussian Mixture Model |
| Explainability | SHAP + LIME + PDP |
| Experiment tracking | MLflow local backend; no cloud account |
| API | FastAPI + Pydantic + Uvicorn |
| DB | PostgreSQL + SQLAlchemy 2.x + Alembic |
| Dashboard | Streamlit + Plotly |
| Revenue forecast | Holt-Winters Exponential Smoothing with Holt fallback |
| PDF report export | ReportLab |
| Tests | Pytest + pytest-cov + HTTPX where needed |
| Lint/format | Ruff + Black |
| Deployment | Docker + Docker Compose |

Exact package versions are resolved and pinned during Step 01 after installation compatibility is verified. Changing a library family is not permitted.

## Dataset and identifiers

- canonical input: `data/raw/online_retail_II.xlsx`;
- raw file is immutable;
- identifiers (`invoice`, `stock_code`, `customer_id`) are strings after normalization;
- interim/processed tables use Parquet;
- raw XLSX is not committed to Git by default.

## Snapshot and target semantics

### General

- `observation_end` = maximum valid `InvoiceDate` in the supplied dataset;
- features use transactions with `invoice_date < cutoff`;
- target windows begin at cutoff and must be fully observable before `observation_end`.

### Churn

- horizon: 90 days;
- `churn = 1` when a customer has no valid positive merchandise purchase in the next 90 days;
- returns alone do not count as purchases;
- canonical churn cutoff is derived as `observation_end - 90 days`.

### CLV

- internal target name: `clv_180d_target`;
- business label: **Predicted 180-Day Customer Value**;
- interpretation: 180-day forward net-revenue-based CLV proxy, not literal infinite-lifetime profit;
- canonical cutoff: `observation_end - 180 days`;
- target uses positive merchandise revenue minus returns in horizon, clipped at zero;
- Ridge baseline + XGBRegressor candidate;
- predicted CLV is not fed into churn model.

### LSTM

- predicts purchase within next 30 days;
- uses monthly/controlled rolling historical snapshots with complete 30-day observation window;
- last 20 invoice events;
- event features include `log1p(order_value)`, `gap_days`, dominant product category;
- all snapshots for a customer remain in that customer's global partition.

## Split and CV

- customer-level train/validation/test = 70/15/15;
- seed = 42;
- split is created once, persisted, reused by all models;
- churn/next-category CV: 5-fold stratified training-only;
- CLV CV: 5 folds stratified on quantile bins of `log1p(CLV)` only for fold balancing;
- LSTM CV: 5-fold StratifiedGroupKFold or equivalent customer-group-preserving strategy;
- held-out test is evaluated exactly once after selection is frozen.

## Imbalance

Primary handling is class weighting. SMOTE is optional experiment only inside training folds and never validation/test.

## Feature/model relationship

The complete customer feature table contains all PRD-required business features. Individual models use a documented non-redundant subset. ANN uses the same final churn feature set as classical churn models.

## Product taxonomy

- canonical description per StockCode derived deterministically;
- exclude administrative/non-product StockCodes;
- taxonomy: TF-IDF word/bigram -> TruncatedSVD -> MiniBatchKMeans;
- candidate `k`: 12, 16, 20, 24, 30;
- selection uses silhouette, cluster balance and interpretability;
- resulting taxonomy is frozen and versioned;
- no reclustering at inference.

## Discount feature

Dataset lacks explicit promotion flag. Use a documented **Markdown Affinity Proxy**:

- training-history median valid price per StockCode = reference price;
- require at least 5 eligible historical observations;
- markdown-like if price <= 90% of reference price;
- customer feature = markdown-like eligible orders / eligible orders;
- do not claim true causal discount sensitivity.

## Required churn models

1. Logistic Regression with L1/L2 search
2. Decision Tree
3. Random Forest
4. XGBoost
5. LightGBM
6. SVM
7. PyTorch ANN comparison

Production model selected by validation evidence, not pre-assigned.

Eligibility preference:

1. churn recall >=0.70;
2. highest validation ROC-AUC;
3. within 0.01 ROC-AUC, prefer higher recall, then latency, explainability, operational simplicity.

Classification threshold is selected on validation data using F2-oriented logic while targeting recall >=0.70; threshold is frozen before test.

## ANN

Default architecture family: Input -> 128 + BatchNorm + ReLU + Dropout(0.30) -> 64 + BatchNorm + ReLU + Dropout(0.20) -> 1. BCEWithLogitsLoss, AdamW, early stopping. Exact training constants live in config.

## Autoencoder

Scaled spending/behavior inputs; compact encoder/latent/decoder; MSE reconstruction loss; default anomaly threshold = 99th percentile of validation reconstruction error. Output means anomaly/manual-review candidate, not confirmed fraud.

## Segmentation

K-Means primary; GMM secondary. Primary clustering features remain interpretable behavioral features rather than full high-dimensional affinity vectors. Use business-readable labels. PCA is for visualization, not the clustering input reduction.

## Next-purchase category

Target = dominant category by merchandise value in customer's next valid invoice after cutoff; tie-break by quantity then deterministic category ID. Model = LightGBM multiclass. Report macro-F1, Top-1 and Top-3 accuracy plus most-popular-category baseline.

## Recommender

Item-to-item implicit collaborative filtering using customer x StockCode sparse interactions, log1p quantity weighting and cosine similarity. Default top 5. Sparse/new-customer fallback = popular products within assigned segment. Evaluate via leave-last-order-out with Recall@5, HitRate@5 and catalog coverage.

## Explainability

- best churn model gets full SHAP + LIME + PDP requirements;
- SHAP explainer family follows actual winning model;
- CLV: SHAP for production regressor;
- next category: multiclass SHAP where practical;
- LSTM: model-agnostic sequence-event perturbation;
- autoencoder: per-feature reconstruction error;
- recommender: similarity/affinity reason;
- segment: dominant profile statistics.

## Backend/UI boundary

Streamlit consumes FastAPI and does not import trained models or own model inference. API owns scoring orchestration. PostgreSQL owns persisted prediction/segment state. Fitted artifacts are loaded once at API startup where practical.

## Priority score

Dashboard retention priority = normalized churn probability × normalized predicted 180-day value. Underlying values remain visible.

## Configuration precedence

Runtime environment variables override YAML defaults for approved runtime values such as DB URL, artifact directory, log level and app environment. No secret may be stored in YAML.

## No silent change rule

If implementation evidence suggests one of these choices is genuinely impossible, Codex must stop and request an owner-governance amendment. It may not substitute an alternative on its own.
