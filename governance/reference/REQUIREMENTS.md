# REQUIREMENTS.md — Vantara Actionable Requirements Baseline

## R-01 Reproducible data foundation

The project shall load the supplied Online Retail II workbook, both sheets, into a chronologically ordered standardized transaction table, validate it, clean it without mutating source data, and produce reproducible interim/processed outputs.

Acceptance evidence: Step 03 `python -m src.pipeline` succeeds from canonical raw input with no notebook intervention.

## R-02 Data quality handling

The pipeline shall explicitly handle:

- missing Customer ID;
- exact duplicate invoice lines;
- returns/cancellations;
- non-positive/adjustment prices;
- administrative/non-product StockCodes;
- inconsistent descriptions;
- quantity/price outliers with IQR + domain logic;
- source date/schema/null-rate validation.

## R-03 Required engineered feature table

The customer-level feature table shall include all PRD-required feature families documented in `FEATURE_CONTRACT.md` and use point-in-time cutoff discipline.

## R-04 Leakage prevention

Leakage prevention is an acceptance requirement. Unit tests must prove future transactions, future population statistics, cross-partition customer reuse, preprocessing fit, taxonomy/reference-price fit, and target-window data cannot contaminate features.

## R-05 Churn prediction

Train and compare exactly these six required classical models:

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. XGBoost
5. LightGBM
6. SVM

Train the required ANN on the same final engineered churn feature set.

Report Accuracy, Precision, Recall, F1, ROC-AUC and Confusion Matrix. Target final held-out ROC-AUC >=0.80 and churn recall >=0.70.

## R-06 CLV prediction

Train at least one regression model; audited plan uses Ridge baseline + XGBRegressor candidate. Target is a 180-day forward net-revenue-based CLV proxy. Report MAE, RMSE and R². Target R² >=0.60.

## R-07 Sequential next-purchase behavior

Train a PyTorch LSTM on ordered customer purchase sequences to estimate probability of another purchase within 30 days. Use grouped rolling snapshots to increase training examples without customer leakage.

## R-08 Anomaly detection

Train a PyTorch autoencoder on scaled spending/behavior features. Reconstruction error above a validated threshold produces an anomaly/manual-review flag. Do not claim fraud ground truth.

## R-09 Customer segmentation

Train K-Means and Gaussian Mixture Models. Use data-driven selection, Silhouette and Davies-Bouldin; K-Means additionally uses elbow/inertia, GMM additionally uses BIC. Produce business-readable segment profiles and PCA visualization.

## R-10 Product behavior

The system shall derive a frozen product taxonomy from historical descriptions, customer category-affinity features, a next-purchase-category classifier, and a personalized item-to-item recommender with offline evaluation.

## R-11 Explainability

The best production churn model must have:

- global SHAP summary;
- individual low-risk, borderline and high-risk SHAP outputs;
- one LIME example compared against SHAP for same customer;
- PDPs for 2–3 influential features;
- deterministic plain-language explanation.

Other predictive outputs shall expose a reasonable explanation surface as defined in `MODELING_SPEC.md`.

## R-12 Experiment discipline

All model runs use fixed seed where applicable, logged parameters/metrics/training time, training-only CV, immutable 70/15/15 customer split, and final held-out evaluation once after model selection is frozen.

## R-13 API

FastAPI shall expose at minimum:

- health check;
- model metadata;
- single-customer prediction;
- batch CSV scoring.

Pydantic must return clear validation errors.

## R-14 Persistence

PostgreSQL shall persist scored predictions and segment assignments. Schema shall include customers, transactions, predictions and segments; recommendations may be persisted as an approved extension.

## R-15 Dashboard

Streamlit shall provide:

- segmentation view with segment/country/value-tier filters;
- churn priority leaderboard with high-value/high-risk first;
- sales/revenue trends + simple forecast overlay;
- customer search and explanation panel;
- batch CSV scoring;
- downloadable CSV/PDF report.

## R-16 Performance

- single-customer API p95 <400 ms under the documented local benchmark;
- full segment-view dashboard load <3 s under documented local benchmark.

## R-17 Deployment

PostgreSQL, API and dashboard shall run from `docker-compose.yml` using `docker compose up --build` / compatible `docker-compose up --build` instructions and environment-based runtime configuration.

## R-18 Code quality

- Python 3.11 project target;
- type hints on all function signatures;
- docstrings on public functions/classes;
- Ruff zero errors;
- Black formatting check passes;
- structured logging;
- no hard-coded secrets;
- YAML-owned project configuration;
- clean environment reproduction.

## R-19 Testing

Unit, integration and acceptance tests shall cover feature logic, leakage, API, DB-facing boundaries as practical, artifact loading and end-to-end smoke paths. `src/` coverage must be >=70%; project target is >=80% where meaningful.

## R-20 Documentation/deliverables

Final output shall include:

- required notebooks;
- serialized required model artifacts;
- model comparison report and production choice justification;
- XAI outputs;
- architecture, ER and workflow diagrams;
- coverage report;
- README clean-clone/setup guide;
- final report with 2–4 page math appendix and future enhancements;
- recorded walkthrough checklist/evidence notes.

## R-21 Scope preservation

Explicit PRD deferrals remain deferred. Optional CatBoost, Transformer, supplementary dataset, streaming, RBAC, production CI/CD and A/B testing must not consume build time without owner approval.
