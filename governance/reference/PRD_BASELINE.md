# PRD_BASELINE.md — Structured Baseline of the Supplied Vantara PRD

> Authority note: this file is a faithful implementation-oriented baseline of `governance/source/Vantara_requirements.docx`. It does not replace the source document. If wording appears inconsistent, inspect the source PRD and stop rather than silently rewriting requirements.

## Document identity

- Division: Data Science & Analytics
- Product: Customer Behavior Prediction Platform
- Assignment: single contributor, Data Scientist / ML Engineer intern-to-associate level
- Duration: 4 weeks / 20 working days
- Status: approved for execution
- Classification: internal training/onboarding project

## 1–2. Executive summary and business motivation

The system ingests historical retail transactions and produces churn risk, CLV estimates, next-purchase probability, recommendations, and customer segments. It must be delivered with REST API, interactive dashboard, and full documentation.

Business use includes prioritized retention, discount targeting, demand planning, personalized recommendations, and executive customer-health/revenue reporting.

## 3. Objectives and success metrics

Required objectives:

1. reproducible raw-to-customer feature pipeline;
2. multiple churn models plus CLV regression;
3. supervised deep sequence model and unsupervised anomaly model;
4. at least two clustering algorithms with actionable profiles;
5. global and individual explainability;
6. REST API + business dashboard.

Success targets:

- churn held-out ROC-AUC >= 0.80;
- CLV test R² >= 0.60;
- churn recall >= 0.70;
- single-customer API p95 < 400 ms;
- full segment-view dashboard load < 3 s;
- production pipeline 100% scripted, no notebook-only production step.

Accuracy is not the primary churn metric; precision, recall, F1 and ROC-AUC must be reported together.

## 4. Scope

Required classical churn models:

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost
- LightGBM
- KNN or SVM (one)

Required deep models:

- feed-forward ANN on the same engineered churn feature set;
- LSTM on time-ordered purchase sequences;
- autoencoder for unsupervised spending-pattern anomaly detection.

Required clustering:

- K-Means;
- DBSCAN or Gaussian Mixture Model.

Required XAI/deployment:

- SHAP for best churn model;
- at least one LIME local example;
- FastAPI;
- Streamlit;
- Docker local deployment.

Explicitly deferred: Transformers, Kafka/event streaming, multi-tenant auth/RBAC, production CI/CD, live A/B testing.

## 5. Dataset

Primary dataset: **UCI Online Retail II**, UK online retailer transactions from 01 Dec 2009 through 09 Dec 2011.

Approximate source facts:

- 1,067,371 invoice-line rows;
- two sheets: `Year 2009-2010`, `Year 2010-2011`;
- fields: Invoice, StockCode, Description, Quantity, InvoiceDate, Price, Customer ID, Country;
- about 5,900 identified customers;
- targets are derived, including churn = no purchase in next 90 days, CLV, next-purchase category.

Known quality issues include missing Customer IDs, negative Quantity returns/cancellations, zero/negative price adjustments, administrative StockCodes, inconsistent descriptions, outliers/bulk orders, and multiple seasons/leap-year coverage.

Optional Customer Personality Analysis data is a stretch goal only and has no shared customer key.

## 6. Data pipeline

Must:

- load both source sheets;
- standardize columns/types;
- combine chronologically before cleaning;
- handle missing IDs appropriately;
- deduplicate exact duplicate lines without removing legitimate repeated purchases;
- detect Quantity/Price outliers using IQR + domain rules;
- normalize product descriptions using StockCode lookup;
- fail loudly on schema/null/date validation breaches;
- implement reusable Pandas/NumPy functions, not notebook-only wrangling;
- one-hot encode low-cardinality categorical fields;
- use target/frequency-style handling for high-cardinality StockCode-derived features;
- scale scale-sensitive models only;
- persist fitted encoders/scalers.

EDA must include spend/frequency/recency distributions, correlation, VIF, country/seasonality analysis, and written hypotheses.

## 7. Required customer features

- recency;
- frequency;
- total/average monetary value;
- historical + predicted CLV concept;
- average basket size;
- purchase-frequency trend;
- variance in time between purchases;
- seasonal purchase concentration;
- product-category affinity vector;
- return rate;
- discount/markdown sensitivity;
- engagement score based on RFM.

All prediction features must obey point-in-time cutoff discipline and leakage prevention must be unit-tested.

## 8. Modeling discipline

Classical model purposes and tuning follow the PRD. ANN must use batch normalization, dropout and early stopping. LSTM uses event sequences including amount, gaps and category. Autoencoder uses scaled spending features and reconstruction error.

All models use fixed seed. Recommended split is stratified 70/15/15. Hyperparameter search uses training data only. Test data is touched only once at final reporting. Every run logs parameters, metrics and training time.

## 9. Segmentation

K-Means k must be selected using elbow + silhouette. GMM/DBSCAN tests assumptions beyond K-Means. Every segment needs business-readable name and summary statistics. Hierarchical clustering is optional exploratory work only.

## 10. Explainability

Required:

- global SHAP summary;
- SHAP force plots for low-risk, high-risk and borderline customers;
- at least one LIME example compared with SHAP for the same customer;
- PDPs for 2–3 influential features;
- plain-language "why flagged" explanation.

## 11. Evaluation

Churn: Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix.

CLV: MAE, RMSE, R².

Clustering: Silhouette, Davies-Bouldin.

All supervised models: 5-fold cross-validation on training data before final test evaluation.

A consolidated model comparison table must rank churn models by ROC-AUC and recall and justify the production model.

## 12. API/dashboard/deployment

FastAPI must expose:

- single-customer prediction;
- batch CSV scoring;
- model metadata;
- health check.

PostgreSQL stores scored predictions and segment assignments. Pydantic validates inputs.

Dashboard requires:

- segmentation view with segment/country/value filters;
- churn leaderboard with high-value high-risk customers first;
- sales/revenue trends with simple forecast overlay;
- per-customer feature importance/SHAP search by Customer ID;
- CSV batch scoring;
- downloadable PDF/CSV results.

API, dashboard and database must run with a single Docker Compose command and environment-based runtime configuration.

## 13. Non-functional requirements

- type hints on all function signatures;
- docstrings on public functions/classes;
- Ruff or Flake8 zero errors;
- structured logging, not prints;
- unit + API integration tests, target >=70% source coverage;
- YAML config for paths/hyperparameters/thresholds;
- fixed seed and reproducible requirements/pyproject;
- no committed credentials; use env/.env ignored by Git.

## 14–15. Architecture and repository

Four layers: data, processing, modeling, serving. Batch scoring persists predictions; dashboard must not recompute models itself.

PRD repository directories and canonical files must be preserved. Deviations require README explanation.

## 16. Mathematical appendix

Final report needs 2–4 pages tied to actual implementation: statistics/probability, PCA, Logistic Regression/cross-entropy, gradient descent/backpropagation, regularization, and evaluation metrics. ANN/LSTM loss curves should demonstrate convergence/overfitting behavior.

## 17. Four-week plan

Week 1: data + EDA + features.
Week 2: classical ML + CLV + segmentation.
Week 3: ANN + LSTM + autoencoder + SHAP/LIME/PDP.
Week 4: FastAPI + PostgreSQL + Streamlit + Docker + documentation.

This governance pack compresses that schedule into 3 milestones / 9 build steps without removing scope.

## 18–20. Deliverables, risks and acceptance

Deliverables include datasets/pipeline scripts, serialized models, comparison report, XAI, FastAPI, Streamlit, Docker deployment, diagrams, tests/coverage, final report/math/future enhancements, and clean-clone README.

Acceptance requires:

- single-command raw-to-feature pipeline;
- at least six classical and all three deep models trained/logged/compared;
- business-readable segments;
- dashboard-accessible SHAP/LIME;
- FastAPI + Streamlit via Docker Compose on clean machine;
- >=70% `src/` coverage and leakage tests passing;
- success metrics met or honestly documented when missed.

## 21. Technology allowance

The PRD requires/permits Python 3.10+, Pandas, NumPy, Matplotlib, Seaborn, Plotly, scikit-learn, XGBoost, LightGBM, imbalanced-learn, TensorFlow or PyTorch, SHAP, LIME, FastAPI, Uvicorn, Pydantic, PostgreSQL or MongoDB, SQLAlchemy, Docker, Streamlit or React, Pytest, Coverage.py, Ruff/Flake8, Black, MLflow or structured logging, Jupyter, and diagram tooling.

The audited selections are frozen in `DECISIONS_LOCK.md`.
