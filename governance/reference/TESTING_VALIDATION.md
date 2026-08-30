# TESTING_VALIDATION.md — Vantara Verification Contract

## Principle

No step is complete because code exists. A step is complete only when its exit checks have actually run and passed, or a permitted PRD metric miss is explicitly evidenced/documented at final acceptance.

## 1. Standard code-quality gates

Once Step 01 tooling is established, every implementation step should run the relevant subset plus final full gates:

```bash
ruff check .
black --check .
pytest -q
```

Final coverage gate:

```bash
pytest --cov=src --cov=api --cov-report=term-missing --cov-report=html --cov-report=xml --cov-fail-under=70
```

Target meaningful coverage: >=80%; hard PRD acceptance floor: >=70% on `src/`.

## 2. Mandatory data tests

At minimum:

- source schema/header validation;
- sheet load/combined row sanity;
- canonical dtype conversion;
- duplicate rule;
- return/cancellation handling;
- admin-code exclusion from product behavior;
- canonical description mapping determinism;
- outlier flags;
- point-in-time feature cutoff;
- churn-label 90-day horizon;
- CLV 180-day target horizon;
- split disjointness;
- training-only preprocessing fit.

## 3. Mandatory leakage attack tests

Explicitly test:

1. adding future transactions cannot alter historical RFM/features;
2. future rows cannot alter reference prices for a fitted historical snapshot;
3. future rows cannot alter fitted product taxonomy;
4. validation/test population cannot alter training-fitted engagement percentiles;
5. validation/test cannot affect scaler/imputer/encoder fit;
6. same customer cannot exist across train/validation/test;
7. same customer's LSTM snapshots cannot cross grouped folds;
8. SMOTE cannot process validation/test;
9. CLV target-window rows cannot enter CLV features;
10. churn target-window rows cannot enter churn features;
11. tuning code cannot access final-test metrics.

Any failure invalidates downstream model evidence until corrected and rerun.

## 4. EDA validation

Step 03 must show:

- univariate/bivariate spend, frequency, recency analysis;
- class balance;
- correlations;
- VIF;
- country analysis;
- seasonality analysis;
- outlier review;
- explicit modeling hypotheses at notebook end.

## 5. Modeling validation

### Churn

For all six classical models and ANN record Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix.

### CLV

Record MAE, RMSE, R².

### Clustering

Record Silhouette and Davies-Bouldin; add inertia/elbow for K-Means and BIC for GMM.

### LSTM

Record binary classification metrics + training/validation loss.

### Autoencoder

Record train/validation reconstruction loss, threshold, anomaly rate, error distribution.

### Next category

Record macro-F1, Top-1, Top-3 and baseline.

### Recommender

Record Recall@5, HitRate@5, catalog coverage.

## 6. API validation

Required tests/smoke checks:

- FastAPI starts;
- `/health` succeeds;
- `/models/metadata` succeeds;
- valid known customer scoring;
- invalid/unknown customer path;
- malformed request handling;
- valid batch upload;
- malformed batch schema;
- prediction persistence;
- XAI/recommendation customer path used by dashboard.

## 7. Dashboard validation

Verify each required view and filter. Test customer search, leaderboard ordering, revenue/forecast, XAI, CSV upload and CSV/PDF downloads. Confirm frontend does not import models directly.

## 8. Performance benchmarks

### API

Warm service, run >=100 representative single-customer requests where practical, report p50/p95/max, environment and request count. PRD target p95 <400 ms.

### Dashboard

Measure full segment-view initial execution/render data path in a documented local method. PRD target <3 seconds.

Do not estimate performance by eye.

## 9. Docker/clean-run validation

Final step:

```bash
docker compose up --build
```

Verify:

- PostgreSQL healthy;
- migrations/init succeed;
- API healthy;
- dashboard reachable;
- dashboard can retrieve real persisted data;
- no secret/hard-coded local path dependency.

Also document compatibility form `docker-compose up --build` if environment uses classic Compose.

## 10. Regression rule

Any bug fix must add/update a test when practical and rerun the affected gate. Before final submission rerun the complete suite and acceptance matrix.

## 11. Evidence honesty

`STATUS.md` may mark a gate PASS only with actual command/evidence. Use FAIL/BLOCKED/NOT_RUN truthfully. Never transform a NOT_RUN into PASS based on code inspection.
