# STATUS.md — Vantara Live Project State

> MUTABLE STATE FILE. Codex may edit this file. It must contain observed facts and evidence, never optimistic assumptions.

## STEP 04 — Classical ML + CLV + Segmentation + Product Intelligence

- Authorization: owner explicitly approved continuation into STEP 04 with `continue` and later confirmed continuation after context compaction on 2026-08-31.
- Scope status: `STEP_COMPLETE_WAITING_FOR_APPROVAL`.
- Scope boundary: no STEP 05 ANN, rolling LSTM, or autoencoder work; no STEP 06 production-model selection, threshold freeze, explainability, or final held-out test evaluation.
- Pre-step HEAD: `af1852eb804a4d34b1edc445d0b390bddf52805a`.
- Canonical command: `.venv\Scripts\python.exe -m src.models.step04_pipeline --config config/config.yaml` — PASS on the full governed M1 artifacts.
- Split discipline: five-fold training-only CV/tuning; validation-only comparison evidence; the final `test` population was neither scored nor used for target generation, preprocessing, selection, or reported metrics.

### Classical churn models

- Exactly six required models trained and serialized on the exact `vantara-churn-features-v1` 47-feature order: Logistic Regression, Decision Tree, Random Forest, XGBoost, LightGBM, and SVM.
- Every run records parameters, CV metrics, validation Accuracy/Precision/Recall/F1/ROC-AUC/confusion matrix, training time, source hash, schema version, split version, seed `42`, and local MLflow run ID.
- Highest validation ROC-AUC: Logistic Regression `0.8038210831`, with recall `0.7331730769`.
- Random Forest validation ROC-AUC `0.8015868334`, recall `0.7788461538`; SVM produced the highest validation recall `0.8221153846` with ROC-AUC `0.7874255250`.
- Production selection and threshold optimization are intentionally deferred to STEP 06; STEP 04 does not freeze a winner.

### CLV regression

- Ridge baseline and XGBRegressor candidate use `log1p` target modeling and five folds stratified on training-only target quantiles.
- Predictions are nonnegative and capped by a bound learned independently from each fit's training-target maximum, preventing log-inverse extrapolation outside observed training support without using validation/test values.
- Ridge validation: MAE `475.2176610896`, RMSE `1083.6206152548`, R2 `0.9684629277`; mean CV R2 `-2.1797938355` shows material fold instability and is carried as a warning rather than hidden.
- XGBRegressor validation: MAE `626.0065210641`, RMSE `5693.9093410618`, R2 `0.1292610877`; mean CV R2 `0.3169980681`.
- These are validation results only; the PRD final held-out CLV target is not claimed as achieved before STEP 06.

### Segmentation and product intelligence

- K-Means candidates `k=3..8` evaluated with inertia, Silhouette, and Davies-Bouldin; selected `k=8` by highest training silhouette `0.3052447809` (Davies-Bouldin `1.1009530648`).
- GMM candidates `2..8` evaluated with BIC, Silhouette, and Davies-Bouldin; selected `8` by lowest BIC `-35988.1604784577` (Silhouette `0.1086794699`, Davies-Bouldin `2.4075170649`).
- Business-readable profiles and PCA(2) visualization data were generated; clustering used ten interpretable standardized behavioral features, not PCA coordinates or full affinity vectors.
- Next-category target uses each eligible train/validation customer's first valid invoice after the cutoff, frozen taxonomy plus explicit unknown category, and deterministic value/quantity/category tie-breaking.
- Next-category LightGBM validation: macro-F1 `0.0898417858`, Top-1 `0.3312883436`, Top-3 `0.5858895706`; most-popular baseline: macro-F1 `0.0194515306`, Top-1 `0.3742331288`, Top-3 `0.5521472393`. Six of 31 training classes have fewer than five examples, limiting stratification and macro performance; categories were not merged or relabeled to improve metrics.
- Item-to-item recommender: training customers only, `log1p(quantity)` implicit weights, cosine similarity, segment-popularity fallback, leave-last-order-out on `2,670` eligible customers. Recall@5 `0.0396743526`, HitRate@5 `0.3539325843`, catalog coverage `0.2412579723` across `4,569` catalog items.

### Evidence, artifacts, and notebook

- Local MLflow: `23` successful runs (6 churn, 2 CLV, 13 segmentation candidates, 1 next-category, 1 recommender); local runtime store remains ignored and untracked, with a tracked run-summary export under `reports/modeling/`.
- Eleven serialized STEP 04 artifacts were reloaded successfully and exercised; largest artifact is `next_category_lightgbm.joblib` at approximately `10.51 MB`, below the governance size gate.
- `notebooks/03_model_experiments.ipynb`: PASS — 15 cells, 7/7 code cells executed, zero error outputs; tests prohibit notebook-owned fitting logic.
- Generated tracked evidence includes churn/CLV comparisons, all confusion matrices, segmentation selection/profiles/PCA sample, next-category baseline comparison, recommender metrics, MLflow summary, reload audit, and STEP 04 summary.
- Environment warnings: LightGBM/scikit-learn emitted feature-name and rare-multiclass warnings; Joblib used logical-core fallback because WMIC is unavailable; Jupyter used a temporary local runtime and emitted the Windows Proactor selector-thread warning. None failed a final gate.

### STEP 04 validation

- `python -m src.models.step04_pipeline --config config/config.yaml`: PASS on the real full M1 artifacts after the final held-out-customer filtering change.
- `pytest --cov=src --cov-report=term-missing --cov-fail-under=70 -q`: PASS — `47 passed`, source coverage `87.15%`.
- Synthetic orchestration test: PASS — all six churn families, both CLV families, K-Means, GMM, next-category, recommender, 12 temporary MLflow runs, 11 reload checks, and no final-test access exercised from temporary data.
- Ruff: PASS for repository Python and all three executed notebooks; Black: PASS for Python source.
- `pip check`: PASS; `compileall`: PASS.
- Governance reference lock: PASS — 30 immutable files verified.
- Raw source/canonical-copy SHA-256 and Git protection: PASS — exact expected hash on both; neither workbook tracked; both ignored.
- Artifact/runtime audit: PASS — 11 tracked joblib artifacts total `16,447,430` bytes, maximum `10,509,998` bytes; MLflow runtime store, generated processed data, caches, and temporary Jupyter runtime remain ignored/untracked.

## STEP 03 — EDA + Data Freeze

- Authorization: owner explicitly approved continuation into STEP 03 on 2026-08-31.
- Scope status: `STEP_COMPLETE_WAITING_FOR_APPROVAL`.
- Scope boundary: no STEP 04 model training, MLflow experiments, segmentation, next-category model, or recommender was implemented.
- Pre-step HEAD: `3430014657b6184ee48e14cdabe669705bc4abce`.

### End-to-end M1 pipeline

- Canonical command: `.venv\Scripts\python.exe -m src.pipeline --config config\config.yaml`.
- Full immutable-workbook execution: PASS — loader, source validation, cleaning, interim Parquet, snapshots, targets, feature tables, training-only product artifacts, shared customer split, preprocessing contracts, EDA, and schema freeze completed without notebook intervention.
- Final full-run rows: `1,033,036` cleaned transactions and `4,952` eligible customer rows.
- STEP 02 generated data fingerprints remained stable through the M1 run.
- Raw source/canonical copy were not modified or tracked.

### Required notebooks and EDA

- `notebooks/01_eda.ipynb`: PASS — 14 cells, 7/7 code cells executed, zero error outputs.
- `notebooks/02_feature_engineering.ipynb`: PASS — 11 cells, 5/5 code cells executed, zero error outputs.
- Both notebooks are consumers of `src.analysis.eda` and persisted evidence; tests prohibit notebook-owned groupby/split/taxonomy production logic.
- Required analyses present: spend/frequency/recency distributions, churn class balance, training-only correlations, VIF, country analysis, monthly/quarter seasonality, and outlier/data-quality review.
- Seven explicit modeling hypotheses are recorded at `reports/eda/hypotheses.md`.
- Observed EDA highlights: median recency `291.9` days for churned versus `78.9` for active customers; median order frequency `2.0` versus `6.0`; observed gross-revenue peak month `2011-11`; United Kingdom share among displayed top-country revenue `86.6%`.
- Five static figures and six evidence tables were generated under `reports/eda/` and visually inspected.

### Correlation, VIF, and frozen schema

- Correlation/VIF fitting population: shared training partition only (`3,466` customers).
- Pearson high-correlation threshold: `0.95`; observed high-correlation pairs: `3`.
- VIF threshold: `10.0`; maximum final VIF: `9.5125581744` — PASS.
- Canonical schema: `models_artifacts/churn_feature_schema.json`, version `vantara-churn-features-v1`.
- Frozen churn model feature count/order: `47`; this exact schema is required for all six classical churn models and the ANN.
- Documented exclusions: `category_affinity_unknown` as the reference category; exact duplicate `historical_customer_value`; highly correlated `gross_spend` in favor of return-aware `net_spend`; `category_affinity_00` at training VIF `45.749312`; and `engagement_score` at training VIF `15.896621`.
- The complete 52-feature business table remains unchanged; exclusions affect only the final churn model input schema.
- Evidence: correlation matrix, three high-correlation pairs, initial/final VIF tables, and data-freeze summary under `reports/data_freeze/`.
- Determinism: all 18 tracked EDA/data-freeze/schema evidence files produced identical SHA-256 values after a repeated STEP 03 analysis run.

### STEP 03 validation and M1 acceptance

- `pytest --cov=src --cov-report=term-missing --cov-fail-under=70 -q`: PASS — `35 passed`, source coverage `85.10%`.
- Ruff: PASS, including notebook code cells.
- Black check: PASS for Python source (`34` files unchanged); Black reported notebooks skipped because its optional Jupyter extra is not installed, while Ruff and executed-notebook validation passed.
- `pip check`: PASS; compileall: PASS.
- Governance reference lock: PASS — 30 immutable files verified before and after the owner-authorized commit/push governance amendment.
- Owner governance amendment: PASS — on 2026-08-31 the owner explicitly established that approval of a roadmap step also authorizes its green atomic commit(s) and normal push without a second chat confirmation. `AGENTS.md` and `governance/reference/GIT_WORKFLOW.md` now record that standing rule, and `governance/REFERENCE_LOCK.json` was refreshed for those exact authorized changes.
- M1 acceptance: PASS — workbook, both-sheet ingestion, cleaning, required features, churn/CLV targets, taxonomy/affinities, shared split, leakage tests, required notebooks, EDA, correlation/VIF evidence, and single-command pipeline all have executed evidence.
- Environment warnings: Joblib used logical-core fallback because WMIC physical-core discovery is unavailable; Matplotlib dependency deprecation warnings occurred during tests; Jupyter required temporary workspace/local-temp runtime directories because user-profile writes are sandbox-blocked. None caused a failed final gate.

## STEP 02 — Cleaning + Snapshot + Feature Pipeline

- Authorization: owner explicitly approved continuation into STEP 02 on 2026-08-30.
- Scope status: `STEP_COMPLETE_WAITING_FOR_APPROVAL`.
- Scope boundary: no STEP 03 notebooks, EDA, correlation/VIF decision, final feature-schema freeze, or `src.pipeline` orchestration was implemented.
- Pre-step HEAD: `01081363d01bfb45e9d4d60404fb2c43c9e313d4`.

### Cleaning and interim data

- Full raw input rows: `1,067,371`; exact duplicate invoice lines removed: `34,335`; cleaned/audit rows: `1,033,036`.
- Missing-customer rows retained and flagged: `235,151`.
- Return rows retained and flagged: `22,496`; cancelled-invoice rows: `19,104`.
- Non-positive-price rows retained and flagged: `6,019`.
- Administrative/non-product rows retained and flagged: `5,821`; configured exact-code and regex rules are version-controlled in `config/config.yaml`.
- Statistical IQR outlier rows retained and flagged: `81,830`; likely domain-error rows at configured absolute limits: `0`.
- IQR thresholds were fitted only on training-customer history before the CLV cutoff: quantity `[-28, 42]`, price `[-6.25, 11.25]`; fixed domain limits are absolute quantity `100,000` and price `50,000`.
- Product descriptions are normalized deterministically while original descriptions remain preserved.
- Clean output: ignored local `data/interim/transactions_clean.parquet`; outlier audit: ignored local `data/interim/outlier_audit.parquet`.
- Clean-table checks: 24 columns, zero exact duplicates, chronological order PASS, returns never count as positive purchase events.

### Snapshots, targets, and features

- Observation end derived from data: `2011-12-09 12:50:00`.
- Canonical churn cutoff: `2011-09-10 12:50:00`; feature history is strictly earlier; 90-day target window is fully observable.
- Canonical CLV cutoff: `2011-06-12 12:50:00`; feature history is strictly earlier; 180-day forward net-revenue proxy is clipped at zero.
- Eligible shared modeling population: `4,952` customers with a valid positive merchandise purchase before the earlier CLV cutoff.
- Churn feature table: `4,952 x 57`; churn rate `0.5714862682`; labels contain only `0/1`.
- CLV feature table: `4,952 x 57`; positive-target rate `0.5179725363`; minimum target `0.0`.
- Required RFM/monetary, basket, timing/trend, seasonality, return, markdown proxy, engagement, training-only product-frequency, and full product-category affinity families are present.
- Full affinity vector: 30 frozen taxonomy categories plus explicit unknown-product affinity; every eligible customer vector sums to `1.0`.
- Engagement percentiles use the locked 40/30/30 RFM weights and are fitted on training customers only.
- Scaled linear/distance/ANN and unscaled tree/boosting preprocessing contracts were fitted separately using the same `3,466` training customers and 52 numerical features.

### Product artifacts and split

- Customer split version: `vantara-customer-split-v1`, seed `42`, persisted once under ignored `data/processed/`.
- Split counts: train `3,466`, validation `742`, test `744`; `4,952` unique customers; all partitions are disjoint.
- Product taxonomy version: `vantara-taxonomy-v1`; training customers/history only; TF-IDF word/bigram -> TruncatedSVD -> MiniBatchKMeans.
- All locked candidate cluster counts `12, 16, 20, 24, 30` were evaluated; selected `k=30` by the configured silhouette/balance rule.
- Frozen taxonomy rows: `4,219`; training frequency-encoded products: `4,210`; eligible reference-price products with at least five observations: `3,665`.
- Markdown proxy uses `price <= 0.90 * training-history median reference price`; it is not represented as causal discount sensitivity.
- Generated split, feature, taxonomy, reference-price, frequency-encoding, preprocessing, and metadata outputs remain ignored and untracked by default.

### Leakage and deterministic validation

- Explicit tests PASS for strict point-in-time cutoff, target-window exclusion, future-transaction insertion, future-insensitive frozen taxonomy/reference prices, validation/test-insensitive engagement percentiles and preprocessors, training-only outlier thresholds, and customer partition disjointness.
- LSTM grouped-fold, SMOTE-boundary, and final-test-access tests remain correctly deferred until those systems exist in Steps 04–06.
- Two full immutable-workbook executions completed successfully and produced identical content fingerprints:
  - cleaned transactions: `0ee6e6ce421848cf3fa2a76071f6366b8a8554af108ee91f3758e7863fe1cbc3`;
  - customer split: `8177d4d9015d49303d2aae9fc86a2ed56e3db847643dff1f5bb7cf698cf960d3`;
  - churn features: `80ee96733c58019b9fa78fb919e7346b5546d084c6240950f6b4eb0d0edf174e`;
  - CLV features: `96cab6da3e212ecc899bf6f7d5aebc58c7f4255580e127f60fc99cba46ab3c19`;
  - product taxonomy: `3a161eadc36888a66eabad0b05d470abf812e5ca6e44efee331e0d6fefdfffc8`.
- `pytest --cov=src --cov-report=term-missing --cov-fail-under=70 -q`: PASS — `29 passed`, source coverage `85.36%`.
- Ruff: PASS; Black check: PASS (`27` files unchanged); `pip check`: PASS; compileall: PASS.
- Governance reference lock: PASS — 30 immutable files verified.
- Environment warning: Joblib could not query physical cores through WMIC and used logical-core count; both full runs and deterministic fingerprints passed.

## STEP 01 — Bootstrap + Raw Ingestion

- Authorization: owner explicitly approved STEP 01 on 2026-08-30.
- Scope status: `STEP_COMPLETE_WAITING_FOR_APPROVAL`.
- Scope boundary: no STEP 02 cleaning, snapshots, labels, feature engineering, taxonomy, splitting, interim data, or processed data was implemented.

### Environment and dependencies

- Project runtime: Python `3.11.9` in ignored local `.venv`.
- Environment tool: standard-library `venv` + `pip 26.2.1`.
- Direct dependency specification: `requirements.in`, exact pins for all governance-approved technology families.
- Fully resolved lock: `requirements.txt`, 201 exact installed package pins including transitive dependencies.
- Dependency installation: PASS.
- `python -m pip check`: PASS — no broken requirements.
- Major import/version smoke: PASS for NumPy, Pandas, PyArrow, OpenPyXL, Pandera, scikit-learn, XGBoost, LightGBM, imbalanced-learn, PyTorch, SHAP, LIME, MLflow, statsmodels, FastAPI, Uvicorn, Pydantic, SQLAlchemy, Alembic, psycopg, HTTPX, Streamlit, Plotly, ReportLab, Matplotlib, Seaborn, JupyterLab, Pytest, Ruff, and Black.
- Host note: Docker CLI is not installed; Docker is not a STEP 01 validation gate and belongs to later deployment work.

### Repository bootstrap

- `.gitignore` was created and verified before any staging operation.
- Protected paths verified ignored: root/canonical raw XLSX, `.venv`, `.env`, database files/volumes, MLflow stores, Python/test caches, temporary files, logs, and generated interim/processed data.
- Created governed scaffold paths for `data/`, `notebooks/`, `src/`, `api/`, `frontend/`, `models_artifacts/`, `config/`, `tests/`, `docs/`, `reports/`, `scripts/`, and `migrations/` without implementing later-step functionality.
- Added `.env.example`, `.gitattributes`, `pyproject.toml`, root `README.md`, and raw-data placement instructions.

### Raw data placement and ingestion

- Source path: `online_retail_II.xlsx`.
- Canonical raw path: `data/raw/online_retail_II.xlsx`.
- Copy method: byte-for-byte `Copy-Item`; no transformation, cleaning, rewrite, normalization in-place, rename, or move.
- Source bytes: `45,622,278`; canonical copy bytes: `45,622,278`.
- Source and copy SHA-256: `bcbe73b35f5b7babf197fb0cb983a11f5d9ff929078d4aa53d171b1f2df2e980` — MATCH.
- Git tracking: neither workbook is tracked; both exact paths are ignored.
- Sheets loaded: `Year 2009-2010` = `525,461` rows; `Year 2010-2011` = `541,910` rows.
- Combined rows: `1,067,371`; normalization preserved all rows.
- Canonical columns: `invoice`, `stock_code`, `description`, `quantity`, `invoice_date`, `price`, `customer_id`, `country`.
- Canonical dtypes: string identifiers/text, nullable `Int64` quantity, `datetime64[ns]` invoice date, `float64` price.
- Null rates: description `0.00410541`, customer ID `0.22766873`, all other canonical columns `0.0`; all within configured thresholds.
- Date bounds: `2009-12-01 07:45:00` through `2011-12-09 12:50:00` — exact configured bounds.
- Chronological merge: PASS — stable sorted output is monotonic by `invoice_date`.
- Cleaning behavior: NONE — negative quantities, non-positive prices, anonymous rows, duplicates, and administrative codes remain available for STEP 02 handling.

### STEP 01 validation evidence

- `python -m src.data.loader --config config/config.yaml` — PASS — exact hash, two sheets, exact rows/schema/date bounds, 1,067,371-row chronological merge.
- `python -m pytest -q` — PASS — 15 passed.
- `python -m pytest --cov=src --cov-report=term-missing --cov-fail-under=70 -q` — PASS — 15 passed, `83.21%` total source coverage.
- `ruff check .` — PASS — zero errors.
- `black --check .` — PASS — 17 Python files unchanged.
- `python -m pip check` — PASS — no broken requirements.
- `python -m compileall -q src tests` — PASS.
- `python governance/tools/verify_reference_lock.py` — PASS — 30 immutable files verified.
- Raw source/copy size and SHA-256 comparison — PASS.
- Git ignore/tracking audit — PASS — raw tracked-file count `0`; all required local/runtime exclusions matched `.gitignore`.

## Governance

- Governance pack: `Vantara_Governance_Pack_v1.0`
- STEP 00 read order: PASS — `AGENTS.md`, `GOVERNANCE_INDEX.md`, every numbered reference in the required order, `STATUS.md`, `NEXT_STEPS.md`, the source PRD, and the remaining immutable pack/manifest files were inspected.
- Reference lock: PASS — `python governance/tools/verify_reference_lock.py` exited `0` with `REFERENCE LOCK PASS: 30 immutable files verified`.
- Immutable governance discrepancies: NONE OBSERVED.

## Source PRD

- Authoritative path: `governance/source/Vantara_requirements.docx`.
- Accessibility/inspection: PASS — opened read-only and structurally inspected end-to-end (301 paragraphs, 11 tables, 21 numbered sections).
- Size: `29,899` bytes — MATCHES `SOURCE_MANIFEST.md`.
- SHA-256: `775284832513c27552d8c06c14456ba7adb8526ed5a898a49c0bbd106d7426e9` — MATCHES `SOURCE_MANIFEST.md`.
- Baseline/traceability comparison: PASS — no unresolved product-requirement conflict was found among the source PRD, `PRD_BASELINE.md`, `PRD_TRACEABILITY.md`, the locked decisions, architecture, or roadmap. Locked technology/semantic selections fall within PRD alternatives or resolve PRD underspecification.
- Visual render note: FAIL (environment) — the bundled DOCX renderer was attempted but LibreOffice/`soffice` was unavailable. This does not affect the successful structural/content/hash inspection; no visual page-layout claim is made.
- Source file modified: NO.

## Dataset

- Supplied STEP 00 path: `online_retail_II.xlsx` at repository root.
- Canonical eventual path: `data/raw/online_retail_II.xlsx` (placement belongs to STEP 01 and was not performed).
- Accessibility: PASS — XLSX ZIP/XML package opened read-only and both worksheets streamed successfully.
- Filename/source identity: PASS — `online_retail_II.xlsx`, exact manifest identity.
- Size: `45,622,278` bytes — MATCHES `DATASET_MANIFEST.md`.
- SHA-256: `bcbe73b35f5b7babf197fb0cb983a11f5d9ff929078d4aa53d171b1f2df2e980` — MATCHES `DATASET_MANIFEST.md`.
- Sheets: PASS — `Year 2009-2010`, `Year 2010-2011`.
- Schema on both sheets: PASS — `Invoice`, `StockCode`, `Description`, `Quantity`, `InvoiceDate`, `Price`, `Customer ID`, `Country`.
- Row counts: PASS — `525,461` + `541,910` data rows = `1,067,371` combined; 8 columns per sheet.
- Observed date range: `2009-12-01 07:45:00` through `2011-12-09 12:50:00`.
- Identified customers across both sheets: `5,942`, consistent with the PRD's approximate 5,900.
- Identity verdict: PASS — exact manifest hash, two expected yearly sheets, exact schema/rows, and two-year date range confirm UCI Online Retail II rather than the older single-year Online Retail dataset.
- Dataset modified, renamed, moved, copied, cleaned, committed, or pushed: NO.

## Repository and Git

- Pre-step repository state: directory was not a Git repository; no branch, HEAD, commits, tracked files, remotes, or ignore rules existed.
- Existing product implementation: NONE — repository contained only the governance pack/state files and root source workbook.
- Architecture/roadmap conflict with existing contents: NONE. The root workbook placement is an explicitly owner-supplied STEP 00 source location; governed byte-for-byte placement under `data/raw/` remains STEP 01 work.
- Local Git repository: INITIALIZED during STEP 00 as required Git metadata only.
- Target/current branch: unborn `main`.
- Pre-step HEAD: NONE (no Git repository).
- Post-step HEAD: NONE (unborn branch; no commits created).
- Tracked files: NONE.
- Working tree: all project files are untracked because no initial commit exists.
- `.gitignore`: PRESENT; both raw workbook paths and all required local/runtime outputs are ignored and verified before staging.
- Environment warning: external/elevated Git commands require a command-scoped `safe.directory` override because the sandbox-created `.git` metadata and host user have different ownership identities. No global Git configuration was changed.

## Remote

- Owner-provided shorthand: `atrx07/Vantara-internmo-3.git`.
- Bound remote: `origin` -> `https://github.com/atrx07/Vantara-internmo-3.git` (fetch and push URLs).
- Connectivity: PASS after approved network access.
- Fetch: PASS.
- Remote refs/history: `refs/heads/main` exists at the completed STEP 03 governance/state commit before the STEP 04 push.
- Remote default branch: `main`.
- Local/remote history conflict: NONE.
- Pushes during STEP 00: NONE.

## Execution

- Project status: INCOMPLETE
- Current milestone: M2 — Intelligence Ready
- Current step: STEP 04 — Classical ML + CLV + Segmentation + Product Intelligence
- Step state: `STEP_COMPLETE_WAITING_FOR_APPROVAL`
- Last owner-approved step: STEP 04
- M0 Governance Ready: PASS
- M1 Data Ready: PASS
- M2 Intelligence Ready: IN_PROGRESS
- M3 Product Ready: NOT_STARTED

## Validation evidence

- Mandatory governance/reference/source read — PASS — complete pack inspected in required order.
- `python governance/tools/verify_reference_lock.py` — PASS — 30 immutable files verified.
- PRD size/SHA-256 and structured content/table inspection — PASS.
- Bundled DOCX render attempt — FAIL (environment) — LibreOffice/`soffice` was unavailable; no source mutation.
- Dataset size/SHA-256 — PASS.
- Dataset read-only sheet/header/row/date/customer streaming inspection — PASS.
- `git ls-remote --symref ...` and full `git ls-remote ...` — PASS — remote exists and is empty.
- `git fetch origin` — PASS — no refs fetched because remote is empty.
- Final governance-lock, source-hash, dataset-hash, and working-tree rechecks — PASS — lock still verifies all 30 immutable files; source/dataset hashes and dataset timestamp remain unchanged; only the two permitted state files were edited in the working tree, with required Git metadata initialized separately.
- STEP 04 full modeling command — PASS — 23 local MLflow runs, 11 reload-validated artifacts, all required validation evidence, and no final-test metrics.
- STEP 04 notebook execution — PASS — 7/7 code cells executed with zero error outputs.

## Commits / pushes

- Commits created during STEP 00: NONE.
- Refs/heads pushed during STEP 00: NONE.
- STEP 01 implementation commit: `4c152c199216ad74a04476776ed849b34d6f552a` — `step-01: bootstrap project and validate raw ingestion`.
- STEP 01 implementation push: PASS — created `refs/heads/main` on `origin` at `4c152c199216ad74a04476776ed849b34d6f552a`; local `HEAD` and `origin/main` matched immediately after fetch verification.
- STEP 01 state-finalization commit: this status record is committed separately after the implementation push; Git assigns its hash after the file content is fixed, so the exact hash/pushed ref is recorded in the chat handoff.
- STEP 02 implementation commit: `bf28920275119181040ebb2920ba0b68f6d52b48` — `step-02: implement cleaning snapshots and feature pipeline`.
- STEP 02 implementation push: PASS — advanced `refs/heads/main` from `01081363d01bfb45e9d4d60404fb2c43c9e313d4` to `bf28920275119181040ebb2920ba0b68f6d52b48`.
- STEP 02 state-finalization commit: this status record is committed separately after the implementation push; its exact hash and pushed ref are recorded in the chat handoff because a commit cannot contain its own hash.
- STEP 03 implementation commit: `f7fa9782ee57206ef57b0fd4c8b124741ceccdd0` — `step-03: complete EDA and freeze data foundation`.
- STEP 03 implementation push: PASS — advanced `refs/heads/main` from `3430014657b6184ee48e14cdabe669705bc4abce` to `f7fa9782ee57206ef57b0fd4c8b124741ceccdd0`.
- STEP 03 governance/state-finalization commit: `af1852eb804a4d34b1edc445d0b390bddf52805a` — `governance: authorize green pushes and finalize step 03`; push advanced `refs/heads/main` from `f7fa9782ee57206ef57b0fd4c8b124741ceccdd0` to this hash.
- STEP 04 implementation/state commit: this record is committed with the green STEP 04 implementation; its exact hash and pushed ref are reported in the chat handoff because a commit cannot contain its own hash.

## Blockers / warnings

- Blocking PRD/governance/dataset/repository conflict: NONE.
- Raw data protection: PASS — both workbook paths are ignored and untracked.
- Remote branch: PASS — `origin/main` is established and is the remote default branch.
- Warning: DOCX visual rendering was unavailable; structural/content/hash inspection completed successfully.
- Warning: use command-scoped Git `safe.directory` in this Codex environment; no global exception was installed.
- Warning: next-category data contain 31 classes, including 6 training classes with fewer than 5 examples; the locked five-fold stratification therefore has sparse class support, and actual macro metrics are preserved.
- Warning: Ridge CLV validation R2 is high while mean training-CV R2 is negative, indicating sensitivity to the extreme-value distribution; no final-test or production claim is made.
- Warning: both segmentation selection rules chose the upper configured candidate boundary (`8`); this is recorded evidence, not an unreported search expansion.

## Readiness and authorization boundary

- STEP 01 implementation, validation, and implementation push: COMPLETE.
- STEP 02 implementation, validation, and all pushes: COMPLETE.
- STEP 03 implementation, validation, and all pushes: COMPLETE.
- STEP 04 implementation, validation, evidence, artifacts, and notebook: COMPLETE. This record accompanies the owner-authorized green STEP 04 commit/push; its exact Git evidence is reported in the chat handoff.
- STEP 05 is NOT AUTHORIZED.
- After a green push, next authorized action: WAIT FOR OWNER APPROVAL.
