# Vantara Customer Behavior Prediction Platform

Vantara is a governed, batch-oriented retail analytics prototype. Milestone M1 provides immutable ingestion, cleaning, point-in-time targets/features, a shared customer split, frozen product artifacts, reproducible EDA, and a versioned churn feature schema. STEP 04 adds validation-stage classical intelligence, and STEP 05 adds PyTorch ANN, grouped rolling-snapshot LSTM, and reconstruction-anomaly experiments without accessing the final held-out test.

## Development setup

Requirements:

- Python 3.11.x
- `venv` and `pip`
- the exact UCI Online Retail II workbook described in `data/raw/README.md`

Create and activate a local environment on Windows:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Place the source workbook at `data/raw/online_retail_II.xlsx`, verify its hash, and run the read-only ingestion check:

```powershell
python -m src.data.loader --config config/config.yaml
```

Run the STEP 02 data foundation after validating the immutable source:

```powershell
python -m src.data.step02_pipeline --config config/config.yaml
```

The command writes ignored, reproducible local outputs under `data/interim/` and `data/processed/`. It removes exact duplicate lines only; all other source rows remain auditable through explicit quality flags. Predictive features use transactions strictly before their configured snapshot cutoff, and population-learned artifacts use training customers only.

Run the complete single-command M1 pipeline:

```powershell
python -m src.pipeline --config config/config.yaml
```

This regenerates the data foundation, tracked EDA/data-freeze evidence under `reports/`, and the canonical `models_artifacts/churn_feature_schema.json`. The required `notebooks/01_eda.ipynb` and `notebooks/02_feature_engineering.ipynb` are executed analysis consumers; they do not own production transformations.

Run the STEP 04 training and product-intelligence pipeline after the M1 outputs exist:

```powershell
python -m src.models.step04_pipeline --config config/config.yaml
```

The command performs five-fold training-only model searches, validation-only evaluation, local MLflow logging, serialization, and artifact-reload checks. Tracked evidence is written under `reports/modeling/`, serving candidates under `models_artifacts/step04/`, and the required executed `notebooks/03_model_experiments.ipynb` remains a consumer of source-generated evidence. The local `mlruns/` store is intentionally ignored. Production churn selection, threshold freezing, explainability, and the one-time final held-out evaluation remain deferred to STEP 06.

Run the STEP 05 PyTorch training pipeline after the STEP 02 processed inputs exist:

```powershell
python -m src.models.step05_pipeline --config config/config.yaml
```

The command trains the governed ANN on the exact frozen 47-feature churn schema, creates controlled monthly rolling 20-event sequences for 30-day purchase prediction, executes five customer-grouped LSTM folds with fold-local preprocessing, and trains the behavioral autoencoder. It writes tracked loss/metric/reconstruction evidence under `reports/deep_learning/`, three safe-reload PyTorch artifacts under `models_artifacts/step05/`, and updates the executed model-experiments notebook as an evidence consumer. Test customers are excluded from model fitting, validation, rolling-sequence construction, and reported metrics.

Run the current quality gates:

```powershell
python governance/tools/verify_reference_lock.py
ruff check .
black --check .
pytest -q
```

The raw workbook, local environments, secrets, database files, generated datasets, caches, logs, and MLflow runtime stores are intentionally excluded from Git.

## Governance

Read `AGENTS.md` before making changes. The full build sequence and immutable contracts are under `governance/reference/`. Do not begin a roadmap step without explicit owner approval.
