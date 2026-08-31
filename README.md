# Vantara Customer Behavior Prediction Platform

Vantara is a governed, batch-oriented retail analytics prototype. The repository is built through owner-approved roadmap steps; Milestone M1 now provides immutable ingestion, cleaning, point-in-time targets/features, a shared customer split, frozen product artifacts, reproducible EDA, and a versioned churn feature schema.

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
