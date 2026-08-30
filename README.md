# Vantara Customer Behavior Prediction Platform

Vantara is a governed, batch-oriented retail analytics prototype. The repository is built through owner-approved roadmap steps; current implementation scope is limited to STEP 01 raw ingestion and validation.

## STEP 01 development setup

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

Run STEP 01 quality gates:

```powershell
python governance/tools/verify_reference_lock.py
ruff check .
black --check .
pytest -q
```

The raw workbook, local environments, secrets, database files, generated datasets, caches, logs, and MLflow runtime stores are intentionally excluded from Git.

## Governance

Read `AGENTS.md` before making changes. The full build sequence and immutable contracts are under `governance/reference/`. Do not begin a roadmap step without explicit owner approval.
