# CODE_STANDARDS.md — Vantara Python and Implementation Rules

## Python

Target Python 3.11.

All function signatures must have type hints. All public functions/classes require meaningful docstrings.

Prefer small explicit modules and pure/testable functions around transformations.

## Formatting/linting

Required final gates:

```bash
ruff check .
black --check .
```

Ruff must report zero errors at step exit once configured.

## Logging

Production pipeline/API code uses structured logging. Do not use `print()` for operational logs. Notebook display/output is exempt when appropriate for EDA.

Recommended levels:

- INFO: pipeline stages, model runs, API request summary without sensitive payloads;
- WARNING: recoverable validation/data-quality concerns;
- ERROR: failed validations, artifact incompatibility, unrecoverable stage failure.

## Configuration

Project behavior belongs in `config/config.yaml` when it is a path, horizon, model hyperparameter/range, threshold, seed, taxonomy search space or similar configurable value.

Approved environment variables override runtime configuration such as DB URL, artifact directory, log level and app environment.

No hidden duplicate constants scattered across notebooks/scripts.

## Error handling

- fail loudly on invalid required data/schema/artifacts;
- provide actionable exceptions/messages;
- do not `except Exception: pass`;
- do not silently fill missing required columns;
- do not continue using stale artifacts after metadata incompatibility.

## DataFrame practices

- avoid mutation chains that obscure provenance;
- make row-filtering/quality rules explicit;
- deterministic sorts before operations dependent on order;
- preserve identifiers as strings;
- avoid full-data copies inside loops on million-row transaction tables;
- prefer vectorized Pandas/NumPy operations where clear.

## ML practices

- use sklearn `Pipeline`/`ColumnTransformer` where appropriate;
- fit preprocessing only on training data;
- persist preprocessing with model artifacts;
- every experiment has deterministic seed metadata where supported;
- do not use test data for tuning/threshold decisions;
- no metric fabrication/rounding that hides failure.

## PyTorch practices

- deterministic seed setup where practical;
- device auto-selection without hard-coded GPU requirement;
- training and validation loops clearly separated;
- early stopping based on validation evidence;
- save model state/config/feature metadata needed for reload;
- no huge opaque framework wrapper merely to shorten code.

## SQL/API

- SQLAlchemy ORM/core usage with parameterization; no unsafe string-built SQL;
- Pydantic validation at HTTP boundary;
- no DB credentials in source;
- API routers remain thin; business logic stays reusable.

## Notebooks

Notebooks may explore and visualize, but must import production logic from `src/`. A notebook-only transformation/model required for final behavior is a failure.

Required filenames from PRD remain:

- `01_eda.ipynb`
- `02_feature_engineering.ipynb`
- `03_model_experiments.ipynb`

## Simplicity

Do not introduce abstraction layers unless they remove real duplication or enforce a required contract. No premature plugin systems, dependency injection frameworks, event buses or generalized model registries beyond what this project needs.
