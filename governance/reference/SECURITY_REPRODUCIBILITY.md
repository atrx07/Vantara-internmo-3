# SECURITY_REPRODUCIBILITY.md — Vantara Safety and Reproducibility Contract

## Secrets

Never commit:

- database passwords;
- tokens;
- Git credentials;
- API keys;
- private environment values.

Use `.env` locally and keep it ignored. Commit `.env.example` containing placeholders only.

## Runtime configuration

Approved runtime variables include:

```text
DATABASE_URL
MODEL_ARTIFACT_DIR
LOG_LEVEL
APP_ENV
```

YAML contains non-secret defaults/experiment configuration. Environment variables override approved runtime defaults.

## Dependency reproducibility

Step 01 shall create/pin a validated `requirements.txt`. Exact versions are chosen after successful compatibility testing with Python 3.11 and then frozen.

No unpinned "latest" dependency installation may remain in final setup instructions.

## Randomness

Seed 42 for NumPy, scikit-learn and PyTorch where supported. Record nondeterministic GPU limitations if complete determinism is unavailable.

## Experiment traceability

Every important model run records:

- dataset hash;
- feature schema/version;
- split version;
- seed;
- hyperparameters;
- metrics;
- training time;
- relevant model/preprocessor artifact paths.

## Artifact compatibility

Serving code must validate enough metadata to prevent accidental use of a model with the wrong feature order/taxonomy/preprocessor.

## Privacy

The source dataset is public research data, but do not invent/store new personal identity fields or enrich customers from external sources. Treat Customer ID as an opaque identifier.

## Network/cloud

External network access may be used for package installation or source retrieval when needed, but the final local prototype must not require a paid/cloud model API. PostgreSQL/API/dashboard run locally through Docker Compose.

## Clean-environment reproduction

Final acceptance requires a documented clean-environment procedure covering:

1. clone repository;
2. obtain/place canonical dataset where required for retraining;
3. create environment/install pinned dependencies for development path;
4. run pipeline/training as documented if reproducing experiments;
5. run tests;
6. run Docker Compose serving stack using provided production artifacts/data initialization path.
