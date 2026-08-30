# BACKEND_SPEC.md — Vantara FastAPI Contract

## Role

FastAPI is the only supported application scoring/service boundary consumed by the dashboard. It orchestrates reusable `src/` feature/scoring logic and PostgreSQL persistence.

## Required endpoints

Base prefix: `/api/v1` unless implementation evidence requires a harmless routing adjustment.

### Health

```text
GET /api/v1/health
```

Return service health and safe dependency status. Do not expose secrets.

### Model metadata

```text
GET /api/v1/models/metadata
```

Expose serving model names/versions, relevant cutoffs/thresholds/features and safe evaluation metadata.

### Single-customer prediction

```text
POST /api/v1/predict/customer
```

Preferred request contract:

```json
{
  "customer_id": "12345",
  "as_of_date": "optional ISO timestamp"
}
```

Server owns feature preparation from available transaction history and persisted artifacts. A client should not be required to manually provide engineered feature values.

If `as_of_date` is omitted, use latest valid supported observation/scoring date according to implementation contract.

Response should include available predictive outputs and model/version metadata, not merely churn.

### Batch scoring

```text
POST /api/v1/predict/batch
```

Accept CSV upload in a clearly documented canonical transaction-level schema. Validate input through Pydantic/explicit CSV schema validation, score consistently, persist results as applicable, and return downloadable result reference/response within safe size limits.

## Approved supporting endpoints

Useful read endpoints may include:

```text
GET /api/v1/customers/{customer_id}
GET /api/v1/customers/{customer_id}/explanation
GET /api/v1/customers/{customer_id}/recommendations
GET /api/v1/segments
GET /api/v1/segments/{segment_id}
GET /api/v1/analytics/revenue
```

Do not expand into a large REST surface without dashboard need.

## Artifact loading

Load production model/preprocessing/taxonomy artifacts once during application startup/lifespan where practical. Do not reload serialized models per request.

Validate artifact metadata compatibility at startup and fail loudly on missing/incompatible model contract.

## Input/output validation

- Pydantic request/response models;
- clear 4xx errors for malformed user input;
- structured server-side error logging;
- no stack traces/secrets in normal client errors;
- identifier fields treated as strings.

## Performance

Step 09 must benchmark warmed single-customer prediction and demonstrate/document p95 <400 ms or honestly report why missed.

Benchmark result must identify environment, request count and measured p50/p95/max.

## Testing

API tests must cover at least:

- health;
- metadata;
- valid single-customer prediction;
- unknown customer behavior;
- malformed input;
- batch schema validation;
- artifact-loading failure path where testable;
- DB persistence integration boundary;
- explanation/recommendation endpoint smoke paths used by dashboard.
