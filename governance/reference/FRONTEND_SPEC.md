# FRONTEND_SPEC.md — Vantara Streamlit Dashboard Contract

## Role

Streamlit is a business-facing analytics UI for non-technical Marketing/Retention users. It consumes FastAPI; it does not import trained models or implement independent scoring logic.

Use Plotly for interactive dashboard visualizations. EDA notebooks may use Matplotlib/Seaborn as required by the PRD.

## Required views

### 1. Executive Overview

Show concise customer-health/revenue summary using persisted/scored data. Avoid ML jargon as the primary language.

### 2. Customer Segments

Required filters:

- segment;
- country;
- value tier.

Show business-readable segment names and summary stats; include interpretable segment visualization/PCA view where useful.

### 3. Churn Risk Leaderboard

Default priority surfaces high-value + high-risk customers first.

Use locked retention-priority formula:

```text
normalized(churn_probability) * normalized(predicted_clv_180d)
```

Show underlying churn probability and predicted value so users can understand the composite.

### 4. Customer Explorer

Search by Customer ID and show:

- churn risk;
- predicted 180-day value;
- next-purchase probability;
- next category;
- segment;
- anomaly score/flag;
- recommendations;
- feature importance / plain-language explanation;
- required individual SHAP explanation access.

### 5. Revenue Trends + Forecast

Show historical sales/net-revenue trends and a simple forecast overlay using the locked Holt-Winters approach with documented fallback.

### 6. Batch Scoring

Upload canonical CSV, display validation errors clearly, call API batch scoring, show results and enable required CSV/PDF downloads.

### 7. Model Insights / Explainability

Expose global feature importance/SHAP, model metadata/comparison summary as appropriate, and PDP/representative explanation artifacts without overwhelming business users.

## UI rules

- no model jargon without plain-language labels/tooltips;
- risk/value units and prediction horizons must be visible;
- never label the autoencoder output as confirmed fraud;
- never call markdown affinity proven discount sensitivity;
- never imply a forecast is guaranteed;
- clearly distinguish historical vs predicted values.

## Performance

Full segment view target: <3 seconds under the documented local benchmark.

Use API-side/persisted aggregates and safe caching for expensive requests. Do not solve performance by bypassing API and importing model/DB internals directly.

## Downloadable reports

CSV and PDF outputs must state scoring timestamp/model version where practical and preserve enough identifiers for auditability.

## Error handling

API unavailable, invalid CSV, unknown customer and missing explanation/artifact errors should be user-readable and non-destructive. Do not show raw tracebacks as the normal UX.
