# FEATURE_CONTRACT.md — Required Vantara Feature Definitions

## Principle

The complete customer feature table must contain the PRD-required business features. A model may use a documented subset to avoid multicollinearity/redundancy. "Required to compute" does not mean "blindly feed every feature into every model."

Each feature must have a one-line predictive/business justification in generated documentation.

## Required feature families

### RFM and monetary

- `recency_days` — days since last valid positive purchase before cutoff;
- `frequency_orders` — count of distinct valid positive purchase invoices before cutoff;
- `gross_spend` — positive merchandise spend;
- `net_spend` — positive spend adjusted for attributable returns where possible;
- `avg_order_value` — valid historical order spend average;
- historical customer value/revenue summary.

### Basket behavior

- average basket units;
- average distinct products per order;
- unique product count.

### Purchase timing / trend

- customer tenure days;
- mean interpurchase gap;
- variance of interpurchase gap;
- purchase-frequency trend over a fixed recent period, implemented as a time slope using only pre-cutoff history.

### Seasonality

A deterministic seasonal purchase concentration feature, e.g. largest quarterly share of valid orders, with definition fixed in config/docs.

### Product affinity

Full product-category affinity vector derived from the frozen taxonomy. Preserve full vector in feature table. For scale-sensitive linear models, drop/reference one category if required to avoid exact dependency.

### Return behavior

A documented return rate such as absolute returned quantity divided by positive purchased quantity, with zero-denominator handling.

### Markdown affinity proxy

Because source data lacks a promotion flag:

1. historical reference price per product = training-history median valid price;
2. at least 5 eligible observations;
3. `markdown_like = price <= 0.90 * reference_price`;
4. customer proxy = markdown-like eligible orders / eligible orders.

Name/report it as a proxy, not causal sensitivity.

### Engagement score

Interpretable 0–100 RFM-based score. Default audited weighting:

- 40% inverted recency percentile;
- 30% frequency percentile;
- 30% monetary percentile.

Percentile mapping is fitted on training population only.

### StockCode-derived frequency encoding

To satisfy high-cardinality product handling without target leakage, derive training-only product popularity/frequency values and aggregate them at customer level, e.g. mean/median/max product popularity and rare-product share.

## Model input contract

Before finalizing churn feature set:

1. compute correlation analysis;
2. compute VIF on appropriate numerical design matrix;
3. document redundant/derived relationships;
4. select one final churn feature schema;
5. persist feature names/order/version;
6. use exactly that schema for all six classical churn models and the ANN, aside from model-specific preprocessing.

Do not remove a required business feature from the feature table merely because it is excluded from a model.

## Leakage tests

At minimum prove:

- future transaction insertion does not change historical features;
- future transaction insertion does not change taxonomy/reference prices for an already-fitted snapshot;
- target-window transactions do not enter model features;
- test customers do not change fitted percentiles/scalers/encoders;
- split membership remains disjoint.
