# DATA_CONTRACT.md — Vantara Data Pipeline Contract

## 1. Canonical source mapping

Immediately after loading, map source names to:

```text
Invoice      -> invoice
StockCode    -> stock_code
Description  -> description
Quantity     -> quantity
InvoiceDate  -> invoice_date
Price        -> price
Customer ID  -> customer_id
Country      -> country
```

Canonical semantic types:

```text
invoice       string
stock_code    string
description   string / null
quantity      integer
invoice_date  datetime
price         numeric
customer_id   string / null
country       string
```

Identifiers are never treated as numerical magnitude.

## 2. Raw preservation

- never edit or save over `online_retail_II.xlsx`;
- load both sheets before cleaning;
- combine and sort chronologically;
- preserve a reproducible source hash in metadata;
- write cleaned transaction data to `data/interim/` as Parquet.

## 3. Cleaning rules

### Missing Customer ID

- retain in product-level/audit transaction table;
- exclude from customer-level modeling where identity is required;
- never impute/fabricate IDs.

### Returns/cancellations

- negative quantity and/or cancellation invoice indicators must be explicitly represented;
- create flags such as `is_return` / `is_cancelled_invoice`;
- returns remain available as predictive behavior;
- returns alone do not count as positive purchase events.

### Non-positive prices

- preserve/flag adjustment rows in interim audit data;
- exclude invalid/non-merchandise price rows from monetary model calculations according to documented rules;
- never silently turn non-positive values positive.

### Duplicates

Remove only exact duplicate invoice-line records. Preserve legitimate repeated SKU purchases in the same invoice.

### Product descriptions

Normalize case/whitespace/formatting. Create deterministic canonical description lookup keyed by StockCode using fitting history. Recommended tie-break: modal normalized description, then most recent candidate, then lexical order.

### Administrative StockCodes

Create explicit `is_product` / `is_administrative_line` rule using a versioned configuration list/pattern audit. Exclude administrative lines from product taxonomy, affinity, next-category targets, recommender interactions and LSTM category events.

### Outliers

Detect quantity/price outliers using IQR plus domain rules. Create flags; do not blindly delete wholesale behavior. Separate likely data-quality errors from legitimate extreme orders. Produce an outlier audit artifact.

## 4. Validation gates

Before feature engineering, validate at least:

- exact required source columns;
- accepted canonical dtypes/coercion behavior;
- non-empty workbook/sheets;
- date range sanity;
- null-rate thresholds from config;
- duplicate statistics;
- price/quantity invalid-rate statistics;
- chronological order after merge;
- source hash recorded.

Critical schema/date failures stop the pipeline with non-zero exit.

## 5. Snapshot contract

`CustomerSnapshot(customer_id, cutoff_timestamp)` is the core supervised-sample abstraction.

Feature builders receive only history where:

```text
invoice_date < cutoff_timestamp
```

Label builders receive the defined future window separately.

No helper may default to "all transactions" when building a predictive snapshot.

## 6. Canonical cutoffs

Derive from actual valid source data:

```text
observation_end = max(invoice_date)
churn_cutoff = observation_end - 90 days
clv_cutoff = observation_end - 180 days
```

Do not hard-code calendar dates in model code.

## 7. Split contract

Create one customer-level split table once:

```text
train 70%
validation 15%
test 15%
seed 42
```

Persist under `data/processed/`. All model families consume this same partition contract.

A customer must never appear in more than one global partition.

## 8. Population-fit rule

Anything that learns population statistics is fitted on training customers/history only, including:

- scalers;
- imputers;
- encoders;
- winsorization thresholds;
- StockCode frequency encodings;
- product taxonomy;
- reference prices for markdown proxy;
- engagement percentile transforms;
- feature-selection/VIF decisions used to create the model input contract.

Validation/test are transform-only.

## 9. Required pipeline command

By Step 03, implement a single documented command:

```bash
python -m src.pipeline
```

It must run source validation -> cleaning -> snapshot/feature generation -> processed outputs without notebook intervention.

## 10. Data output policy

Recommended deterministic outputs include:

```text
data/interim/transactions_clean.parquet
data/processed/customer_features_churn.parquet
data/processed/customer_features_clv.parquet
data/processed/customer_split.parquet
```

Exact additional filenames may be implementation-level choices, but no duplicate competing "final" datasets are allowed.
