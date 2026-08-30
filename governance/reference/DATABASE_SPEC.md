# DATABASE_SPEC.md — Vantara PostgreSQL Contract

## Role

PostgreSQL is the serving persistence layer for customer/scoring/segment state. It is not a replacement for the immutable training workbook and processed training artifacts.

Use SQLAlchemy 2.x and Alembic migrations. Runtime connection information comes from environment variables.

## Required core entities

### `customers`

Minimum concepts:

- `customer_id` primary/business key;
- country/current summary fields if needed for serving;
- timestamps/metadata as useful.

### `transactions`

Minimum concepts:

- surrogate primary key;
- invoice identifier;
- StockCode;
- customer foreign key when known;
- quantity;
- price;
- invoice timestamp;
- quality/product flags needed by scoring path.

Avoid pretending anonymous source rows have a customer FK.

### `predictions`

Store versioned scoring output including:

- customer_id;
- model version;
- scored/as-of timestamp;
- churn probability/label;
- selected churn threshold/version;
- predicted 180-day value;
- next-purchase probability;
- next-category output where available;
- anomaly score/flag where available.

### `segments`

Store:

- customer_id;
- segment identifier;
- business-readable segment name;
- model/version;
- assignment timestamp.

### `recommendations` — approved extension

Store ranked recommendation output where useful:

- customer_id;
- stock_code;
- rank;
- score;
- recommendation/version timestamp.

## Integrity and indexing

Use appropriate foreign keys/unique constraints. Add indexes based on actual dashboard/API query paths, especially customer ID, segment, country/value/risk filters, scored timestamp and risk sorting where useful.

Do not add indexes speculatively in bulk; validate with query path/performance evidence.

## Migrations

- all schema changes through Alembic once DB implementation begins;
- migration files are committed;
- do not hand-edit production DB schema as the only source of truth;
- Step 09 clean-compose test must create/migrate a fresh DB successfully.

## Secrets and volumes

- `DATABASE_URL` from environment;
- `.env` ignored;
- Docker PostgreSQL data volume ignored/not committed;
- no passwords in README/governance/config.

## Seed/load behavior

Provide a deterministic local path to load required scored/customer data for dashboard evaluation without forcing full model retraining on every compose startup.
