# DATASET_MANIFEST.md — Supplied Online Retail II Workbook

## Canonical supplied file

Expected local project path:

```text
data/raw/online_retail_II.xlsx
```

Source dataset: UCI Machine Learning Repository — **Online Retail II**.

Supplied conversation file identity verified during governance creation:

- filename: `online_retail_II.xlsx`
- size: `45,622,278` bytes
- SHA-256: `bcbe73b35f5b7babf197fb0cb983a11f5d9ff929078d4aa53d171b1f2df2e980`

Expected sheets and data-row counts:

| Sheet | Data rows | Columns |
|---|---:|---:|
| `Year 2009-2010` | 525,461 | 8 |
| `Year 2010-2011` | 541,910 | 8 |
| Combined | 1,067,371 | 8 |

Expected source columns:

```text
Invoice
StockCode
Description
Quantity
InvoiceDate
Price
Customer ID
Country
```

The PRD describes transactions spanning 01 Dec 2009 through 09 Dec 2011 and approximately 5,900 identified customers.

## Step 00 verification

Codex must verify, without modifying the workbook:

1. file exists at owner-provided/canonical path;
2. SHA-256;
3. workbook sheet names;
4. headers;
5. workbook can be opened read-only;
6. row counts are consistent with this manifest.

If the owner provides the file outside the repository, Step 01 may copy it into `data/raw/online_retail_II.xlsx` as a byte-for-byte copy. Do not transform it during copy.

## Git rule

The raw XLSX is local source data and should be ignored by Git by default. The repository must contain a `data/raw/README.md` explaining the authoritative UCI source and expected placement/hash. Do not commit the 45 MB workbook unless the owner explicitly directs it.

## Mismatch rule

A hash/structure mismatch is not permission to substitute another similarly named dataset. Stop and report. The PRD explicitly warns not to substitute the older single-year Online Retail dataset.
