# FILE_HANDLING.md — Vantara Source, Artifact and Save Rules

## 1. Never overwrite supplied source/reference files

Read-only inputs:

- `governance/source/Vantara_requirements.docx`;
- all governance/reference/tool files;
- supplied `online_retail_II.xlsx` once placed under `data/raw/`.

Generated outputs must never be written over these files.

## 2. Canonical destinations

| Content | Destination |
|---|---|
| Raw source workbook | `data/raw/online_retail_II.xlsx` |
| Raw-source placement instructions | `data/raw/README.md` |
| Clean transaction data | `data/interim/` |
| Feature/model-ready tables | `data/processed/` |
| Notebooks | `notebooks/` |
| Reusable implementation | `src/` |
| API | `api/` |
| Streamlit | `frontend/` |
| Serialized models/preprocessors/taxonomy | `models_artifacts/` |
| YAML config | `config/` |
| Tests | `tests/` |
| Generated evaluation evidence | `reports/` |
| Final diagrams/report | `docs/` |
| DB migrations | `migrations/` |
| One-off supported operational scripts | `scripts/` |

## 3. No duplicate-final-file pattern

Never create filenames/directories such as:

```text
final2.py
model_new.pkl
cleaned_final_final.csv
src_new/
backup_src/
frontend_fixed/
```

When a canonical file needs a correction, edit the canonical implementation file under the authorized step and let Git preserve history.

## 4. Git inclusion policy

### Commit

- source code;
- tests;
- migrations;
- config defaults without secrets;
- governance pack exactly as provided;
- README/docs;
- diagrams;
- final evidence tables/reports needed for submission;
- required model artifacts if within sane Git size limits;
- `.env.example` with placeholders only.

### Ignore by default

- `.venv/`;
- `.env`;
- Python caches;
- Jupyter checkpoints;
- raw XLSX dataset;
- interim/processed generated datasets unless owner explicitly wants them versioned;
- PostgreSQL volumes;
- MLflow local DB/run directory;
- temporary exports;
- coverage HTML/XML except final evidence if intentionally preserved;
- local logs;
- OS/IDE temp files.

## 5. Model artifact size gate

The PRD expects serialized required models. Before Step 09 final push:

1. inventory every required serialized artifact and size;
2. ensure no individual Git-tracked file approaches GitHub's 100 MB hard limit;
3. if any artifact >90 MB or the total serving/required artifact set becomes unreasonable for normal Git, STOP and ask owner before adding Git LFS or another distribution strategy;
4. do not silently add Git LFS or omit required deliverables.

## 6. Dataset handling

- do not download/substitute data without need when the supplied workbook is available;
- copying source workbook into `data/raw/` must be byte-for-byte;
- validate hash after copy;
- do not commit raw workbook by default;
- do not save cleaned rows back into XLSX;
- use Parquet for interim/processed tables.

## 7. PRD handling

The supplied PRD copied into `governance/source/` is reference evidence. Do not convert and replace it. Derived implementation docs may summarize it but must never delete the original.

## 8. State file handling

Only `STATUS.md` and `NEXT_STEPS.md` are mutable governance state.

`STATUS.md` = facts/history/evidence.

`NEXT_STEPS.md` = current authorized next action only; keep it concise and do not turn it into another roadmap.

## 9. Destructive operations

Before deleting, moving, bulk-renaming, schema-dropping, history-rewriting, or replacing generated artifacts required by earlier completed steps, confirm the action is explicitly part of the current roadmap step. Otherwise stop and ask.
