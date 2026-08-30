# BACKEND_FRONTEND_DB_CHECKLIST.md — Cross-Layer Integration Guard

This concise checklist exists because earlier agent-driven projects often failed at seams rather than individual files.

## Backend

- [ ] API owns inference orchestration.
- [ ] Training preprocessing is reused at inference.
- [ ] Artifacts load once/startup where practical.
- [ ] Required four PRD endpoints exist.
- [ ] Pydantic errors are clear.
- [ ] No secret/path hard-coding.
- [ ] DB writes are versioned/auditable.

## Database

- [ ] Alembic creates fresh schema.
- [ ] customer identifiers remain strings.
- [ ] predictions store model version/scored time.
- [ ] segments have readable labels.
- [ ] DB volume is not committed.
- [ ] dashboard query paths have appropriate indexes.

## Frontend

- [ ] Streamlit calls FastAPI only for product data/scoring.
- [ ] no direct model import.
- [ ] required filters/views exist.
- [ ] prediction horizons/units are visible.
- [ ] anomaly != confirmed fraud.
- [ ] markdown proxy != proven sensitivity.
- [ ] error states do not expose traceback as normal UX.

## Integration

- [ ] same model/version appears in API metadata and dashboard.
- [ ] customer explorer values match API response.
- [ ] batch CSV through UI matches API batch path.
- [ ] persisted predictions can be read after service restart.
- [ ] Compose startup order/health works on fresh volumes.
