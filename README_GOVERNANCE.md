# Vantara Governance Pack v1.0

This pack is intended to be placed at the repository root before Codex begins implementation.

## Start

1. Place the supplied `online_retail_II.xlsx` where Codex can access it; the canonical project destination will be `data/raw/online_retail_II.xlsx`.
2. Give Codex this repository/workspace plus the Git remote URL in chat.
3. Tell Codex to execute **Step 00 only** from `NEXT_STEPS.md`.
4. Do not approve Step 01 until the Step 00 report is correct.

## Codex starter message

Paste this, replacing the placeholder:

```text
Execute Vantara Step 00 only. Read AGENTS.md and the entire governance pack exactly as instructed before doing anything else. The Git remote URL for this project is: <REMOTE_URL>. Inspect and verify the supplied PRD, governance lock, dataset, repository and remote. Do not implement product code and do not push in Step 00. Update only STATUS.md and NEXT_STEPS.md, report using the required step report template, then stop and wait for my approval for Step 01.
```

## Immutable vs mutable

Codex may modify only `STATUS.md` and `NEXT_STEPS.md` among governance/control files. All other governance/reference/source/tool files are locked and verified by `governance/REFERENCE_LOCK.json`.
