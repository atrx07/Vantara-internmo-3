# STEP_REPORT_TEMPLATE.md — Required Codex Chat Report

Use this structure after every roadmap step.

```text
VANTARA STEP REPORT

Milestone:
Step:
Authorization received:
Branch:
Remote:
Pre-step HEAD:

1. Completed
- ...

2. PRD / governance coverage
- ...

3. Files / artifacts changed or created
- ...

4. Validation evidence
- <command/check> — PASS/FAIL/NOT_RUN — <key result>
- ...

5. Git evidence
Commits created this step:
- <hash> <subject>
- ...

Refs/heads pushed this step:
- <remote ref -> hash>

Final local HEAD:
Final origin/main HEAD:
Remote/local synchronized: YES/NO
Working tree clean: YES/NO

6. Known issues / blockers
- ...

7. State
STEP_COMPLETE_WAITING_FOR_APPROVAL / STEP_BLOCKED_WAITING_FOR_OWNER / STEP_FAILED_NOT_PUSHED

8. Proposed next step
Step NN — <title>: <one-line objective>

Waiting for owner approval before starting Step NN.
```

Step 00 omits commit/push entries or explicitly states `No push by governance rule`.
