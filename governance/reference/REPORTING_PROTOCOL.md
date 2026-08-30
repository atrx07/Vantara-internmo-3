# REPORTING_PROTOCOL.md — Mandatory Codex Step Handoff

## Core rule

At the end of every roadmap step, Codex must stop and report. It may not begin the next step until the owner explicitly approves it.

## Required report content

### 1. Step identity

- milestone;
- step number/title;
- approval that authorized this step;
- branch/remote.

### 2. Work completed

Concise but complete summary of implementation, files/areas changed and artifacts produced.

### 3. PRD/governance coverage

List the requirement/PRD areas satisfied or advanced in this step. Note any deviation/blocker truthfully.

### 4. Validation evidence

For each required command/check:

```text
COMMAND / CHECK | PASS / FAIL / NOT_RUN | key result
```

Never claim inferred success.

### 5. Git evidence

From Step 01 onward list:

- pre-step HEAD;
- every commit created in step: full/short hash + subject;
- every ref/head pushed;
- final local HEAD;
- final `origin/main` HEAD;
- remote synchronization result;
- working tree cleanliness.

If no commit/push occurred because validation failed, say so explicitly.

### 6. Status

State one of:

- `STEP_COMPLETE_WAITING_FOR_APPROVAL`
- `STEP_BLOCKED_WAITING_FOR_OWNER`
- `STEP_FAILED_NOT_PUSHED`

### 7. Known issues

Carry forward unresolved issues. Do not bury them because the step is mostly successful.

### 8. Next step proposal

Name exactly one next roadmap step and its objective. Do not start it.

### 9. Stop line

End with a clear equivalent of:

```text
Waiting for owner approval before starting Step NN.
```

## State synchronization

Before sending the chat report, update:

- `STATUS.md` with observed facts/evidence;
- `NEXT_STEPS.md` with the single waiting next action.

Do not create a third status/next-step file.
