# GIT_WORKFLOW.md — Vantara Atomic Commit and Push Protocol

## 1. Goal

Move fast on `main` without sacrificing recoverability. Every remote change after Step 00 must be a green, coherent, auditable unit.

## 2. Step 00 remote binding

The owner supplies the remote URL directly in chat during Step 00.

Codex must:

- inspect whether repository is already initialized;
- inspect current branch and commits;
- inspect existing `origin`;
- safely add/verify `origin`;
- fetch remote refs;
- confirm/establish target branch `main` without overwriting unrelated history;
- never store credentials in files;
- never push during Step 00 unless separately ordered.

If an existing remote URL differs from owner-provided remote, STOP and ask rather than replacing it.

## 3. Main-only fast path

Use `main` directly unless owner explicitly changes branch strategy. No PR ceremony is required.

`main` must remain runnable/validated at every pushed commit.

## 4. Atomic commits

Default preference: one final coherent commit per roadmap step.

Multiple commits are allowed when the step naturally contains separable atomic units, but **every pushed commit must independently pass the relevant validation available at that point**.

No WIP/debug/failing commits on remote.

Suggested subjects:

```text
step-01: scaffold project and validate raw ingestion
step-02: implement cleaning snapshots and feature pipeline
step-03: complete EDA and freeze data foundation
...
```

## 5. Before each commit

1. inspect `git status --short`;
2. confirm only current-step files are included;
3. confirm immutable reference lock passes;
4. run required step validation;
5. update `STATUS.md` factually;
6. update `NEXT_STEPS.md` only when step state warrants it;
7. inspect diff for secrets/raw dataset/large accidental files;
8. commit with coherent message.

Because `STATUS.md` needs real commit hashes, when necessary use a two-stage factual entry approach without amending a pushed commit: record step evidence and commit subject before commit; after commit, add hash in the final state commit if the step uses multiple commits. Prefer designing state update so the final step commit can include all evidence without rewriting remote history.

## 6. Before push

- `git fetch origin`;
- verify local branch relationship to `origin/main`;
- if remote advanced unexpectedly, stop/report rather than force-push;
- run final required step validation against exact HEAD being pushed;
- verify `git status` clean except intentionally ignored local data;
- push normally.

Never use `--force`, `--force-with-lease`, destructive rebase, history reset, branch deletion, or credential embedding without explicit owner instruction.

## 7. After push verification

Record:

- remote alias and sanitized URL/host;
- branch;
- pre-step HEAD;
- every commit hash + subject created during step;
- every pushed ref/head (normally `refs/heads/main`);
- final local `HEAD`;
- final `origin/main` hash;
- confirmation local and remote head match;
- validation commands/results;
- working-tree status.

The step report must include all of these.

## 8. Governance pack tracking

The governance pack may be committed exactly as provided. Codex must never modify immutable references to make them fit the repo. The raw dataset remains ignored by default.

## 9. Large files

Before commit/push inspect file sizes. Do not accidentally push the raw XLSX, local DB, MLflow stores or oversized model artifacts. Follow `FILE_HANDLING.md` artifact-size gate.

## 10. Tags/releases

Do not create a release tag until Step 09 is accepted by the owner unless explicitly requested.
