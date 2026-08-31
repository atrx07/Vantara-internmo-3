# AGENTS.md — Vantara Codex Governance Controller

## 0. Purpose and authority

This file is the primary execution contract for any coding agent working on **Vantara Retail Solutions — Customer Behavior Prediction Platform**. It governs workflow, permissions, scope, validation, Git behavior, file ownership, step transitions, and handoff discipline.

The project is based on the supplied internship PRD `governance/source/Vantara_requirements.docx`. The PRD remains the primary product-requirements authority. The governance pack converts that PRD into an implementation-safe, audited, explicit build contract.

### Source precedence

When instructions appear to conflict, use this order:

1. The project owner's latest explicit instruction in the active conversation.
2. The supplied internship PRD for product requirements and acceptance intent.
3. This `AGENTS.md` for workflow, permissions, governance, Git, approval gates, and reference immutability.
4. `governance/reference/DECISIONS_LOCK.md` for choices deliberately locked where the PRD allowed alternatives or was underspecified.
5. `governance/reference/REQUIREMENTS.md` and `governance/reference/PRD_TRACEABILITY.md`.
6. `governance/reference/ARCHITECTURE.md` and the subsystem specifications.
7. `governance/reference/ROADMAP.md`.
8. `STATUS.md` and `NEXT_STEPS.md` for current observed state and immediate continuation only.
9. Existing implementation reality.
10. README or descriptive implementation documentation.

Existing code never overrides a locked requirement. If code conflicts with a higher authority, fix the code.

## 1. Mandatory Step 00 — read before building

No implementation may begin until **Step 00 — PRD + Governance + Repository Scan** is completed and the owner explicitly approves Step 01.

During Step 00 the agent must:

1. Read this file completely.
2. Read `governance/reference/GOVERNANCE_INDEX.md` and every immutable reference file listed there.
3. Inspect the supplied PRD itself at `governance/source/Vantara_requirements.docx` and compare it against `PRD_BASELINE.md` and `PRD_TRACEABILITY.md`.
4. Verify the governance lock with `python governance/tools/verify_reference_lock.py`.
5. Locate the supplied dataset or the owner-provided dataset path and verify it against `DATASET_MANIFEST.md` without modifying it.
6. Inspect the repository tree, Git status, current branch, existing commits, tracked/untracked files, ignored files, and any existing implementation.
7. Receive the Git remote URL directly from the owner in chat. Never write credentials or private tokens into repository files.
8. Bind or verify `origin` safely. Fetch remote refs if available. Do not overwrite a different existing remote, rewrite history, delete branches, or force-push.
9. Confirm the target branch is `main`. If the remote's established default branch is incompatible or the remote contains unrelated history, stop and report instead of rewriting it.
10. Update only `STATUS.md` and `NEXT_STEPS.md` with observed Step 00 facts.
11. Report the scan, discrepancies, remote/branch state, dataset verification, governance-lock result, and the exact proposed Step 01.
12. **STOP and wait for explicit owner approval.**

Step 00 performs no product implementation and no Git push unless the owner separately orders one.

## 2. Immutable reference contract

The coding agent MUST NOT edit, rewrite, rename, delete, regenerate, auto-format, or silently "improve" any immutable governance/reference file.

Immutable content includes:

- this `AGENTS.md`;
- all files under `governance/reference/`;
- all files under `governance/source/`;
- all files under `governance/tools/`;
- `governance/REFERENCE_LOCK.json`;
- `README_GOVERNANCE.md`;
- `PACK_MANIFEST.md`.

The only governance state files the agent may edit during normal project execution are:

- `STATUS.md` — factual observed state, validation evidence, commit/push records, blockers;
- `NEXT_STEPS.md` — exactly one authorized next step / continuation queue.

No other "state", "decision", "notes", "scratch governance", or competing roadmap file may be created to bypass this rule.

If an immutable reference appears wrong, incomplete, contradictory, or incompatible with reality:

1. do not edit it;
2. stop before building around the conflict;
3. report the exact conflict, affected file/section, implementation consequence, and recommended amendment;
4. wait for the owner to provide an updated governance pack or explicit replacement instruction.

The agent may never weaken tests, acceptance thresholds, leakage rules, source precedence, or scope restrictions merely to make progress easier.

## 3. Scope lock

Build only the PRD-required prototype and the implementation choices explicitly locked in this pack.

### Required product surface

The final system must provide:

- reproducible raw-to-feature data pipeline;
- churn prediction;
- 180-day forward revenue-based CLV proxy prediction;
- next-purchase probability through an LSTM sequence model;
- next-purchase category prediction;
- product recommendations;
- customer segmentation;
- anomaly detection through an autoencoder;
- global and individual explainability;
- FastAPI REST service;
- PostgreSQL persistence;
- Streamlit business dashboard;
- Docker Compose local deployment;
- tests, benchmarks, diagrams, README, final report, math appendix, and handover evidence.

### Explicitly forbidden unless the owner changes scope

Do not add:

- Transformer sequence models;
- Kafka, event streaming, or real-time ingestion infrastructure;
- multi-tenant authentication or RBAC;
- production-grade CI/CD or GitHub Actions merely for convenience;
- A/B testing framework;
- cloud-only services or paid APIs;
- external LLM calls;
- MongoDB in place of PostgreSQL;
- TensorFlow/Keras in place of PyTorch;
- KNN in place of SVM;
- DBSCAN in place of GMM;
- React in place of Streamlit;
- CatBoost unless the owner explicitly authorizes the optional comparison;
- the optional Customer Personality Analysis dataset unless the owner explicitly authorizes the stretch goal;
- extra microservices, queues, orchestration systems, vector databases, or unnecessary frameworks;
- arbitrary feature/target changes made only to improve metrics.

## 4. Locked execution model

The project is executed as three milestones and ten gated steps including Step 00:

- **M0 Governance:** Step 00
- **M1 Data Ready:** Steps 01–03
- **M2 Intelligence Ready:** Steps 04–06
- **M3 Product Ready:** Steps 07–09

`governance/reference/ROADMAP.md` is the canonical roadmap. Do not merge, split, skip, reorder, or silently expand steps.

### Approval gate

At the end of **every** step:

1. finish only that step's authorized scope;
2. run the step's required validation and tests;
3. update `STATUS.md` with facts and evidence;
4. update `NEXT_STEPS.md` so it points to the next step but marks it `WAITING_FOR_OWNER_APPROVAL`;
5. from Step 01 onward, create and push only green atomic commit(s) as defined in `GIT_WORKFLOW.md`;
6. report exactly what was done, validation results, changed files, commits, pushed refs/heads, current HEAD, blockers, status, and the proposed next step;
7. **STOP**;
8. wait for explicit owner approval before beginning the next step.

Approval for one step does not imply approval for later steps. Never auto-chain steps.

### Commit and push authorization

Explicit owner approval of a roadmap step authorizes the agent to create and normally push that step's green atomic commit(s) without requesting a separate commit or push approval in chat. Before pushing, the agent must still complete the required validation, fetch and compare the remote, confirm raw-data and secret protections, and satisfy every Git safety rule. This standing authorization does not authorize a later roadmap step, a force-push, a history rewrite, or bypassing a platform-required security approval.

## 5. Git safety contract

The fast-path branch is `main`; no feature-branch/PR workflow is required for this solo internship project.

From Step 01 onward:

- each pushed commit must be coherent and validated;
- do not push failing or knowingly incomplete code;
- do not force-push;
- do not rewrite or amend already-pushed commits;
- do not delete remote branches/tags;
- do not commit credentials, `.env`, raw database volumes, caches, notebooks outputs, MLflow run stores, or temporary artifacts;
- do not commit the raw XLSX dataset unless the owner explicitly orders it;
- preserve user changes and unknown existing work;
- before changing a file, inspect it and understand its role;
- before pushing, fetch/compare remote state and stop on divergence that cannot be resolved safely without owner choice.

Every step report must list **all commit hashes and all pushed refs/heads created or advanced during that step**.

## 6. Truthful validation rule

Never claim a test, benchmark, build, model metric, coverage threshold, Docker startup, API response, dashboard view, or clean-clone run passed unless it was actually executed successfully.

A code path existing is not evidence of acceptance.

If a PRD metric is missed honestly, preserve the real result and document the reason. Never manipulate labels, use future information, alter test partitions, cherry-pick test runs, or change evaluation semantics to manufacture compliance.

## 7. Point-in-time and held-out-test safety

Target leakage prevention is a non-negotiable project invariant.

- Features for a snapshot may use only transactions strictly before its cutoff timestamp.
- Labels use only the defined future target window.
- Training-learned transforms may not fit on validation/test customers.
- A customer may never cross train/validation/test partitions.
- Rolling LSTM snapshots remain grouped by customer.
- SMOTE, if experimentally used, may touch training folds only.
- The final held-out test partition is evaluated exactly once after model, hyperparameter, and threshold choices are frozen.

If any code or notebook violates these invariants, the affected model evidence is invalid and must be regenerated.

## 8. File and data preservation

Follow `FILE_HANDLING.md` exactly.

Key rules:

- supplied PRD and dataset are read-only source inputs;
- raw source data is never cleaned in place;
- intermediate/processed data is written to the documented directories;
- notebooks are consumers of source modules, not the production implementation;
- do not create duplicate "final", "final2", "new", "fixed", or ad-hoc parallel source trees;
- use configuration and deterministic artifact paths;
- never save generated files over source/reference inputs;
- before destructive operations, stop unless the roadmap explicitly authorizes them.

## 9. Codex implementation discipline

Prefer explicit, boring, testable Python over clever abstractions.

- public functions/classes: type hints and docstrings;
- structured logging instead of `print()` in production paths;
- paths/hyperparameters/thresholds in YAML or approved environment variables;
- deterministic random seed `42` wherever applicable;
- reusable `src/` functions; notebooks never own unique production logic;
- fitted preprocessors are serialized and reused unchanged at inference;
- frontend does not import model artifacts directly;
- no silent architecture replacement;
- no TODO placeholders accepted as completed requirements;
- no broad refactor outside the current step unless required to make the current step correct.

## 10. Definition of Done

Vantara is complete only when all items in `ACCEPTANCE_MATRIX.md` are satisfied or any missed PRD success metric is explicitly documented with genuine evidence and a reason, as the PRD permits.

Until Step 09 final acceptance is complete, `STATUS.md` must report the project as incomplete.
