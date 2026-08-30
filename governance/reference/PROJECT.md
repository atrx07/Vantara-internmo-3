# PROJECT.md — Vantara Project Contract

## Product

**Vantara Retail Solutions — Customer Behavior Prediction Platform**

A four-week internship/training prototype that converts historical retail transactions into customer-level predictive intelligence for Marketing and Retention users.

## Core business outputs

The platform must provide:

- churn risk scores;
- predicted 180-day customer value proxy;
- next-purchase probability;
- next-purchase category;
- personalized product recommendations;
- customer segments with business-readable names;
- anomaly flags for unusual spending behavior;
- global and customer-level explanations;
- cohort/revenue reporting and a simple forecast overlay.

The PRD frames the system as both a working internal analytics prototype and a reference implementation of applied ML capability.

## Primary users

- Marketing analysts
- Retention analysts
- Internal evaluator/mentor
- Engineer/trainee taking over the project

The dashboard must be understandable without ML expertise.

## Product goals

1. Replace manual rule-only customer prioritization with evidence-based predictive scoring.
2. Prioritize retention attention using both churn risk and expected value.
3. Demonstrate a correct, reproducible end-to-end ML/DL lifecycle.
4. Keep predictions explainable.
5. Make the system locally deployable and handover-ready.

## Non-goals

The prototype does not include:

- real-time Kafka/event ingestion;
- production multi-tenant authentication/RBAC;
- production-grade CI/CD;
- live campaign A/B testing;
- transformer sequence models;
- cloud dependency;
- a production commerce integration.

## Quality philosophy

Correctness and auditability beat inflated metrics. A missed success metric with a documented, reproducible reason is acceptable under the PRD; manipulated evaluation is not.

The final repository must make it possible for a new engineer to understand what data was used, how targets/features were defined, how models were compared, how predictions are served, and how the final evidence was produced.
