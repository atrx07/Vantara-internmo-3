# MODELING_SPEC.md — Vantara Model, Evaluation and XAI Contract

## 1. Experiment invariants

- random seed: 42 where applicable;
- one persisted 70/15/15 customer split;
- training-only hyperparameter search/CV;
- validation for model/threshold choice;
- held-out test exactly once after choices freeze;
- every run logs parameters, metrics and training time;
- final comparison table is generated from logs/evidence, never retyped by hand.

MLflow uses a local backend. MLflow development stores are not a production runtime dependency.

## 2. Churn models

Required classical models:

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. XGBoost
5. LightGBM
6. SVM

Required deep comparison:

7. feed-forward PyTorch ANN on same final churn feature schema.

### Preprocessing

Scale-sensitive family (Logistic Regression, SVM, ANN): imputation as required -> training-only approved outlier treatment -> StandardScaler -> approved categorical encoding.

Tree family: no StandardScaler; persist deterministic categorical/preprocessing contract.

All inference uses serialized training-fitted transforms.

### Imbalance

Primary: class weights / model-equivalent positive weighting. SMOTE only as explicit training-fold experiment and never validation/test.

### Tuning

Prefer bounded RandomizedSearchCV rather than exhaustive giant grids. XGBoost/LightGBM may use early stopping on validation during final fitting.

### Metrics

Report on required evaluation sets:

- Accuracy
- Precision
- Recall
- F1
- ROC-AUC
- Confusion Matrix

Target final held-out ROC-AUC >=0.80 and recall >=0.70.

### Production selection

Select using validation evidence. Prefer models meeting recall >=0.70, then highest ROC-AUC. If within 0.01 ROC-AUC, prefer recall, latency, explainability and operational simplicity in that order.

Choose classification threshold on validation only, using F2-oriented selection while preserving recall target. Freeze threshold before final test.

## 3. ANN

PyTorch architecture family:

```text
Input
 -> Linear(128) -> BatchNorm -> ReLU -> Dropout(0.30)
 -> Linear(64)  -> BatchNorm -> ReLU -> Dropout(0.20)
 -> Linear(1)
```

Use BCEWithLogitsLoss with positive weighting, AdamW, early stopping, configured learning rate/batch/patience. Save train/validation loss curves.

## 4. CLV regression

Target: `clv_180d_target`, a 180-day forward net-revenue-based proxy.

Models:

- Ridge baseline;
- XGBRegressor primary candidate.

Use `log1p` target transform where justified; inverse-transform predictions before business metrics. Report MAE, RMSE, R². Target R² >=0.60.

Predicted CLV is used for prioritization/reporting, not as an input to churn model.

## 5. LSTM

Goal: probability of a valid positive purchase in next 30 days.

Training examples: rolling historical snapshots with complete future 30-day observation and customer-group isolation.

Sequence: last 20 valid invoice events, with features including:

- log1p order amount;
- gap days;
- dominant product category embedding.

Default family: single LSTM layer, hidden size around 64, category embedding around 8, dropout where effective, binary output head. Exact constants live in YAML.

Metrics: Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix. Use grouped CV by customer.

## 6. Autoencoder

Input: scaled numerical spending/behavior features.

Compact symmetric architecture (e.g. input -> 32 -> 8 latent -> 32 -> input) with MSE reconstruction loss.

Default anomaly threshold: 99th percentile of validation reconstruction error unless validation evidence requires a documented adjustment within the locked method.

Report reconstruction error distribution, threshold, flagged count/rate and top contributing reconstructed features. Never claim fraud classification accuracy without labels.

## 7. Segmentation

### K-Means

Search k using elbow/inertia + silhouette. Report Davies-Bouldin.

### GMM

Search component count using BIC primarily; also report silhouette and Davies-Bouldin.

Primary feature space should use interpretable standardized behavior features such as recency, frequency, monetary, basket, trend, gap variance, return rate, markdown proxy and seasonality.

Create business-readable labels based on profile statistics. PCA(2) is required for visualization only.

## 8. Product taxonomy

Historical canonical descriptions -> TF-IDF words/bigrams -> TruncatedSVD -> MiniBatchKMeans.

Candidate category counts: `{12,16,20,24,30}`. Choose using silhouette, cluster balance and interpretability; persist/freeze taxonomy.

## 9. Next-purchase category

For each eligible snapshot, find customer's next valid invoice. Target is category with highest merchandise value in that invoice; ties by quantity then deterministic category ID.

Model: LightGBM multiclass.

Report:

- macro-F1;
- Top-1 accuracy;
- Top-3 accuracy;
- comparison with most-popular-category baseline.

## 10. Recommender

Item-to-item implicit collaborative filtering:

- sparse customer x StockCode matrix;
- log1p quantity weighting;
- cosine similarity;
- returned/admin items excluded or appropriately penalized;
- default Top 5;
- new/sparse fallback = popular items in customer's segment.

Offline evaluation: leave-last-order-out with Recall@5, HitRate@5 and catalog coverage.

## 11. Explainability

### Production churn

Required full PRD XAI:

- global SHAP summary;
- force/local output for low-risk, borderline, high-risk customers;
- one LIME explanation on same customer as one SHAP example;
- written SHAP-vs-LIME agreement/divergence note;
- PDP for top 2–3 influential features;
- deterministic plain-language explanation template.

Explainer family follows actual production model; do not assume a tree winner beforehand.

### Other outputs

- CLV: SHAP for production regressor;
- next category: multiclass SHAP where technically practical;
- LSTM: event masking/perturbation effect explanation;
- autoencoder: per-feature reconstruction contribution;
- recommender: similar-item/affinity reason;
- segment: profile statistics.

## 12. Final test lock

The final-test command must be separate from routine tuning. Recommended final interface:

```bash
python -m src.models.final_evaluate
```

Before this command is run, `STATUS.md` must state that model/hyperparameter/threshold choices are frozen and owner has approved Step 06 final evaluation.

After final test is run, changing modeling choices requires owner approval and explicit invalidation/re-baselining of final evidence; Codex may not casually rerun the held-out test.
